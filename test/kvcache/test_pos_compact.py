# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for inference-time position compaction (kvcache/pos_compact.py)."""

import torch

from keys_values.config import Config
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.kvcache.pos_compact import (
    _rotate_half_apply,
    compact_rope_positions,
    set_position_compaction,
)
from keys_values.attention.base import DefaultKeysAndValues
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.pos_encoding import LinearPositionEncoding
from keys_values.utils import VerbosityLevels


def _tiny_config():
    return Config(
        block_size=512,
        vocab_size=256,
        padded_vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_query_groups=2,
        intermediate_size=128,
    )


def test_delta_rotation_is_exact():
    """Rotating a key from position p to rank r via the delta must equal
    rotating the raw key directly at r (rotations compose additively)."""
    config = _tiny_config()
    enc = LinearPositionEncoding(config, device=torch.device("cpu"))
    n_elem = config.rope_n_elem
    cos, sin = enc._cos, enc._sin

    torch.manual_seed(0)
    raw = torch.randn(1, 1, 1, config.head_size)
    p_orig, p_new = 37, 5

    def rope_at(x, p):
        return _rotate_half_apply(x, cos[p, :n_elem], sin[p, :n_elem], n_elem)

    k_at_orig = rope_at(raw, p_orig)
    # Delta rotation by (p_new - p_orig) < 0: cos row |d|, sin negated.
    d = p_orig - p_new
    k_delta = _rotate_half_apply(k_at_orig, cos[d, :n_elem], -sin[d, :n_elem], n_elem)
    k_direct = rope_at(raw, p_new)
    assert (k_delta - k_direct).abs().max().item() < 1e-5


def test_compaction_is_noop_for_contiguous_positions():
    """When retained positions are already 0..T-1 (dense cache case), the
    transform must return query and keys unchanged (delta == 0)."""
    config = _tiny_config()
    enc = LinearPositionEncoding(config, device=torch.device("cpu"))
    B, G, H, T, num = 2, config.n_query_groups, config.n_head, 24, 1

    torch.manual_seed(0)
    query = torch.randn(B, H, num, config.head_size)
    keys = torch.randn(B, G, T, config.head_size)
    values = torch.randn(B, G, T, config.head_size)
    token_positions = torch.arange(T).view(1, 1, T).expand(B, G, T).contiguous()

    new_q, new_kv = compact_rope_positions(
        query=query,
        k_and_v=DefaultKeysAndValues(keys, values),
        token_positions=token_positions,
        input_pos=T - num,
        num=num,
        pos_encoding=enc,
        rope_n_elem=config.rope_n_elem,
    )
    assert (new_q - query).abs().max().item() < 1e-5
    assert (new_kv.keys() - keys).abs().max().item() < 1e-5
    assert torch.equal(new_kv.values(), values)


def _generate(
    gpt_model,
    cache_name,
    cache_length,
    prompt,
    max_new,
    compact,
    return_logits=False,
):
    cache_kwargs = {"grace_period": 2} if cache_name.startswith("h2o") else {}
    deallocate_kv_cache_buffers_of_model(gpt_model)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=cache_name,
            max_batch_size=1,
            cache_length=cache_length,
            dtype=torch.float32,
            cache_kwargs=cache_kwargs,
        )
    )
    set_position_compaction(gpt_model, compact)
    gpt_model.eval()
    gpt_model.max_seq_length = prompt.shape[1] + max_new
    inference_model = LongContextInferenceModel(
        gpt_model, head_model=None, chunk_size=16, verbose=VerbosityLevels.NONE
    )
    tokens = prompt
    with torch.no_grad():
        logits = inference_model(tokens, targets=None)
        out = []
        logit_list = []
        for _ in range(max_new):
            logit_list.append(logits[:, -1])
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            out.append(next_tok)
            logits = gpt_model(next_tok)
    if return_logits:
        return torch.cat(out, dim=1), torch.stack(logit_list, dim=1)
    return torch.cat(out, dim=1)


def test_dense_generation_unchanged_by_compaction():
    """Dense cache retains everything, so compaction is the identity and
    greedy generation must be token-identical."""
    config = _tiny_config()
    torch.manual_seed(0)
    with torch.device("cpu"):
        gpt_model = GPT(config).eval()
    torch.manual_seed(1)
    prompt = torch.randint(0, 256, (1, 40))

    base = _generate(gpt_model, "dense-default", 60, prompt, 8, compact=False)
    comp = _generate(gpt_model, "dense-default", 60, prompt, 8, compact=True)
    assert torch.equal(base, comp)


def test_h2o_compaction_changes_logits():
    """With an evicting cache the compacted phases genuinely differ: the
    path must run cleanly and the decode logits must change (argmax tokens
    of a tiny random model can coincide, so assert on logits)."""
    config = _tiny_config()
    torch.manual_seed(0)
    with torch.device("cpu"):
        gpt_model = GPT(config).eval()
    torch.manual_seed(1)
    prompt = torch.randint(0, 256, (1, 40))

    base, base_logits = _generate(
        gpt_model, "h2o-default", 24, prompt, 8, compact=False, return_logits=True
    )
    comp, comp_logits = _generate(
        gpt_model, "h2o-default", 24, prompt, 8, compact=True, return_logits=True
    )
    assert base.shape == comp.shape
    diff = (base_logits - comp_logits).abs().max().item()
    assert diff > 1e-4, f"compaction had no effect on logits (max diff {diff})"
