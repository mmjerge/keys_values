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
"""
Parity tests for shared-prompt prefill (``keys_values.rl.grpo.prefix``).

The claim under test: prefilling the shared GRPO prompt once at batch 1 and
expanding the retained cache state to the group batch produces *exactly* the
same completions (and matching log-probs) as prefilling the prompt
``group_size`` times, for dense and evicting cache types.
"""

import pytest
import torch

from keys_values.config import Config
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.rl.grpo.loop import grpo_step
from keys_values.rl.grpo.prefix import expand_prefix_to_group
from keys_values.rl.grpo.rollout import generate_completions_with_logprobs
from keys_values.utils import VerbosityLevels

GROUP_SIZE = 4
PROMPT_LEN = 40
MAX_NEW_TOKENS = 12

# (cache_name, cache_length): dense holds everything; the evicting caches are
# shorter than prompt + completion so eviction actually happens during decode.
CACHE_CASES = [
    ("dense-default", PROMPT_LEN + MAX_NEW_TOKENS + 4),
    ("h2o-default", 32),
    ("lastrec-default", 32),
]


def _make_model():
    config = Config(
        block_size=512,
        vocab_size=256,
        padded_vocab_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        n_query_groups=2,
        intermediate_size=128,
    )
    torch.manual_seed(0)
    with torch.device("cpu"):
        gpt_model = GPT(config).eval()
    return gpt_model


def _assign_caches(gpt_model, name, cache_length):
    cache_kwargs = {"grace_period": 2} if name.startswith("h2o") else {}
    deallocate_kv_cache_buffers_of_model(gpt_model)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name=name,
            max_batch_size=GROUP_SIZE,
            cache_length=cache_length,
            dtype=torch.float32,
            cache_kwargs=cache_kwargs,
        )
    )
    gpt_model.eval()
    gpt_model.max_seq_length = PROMPT_LEN + MAX_NEW_TOKENS
    return LongContextInferenceModel(
        gpt_model,
        head_model=None,
        chunk_size=16,
        verbose=VerbosityLevels.NONE,
    )


@pytest.mark.parametrize("cache_name,cache_length", CACHE_CASES)
def test_shared_prefill_matches_batched_prefill(cache_name, cache_length):
    """Greedy decode: shared-prefill completions must be token-identical."""
    gpt_model = _make_model()
    torch.manual_seed(1)
    prompt = torch.randint(0, 256, (1, PROMPT_LEN))

    # Baseline: prompt prefilled GROUP_SIZE times (batched).
    inference_model = _assign_caches(gpt_model, cache_name, cache_length)
    base_completions, base_logps, _ = generate_completions_with_logprobs(
        model=inference_model,
        prompt_ids=prompt.repeat(GROUP_SIZE, 1),
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        eos_token_id=None,
        pad_token_id=0,
    )

    # Shared: batch-1 prefill, expand retained state, decode batched.
    inference_model = _assign_caches(gpt_model, cache_name, cache_length)
    with torch.no_grad():
        logits_1 = inference_model(prompt, targets=None)
    expand_prefix_to_group(gpt_model, GROUP_SIZE)
    shared_completions, shared_logps, _ = generate_completions_with_logprobs(
        model=inference_model,
        prompt_ids=prompt.repeat(GROUP_SIZE, 1),
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        eos_token_id=None,
        pad_token_id=0,
        prefill_logits=logits_1.expand(GROUP_SIZE, -1, -1),
    )

    assert torch.equal(base_completions, shared_completions)
    assert (base_logps - shared_logps).abs().max().item() < 1e-5


@pytest.mark.parametrize("cache_name,cache_length", CACHE_CASES)
def test_grpo_step_share_prompt_prefill(cache_name, cache_length):
    """grpo_step with share_prompt_prefill=True matches the naive schedule."""

    def reward_fn(prompts, completions):
        return completions.float().mean(dim=1)

    results = {}
    for shared in (False, True):
        gpt_model = _make_model()
        cache_kwargs = {"grace_period": 2} if cache_name.startswith("h2o") else {}
        deallocate_kv_cache_buffers_of_model(gpt_model)
        gpt_model.assign_kv_caches(
            KVCacheFactory.create(
                gpt_model=gpt_model,
                name=cache_name,
                max_batch_size=GROUP_SIZE,
                cache_length=cache_length,
                dtype=torch.float32,
                cache_kwargs=cache_kwargs,
            )
        )
        optimizer = torch.optim.SGD(gpt_model.parameters(), lr=1e-3)
        torch.manual_seed(1)
        prompt = torch.randint(0, 256, (1, PROMPT_LEN))
        torch.manual_seed(2)
        results[shared] = grpo_step(
            gpt_model=gpt_model,
            optimizer=optimizer,
            prompt_ids=prompt,
            reward_fn=reward_fn,
            group_size=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            top_k=1,  # greedy: schedules must agree exactly
            chunk_size=16,
            share_prompt_prefill=shared,
        )

    assert (
        abs(results[True]["mean_reward"] - results[False]["mean_reward"]) < 1e-6
    ), "completions diverged between shared and naive prefill schedules"


def test_share_prompt_prefill_rejects_multiple_prompts():
    gpt_model = _make_model()
    deallocate_kv_cache_buffers_of_model(gpt_model)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model,
            name="lastrec-default",
            max_batch_size=2 * GROUP_SIZE,
            cache_length=32,
            dtype=torch.float32,
        )
    )
    optimizer = torch.optim.SGD(gpt_model.parameters(), lr=1e-3)
    prompts = torch.randint(0, 256, (2, PROMPT_LEN))
    with pytest.raises(ValueError, match="single prompt"):
        grpo_step(
            gpt_model=gpt_model,
            optimizer=optimizer,
            prompt_ids=prompts,
            reward_fn=lambda p, c: c.float().mean(dim=1),
            group_size=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            chunk_size=16,
            share_prompt_prefill=True,
        )
