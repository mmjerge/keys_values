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
Task 4.3 step 2b.1: compute per-KV-position H2O scores in-engine (single seq).

Smallest real scoring step. For a single-sequence generation, we wrap
``FlashInferImpl.forward`` and, per layer, accumulate the ``slot_mapping`` across
calls. For one request that enumerates exactly its KV slots in position order, so
we can gather the request's full K/V from the paged ``kv_cache`` and compute
per-position attention mass with the verified ``reference_summed_attention``
(``keys_values.vllm.attention``) - no block table or manager wiring needed yet.

For the first few decode steps of layer 0, it prints the score vector length,
its total mass (a sanity check: ~= number of query heads per KV group), and the
top-scoring positions. This validates the in-engine score signal before we
tackle multi-request batching, the block mapping, and eviction.

Run single-sequence only (one prompt). Usage:
    python examples/vllm_h2o_score_probe.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys
from collections import defaultdict

# Accumulated KV slots per layer-impl instance (id(self) -> list[int]).
_slots_by_layer: dict = defaultdict(list)
_decode_logs = 0
_MAX_DECODE_LOGS = 4
_first_layer_id = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument(
        "--prompt",
        default="The capital of France is Paris. Tell me a short fact about it.",
    )
    p.add_argument("--max-tokens", type=int, default=12)
    return p.parse_args()


def _check_env() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("torch not installed.")
    import torch

    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Run on a GPU box.")
    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit("vllm not installed.")


def _score_decode_step(self, query, kv_cache, slots) -> None:
    """Gather the request's K/V from the paged cache and print H2O scores."""
    import torch

    from keys_values.vllm.attention import reference_summed_attention

    # kv_cache: (num_blocks, 2, block_size, n_kv_heads, head_size)
    num_blocks, two, block_size, n_kv_heads, head_size = kv_cache.shape
    keys_all = kv_cache[:, 0].reshape(-1, n_kv_heads, head_size)
    values_all = kv_cache[:, 1].reshape(-1, n_kv_heads, head_size)
    slot_idx = torch.tensor(slots, device=kv_cache.device, dtype=torch.long)
    k_req = keys_all[slot_idx]  # (seq_len, n_kv_heads, head_size)
    v_req = values_all[slot_idx]
    seq_len = k_req.shape[0]

    # Shape for the reference: query (1, n_heads, 1, head_size);
    # key/value (1, n_kv_heads, seq_len, head_size).
    n_heads = query.shape[1]
    q = query.reshape(1, n_heads, 1, head_size)
    k = k_req.permute(1, 0, 2).unsqueeze(0)
    v = v_req.permute(1, 0, 2).unsqueeze(0)
    _, summed = reference_summed_attention(q, k, v, causal=True)
    summed = summed[0]  # (n_kv_heads, seq_len)
    per_head_mass = summed.sum(dim=-1)  # ~= query heads per group
    top = torch.topk(summed.sum(dim=0), k=min(3, seq_len))
    print(
        f"  [score] seq_len={seq_len} mass_per_kv_head={per_head_mass.tolist()} "
        f"top_positions={top.indices.tolist()} top_scores="
        f"{[round(x, 3) for x in top.values.tolist()]}"
    )


def _wrap_flashinfer_impl() -> None:
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    if getattr(FlashInferImpl.forward, "_h2o_score_wrapped", False):
        return
    original = FlashInferImpl.forward

    def wrapped(self, *args, **kwargs):
        global _decode_logs, _first_layer_id
        attn_metadata = args[5] if len(args) > 5 else None
        if attn_metadata is not None:
            query = args[1]
            kv_cache = args[4]
            slot_mapping = getattr(attn_metadata, "slot_mapping", None)
            if (
                kv_cache is not None
                and kv_cache.numel() > 0
                and slot_mapping is not None
            ):
                lid = id(self)
                if _first_layer_id is None:
                    _first_layer_id = lid
                _slots_by_layer[lid].extend(slot_mapping.tolist())
                num_decode = int(getattr(attn_metadata, "num_decode_tokens", 0))
                num_prefills = int(getattr(attn_metadata, "num_prefills", 0))
                is_single_decode = num_prefills == 0 and num_decode == 1
                if (
                    is_single_decode
                    and lid == _first_layer_id
                    and _decode_logs < _MAX_DECODE_LOGS
                ):
                    print(f"\n[impl] layer0 decode step (log {_decode_logs}):")
                    _score_decode_step(self, query, kv_cache, _slots_by_layer[lid])
                    _decode_logs += 1
        return original(self, *args, **kwargs)

    wrapped._h2o_score_wrapped = True
    FlashInferImpl.forward = wrapped
    print("[probe] wrapped FlashInferImpl.forward for in-engine scoring")


def main() -> None:
    args = parse_args()
    _check_env()
    _wrap_flashinfer_impl()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_prefix_caching=False,
        attention_backend="FLASHINFER",
    )
    out = llm.generate(
        [args.prompt], SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    )
    print(f"\n[probe] Output: {out[0].outputs[0].text!r}")
    print(
        "\nSanity check: mass_per_kv_head should be ~= query heads per KV group "
        "(14/2 = 7 for Qwen2.5-0.5B), and seq_len should grow by 1 each decode "
        "step. If so, we have a correct in-engine H2O score signal."
    )


if __name__ == "__main__":
    main()
