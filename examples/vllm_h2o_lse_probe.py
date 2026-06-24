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
Task 4.3 probe, step 2a: inspect the FlashInfer attention impl.

To compute H2O scores in-engine we need the per-query LSE (or Q + paged K) and
the attention metadata (block tables, seq lens) that maps KV positions to
blocks. The module-level forward hook only exposed Q/output; those tensors live
one level down, inside ``FlashInferImpl.forward``.

This probe wraps ``FlashInferImpl.forward`` and logs, for the first few calls,
the shape/dtype of every positional and keyword argument plus the relevant
attention-metadata fields. It does not change behavior (it calls the original).
The point is to learn exactly what we can reach before writing the LSE -> score
computation.

Usage:
    python examples/vllm_h2o_lse_probe.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys

_MAX_LOGS = 4
_log_count = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument(
        "--prompt",
        default="List three uses of a key-value cache, then summarize them.",
    )
    p.add_argument("--max-tokens", type=int, default=8)
    return p.parse_args()


def _check_env() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("torch not installed.")
    if not torch.cuda.is_available():
        sys.exit("No CUDA device. Run on a GPU box.")
    try:
        import vllm  # noqa: F401
    except ImportError:
        sys.exit("vllm not installed.")


def _describe(x) -> str:
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return f"Tensor{tuple(x.shape)}/{x.dtype}"
    if isinstance(x, (int, float, bool, str)) or x is None:
        return repr(x)
    return type(x).__name__


def _describe_metadata(md) -> str:
    """Pull out the attn-metadata fields we care about for scoring."""
    fields = [
        "num_prefills",
        "num_prefill_tokens",
        "num_decode_tokens",
        "num_actual_tokens",
        "slot_mapping",
        "block_table",
        "block_tables",
        "seq_lens",
        "query_start_loc",
    ]
    parts = []
    for f in fields:
        if hasattr(md, f):
            parts.append(f"{f}={_describe(getattr(md, f))}")
    return "; ".join(parts) if parts else f"(no known fields on {type(md).__name__})"


def _wrap_flashinfer_impl() -> None:
    from vllm.v1.attention.backends.flashinfer import FlashInferImpl

    if getattr(FlashInferImpl.forward, "_h2o_probe_wrapped", False):
        return
    original = FlashInferImpl.forward

    def wrapped(self, *args, **kwargs):
        global _log_count
        if _log_count < _MAX_LOGS:
            print(f"\n[impl] FlashInferImpl.forward call #{_log_count}")
            print(
                f"  can_return_lse_for_decode = "
                f"{getattr(self, 'can_return_lse_for_decode', None)}"
            )
            for i, a in enumerate(args):
                desc = _describe(a)
                print(f"  arg[{i}] = {desc}")
                # The attention metadata is whichever arg looks like it.
                if "Metadata" in type(a).__name__ or hasattr(a, "slot_mapping"):
                    print(f"    metadata: {_describe_metadata(a)}")
            for k, v in kwargs.items():
                print(f"  kw[{k}] = {_describe(v)}")
            _log_count += 1
        return original(self, *args, **kwargs)

    wrapped._h2o_probe_wrapped = True
    FlashInferImpl.forward = wrapped
    print("[probe] wrapped FlashInferImpl.forward")


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
        "\nWhat to look for: which arg is the query, the kv_cache, and the "
        "attn metadata (block_table/slot_mapping/seq_lens). Those determine how "
        "step 2b computes per-position scores and maps them to blocks."
    )


if __name__ == "__main__":
    main()
