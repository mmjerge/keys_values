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
Task 4.3 probe, step 1: foundation for in-engine H2O scoring.

Before the hard parts (capturing per-position attention weights via the
FlashInfer backend's LSE, and reflecting evicted blocks in the attention path),
this probe verifies the plumbing we depend on:

1. vLLM can run with the FlashInfer backend on this GPU
   (``VLLM_ATTENTION_BACKEND=FLASHINFER``), which is the backend that exposes
   LSE (FlashAttention does not).
2. We can install forward hooks on the model's ``Attention`` layers and they
   fire during generation, so a later step can capture the tensors needed to
   compute H2O scores.

It logs, for the first few hook fires, the layer name and the query/output
shapes. It does not yet compute scores. Runs the engine in-process so hooks
apply in this interpreter.

Usage:
    python examples/vllm_h2o_probe.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import sys

_MAX_LOGS = 6
_log_count = 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument(
        "--prompt",
        default="Count to five, then explain why selective KV caching helps.",
    )
    p.add_argument("--max-tokens", type=int, default=16)
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


def _get_attention_layers(vllm_config):
    get_layers = None
    for modpath in ("vllm.config", "vllm.model_executor.models.utils"):
        try:
            mod = __import__(modpath, fromlist=["get_layers_from_vllm_config"])
            get_layers = mod.get_layers_from_vllm_config
            break
        except (ImportError, AttributeError):
            continue
    if get_layers is None:
        raise ImportError("could not locate get_layers_from_vllm_config")
    attention_cls = None
    for modpath in ("vllm.model_executor.layers.attention", "vllm.attention"):
        try:
            attention_cls = __import__(modpath, fromlist=["Attention"]).Attention
            break
        except (ImportError, AttributeError):
            continue
    if attention_cls is None:
        raise ImportError("could not locate the Attention layer class")
    return get_layers(vllm_config, attention_cls)


def _make_hook(layer_name: str):
    def hook(module, inputs, output):
        global _log_count
        if _log_count >= _MAX_LOGS:
            return
        q_shape = None
        if inputs and hasattr(inputs[0], "shape"):
            q_shape = tuple(inputs[0].shape)
        out_shape = tuple(output.shape) if hasattr(output, "shape") else None
        print(f"[hook] {layer_name}: query={q_shape} output={out_shape}")
        _log_count += 1

    return hook


def _install_attention_hooks() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner.load_model, "_h2o_probe_patched", False):
        return
    original_load = GPUModelRunner.load_model

    def patched_load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        try:
            layers = _get_attention_layers(self.vllm_config)
            for name, attn in layers.items():
                attn.register_forward_hook(_make_hook(name))
            print(f"[probe] installed hooks on {len(layers)} attention layers")
        except Exception as exc:  # noqa: BLE001 - probe: surface it
            print(f"[probe][warn] could not install hooks: {exc}")
        return result

    patched_load._h2o_probe_patched = True
    GPUModelRunner.load_model = patched_load


def main() -> None:
    args = parse_args()
    _check_env()
    _install_attention_hooks()

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    out = llm.generate(
        [args.prompt], SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    )
    print(f"\n[probe] Output: {out[0].outputs[0].text!r}")
    print(
        "\nIf you saw the FLASH_ATTN... line say FLASHINFER and [hook] lines "
        "fired, the foundation for in-engine H2O scoring is in place. Next: "
        "capture LSE in the FlashInfer path and compute per-position score sums."
    )


if __name__ == "__main__":
    main()
