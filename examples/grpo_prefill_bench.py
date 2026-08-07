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
Benchmark shared-prompt prefill (``share_prompt_prefill``) vs the naive
per-member prefill in ``grpo_step``.

Timing does not depend on token content, so a synthetic prompt of the target
length is used. Reports gen/grad/total per-step time and peak memory for both
schedules.

Example (A10G, Qwen2.5-0.5B, ~7k prompt, G=8):

    python examples/grpo_prefill_bench.py --prompt-len 7000 --group-size 8 \
        --kv-cache-name h2o-default --cache-length 4096
"""

import argparse
import statistics
import time

import lightning as L
import torch

from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
    load_checkpoint,
)

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.model import GPT
from keys_values.rl.grpo.loop import grpo_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--kv-cache-name", default="h2o-default")
    parser.add_argument("--cache-length", type=int, default=4096)
    parser.add_argument("--prompt-len", type=int, default=7000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--layers-per-cell", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--access-token", default=None)
    args = parser.parse_args()

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(
        devices=1,
        accelerator=args.device,
        precision="32-true" if args.device == "cpu" else "bf16-true",
    )
    is_cuda = args.device == "cuda"

    checkpoint_dir = auto_download_checkpoint(
        model_name=args.model, access_token=args.access_token
    )
    check_valid_checkpoint_dir(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(fabric.device)
    snapshot = {k: v.detach().clone() for k, v in gpt_model.state_dict().items()}

    torch.manual_seed(0)
    prompt = torch.randint(
        0, config.padded_vocab_size, (1, args.prompt_len), device=fabric.device
    )

    def reward_fn(
        prompt_ids: torch.Tensor, completion_ids: torch.Tensor
    ) -> torch.Tensor:
        return completion_ids.float().mean(dim=1)

    def run(shared: bool) -> dict:
        gpt_model.load_state_dict(snapshot, strict=True)
        deallocate_kv_cache_buffers_of_model(gpt_model)
        gpt_model.assign_kv_caches(
            KVCacheFactory.create(
                gpt_model=gpt_model,
                name=args.kv_cache_name,
                max_batch_size=args.group_size,
                cache_length=args.cache_length,
                dtype=dtype,
            )
        )
        optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=args.lr)

        def step() -> dict:
            return grpo_step(
                gpt_model=gpt_model,
                optimizer=optimizer,
                prompt_ids=prompt,
                reward_fn=reward_fn,
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                chunk_size=args.chunk_size,
                layers_per_cell=args.layers_per_cell,
                temperature=1.0,
                share_prompt_prefill=shared,
                profile=True,
            )

        for _ in range(args.warmup):
            step()
        if is_cuda:
            torch.cuda.synchronize(fabric.device)
            torch.cuda.reset_peak_memory_stats(fabric.device)
        gen, grad, total = [], [], []
        for _ in range(args.iters):
            if is_cuda:
                torch.cuda.synchronize(fabric.device)
            t0 = time.perf_counter()
            metrics = step()
            if is_cuda:
                torch.cuda.synchronize(fabric.device)
            total.append((time.perf_counter() - t0) * 1000.0)
            gen.append(metrics.get("gen_time_ms", 0.0))
            grad.append(metrics.get("grad_time_ms", 0.0))
        peak_gb = (
            torch.cuda.max_memory_allocated(fabric.device) / 1e9 if is_cuda else 0.0
        )
        return {
            "gen": statistics.mean(gen),
            "grad": statistics.mean(grad),
            "total": statistics.mean(total),
            "peak_gb": peak_gb,
        }

    print(
        f"\nmodel={args.model}  cache={args.kv_cache_name}@{args.cache_length}  "
        f"prompt_len={args.prompt_len}  G={args.group_size}  "
        f"new_tokens={args.max_new_tokens}  iters={args.iters}\n"
    )
    print(f"{'mode':>9} | {'gen':>9} {'grad':>9} {'total':>9} (ms) | {'peak':>7} (GB)")
    results = {}
    for shared in (False, True):
        r = run(shared)
        results[shared] = r
        mode = "shared" if shared else "baseline"
        print(
            f"{mode:>9} | {r['gen']:9.0f} {r['grad']:9.0f} {r['total']:9.0f}      "
            f"| {r['peak_gb']:7.2f}"
        )
    speedup = results[False]["gen"] / max(results[True]["gen"], 1e-9)
    print(f"\ngen-time speedup (baseline/shared): {speedup:.2f}x")


if __name__ == "__main__":
    main()
