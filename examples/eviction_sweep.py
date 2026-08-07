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
A2: eviction-policy sweep against the sparse-inference accuracy gap.

Evaluates the base model on a HELMET split under a battery of cache
configurations at the same memory budget, to find which retention choices
close the gap to dense. Arms:

- dense reference
- h2o quantized-8 (campaign default config) -- reproduces the gap
- h2o unquantized -- isolates 8-bit quantization loss from eviction loss
- h2o + normalize_scores -- removes the residency bias of cumulative scores
- h2o + keep_initial_fraction -- explicit attention-sink protection
- h2o with large grace (cl/4) -- recency-heavier retention
- h2o-vlen -- value-norm-weighted scores
- lastrec + init grace -- pure recency with protected initial tokens (control)

    python examples/eviction_sweep.py --device cuda --dataset-key trivia_qa \
        --cache-length 4096 --n-eval 100 --disable-flashinfer \
        --out runs/a2_sweep_trivia.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightning as L
import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import (
    auto_download_checkpoint,
    check_valid_checkpoint_dir,
    load_checkpoint,
)

sys.path.insert(0, str(Path(__file__).parent))
from grpo_helmet_crosseval import eval_records  # noqa: E402

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.data.load_helmet_dev_eval import load_helmet_dev_eval
from keys_values.kvcache.factory import (
    KVCacheFactory,
    deallocate_kv_cache_buffers_of_model,
)
from keys_values.model import GPT


def sweep_arms(cache_length: int, dense_length: int):
    """(tag, cache_name, cache_length, cache_kwargs) per arm."""
    g16 = cache_length // 16
    return [
        ("dense", "dense-default", dense_length, {}),
        ("h2o_q8", "h2o-torch-quantized8", cache_length, {"grace_period": g16}),
        ("h2o_fp", "h2o-default", cache_length, {"grace_period": g16}),
        (
            "h2o_q8_norm",
            "h2o-torch-quantized8",
            cache_length,
            {"grace_period": g16, "normalize_scores": True},
        ),
        (
            "h2o_q8_sink",
            "h2o-torch-quantized8",
            cache_length,
            {"grace_period": g16, "keep_initial_fraction": 0.05},
        ),
        (
            "h2o_q8_grace4",
            "h2o-torch-quantized8",
            cache_length,
            {"grace_period": cache_length // 4},
        ),
        (
            "h2o_vlen_q8",
            "h2o-vlen-torch-quantized8",
            cache_length,
            {"grace_period": g16},
        ),
        ("lastrec_sink", "lastrec-default", cache_length, {"init_grace_tokens": g16}),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dataset-key", default="trivia_qa")
    p.add_argument("--max-length", default="8k")
    p.add_argument("--dataset-parent-dir", default=None)
    p.add_argument("--cache-length", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument(
        "--arms", default=None, help="Comma-separated arm tags to run (default: all)."
    )
    p.add_argument("--out", default=None)
    p.add_argument("--disable-flashinfer", action="store_true")
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    if args.disable_flashinfer:
        from keys_values.attention import flashinfer_ops

        flashinfer_ops._available = False

    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(
        devices=1,
        accelerator=args.device,
        precision="32-true" if args.device == "cpu" else "bf16-true",
    )
    checkpoint_dir = auto_download_checkpoint(
        model_name=args.model, access_token=args.access_token
    )
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    prompt_style = (
        load_prompt_style(checkpoint_dir)
        if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )
    pad_id = tokenizer.processor.token_to_id("<|endoftext|>")
    if pad_id is None:
        pad_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else 0
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    load_kwargs = dict(tokenizer=tokenizer.processor, max_length=args.max_length)
    if args.dataset_parent_dir:
        load_kwargs["dataset_parent_dir"] = args.dataset_parent_dir
    _, eval_data = load_helmet_dev_eval(args.dataset_key, **load_kwargs)
    # Question-diverse selection (same policy as grpo_helmet_crosseval).
    by_qid: dict = {}
    for rec in eval_data:
        by_qid.setdefault(rec["query_id"], []).append(rec)
    records = []
    rank = 0
    while len(records) < args.n_eval and any(rank < len(v) for v in by_qid.values()):
        for variants in by_qid.values():
            if rank < len(variants) and len(records) < args.n_eval:
                records.append(variants[rank])
        rank += 1
    lens = [
        int(tokenizer.encode(prompt_style.apply(r["input"])).size(0)) for r in records
    ]
    max_len = max(lens)
    print(
        f"HELMET {args.dataset_key}/{args.max_length}: {len(records)} records "
        f"({len(set(r['query_id'] for r in records))} questions), "
        f"prompt tokens max={max_len}"
    )

    check_valid_checkpoint_dir(checkpoint_dir)
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(fabric.device)

    arms = sweep_arms(args.cache_length, max_len + args.max_new_tokens + 8)
    if args.arms:
        selected = set(args.arms.split(","))
        unknown = selected - {t for t, *_ in arms}
        if unknown:
            raise ValueError(f"Unknown arms: {sorted(unknown)}")
        arms = [a for a in arms if a[0] in selected]

    results = {}
    hdr = f"{'arm':>14} | {'EM':>6} {'F1':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for tag, cache_name, cl, ckw in arms:
        deallocate_kv_cache_buffers_of_model(gpt_model)
        gpt_model.assign_kv_caches(
            KVCacheFactory.create(
                gpt_model=gpt_model,
                name=cache_name,
                max_batch_size=1,
                cache_length=cl,
                dtype=dtype,
                cache_kwargs=ckw,
            )
        )
        em, f1 = eval_records(
            gpt_model,
            records,
            tokenizer,
            prompt_style,
            pad_id,
            eos_id,
            args.max_new_tokens,
            args.chunk_size,
            fabric,
        )
        results[tag] = {
            "em": round(em, 4),
            "f1": round(f1, 4),
            "cache": cache_name,
            "cache_length": cl,
            "cache_kwargs": {k: v for k, v in ckw.items()},
        }
        print(f"{tag:>14} | {em:>6.3f} {f1:>6.3f}", flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "dataset": args.dataset_key,
                    "max_length": args.max_length,
                    "n_eval": len(records),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
