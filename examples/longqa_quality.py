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
Inference-quality precursor: dense vs H2O on long-context key-value retrieval.

Takes ONE checkpoint and measures answer accuracy (substring exact match) on
the same set of long-context QA examples under a full (dense) KV cache vs H2O
at several cache budgets. If H2O already degrades accuracy at inference, RL
won't fix it; if it holds, the dense-vs-H2O GRPO comparison is worth running.

    python examples/longqa_quality.py --device cuda \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --context-len 8192 --n-examples 40 --h2o-budgets 1024,2048,4096
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import lightning as L
import torch
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
from litgpt.utils import auto_download_checkpoint, check_valid_checkpoint_dir, load_checkpoint

sys.path.insert(0, str(Path(__file__).parent))
from longqa_task import build_dataset, left_pad  # noqa: E402

from keys_values.config import Config
from keys_values.data.constants import LIT_MODEL_FNAME
from keys_values.evaluation.metrics import sub_exact_match
from keys_values.kvcache.factory import KVCacheFactory, deallocate_kv_cache_buffers_of_model
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.utils import VerbosityLevels


@torch.no_grad()
def eval_accuracy(gpt_model, examples, cache_name, cache_length, dtype,
                  max_new, chunk_size, batch_size, tokenizer, pad_id, eos_id, fabric):
    deallocate_kv_cache_buffers_of_model(gpt_model)
    gpt_model.assign_kv_caches(
        KVCacheFactory.create(
            gpt_model=gpt_model, name=cache_name, max_batch_size=batch_size,
            cache_length=cache_length, dtype=dtype,
        )
    )
    gpt_model.eval()
    correct, total, gen_lens = 0, 0, []
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompt_ids = left_pad([e.prompt_ids for e in batch], pad_id).to(fabric.device)
        gpt_model.max_seq_length = int(prompt_ids.shape[1]) + max_new
        # A processing chunk cannot exceed the cache's forward capacity, so cap
        # it to the (possibly small) cache length.
        eff_chunk = min(chunk_size, cache_length)
        inf = LongContextInferenceModel(gpt_model, head_model=None,
                                        chunk_size=eff_chunk, verbose=VerbosityLevels.NONE)
        completions = generate_completions(
            model=inf, prompt_ids=prompt_ids, max_new_tokens=max_new,
            temperature=1.0, top_k=1, top_p=1.0, eos_token_id=eos_id, pad_token_id=pad_id,
        )
        for row, ex in zip(completions, batch):
            toks = row[row != pad_id]
            text = tokenizer.decode(toks) if toks.numel() else ""
            gen_lens.append(int(toks.numel()))
            correct += int(sub_exact_match(text, ex.target))
            total += 1
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--context-len", type=int, default=8192)
    p.add_argument("--n-examples", type=int, default=40)
    p.add_argument("--h2o-budgets", default="1024,2048,4096")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    budgets = [int(x) for x in args.h2o_budgets.split(",")]
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(devices=1, accelerator=args.device,
                      precision="32-true" if args.device == "cpu" else "bf16-true")

    checkpoint_dir = auto_download_checkpoint(model_name=args.model, access_token=args.access_token)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    prompt_style = (
        load_prompt_style(checkpoint_dir) if has_prompt_style(checkpoint_dir)
        else PromptStyle.from_config(config)
    )
    pad_id = tokenizer.processor.token_to_id("<|endoftext|>")
    if pad_id is None:
        pad_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else 0
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    check_valid_checkpoint_dir(checkpoint_dir)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(fabric.device)

    examples = build_dataset(tokenizer, prompt_style.apply, args.context_len,
                             args.n_examples, seed=args.seed, device=fabric.device)
    lens = [int(e.prompt_ids.size(0)) for e in examples]
    max_len, avg_len = max(lens), int(sum(lens) / len(lens))
    # Dense cache must hold the LONGEST prompt in any batch (+ generated tokens).
    dense_cl = max_len + args.max_new_tokens + 8
    print(f"\nmodel={args.model}  device={args.device}  target_ctx={args.context_len}  "
          f"actual_ctx: avg~{avg_len} max={max_len}  n={args.n_examples}\n")

    print(f"{'cache':>26} {'cache_len':>10} | {'accuracy':>8}")
    print("-" * 50)
    # dense baseline (full attention)
    acc = eval_accuracy(gpt_model, examples, "dense-default", dense_cl,
                        dtype, args.max_new_tokens, args.chunk_size, args.batch_size,
                        tokenizer, pad_id, eos_id, fabric)
    print(f"{'dense-default':>26} {dense_cl:>10} | {acc:>8.3f}  (baseline)")
    for b in budgets:
        acc = eval_accuracy(gpt_model, examples, "h2o-torch-quantized8", b, dtype,
                            args.max_new_tokens, args.chunk_size, args.batch_size,
                            tokenizer, pad_id, eos_id, fabric)
        evicts = "evicts" if b < max_len else "no-evict"
        print(f"{'h2o-torch-quantized8':>26} {b:>10} | {acc:>8.3f}  ({evicts})")
    print("\nDone.")


if __name__ == "__main__":
    main()
