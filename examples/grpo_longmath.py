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
RLVR on longmath: competition math embedded in long distractor contexts,
trained through a KeysAndValues sparse KV cache.

Dataset: built by ``scripts/build_longmath.py`` (Hendrycks MATH target
problem hidden among distractor problems; the prompt asks to solve one
problem ID). Reward: **exact match on the normalized final boxed answer** --
fully verifiable, style-proof. Sync the canonical splits first:

    aws s3 sync s3://keys-values-helmet-canonical/longmath/ \
        ~/.cache/huggingface/helmet/longmath/ --region us-east-2

Probe (before any training spend):

    python examples/grpo_longmath.py --device cuda --tier 32k \
        --kv-cache-name h2o-torch-quantized8 --cache-length 8192 \
        --eval-only --n-eval 16 --disable-flashinfer

Generation budget is 768 tokens (step-by-step + boxed answer); with
chunk_size=1024 the generated region spans at most 2 backward chunks, which
stays clear of issue #148.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import lightning as L
import torch
from datasets import load_from_disk
from litgpt.prompts import PromptStyle, has_prompt_style, load_prompt_style
from litgpt.tokenizer import Tokenizer
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
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.rl.grpo.loop import grpo_step
from keys_values.rl.grpo.rollout import generate_completions
from keys_values.utils import VerbosityLevels


def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...} content, brace-balanced (same as the builder)."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def normalize_answer(ans: str) -> str:
    ans = ans.strip().strip("$")
    ans = re.sub(r"\\left|\\right", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = re.sub(r"^\\text\{(.+)\}$", r"\1", ans)
    ans = re.sub(r"\\!|\\,", "", ans)
    return ans


def answer_reward(completion: str, targets: list[str]) -> float:
    """1.0 iff the normalized boxed answer matches any target (targets are
    already normalized by the builder)."""
    boxed = extract_boxed(completion)
    if boxed is None:
        return 0.0
    return 1.0 if normalize_answer(boxed) in targets else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--tier", default="32k", choices=["8k", "16k", "32k", "64k"])
    p.add_argument("--data-dir",
                   default="~/.cache/huggingface/helmet/longmath")
    p.add_argument("--kv-cache-name", default="h2o-torch-quantized8")
    p.add_argument("--cache-length", type=int, default=8192)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--prompts-per-update", type=int, default=2)
    p.add_argument("--adv-mode", choices=["grpo", "rloo"], default="rloo")
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--optimizer", choices=["adamw", "paged_adamw8bit"],
                   default="paged_adamw8bit")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--layers-per-cell", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--rescore-every", type=int, default=25,
                   help="Every N steps, rescore old log-probs with a separate "
                        "training-style forward and log decode-vs-forward "
                        "per-token skew tail statistics (0 = never).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/grpo_longmath")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--disable-flashinfer", action="store_true")
    p.add_argument("--access-token", default=None)
    args = p.parse_args()

    if args.disable_flashinfer:
        from keys_values.attention import flashinfer_ops

        flashinfer_ops._available = False

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    fabric = L.Fabric(devices=1, accelerator=args.device,
                      precision="32-true" if args.device == "cpu" else "bf16-true")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = auto_download_checkpoint(
        model_name=args.model, access_token=args.access_token)
    tokenizer = Tokenizer(checkpoint_dir)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    prompt_style = (load_prompt_style(checkpoint_dir)
                    if has_prompt_style(checkpoint_dir)
                    else PromptStyle.from_config(config))
    pad_id = tokenizer.processor.token_to_id("<|endoftext|>")
    if pad_id is None:
        pad_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else 0
    eos_id = int(tokenizer.eos_id) if tokenizer.eos_id is not None else None

    data_dir = Path(args.data_dir).expanduser() / f"longmath_{args.tier}"
    dsd = load_from_disk(str(data_dir))
    dev, ev = dsd["development"], dsd["evaluation"]
    eval_records = list(ev.select(range(min(args.n_eval, len(ev)))))
    train_records = list(dev)
    print(f"longmath_{args.tier}: {len(train_records)} train, "
          f"{len(eval_records)} eval records", flush=True)

    check_valid_checkpoint_dir(checkpoint_dir)
    with fabric.init_module(empty_init=True):
        gpt_model = GPT(config)
    load_checkpoint(fabric, gpt_model, checkpoint_dir / LIT_MODEL_FNAME)
    gpt_model.to(fabric.device)

    cache_kwargs = {}
    if args.kv_cache_name.startswith(("h2o", "qh2o")) and "orig" not in args.kv_cache_name:
        cache_kwargs["grace_period"] = args.cache_length // 16
    gpt_model.assign_kv_caches(KVCacheFactory.create(
        gpt_model=gpt_model, name=args.kv_cache_name,
        max_batch_size=args.group_size, cache_length=args.cache_length,
        dtype=dtype, cache_kwargs=cache_kwargs))

    def encode(rec):
        return tokenizer.encode(prompt_style.apply(rec["input"]),
                                device=fabric.device)

    @torch.no_grad()
    def eval_model(tag: str) -> float:
        gpt_model.eval()
        scores = []
        for rec in eval_records:
            ids = encode(rec).unsqueeze(0)
            gpt_model.max_seq_length = int(ids.shape[1]) + args.max_new_tokens
            inf = LongContextInferenceModel(
                gpt_model, head_model=None, chunk_size=args.chunk_size,
                verbose=VerbosityLevels.NONE)
            comp = generate_completions(
                model=inf, prompt_ids=ids, max_new_tokens=args.max_new_tokens,
                temperature=1.0, top_k=1, top_p=1.0, eos_token_id=eos_id,
                pad_token_id=pad_id, no_inference_mode=True)
            text = tokenizer.decode(comp[0][comp[0] != pad_id])
            scores.append(answer_reward(text, rec["output"]))
        mean = sum(scores) / max(len(scores), 1)
        print(f"[eval @ {tag}] exact_match = {mean:.3f} (n={len(scores)})",
              flush=True)
        return mean

    if args.eval_only:
        eval_model("base")
        return

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=args.lr)
    else:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(gpt_model.parameters(), lr=args.lr)

    eval_model("step 0")
    history = []
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        micro_metrics = []
        for micro in range(args.prompts_per_update):
            rec = train_records[(step * args.prompts_per_update + micro)
                                % len(train_records)]
            prompt_ids = encode(rec).unsqueeze(0)

            def reward_fn(p_ids, completion_ids):
                vals = []
                for row in completion_ids:
                    text = tokenizer.decode(row[row != pad_id])
                    vals.append(answer_reward(text, rec["output"]))
                return torch.tensor(vals, dtype=torch.float32)

            micro_metrics.append(grpo_step(
                gpt_model=gpt_model, prompt_ids=prompt_ids, reward_fn=reward_fn,
                optimizer=optimizer, group_size=args.group_size,
                max_new_tokens=args.max_new_tokens, chunk_size=args.chunk_size,
                layers_per_cell=args.layers_per_cell,
                temperature=args.temperature, eos_token_id=eos_id,
                pad_token_id=pad_id, advantage_mode=args.adv_mode,
                zero_grad=(micro == 0),
                optimizer_step=(micro == args.prompts_per_update - 1),
                grad_scale=1.0 / args.prompts_per_update,
                rescore_old_logps=(args.rescore_every > 0
                                   and step % args.rescore_every == 0),
            ))
        mean_r = sum(m["mean_reward"] for m in micro_metrics) / len(micro_metrics)
        dt = time.perf_counter() - t0
        entry = {"step": step, "reward": mean_r, "sec": dt}
        skew_keys = [k for k in micro_metrics[0] if k.startswith("logp_skew")]
        for key in skew_keys:
            entry[key] = sum(m[key] for m in micro_metrics) / len(micro_metrics)
        skew_msg = (f" | skew p90 {entry['logp_skew_p90']:.3f}"
                    if "logp_skew_p90" in entry else "")
        print(f"step {step:4d} | reward {mean_r:.3f} | {dt:.1f}s{skew_msg}",
              flush=True)
        history.append(entry)
        if step % args.eval_every == 0 and step < args.steps:
            history.append({"step": step, "eval": eval_model(f"step {step}")})

    torch.save(gpt_model.state_dict(), out_dir / "final.pt")
    eval_model("final")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
