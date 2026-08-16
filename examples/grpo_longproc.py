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
GRPO/RLOO on LongProc (long procedural generation) through a KeysAndValues
sparse KV cache.

LongProc (arXiv:2501.05414; princeton-pli/LongProc) tasks require *generating*
long, rule-checkable outputs (0.5k/2k/8k-token tiers) from long inputs:
html_to_tsv (extract tables; scored by row F1), travel_planning (constraint
satisfaction), countdown, tom_tracking, path_traversal, pseudo_to_code. The
programmatic evaluators double as verifiable RL rewards, and the long
*decode* exercises cache eviction during generation -- a regime the HELMET QA
campaigns (long prefill, 32-token decode) never touch.

Setup: clone LongProc next to the repo (data ships in-repo, ~100MB):

    git clone --depth 1 https://github.com/princeton-pli/LongProc.git repos/longproc

Example (7B on a 48GB card, known-good memory recipe):

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python examples/grpo_longproc.py --device cuda \
        --model Qwen/Qwen2.5-7B-Instruct --dataset html_to_tsv_2k \
        --kv-cache-name h2o-torch-quantized8 --cache-length 8192 \
        --group-size 4 --prompts-per-update 2 --optimizer paged_adamw8bit \
        --disable-flashinfer --out-dir runs/longproc_html2k
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
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

# Primary metric per task family: the scalar the evaluator returns that we
# use as the reward (and eval score).
PRIMARY_METRIC = {
    "html_to_tsv": "f1",
    "pseudo_to_code": "accuracy",
    "path_traversal": "accuracy",
    "tom_tracking": "accuracy",
    "countdown": "success",
    "travel_planning": "success",
}


def load_longproc(dataset: str, path: Path):
    sys.path.insert(0, str(path))
    from longproc.longproc_data import load_longproc_data  # noqa: E402

    data, eval_fn = load_longproc_data(dataset, str(path / "data"))
    family = dataset.rsplit("_", 1)[0]
    metric = PRIMARY_METRIC.get(family)
    if metric is None:
        raise ValueError(f"No primary metric registered for {family}")
    return data, eval_fn, metric


def primary_score(eval_fn, metric: str, prediction: str, record: dict) -> float:
    try:
        metrics, _ = eval_fn(prediction, record["item"])
        return float(metrics.get(metric, 0.0))
    except Exception:
        # Evaluators are robust to garbage, but belt-and-braces: a crash in
        # the checker is a zero-reward completion, not a dead training run.
        return 0.0


def shaped_score(eval_fn, metric: str, prediction: str, record: dict,
                 format_bonus: float) -> float:
    """Training reward: primary metric plus a small format bonus.

    Sampled rollouts at temperature 1.0 almost never hit the strict output
    format the checkers require (e.g. a ```tsv fenced block), so the primary
    metric is 0.0 for every group member and RLOO gets no gradient (observed:
    30+ steps of all-zero rewards while greedy eval scores 0.157). The
    evaluators expose format adherence (``extraction_rate`` etc.); paying a
    small bonus for parseable output gives the group a reward *spread* to
    climb out of the cold start. Eval always uses the unshaped primary
    metric.
    """
    try:
        metrics, _ = eval_fn(prediction, record["item"])
        primary = float(metrics.get(metric, 0.0))
        fmt = float(metrics.get("extraction_rate", 0.0))
        return primary + format_bonus * fmt
    except Exception:
        return 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--dataset", default="html_to_tsv_2k")
    p.add_argument("--longproc-path", default="repos/longproc")
    p.add_argument("--kv-cache-name", default="h2o-torch-quantized8")
    p.add_argument("--cache-length", type=int, default=8192)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--prompts-per-update", type=int, default=2)
    p.add_argument("--adv-mode", choices=["grpo", "rloo"], default="rloo")
    p.add_argument("--max-new-tokens", type=int, default=2600,
                   help="Output budget; the _2k tiers need ~2200 tokens.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--optimizer", choices=["adamw", "paged_adamw8bit"],
                   default="paged_adamw8bit")
    p.add_argument("--chunk-size", type=int, default=1024)
    p.add_argument("--layers-per-cell", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Rollout sampling temperature. Long structured outputs "
                        "derail badly at 1.0 (zero parseable rollouts observed).")
    p.add_argument("--format-bonus", type=float, default=0.2,
                   help="Training-reward bonus per unit of format adherence "
                        "(extraction_rate); 0 disables shaping.")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="runs/grpo_longproc")
    p.add_argument("--eval-only", action="store_true",
                   help="Base-model probe: score n-eval records, no training.")
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

    records, eval_fn, metric = load_longproc(args.dataset, Path(args.longproc_path))
    # Deterministic eval/train partition (records ship in fixed order).
    rng = random.Random(42)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    eval_idx = set(idx[: args.n_eval])
    eval_records = [records[i] for i in sorted(eval_idx)]
    train_records = [records[i] for i in idx[args.n_eval:]]
    print(f"{args.dataset}: {len(records)} records -> "
          f"{len(train_records)} train / {len(eval_records)} eval; "
          f"reward metric = {metric}")

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
        return tokenizer.encode(
            prompt_style.apply(rec["input_prompt"]), device=fabric.device)

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
            scores.append(primary_score(eval_fn, metric, text, rec))
        mean = sum(scores) / max(len(scores), 1)
        print(f"[eval @ {tag}] {metric} = {mean:.3f} (n={len(scores)})", flush=True)
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
                    vals.append(shaped_score(eval_fn, metric, text, rec,
                                             args.format_bonus))
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
            ))
        mean_r = sum(m["mean_reward"] for m in micro_metrics) / len(micro_metrics)
        dt = time.perf_counter() - t0
        print(f"step {step:4d} | reward {mean_r:.3f} | {dt:.1f}s", flush=True)
        history.append({"step": step, "reward": mean_r, "sec": dt})
        if step % args.eval_every == 0:
            score = eval_model(f"step {step}")
            history.append({"step": step, "eval": score})
            torch.save(gpt_model.state_dict(), out_dir / f"step{step}.pt")

    torch.save(gpt_model.state_dict(), out_dir / "final.pt")
    eval_model("final")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"done; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
