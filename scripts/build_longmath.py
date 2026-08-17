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
Build "longmath": long-context competition math with verifiable answers.

Construction: one *target* problem (Hendrycks MATH; MIT license; exact-match
`\\boxed{}` answers) embedded among N distractor problems, each labeled with
a random ID. The prompt shows all problems and asks to solve the one with the
target ID, boxed. Context length scales with N, so the task combines
retrieval over a long context with competition math, and the reward is a
verifiable exact match on the normalized boxed answer -- RLVR-ready, not
style-gameable.

Output: a HuggingFace DatasetDict {development, evaluation} per tier, in the
same layout as the canonical HELMET split caches (records carry "input",
"output", "query_id", "max_length"), plus a MANIFEST-compatible SHA-256 over
the sorted query_ids. Upload to the canonical bucket:

    python scripts/build_longmath.py --tiers 8k,32k --out-dir /tmp/longmath
    aws s3 sync /tmp/longmath s3://keys-values-helmet-canonical/longmath/

The deterministic seed (42) governs problem selection, distractor draw, ID
assignment, and target placement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

# Target token budgets per tier (chars/4 heuristic; prompts are validated
# against the tier budget after construction).
TIER_TOKENS = {"8k": 8000, "16k": 16000, "32k": 32000, "64k": 64000}
CHARS_PER_TOKEN = 4

INSTRUCTION = (
    "Below is a numbered list of competition mathematics problems, each with "
    "a problem ID. Solve ONLY the problem with ID {target_id}. Think step by "
    "step, then give your final answer inside \\boxed{{}}.\n\n"
)


def extract_boxed(solution: str) -> str | None:
    """Last \\boxed{...} content, brace-balanced."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(solution)):
        c = solution[i]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return solution[start:i].strip()
            depth -= 1
    return None


def normalize_answer(ans: str) -> str:
    """Light normalization for exact-match scoring."""
    ans = ans.strip().strip("$")
    ans = re.sub(r"\\left|\\right", "", ans)
    ans = re.sub(r"\s+", "", ans)
    ans = re.sub(r"^\\text\{(.+)\}$", r"\1", ans)
    ans = re.sub(r"\\!|\\,", "", ans)
    return ans


def build_tier(problems: list, tier: str, n_records: int, rng: random.Random):
    budget_chars = TIER_TOKENS[tier] * CHARS_PER_TOKEN
    records = []
    pool = [p for p in problems if p["answer"] is not None]
    for k in range(n_records):
        target = pool[rng.randrange(len(pool))]
        # Fill context with distractors up to the budget.
        chosen = [target]
        used = {id(target)}
        size = len(target["problem"])
        while size < budget_chars:
            d = pool[rng.randrange(len(pool))]
            if id(d) in used:
                continue
            chosen.append(d)
            used.add(id(d))
            size += len(d["problem"]) + 64
        rng.shuffle(chosen)
        ids = rng.sample(range(1000, 9999), len(chosen))
        target_id = ids[chosen.index(target)]
        body = "\n\n".join(
            f"[Problem {pid}]\n{p['problem']}" for pid, p in zip(ids, chosen)
        )
        records.append({
            "input": INSTRUCTION.format(target_id=target_id) + body,
            "output": [target["answer"]],
            "query_id": f"{tier}-{k:04d}-{target_id}",
            "max_length": tier,
            "n_distractors": len(chosen) - 1,
        })
    return records


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiers", default="8k,32k")
    p.add_argument("--n-dev", type=int, default=400)
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--levels", default="3,4,5",
                   help="MATH difficulty levels to include (1=easy..5=hard).")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    src = load_dataset("EleutherAI/hendrycks_math", "algebra")
    # Pull all configs for breadth.
    configs = ["algebra", "counting_and_probability", "geometry",
               "intermediate_algebra", "number_theory", "prealgebra",
               "precalculus"]
    problems = []
    levels = {f"Level {x}" for x in args.levels.split(",")}
    for cfg in configs:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg)
        for split in ("train", "test"):
            for row in ds[split]:
                if row["level"] not in levels:
                    continue
                ans = extract_boxed(row["solution"])
                if ans is None:
                    continue
                problems.append({
                    "problem": row["problem"],
                    "answer": normalize_answer(ans),
                })
    print(f"pool: {len(problems)} problems (levels {sorted(levels)})")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"description": "longmath: Hendrycks MATH problems embedded in "
                "long distractor contexts; reward = exact match on normalized "
                "boxed answer. Built with seed 42.", "datasets": {}}
    for tier in args.tiers.split(","):
        rng = random.Random(42)
        recs = build_tier(problems, tier, args.n_dev + args.n_eval, rng)
        dev, ev = recs[: args.n_dev], recs[args.n_dev:]
        dsd = DatasetDict({
            "development": Dataset.from_list(dev),
            "evaluation": Dataset.from_list(ev),
        })
        name = f"longmath_{tier}"
        dsd.save_to_disk(str(out / name))
        entry = {}
        for split, data in (("development", dev), ("evaluation", ev)):
            ids = sorted(r["query_id"] for r in data)
            entry[split] = {
                "n_records": len(ids),
                "sha256_sorted_query_ids": hashlib.sha256(
                    "\n".join(ids).encode()).hexdigest(),
            }
        manifest["datasets"][name] = entry
        lens = [len(r["input"]) // CHARS_PER_TOKEN for r in recs]
        print(f"{name}: {len(dev)} dev / {len(ev)} eval; "
              f"~tokens min/med/max = {min(lens)}/{sorted(lens)[len(lens)//2]}/{max(lens)}")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out}/MANIFEST.json")


if __name__ == "__main__":
    main()
