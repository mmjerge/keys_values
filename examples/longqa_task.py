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
Controlled long-context QA: key-value retrieval ("needle in a haystack").

Builds prompts of a target token length filled with ``The value for key
<key> is <value>.`` lines, one of which is the target. The model is asked for
the target key's value; the answer is verifiable by substring match. This is
the standard probe for whether a sparse KV cache (H2O) evicts the token the
task needs -- exactly the quality question for long-context sparse attention.

Shared by the quality-eval and GRPO-training scripts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from litgpt.tokenizer import Tokenizer


@dataclass
class QAExample:
    prompt_ids: torch.Tensor  # 1D token ids (chat-templated)
    target: str  # the value string to retrieve
    key: str
    needle_frac: float  # relative position of the needle in the filler


def _rand_key(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(6))


def _rand_value(rng: random.Random) -> str:
    return str(rng.randint(100000, 999999))


TASK_MODES = ("needle_last", "needle_first", "recency")


def _filler_line(rng: random.Random) -> str:
    return f"The value for key '{_rand_key(rng)}' is {_rand_value(rng)}."


def build_example(
    tokenizer: Tokenizer,
    apply_prompt_style,
    context_len: int,
    rng: random.Random,
    device=None,
    needle_frac: float | None = None,
    mode: str = "needle_last",
) -> QAExample:
    """Build one key-value QA example of ~``context_len`` tokens.

    Modes probe different points on the sparse-attention quality frontier:

    - ``needle_last``:  records, then the question at the end. The target record
      is at a random position and is NOT known while reading the records --
      H2O's worst case (it can't know which record matters during prefill).
    - ``needle_first``: the question is stated *before* the records, so the model
      (and thus H2O's attention-weight scoring) can attend to the relevant
      record during prefill. Tests whether query-awareness rescues H2O.
    - ``recency``:      the answer is the value in the *last* record. Recency-
      keeping policies (lastrec / H2O's recent window) should retain it.
    """
    assert mode in TASK_MODES, mode
    question_key = _rand_key(rng)
    target = _rand_value(rng)
    needle_line = f"The value for key '{question_key}' is {target}."

    def assemble(prefix: str, suffix: str, place) -> str:
        budget = max(context_len - int(tokenizer.encode(prefix + suffix).size(0)) - 32, 32)
        lines: List[str] = []
        tok = 0
        placed = False
        while tok < budget:
            if place == "here" and not placed and tok >= (needle_frac or 0.5) * budget:
                line, placed = needle_line, True
            elif place == "end":
                line = _filler_line(rng)
            else:
                line = _filler_line(rng)
            lines.append(line)
            tok += int(tokenizer.encode(" " + line).size(0))
        if place == "here" and not placed:
            lines.insert(len(lines) // 2, needle_line)
        if place == "end":
            lines.append(needle_line)  # the needle IS the most recent record
        return " ".join(lines)

    if mode == "needle_last":
        if needle_frac is None:
            needle_frac = rng.uniform(0.15, 0.85)
        prefix = ("Below is a list of key-value records. Read them and answer "
                  "the question at the end.\n\n")
        suffix = (f"\n\nQuestion: what is the value for key '{question_key}'? "
                  "Reply with only the value.\nAnswer:")
        body = prefix + assemble(prefix, suffix, "here") + suffix
    elif mode == "needle_first":
        needle_frac = rng.uniform(0.15, 0.85)
        prefix = (f"Find the value for key '{question_key}' in the records below.\n\n")
        suffix = (f"\n\nThe value for key '{question_key}' is:")
        body = prefix + assemble(prefix, suffix, "here") + suffix
    else:  # recency
        needle_frac = 1.0
        prefix = ("Below is a list of key-value records.\n\n")
        suffix = ("\n\nQuestion: what is the value in the LAST record above? "
                  "Reply with only the value.\nAnswer:")
        body = prefix + assemble(prefix, suffix, "end") + suffix

    ids = tokenizer.encode(apply_prompt_style(body), device=device)
    return QAExample(prompt_ids=ids, target=target, key=question_key,
                     needle_frac=needle_frac or 0.5)


def build_dataset(
    tokenizer: Tokenizer,
    apply_prompt_style,
    context_len: int,
    n_examples: int,
    seed: int = 0,
    device=None,
    mode: str = "needle_last",
) -> List[QAExample]:
    rng = random.Random(seed)
    return [
        build_example(tokenizer, apply_prompt_style, context_len, rng,
                      device=device, mode=mode)
        for _ in range(n_examples)
    ]


def left_pad(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(int(s.size(0)) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, max_len - seq.size(0):] = seq
    return out
