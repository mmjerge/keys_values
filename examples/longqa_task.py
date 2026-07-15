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


def build_example(
    tokenizer: Tokenizer,
    apply_prompt_style,
    context_len: int,
    rng: random.Random,
    device=None,
    needle_frac: float | None = None,
) -> QAExample:
    """Build one key-value retrieval example of ~``context_len`` tokens."""
    question_key = _rand_key(rng)
    target = _rand_value(rng)
    if needle_frac is None:
        needle_frac = rng.uniform(0.15, 0.85)

    question = (
        f"\n\nQuestion: what is the value for key '{question_key}'? "
        "Reply with only the value.\nAnswer:"
    )
    # Budget for filler tokens (leave room for question + chat template).
    q_tokens = int(tokenizer.encode(question).size(0))
    filler_budget = max(context_len - q_tokens - 32, 32)

    lines: List[str] = []
    tok_count = 0
    # Placeholder for where the needle goes; fill around it.
    needle_line = f"The value for key '{question_key}' is {target}."
    needle_inserted = False
    while tok_count < filler_budget:
        # Insert the needle once we pass the requested fraction.
        if not needle_inserted and tok_count >= needle_frac * filler_budget:
            line = needle_line
            needle_inserted = True
        else:
            line = f"The value for key '{_rand_key(rng)}' is {_rand_value(rng)}."
        lines.append(line)
        tok_count += int(tokenizer.encode(" " + line).size(0))
    if not needle_inserted:
        lines.insert(len(lines) // 2, needle_line)

    body = (
        "Below is a list of key-value records. Read them and answer the "
        "question at the end.\n\n" + " ".join(lines) + question
    )
    prompt_text = apply_prompt_style(body)
    ids = tokenizer.encode(prompt_text, device=device)
    return QAExample(prompt_ids=ids, target=target, key=question_key,
                     needle_frac=needle_frac)


def build_dataset(
    tokenizer: Tokenizer,
    apply_prompt_style,
    context_len: int,
    n_examples: int,
    seed: int = 0,
    device=None,
) -> List[QAExample]:
    rng = random.Random(seed)
    return [
        build_example(tokenizer, apply_prompt_style, context_len, rng, device=device)
        for _ in range(n_examples)
    ]


def left_pad(sequences: List[torch.Tensor], pad_id: int) -> torch.Tensor:
    max_len = max(int(s.size(0)) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, max_len - seq.size(0):] = seq
    return out
