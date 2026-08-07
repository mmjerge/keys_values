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
Shared-prompt prefill for GRPO groups (LongStraw-style schedule).

In GRPO, every group member shares the same prompt. The naive rollout
prefills the prompt ``group_size`` times (batched); at long contexts the
prefill dominates generation cost. Following the schedule validated by
LongStraw (arXiv:2607.14952, "response-only update": capture the shared
prompt once, restore its retained state per group member), this module
prefills at batch size 1 and then *expands* the retained cache state to the
group batch size, so decoding stays batched (single-token decode is
launch-bound; serializing it would forfeit the win).

Because KeysAndValues caches are *bounded*, the retained state is small: the
KV buffers up to ``cache_length``, per-slot token positions, and (for
score-based policies) the score accumulators. Expansion is a row-0 broadcast
into rows ``1..G-1`` of buffers that are already allocated at
``max_batch_size >= G``.

Supported: caches built on :class:`DefaultKVCacheBuffers` (``dense-default``,
``lastrec-default``, ``h2o-default``, ``h2o-vlen-default``, ...). Quantized
buffers hold per-row quantization state and are not yet supported.
"""

from __future__ import annotations

import torch

from keys_values.kvcache.buffers import DefaultKVCacheBuffers
from keys_values.model import GPT


def expand_prefix_to_group(gpt_model: GPT, group_size: int) -> None:
    """Expand batch-1 prefilled KV caches to ``group_size`` identical rows.

    Call directly after a batch-1 prompt prefill (e.g.
    ``LongContextInferenceModel(...)(prompt, targets=None)``); afterwards the
    caches behave exactly as if the prompt had been prefilled at batch size
    ``group_size``.

    Args:
        gpt_model: Model whose per-layer caches were prefilled at batch 1.
            Caches must have ``max_batch_size >= group_size``.
        group_size: Target batch size (``G``).

    Raises:
        NotImplementedError: For caches not built on
            :class:`DefaultKVCacheBuffers` (e.g. quantized buffers).
        ValueError: If the cache was not prefilled at batch size 1, or
            ``max_batch_size < group_size``.
    """
    for block_idx, cache in enumerate(gpt_model.get_kv_caches()):
        if cache is None:
            raise ValueError(f"Block {block_idx}: no KV cache assigned")
        buffers = getattr(cache, "kv_buffers", None)
        if not isinstance(buffers, DefaultKVCacheBuffers):
            raise NotImplementedError(
                f"Block {block_idx}: shared-prompt prefill supports caches on "
                f"DefaultKVCacheBuffers; got {type(buffers).__name__}. Use a "
                "'-default' cache variant (quantized buffers not yet supported)."
            )
        if buffers.batch_size != 1:
            raise ValueError(
                f"Block {block_idx}: expected batch-1 prefill before expansion, "
                f"got batch_size={buffers.batch_size}"
            )
        if buffers.max_batch_size < group_size:
            raise ValueError(
                f"Block {block_idx}: max_batch_size={buffers.max_batch_size} "
                f"< group_size={group_size}"
            )

        # KV buffers: broadcast row 0 into rows 1..G-1. The buffers are
        # allocated lazily at the *prefill* batch size (see
        # ``DefaultKVCacheBuffers._allocate_buffers``), so after a batch-1
        # prefill ``k.shape[0] == 1`` and we must reallocate at ``group_size``
        # first (matching what ``_allocate_buffers`` would do for a batch-G
        # prefill).
        length = buffers.current_length
        if buffers.k.shape[0] < group_size:
            shape = (
                group_size,
                buffers.n_query_groups,
                buffers.cache_length,
                buffers.head_size,
            )
            new_k = torch.zeros(shape, device=buffers.k.device, dtype=buffers.k.dtype)
            new_v = torch.zeros(shape, device=buffers.v.device, dtype=buffers.v.dtype)
            new_k[:, :, :length, :] = buffers.k[0:1, :, :length, :]
            new_v[:, :, :length, :] = buffers.v[0:1, :, :length, :]
            buffers.k = new_k
            buffers.v = new_v
        else:
            buffers.k[1:group_size, :, :length, :] = buffers.k[0:1, :, :length, :]
            buffers.v[1:group_size, :, :length, :] = buffers.v[0:1, :, :length, :]
        buffers.batch_size = group_size

        # Per-slot token positions (H2O family: 3D with a batch dim; lastrec
        # keeps a batch-agnostic 1D token_pos, which needs no expansion).
        token_pos = getattr(cache, "token_pos", None)
        if isinstance(token_pos, torch.Tensor) and token_pos.ndim == 3:
            token_pos[1:group_size] = token_pos[0:1]

        # Score accumulators of attention-weight policies (H2O etc.).
        score_buffers = getattr(cache, "_score_buffers", None)
        if callable(score_buffers):
            for buf, _name in score_buffers():
                if (
                    isinstance(buf, torch.Tensor)
                    and buf.ndim >= 1
                    and buf.shape[0] >= group_size
                ):
                    buf[1:group_size] = buf[0:1]

        # Pending eviction slots computed during prefill (batch-shaped).
        next_positions = getattr(cache, "_next_positions", None)
        if isinstance(next_positions, torch.Tensor) and next_positions.shape[0] == 1:
            cache._next_positions = next_positions.repeat(group_size, 1, 1)
