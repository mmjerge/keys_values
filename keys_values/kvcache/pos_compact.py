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
Position compaction for evicting KV caches (inference-time A/B).

After eviction, the retained tokens keep the RoPE rotations of their
*original* absolute positions, so attention sees a "holey" position layout
(e.g. 0, 3, 47, 812, ...) that the model never encountered in pretraining.
This module re-expresses queries and cached keys at *compacted* positions:
each retained token's position becomes its rank among the retained tokens.
Relative order is preserved, so the causal mask is unchanged; only the
rotary phases move.

Because RoPE rotations compose additively, a key stored with rotation at
position ``p`` can be re-expressed exactly at position ``r`` by applying the
rotation for ``r - p`` (always ``<= 0`` here, since a token's rank among
survivors cannot exceed its original position). No un-rotated keys need to
be stored; the transform gathers cos/sin rows from the existing RoPE tables.

This is an accuracy experiment, not a perf-optimized path: the cached keys
are rotated into a temporary copy each forward (the cache buffers themselves
keep original rotations and are never mutated).
"""

from typing import Tuple

import torch

from keys_values.attention.base import DefaultKeysAndValues, KeysAndValues
from keys_values.pos_encoding import PositionEncoding


def _rotate_half_apply(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_n_elem: int,
) -> torch.Tensor:
    """Apply RoPE with per-position cos/sin that broadcast against ``x``.

    Same rotate-half convention as ``litgpt.model.apply_rope``, but accepts
    cos/sin of any broadcast-compatible shape (the litgpt helper requires 3D)
    and only rotates the first ``rope_n_elem`` features.
    """
    x_rope = x[..., :rope_n_elem]
    half = rope_n_elem // 2
    x1 = x_rope[..., :half]
    x2 = x_rope[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x_rope * cos + rotated * sin).to(dtype=x.dtype)
    if rope_n_elem == x.shape[-1]:
        return roped
    return torch.cat((roped, x[..., rope_n_elem:]), dim=-1)


def _rope_tables(
    pos_encoding: PositionEncoding, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = getattr(pos_encoding, "_cos", None)
    sin = getattr(pos_encoding, "_sin", None)
    if cos is None or sin is None:
        raise NotImplementedError(
            "Position compaction requires a position encoding with "
            "precomputed _cos/_sin tables (LinearPositionEncoding family); "
            f"got {type(pos_encoding).__name__}"
        )
    if cos.ndim != 2:
        raise NotImplementedError(
            "Position compaction does not support per-layer RoPE tables "
            "(rope_local_base_freq / rope_indices)"
        )
    if cos.device != device:
        cos = cos.to(device)
        sin = sin.to(device)
    return cos, sin


def compact_rope_positions(
    query: torch.Tensor,
    k_and_v: KeysAndValues,
    token_positions: torch.Tensor,
    input_pos: int,
    num: int,
    pos_encoding: PositionEncoding,
    rope_n_elem: int,
) -> Tuple[torch.Tensor, KeysAndValues]:
    """Re-express queries and cached keys at compacted (rank) positions.

    Args:
        query: RoPE'd queries at original positions ``input_pos .. input_pos +
            num - 1``, shape ``(batch, n_head, num, head_size)``.
        k_and_v: Cache contents *including* the tokens written this step; keys
            carry rotations of their original positions.
        token_positions: Original position of each slot, shape
            ``(batch, n_query_groups, T)``. Values are unique per ``(b, g)``.
        input_pos: Original position of the first query token.
        num: Number of query tokens in this chunk.
        pos_encoding: Encoding whose cos/sin tables are reused for the delta
            rotations.
        rope_n_elem: Number of leading head features RoPE applies to.

    Returns:
        ``(query, k_and_v)`` with rotary phases moved to compacted positions.
        Values are unchanged. The rotated keys are a copy; cache buffers are
        not modified.
    """
    if rope_n_elem <= 0:
        return query, k_and_v
    keys = k_and_v.keys()
    device = keys.device
    cos_table, sin_table = _rope_tables(pos_encoding, device)

    # Rank of each retained token among survivors (per batch row and query
    # group). token_positions are unique per row, so a double argsort yields
    # the rank = number of surviving tokens with smaller original position.
    order = torch.argsort(token_positions, dim=-1)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        -1,
        order,
        torch.arange(order.shape[-1], device=device, dtype=order.dtype)
        .expand_as(order)
        .contiguous(),
    )

    # Keys: rotate by (rank - original_position) <= 0. For a rotation by a
    # negative offset d: cos(d*theta) = cos(|d|*theta), sin(d*theta) =
    # -sin(|d|*theta).
    neg_delta = (token_positions - ranks).to(torch.long)  # >= 0
    cos_k = cos_table[neg_delta][..., :rope_n_elem]  # (B, G, T, n_elem)
    sin_k = -sin_table[neg_delta][..., :rope_n_elem]
    new_keys = _rotate_half_apply(keys, cos_k, sin_k, rope_n_elem)

    # Queries: the fresh tokens written this step occupy the top `num` ranks
    # (they are the newest surviving tokens), so query token i moves from
    # position input_pos + i to rank T - num + i -- a single scalar shift
    # d_q = T - num - input_pos <= 0 for the whole chunk, identical across
    # batch rows and heads.
    total = token_positions.shape[-1]
    neg_delta_q = input_pos + num - total  # >= 0
    cos_q = cos_table[neg_delta_q, :rope_n_elem]  # (n_elem,)
    sin_q = -sin_table[neg_delta_q, :rope_n_elem]
    new_query = _rotate_half_apply(query, cos_q, sin_q, rope_n_elem)

    return new_query, DefaultKeysAndValues(new_keys, k_and_v.values())


def set_position_compaction(gpt_model, enabled: bool = True) -> None:
    """Enable/disable position compaction on all KV caches of a model."""
    for cache in gpt_model.get_kv_caches():
        if cache is not None:
            cache.compact_positions = enabled
