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
Attention-backend selection for the RL drivers.

:class:`MultiHeadSelfAttention` only considers FlexAttention when it is handed
a :class:`FlexAttentionArgs`; its default is ``None``. The finetune scripts
always construct one (``finetune/longcontext_full.py::_mha_kwargs``), but the
RL examples built ``GPT(config)`` bare, so whenever the FlashInfer extension was
not built, caches that need attention weights (H2O) fell through to the eager
SDPA. This module mirrors the finetune defaults so that eager is never reached
silently:

* ``auto``: FlashInfer if the extension is built, else the 2x FlexAttention
  baseline for attention weights, Flex for everything else. Never eager.
* ``flex``: FlexAttention even if FlashInfer is built.
* ``eager``: the naive implementation (baselines and debugging only).

The returned kwargs must go to BOTH ``GPT(config, **mha_kwargs)`` and
``cache_kwargs`` for :meth:`KVCacheFactory.create`: the caches build their own
``MultiHeadSelfAttention`` and the chunked-gradient cells reuse ``kv_cache.mha``.
"""

from typing import Any, Dict, Optional

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import flashinfer_ops
from keys_values.attention.flex_attention import FlexAttentionArgs, choose_q_lens

ATTN_BACKENDS = ("auto", "flex", "eager")

# FlexAttention compiles a block-mask variant per (kv_len, q_len) shape. RL
# prompts vary in length, so the stock dynamo cache limits are far too small
# and compilation would loop. Same values as finetune's SDPAArgs defaults.
DYNAMO_CACHE_SIZE_LIMIT = 32
DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT = 128


def needs_attn_weights(kv_cache_name: str) -> bool:
    return kv_cache_name.startswith(("h2o", "qh2o"))


def attention_mha_kwargs(
    backend: str,
    kv_cache_name: str,
    chunk_size: int,
    device: torch.device,
    attn_temp_gb: float = 0.0,
    num_q_lens: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Keyword arguments for :class:`MultiHeadSelfAttention` selecting the attention
    backend. Pass to ``GPT(config, **kwargs)`` and merge into ``cache_kwargs``.

    Args:
        backend: One of :data:`ATTN_BACKENDS`.
        kv_cache_name: Cache name; decides whether attention weights are needed.
        chunk_size: Prefill / gradient chunk size, anchors the Flex ``q_lens``.
        device: Flex needs CUDA; on CPU the result is eager regardless.
        attn_temp_gb: If ``> 0``, bound eager attention-weight temporaries (GiB).
            Only matters on the eager path; harmless otherwise.
        num_q_lens: Number of anchor ``q_len`` values for Flex compilation.
        verbose: Print the chosen backend.
    """
    if backend not in ATTN_BACKENDS:
        raise ValueError(f"backend={backend!r}, must be one of {ATTN_BACKENDS}")
    weights = needs_attn_weights(kv_cache_name)
    kwargs: Dict[str, Any] = {}
    if attn_temp_gb > 0:
        kwargs["tmp_array_limit_gb"] = TemporaryArrayLimit(
            init_val=attn_temp_gb, name="attention_forward_temp_size_gb"
        )
    if device.type != "cuda" or backend == "eager":
        # No flexatt_args: _sdpa_mode falls through to eager when attention
        # weights are needed, and to PyTorch SDPA otherwise. Do NOT set
        # use_eager_sdpa_always: the training replay cache rejects it.
        if backend == "eager" and flashinfer_ops._available:
            flashinfer_ops._available = False
        chosen = "eager" if weights else "pytorch-sdpa"
    else:
        if backend == "flex":
            flashinfer_ops._available = False
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, DYNAMO_CACHE_SIZE_LIMIT
        )
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit,
            DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT,
        )
        kwargs["flexatt_args"] = FlexAttentionArgs(
            extend_kv=False,
            q_lens=choose_q_lens(chunk_size=chunk_size, num_q_lens=num_q_lens),
            # 2x Flex call returning LSE, so H2O gets its per-slot weights
            # without the eager path. Ignored by _sdpa_mode if FlashInfer wins.
            forward_return_lse=weights,
        )
        flashinfer = flashinfer_ops._available and backend == "auto"
        if weights:
            chosen = "flashinfer" if flashinfer else "flex (2x, return_lse)"
        else:
            chosen = "flex" + (" + flashinfer" if flashinfer else "")
    if verbose:
        print(
            f"attention backend: {chosen} (requested {backend}; "
            f"attention weights needed: {weights})",
            flush=True,
        )
    return kwargs


def describe_backend(mha_kwargs: Dict[str, Any]) -> Optional[str]:
    """Short label for logs/results, derived from the kwargs."""
    if "flexatt_args" in mha_kwargs:
        return "flashinfer" if flashinfer_ops._available else "flex"
    return "eager"
