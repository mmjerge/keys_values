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

Caches that need attention weights (H2O) have three SDPA implementations in
:meth:`MultiHeadSelfAttention._sdpa_mode`: FlashInfer (vendored kernels, must
be built with ``python build_ext.py``), the 2x FlexAttention baseline (only if
a :class:`FlexAttentionArgs` is passed), and eager. The RL examples used to
build ``GPT(config)`` bare and so silently ran eager whenever FlashInfer was
not built.

Why ``auto`` does NOT fall back to FlexAttention: the Flex manager compiles one
kernel per exact ``kv_len`` (``FlexAttnForPrefillManager._get_args``) and calls
``torch.compile`` on every miss. In token-by-token decoding ``kv_len`` changes
every step until the cache is full, so every decode step recompiles (observed:
a 2-step smoke test did not finish in 25 minutes). The finetune scripts never
generate, which is why they can default to Flex. Until Flex buckets ``kv_len``,
FlashInfer is the only fast path for RL rollouts, and the right fix for a
worker without it is to build it (``scripts/provision_worker.sh``), not to
pick a different kernel.

* ``auto``:  FlashInfer if built. Otherwise eager, with a loud warning.
* ``flex``:  FlexAttention (gradient-pass experiments only; decode recompiles).
* ``eager``: the naive implementation (baselines and debugging).

Returned kwargs go to BOTH ``GPT(config, **mha_kwargs)`` and ``cache_kwargs``
of :meth:`KVCacheFactory.create`: the caches build their own
``MultiHeadSelfAttention`` and the chunked-gradient cells reuse ``kv_cache.mha``.
"""

import warnings
from typing import Any, Dict

import torch

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.attention import flashinfer_ops
from keys_values.attention.flex_attention import FlexAttentionArgs, choose_q_lens

ATTN_BACKENDS = ("auto", "flex", "eager")

# FlexAttention compiles a block-mask variant per (kv_len, q_len) shape; the
# stock dynamo limits are far too small. Same values as finetune's SDPAArgs.
DYNAMO_CACHE_SIZE_LIMIT = 32
DYNAMO_ACCUMULATED_CACHE_SIZE_LIMIT = 128

EAGER_WARNING = (
    "FlashInfer extension is not built: KV cache '{name}' needs attention "
    "weights and will run the EAGER SDPA (several times slower in decode). "
    "Build it with `pip install flashinfer-python && python build_ext.py` "
    "(see scripts/provision_worker.sh)."
)


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
    Keyword arguments for :class:`MultiHeadSelfAttention` selecting the
    attention backend. Pass to ``GPT(config, **kwargs)`` and merge into
    ``cache_kwargs``.

    Args:
        backend: One of :data:`ATTN_BACKENDS`.
        kv_cache_name: Cache name; decides whether attention weights are needed.
        chunk_size: Prefill / gradient chunk size, anchors the Flex ``q_lens``.
        device: FlashInfer and Flex need CUDA; on CPU the result is eager.
        attn_temp_gb: If ``> 0``, bound eager attention-weight temporaries (GiB).
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
    cuda = device.type == "cuda"
    if backend == "flex" and cuda:
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
            forward_return_lse=weights,
        )
        chosen = "flex" + (" (2x, return_lse)" if weights else "")
        warnings.warn(
            "FlexAttention recompiles for every new kv_len; token-by-token "
            "decoding will be extremely slow. Use for gradient-pass experiments.",
            stacklevel=2,
        )
    else:
        # No flexatt_args, no use_eager_sdpa_always (the training replay
        # cache rejects the latter). _sdpa_mode then picks FlashInfer when
        # available and the call needs it, PyTorch SDPA for causal prefill,
        # and eager only for attention-weight calls without FlashInfer.
        if backend == "eager":
            flashinfer_ops._available = False
        flashinfer = cuda and flashinfer_ops._available
        if weights:
            chosen = "flashinfer" if flashinfer else "eager"
            if not flashinfer and backend == "auto" and cuda:
                warnings.warn(EAGER_WARNING.format(name=kv_cache_name), stacklevel=2)
        else:
            chosen = "pytorch-sdpa" + (" + flashinfer decode" if flashinfer else "")
    if verbose:
        print(
            f"attention backend: {chosen} (requested {backend}; "
            f"attention weights needed: {weights})",
            flush=True,
        )
    return kwargs


def describe_backend(mha_kwargs: Dict[str, Any]) -> str:
    """Short label for logs/results, derived from the kwargs."""
    if "flexatt_args" in mha_kwargs:
        return "flex"
    return "flashinfer" if flashinfer_ops._available else "eager"
