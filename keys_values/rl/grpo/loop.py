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
Standalone GRPO training loop built entirely on KeysAndValues components.

This runs the full GRPO pipeline end-to-end, with every memory-heavy step
routed through KeysAndValues' KV-cache infrastructure:

1. **Generation + old log-probs** via
   :func:`generate_completions_with_logprobs` (chunked KV-cache decode; the
   sampling log-probs are captured in the same forward pass, so no separate
   scoring pass is needed).
2. **Reward** via a user-supplied reward function.
3. **Group-relative advantages** via :func:`compute_group_advantages`.
4. **Policy gradient** via :class:`GRPOLossHeadModel` +
   :class:`LongContextGradientModel` (memory-bounded backward).
5. **Optimizer step**.

The loop uses only a ``keys_values.model.GPT``, so it runs anywhere the rest
of the library does (including CPU). This is the path that actually exercises
sparse KV caches through generation, scoring, and the policy gradient -- which
is the reason to use KeysAndValues for RL in the first place.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import os

import torch

import time
from contextlib import contextmanager

from keys_values.array_limit import TemporaryArrayLimit
from keys_values.finetune.utils import may_match_twice_flex_attention_sdpa
from keys_values.rl.grpo.loss import GRPOLossHeadModel
from keys_values.rl.grpo.rollout import generate_completions_with_logprobs
from keys_values.kvcache.gradient.main import LongContextGradientModel
from keys_values.rl.logprobs import compute_logprobs
from keys_values.long_context import LongContextInferenceModel
from keys_values.model import GPT
from keys_values.utils import VerbosityLevels


@contextmanager
def _phase_timer(store: Dict[str, float], key: str, device: torch.device):
    """Record wall-clock time (ms) for a phase, synchronizing CUDA if needed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        store[key] = (time.perf_counter() - start) * 1000.0


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-8,
    mode: str = "grpo",
) -> torch.Tensor:
    """Compute group-relative advantages.

    Rewards are assumed laid out as ``num_groups`` contiguous groups of
    ``group_size`` completions each (the standard GRPO layout: ``G``
    completions per prompt).

    Parameters
    ----------
    rewards : torch.Tensor
        Shape ``(num_groups * group_size,)``.
    group_size : int
        Number of completions per prompt (``G`` in the GRPO paper).
    eps : float
        Numerical stabilizer for the per-group std (``grpo`` mode).
    mode : str
        ``"grpo"``: rewards normalized to zero mean / unit std within the
        group (Shao et al., DeepSeekMath).
        ``"rloo"``: leave-one-out baseline, ``a_i = r_i - mean(r_j, j != i)``
        (Ahmadian et al., RLOO) -- unbiased, no std division, useful when
        reward scales are meaningful.

    Returns
    -------
    torch.Tensor
        Advantages, same shape as ``rewards``.
    """
    if mode not in ("grpo", "rloo"):
        raise ValueError(f"mode = {mode}, must be 'grpo' or 'rloo'")
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1D, got shape {tuple(rewards.shape)}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"rewards length {rewards.numel()} not divisible by group_size {group_size}"
        )
    grouped = rewards.view(-1, group_size)
    if mode == "rloo":
        if group_size < 2:
            raise ValueError("rloo mode requires group_size >= 2")
        total = grouped.sum(dim=-1, keepdim=True)
        loo_mean = (total - grouped) / (group_size - 1)
        advantages = grouped - loo_mean
    else:
        mean = grouped.mean(dim=-1, keepdim=True)
        std = grouped.std(dim=-1, keepdim=True)
        advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def _dense_baseline_backward(
    gpt_model: GPT,
    head: GRPOLossHeadModel,
    model_input_ids: torch.Tensor,
    completions: torch.Tensor,
    grad_scale: float,
) -> torch.Tensor:
    """
    Standard (non-chunked) dense backward, for the dense-RL baseline.

    This is the comparison arm for the memory-bounded chunked path: same
    model, same rollout, same advantages, same loss head, same
    normalization. The *only* difference is how the gradient is computed --
    one full-sequence forward with all activations retained, and a plain
    ``backward()``, which is what standard RL implementations do.

    Equivalence with :class:`LongContextGradientModel` (verified in
    `test/rl/grpo/test_loop.py::test_dense_baseline_matches_chunked_gradient`):

    * The KV caches are detached for the duration of the forward, so
      attention runs as default causal self-attention over the full
      sequence (see :meth:`GPT.forward`: training mode = no caches).
    * The head returns the per-sequence *sum* of per-token losses; we then
      apply ``scale_factor / num_target_entries`` exactly as
      :meth:`LongContextInferenceModel._forward_with_targets` does, with
      ``average_loss_per_batch=True`` semantics (mean over the batch), which
      is the default of :class:`LongContextGradientModel`.
    * The chunked path's ``loss.backward()`` on the per-sequence loss vector
      accumulates gradients of the *mean* over sequences, so we call
      ``.mean().backward()`` here. (Verified empirically: using ``.sum()``
      made every gradient exactly ``batch_size`` times too large.)

    Returns the per-sequence loss vector (detached-comparable to the chunked
    path's return value).
    """
    kv_caches = gpt_model.get_kv_caches()
    gpt_model.clear_kv_caches()
    try:
        gpt_model.max_seq_length = int(model_input_ids.shape[1])
        logits = gpt_model(model_input_ids)
        # `head.forward` slices the last `completions.shape[1]` positions, so
        # passing the full logits aligns targets with their predictors
        per_seq_sum = head(logits, completions, input_pos=0)
        num_entries = head.num_target_entries(completions)
        if num_entries is not None:
            # `average_loss_per_batch=True` semantics
            num_entries = num_entries.to(dtype=torch.float32).mean()
            scale = grad_scale / num_entries.to(device=per_seq_sum.device)
        else:
            scale = grad_scale
        loss = per_seq_sum * scale
        loss.mean().backward()
    finally:
        gpt_model.assign_kv_caches(kv_caches)
    return loss


def grpo_step(
    gpt_model: GPT,
    prompt_ids: torch.Tensor,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    group_size: int = 16,
    max_new_tokens: int = 256,
    chunk_size: int = 64,
    layers_per_cell: int = 1,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    pad_token_id: int = 0,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    rescore_old_logps: bool = False,
    profile: bool = False,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    grad_scale: float = 1.0,
    advantage_mode: str = "grpo",
    backward_tmp_gb: float = 0.0,
    dense_baseline: bool = False,
    verbose: VerbosityLevels = VerbosityLevels.NONE,
) -> Dict[str, float]:
    """Run one GRPO optimization step end-to-end on a KeysAndValues model.

    The ``gpt_model`` must have (non-dense) KV caches assigned, e.g. via
    ``KVCacheFactory.create`` + ``gpt_model.assign_kv_caches(...)``. These
    caches are used for generation, old-log-prob scoring, and the gradient
    pass.

    Old log-probs are captured *during* generation by default (the rollout
    forward pass already produces the exact sampling logits), so the separate
    scoring forward pass is skipped. Set ``rescore_old_logps=True`` to instead
    recompute them with :func:`compute_logprobs` -- useful for A/B comparison
    or to force old==new-policy semantics on the first inner step with a dense
    cache.

    Parameters
    ----------
    gpt_model : GPT
        Policy model with KV caches assigned.
    prompt_ids : torch.Tensor
        Left-padded prompts, shape ``(num_prompts, prompt_len)``. Each prompt
        is expanded into ``group_size`` completions.
    reward_fn : callable
        Maps ``(prompt_ids, completion_ids)`` to a reward tensor of shape
        ``(num_prompts * group_size,)``.
    optimizer : torch.optim.Optimizer
        Optimizer over ``gpt_model`` parameters.
    group_size : int
        Completions sampled per prompt.
    max_new_tokens : int
        Completion length cap.
    chunk_size, layers_per_cell : int
        Control the chunked gradient computation memory/speed tradeoff.
    rescore_old_logps : bool
        If ``True``, recompute old log-probs with a separate scoring pass
        instead of using the values captured during generation.
    profile : bool
        If ``True``, include per-phase wall-clock timings (ms) in the returned
        metrics: ``gen_time_ms``, ``score_time_ms``, ``grad_time_ms``.
    zero_grad, optimizer_step, grad_scale
        Gradient-accumulation controls. To accumulate K prompts into one
        optimizer update, call ``grpo_step`` K times with
        ``zero_grad=(k == 0)``, ``optimizer_step=(k == K-1)`` and
        ``grad_scale=1/K``; gradients sum across the calls.

    Returns
    -------
    dict
        Metrics: ``loss``, ``mean_reward``, ``mean_advantage``,
        ``advantage_std``, ``completion_len``, ``total_completions`` (plus
        timing keys if ``profile=True``).
    """
    device = next(gpt_model.parameters()).device
    num_prompts, prompt_len = prompt_ids.shape
    times: Dict[str, float] = {}

    # A processing chunk cannot exceed the caches' forward capacity. Grace
    # slots reserved by the factory's eviction defaults reduce it; compute the
    # static bound cache_length - grace (kv_cache_max_forward_length() cannot
    # be used here, as it requires an initialized/prefilled cache).
    caps = [
        kvc.cache_length
        - (getattr(kvc, "grace_period", 0) or getattr(kvc, "init_grace_tokens", 0) or 0)
        for kvc in gpt_model.get_kv_caches()
        if kvc is not None
    ]
    if caps:
        chunk_size = max(min(chunk_size, min(caps)), 1)

    # Expand each prompt into `group_size` completions (GRPO group layout).
    expanded_prompts = prompt_ids.repeat_interleave(group_size, dim=0).to(device)
    total = expanded_prompts.shape[0]

    # 1. Generation (chunked KV-cache decode) + old log-prob capture.
    gpt_model.eval()
    gpt_model.max_seq_length = prompt_len + max_new_tokens
    inference_model = LongContextInferenceModel(
        gpt_model=gpt_model,
        head_model=None,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    with _phase_timer(times, "gen_time_ms", device):
        completions, gen_logps, mask = generate_completions_with_logprobs(
            model=inference_model,
            prompt_ids=expanded_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    completion_len = completions.shape[1]

    # 2. Reward + 3. group-relative advantages.
    rewards = reward_fn(expanded_prompts, completions).to(device)
    advantages = compute_group_advantages(rewards, group_size, mode=advantage_mode)

    # Scoring/gradient passes are fed the sequence without its last token, so
    # next-token prediction aligns each completion token `p` with the logits at
    # position `p-1` -- the same distribution it was sampled from. This matches
    # the log-probs captured during generation.
    full_ids = torch.cat([expanded_prompts, completions], dim=1)
    model_input_ids = full_ids[:, :-1]

    # 4. Old (sampling) log-probs. Default (single-epoch GRPO): none -- the
    #    loss head reuses the gradient pass's own forward (policy_logp.detach()),
    #    so the ratio is exactly 1 and no separate scoring pass runs. Set
    #    rescore_old_logps=True for multi-epoch updates / comparison.
    times["score_time_ms"] = 0.0
    old_logps = None
    if rescore_old_logps:
        with _phase_timer(times, "score_time_ms", device):
            with torch.no_grad():
                old_logps, _ = compute_logprobs(
                    gpt_model=gpt_model,
                    input_ids=model_input_ids,
                    targets=completions,
                    chunk_size=chunk_size,
                    temperature=temperature,
                    verbose=verbose,
                )

    # 5. Policy gradient via memory-bounded chunked backward.
    head = GRPOLossHeadModel(
        gpt_model.config,
        epsilon_low=epsilon_low,
        epsilon_high=epsilon_high,
    )
    head.set_batch(advantages=advantages, old_logps=old_logps, mask=mask)
    # With the new training replay cache, "ext-*" annotations match twice;
    # without `may_match_twice`, the second save is left as an unmatched pack
    # argument, which can stall the annotation chain in the chunked backward
    # (issue #148). This mirrors the finetune path (`may_match_twice_factory`).
    autograd_hooks_kwargs: Dict[str, Any] = dict(
        may_match_twice=may_match_twice_flex_attention_sdpa,
    )
    # Env-gated annotation tracing for debugging the chunked backward
    # (issue #148); prints every annotation created/matched/unpacked.
    if os.environ.get("KV_DEBUG_ANNOTATIONS") == "1":
        autograd_hooks_kwargs["debug_print_annotations"] = True
    # Bound the size of temporary device arrays in the backward. Without
    # this, single attention temporaries at 32k contexts exceed 2 GiB and
    # OOM an otherwise-fitting configuration. Mirrors the finetune path.
    _grad_limit_kwargs: Dict[str, Any] = {}
    if backward_tmp_gb > 0:
        _grad_limit_kwargs["backward_tmp_array_limit_gb"] = TemporaryArrayLimit(
            init_val=backward_tmp_gb,
            name="backward_tmp_array_limit_gb",
        )
    grad_model = LongContextGradientModel(
        gpt_model=gpt_model,
        head_model=head,
        layers_per_cell=layers_per_cell,
        chunk_size=chunk_size,
        verbose=verbose,
        autograd_hooks_kwargs=autograd_hooks_kwargs,
        **_grad_limit_kwargs,
    )
    grad_model.train()
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)

    # 6. Backward (+ optimizer step unless accumulating).
    with _phase_timer(times, "grad_time_ms", device):
        if dense_baseline:
            loss = _dense_baseline_backward(
                gpt_model=gpt_model,
                head=head,
                model_input_ids=model_input_ids,
                completions=completions,
                grad_scale=grad_scale,
            )
        else:
            loss = grad_model(model_input_ids, completions, scale_factor=grad_scale)
            loss.backward()
        if optimizer_step:
            optimizer.step()

    metrics = {
        "loss": float(loss.detach().mean().item()),
        "mean_reward": float(rewards.mean().item()),
        "mean_advantage": float(advantages.mean().item()),
        "advantage_std": float(advantages.std().item()),
        "completion_len": completion_len,
        "total_completions": total,
        "mean_completion_tokens": float(mask.sum(dim=-1).mean().item()),
    }
    if rescore_old_logps:
        # Quantify the rollout (decode) vs. training-forward log-prob skew over
        # real completion tokens -- a measure of the train/inference gap.
        with torch.no_grad():
            skew = ((gen_logps - old_logps).abs() * mask).sum() / mask.sum().clamp_min(1.0)
        metrics["logp_skew_decode_vs_forward"] = float(skew.item())
    if profile:
        metrics.update(times)
    return metrics
