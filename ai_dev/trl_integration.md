# RL / GRPO Integration - AI Development Notes

## Direction

The RL work is built as a **standalone GRPO loop** on top of KeysAndValues
components (no TRL dependency), under `keys_values/rl/grpo/`:

- `loop.py` — `grpo_step` (one end-to-end GRPO step) and
  `compute_group_advantages`.
- `loss.py` — `GRPOLossHeadModel`, the clipped GRPO policy-gradient loss
  expressed as a `HeadModel` so the backward runs through the
  memory-bounded chunked gradient path.
- `rollout.py` — `generate_completions`, KV-cache rollout generation.

Per-token log-probs come from `keys_values/logprobs.py::compute_logprobs`.

The earlier TRL `GRPOTrainer` subclass was removed: TRL owns generation and the
model, so the sparse KV cache was never exercised during rollouts, which is the
whole reason to use KeysAndValues for RL.

## AI Usage

AI was used to help reconstruct the standalone loop modules and to generate
docstrings in `keys_values/logprobs.py` and `keys_values/rl/grpo/*.py`.
