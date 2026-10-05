# Decode throughput vs. context: dense vs. bounded evicting cache

Run `aa0_decode_bench_7b` (2026-10-01). Qwen2.5-7B-Instruct bf16, one L40S
48 GB (g6e.2xlarge), batch 8 (= GRPO group), 256 decoded tokens after a
random prompt of the given length, FlashInfer not compiled (eager SDPA).
Reproduce with `examples/decode_bench.py`; raw numbers in
`s3://keys-values-rl-results/runs/aa0_decode_bench_7b/results.json`.

## Throughput (decode tokens/s, batch 8) and peak memory

| context | dense          | H2O int8 K=8192 | H2O int8 + evict_every=64 | H2O bf16 K=8192 | H2O bf16 + evict_every=64 |
|--------:|---------------:|----------------:|--------------------------:|----------------:|--------------------------:|
|  4,096  | 59.1 / 21.8 GB | 47.6 / 21.0 GB  | 47.7 / 21.0 GB            | 58.3 / 21.8 GB  | 58.3 / 21.8 GB            |
|  8,192  | 32.4 / 29.3 GB | 22.5 / 27.7 GB  | 22.8 / 27.7 GB            | 32.3 / 29.3 GB  | 32.8 / 29.2 GB            |
| 16,384  | **OOM**        | 21.9 / 27.7 GB  | 22.1 / 27.7 GB            | 32.3 / 29.3 GB  | 32.7 / 29.2 GB            |
| 32,768  | **OOM**        | 21.9 / 27.8 GB  | 22.1 / 27.7 GB            | 32.1 / 29.3 GB  | 32.7 / 29.2 GB            |

Prefill (chunked, 1024) is the same for all caches at a given context:
~3 s at 4k, ~6 s at 8k, ~38 s at 16k, ~101 s at 32k.

## Takeaways

1. **Dense decode cost grows with context, then dies.** 59 -> 32 tok/s from
   4k to 8k (every token reads the whole KV cache of every layer), OOM at
   16k with batch 8. The bounded cache is flat in both tokens/s and memory
   from 8k to 32k: once the budget is reached, context length stops
   mattering for decode.
2. **Block eviction (`evict_every`) is a no-op here** (<1%). Recomputing the
   argsort every token was not the per-token cost.
3. **The int8 KV cache costs a third of decode throughput** (22 vs 32 tok/s)
   for a 1.5 GB saving at K=8192. The profile (16k, int8) is dominated by
   `aten::copy_` (27% of CUDA time, 162k calls) and elementwise kernels
   (dequantize / write back each step) plus `masked_fill_`, `_softmax`,
   `mean` from the eager attention path that returns weights for H2O
   scoring. The bf16 H2O cache matches dense tokens/s where dense fits.
4. **Where the remaining overhead is:** the eager attention that returns
   per-slot attention weights (needed for H2O scores). A fused decode kernel
   that emits the score sums, or skipping score updates on some steps, is the
   next lever. Not the eviction ranking.

## Decision

RL drivers default to `h2o-default` (bf16) for the decode-heavy RLVR setting.
int8 stays available (`h2o-torch-quantized8`) for the gradient pass where
2x slots at equal memory buys accuracy (`docs/EVICTION_SWEEP_A2.md`).

## Qwen3-1.7B base probes (same day, 8k generation cutoff, avg@1)

| cache              | MATH500 (n=50) | AIME24 (n=30) | truncated in `<think>` | wall |
|--------------------|---------------:|--------------:|-----------------------:|-----:|
| dense              | 0.76           | 0.167         | 20% / 80%              | 5.8 h |
| H2O int8 K=4096    | 0.78           | 0.167         | 14% / 83%              | 14.6 h |

No eviction gap on either set at K=4096 with 5-8k-token completions. The
8k cutoff is the problem: 80% of AIME samples run out of budget mid-thought.
Default cutoff raised to 16k.
