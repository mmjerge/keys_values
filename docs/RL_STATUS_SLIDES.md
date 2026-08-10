% RL Through Sparse Attention
% Michael Jerge
% August 10, 2026

# The goal

**Paper target (option 1):** tangible RL gains on a hard long-context
problem, under resource constraints.

Value proposition: RL on long contexts, on hardware where full attention
**cannot run at all**.

# What works today

- Standalone GRPO / RLOO / SFT through an actively evicting KV cache
  (chunked prefill + bounded-memory chunked backward)
- Single-epoch cache reuse: sampling pass log-probs reused, no scoring pass
- Shared-prompt prefill: token-exact, 1.5x generation speedup at 7k prompts
- Two upstream bugs found + fixed on the way (#139, #140)
- PR #142 (RL core) reviewed, comments addressed, awaiting merge

# 0.5B campaign @8k: the parity result

- Training **through** the evicting cache = dense-attention training,
  within noise, on every paired comparison (2 seeds) -- at ~half the memory
- Best single result: RLOO+EM through evicting H2O, nq **0.43 vs 0.33 base**,
  no answer-style drift
- RL > SFT on both tasks
- Known artifact documented: substring-EM is style-sensitive; the
  EM-anchored reward avoids it

# Why sparse inference loses accuracy (A1/A2)

- Position compaction (exact RoPE re-indexing of survivors): **no recovery**
- Every eviction knob at fixed slots (normalization, sink, grace, v-norm):
  **no recovery**
- The gap is **information loss from eviction** -- evicted answers can't
  be recovered downstream

# The lever that works: quantize, keep more

| arm (trivia_qa @8k) | EM |
|---|---:|
| dense bf16 | 0.620 |
| h2o quantized-8 @ 4096 slots | 0.540 |
| **h2o quantized-8 @ 8192 slots** | **0.640** |

8-bit KV is accuracy-free (verified: dense-q8 == h2o-q8 exactly).
Equal memory = 2x slots. Cache covers prompt = dense accuracy at half
the memory. Under real eviction, degradation is graceful (~25% retention
costs ~9pp).

# 7B memory frontier (48GB L40S)

| context | dense | bounded H2O-q8 @8192 |
|---:|---|---|
| 16k | 40.8 GB | fits |
| 24k | **OOM** | 35.5 GB |
| 32k | **OOM** | 35.7 GB |
| 65k | **OOM** | **35.7 GB** |

Dense GRPO dies at 24k. The bounded evicting cache trains at **65k, flat
memory** -- the only configuration that runs in this regime.

# Benchmark hygiene

- HELMET dev/eval split was drawn unseeded (our port) -- cross-host splits
  shared only 8-10/100 questions; found by replication, fixed (PR #145,
  merged)
- All campaign numbers: one box, one cache -> paired comparisons stand
- Canonical splits now on S3 + GitHub release with SHA-256 manifests
- Related HELMET upstream nondeterminism: issue filed (#43), fix ready

# In flight: 7B @32k flagship

- Base probes (n=50): nq **0.80 dense / 0.52 through-cache**;
  hotpot 0.40 / 0.24 -- a 16-28pp eviction gap to close
- 4 RLOO runs (nq, hotpot x 2 seeds) on a 3-worker L40S fleet, ~350s/step,
  ETA ~2 days
- Dense **training** cannot run here at all -- every gain is a
  "sparse-only regime" gain

# Infrastructure (optional, checked in)

- S3 job queue + self-provisioning workers + self-stop when queue drains
- Terraform module: autoscaling across **all AZs x instance types**
  (single-GPU capacity is scarce; manual placement was costing hours)
- Canonical data + run artifacts on S3, collaborator read access granted
- Validated 7B/48GB recipe encoded in one job-generator script

# Positioning vs related work

- **LongStraw** (2607.14952): wrapper not adoptable (review-only,
  GLM/Megatron-specific) -- but we adopted its validated *schedule* as
  shared-prompt prefill (token-exact, 1.5x gen speedup)
- **OOMB**: keeps full KV, hides the cost (paging/offload). We *bound*
  the state and pay in accuracy -- hence the accuracy-gap workstream;
  bounded state composes better with group reuse at extreme context
- **Spherical KV** (2605.18856): rate-distortion retention -- our A2
  finding (retention determines accuracy; quantization is free) confirms
  the framing independently; their policy = next A2 candidate
- **Needle-in-a-haystack / RULER**: the diagnostic for what eviction
  destroys; difficulty dial + verifiable rewards

# Next: a genuinely hard task

1. **LongProc** -- long procedural generation; outputs 2-8k tokens,
   rule-checkable (verifiable rewards, no style gaming); long *decode*
   stresses eviction during generation. RL driver ready (mmjerge#6);
   base-7B probes queued.
2. infinite_bench_qa / narrative_qa at 64k+ (probes running)
3. RULER needle variants (tunable difficulty; synthetic)

Gate: base capability low but nonzero (json_kv's 0.00 stays excluded).

# Asks

- Review of PR #142 when convenient
- Read on the hard-task ranking (LongProc first?)
- Both S3 buckets readable from your account; report:
  `docs/RL_STATUS_REPORT.md`
