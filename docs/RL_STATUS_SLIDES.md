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

# Flagship result: 7B @32k, 3 seeds

Through-cache EM (deployment condition), dense training **OOMs** here:

| task | base | trained (s0/s1/s2) |
|---|---:|---|
| nq | 0.52 | 0.68 / 0.56 / 0.58 |
| hotpot_qa | 0.24 | 0.36 / 0.38 / 0.36 |

- **6/6 arms positive**; ~half the eviction penalty recovered
- Eviction gap shrinks: nq 28pp -> 12-20pp; hotpot 16pp -> 6-14pp
- **Not** a "sparse beats dense" claim -- dense inference still wins on
  every checkpoint
- nq dense-eval mixed across seeds: answer-style vs substring-EM artifact
  (inspected, documented, not quoted)

# In flight: LongProc (the hard task)

- Gate passed: html_to_tsv base F1 **0.157**; travel_planning 0.000
  (excluded, like json_kv)
- 2 RLOO seeds training; seed 0 past step 100 at reward ~0.62 vs 0.157 base
- **Cold start found + fixed**: at temp 1.0 no rollout hit the strict
  output format -> all-zero rewards, no gradient. Added format-adherence
  bonus to the training reward (eval unshaped), temp 0.7
- **Real bug surfaced** (issue #148): chunked backward breaks when a
  *generated* region spans 3+ chunks; invisible to 32-token QA runs

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

# Next

- LongProc final evals (~1 day) -> first hard-task result
- Probe infinite_bench_qa / narrative_qa at **64k+** (we train at 65k;
  needs a 48GB worker)
- RULER needle variants: difficulty dial + diagnostic (accuracy vs needle
  depth at fixed cache budget)
- Fix #148 properly (or upstream fix) so long generations don't need
  `chunk_size >= max_new_tokens`
- Style-robust reward (normalize articles/prepositions) to kill the nq
  substring-EM artifact at the source

# Asks

- Review of PR #142 when convenient
- Read on the hard-task ranking (LongProc first?)
- Both S3 buckets readable from your account; report:
  `docs/RL_STATUS_REPORT.md`
