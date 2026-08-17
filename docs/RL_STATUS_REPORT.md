# RL-through-sparse-attention: status report

*2026-08-10. Covers everything since the RL effort started; written for the
paper-planning discussion (option 1: tangible gains on a hard RL problem
under resource constraints).*

## What works today

**Training loop.** Standalone GRPO/RLOO (`keys_values/rl/grpo/`) on
KeysAndValues primitives: chunked prefill + bounded-memory chunked backward
through an actively evicting KV cache. Single-epoch cache reuse (the
sampling pass's log-probs are reused; no separate scoring forward), gradient
accumulation, completion masking, RLOO leave-one-out advantages, SFT
baseline driver. PR #142 (review comments addressed, awaiting merge);
RLOO/SFT on stacked branches. Shared-prompt prefill (LongStraw-style
response-only schedule) implemented and token-exact vs the naive schedule:
1.5x generation speedup at 7k prompts (`prompt-reuse` branch).

**Scale.** Two upstream bugs found and fixed along the way (#139 ln_f
before head; #140 eviction defaults destroying generation quality --
measured 0.00 -> 0.90 recovery). 8-bit paged optimizer support added for
7B+ full fine-tuning on 48GB cards.

## Results so far (all HELMET, question-diverse eval)

**0.5B campaign @8k** (`docs/GRPO_HELMET_RESULTS.md`): training through the
evicting cache is stable across GRPO/RLOO/SFT and matches dense-attention
training at ~half the memory (parity within noise on every paired
comparison, 2 seeds on the round-2 recipe). Best single result: RLOO+EM
through evicting H2O, nq 0.43 EM vs 0.33 base, with no answer-style drift.
RL > SFT throughout. Known artifact documented: substring-EM is
style-sensitive; the EM-anchored `em_f1` reward avoids it.

**Accuracy-gap analysis (A1/A2)** (`docs/POSITION_COMPACTION_A1.md`,
`docs/EVICTION_SWEEP_A2.md`): the sparse-INFERENCE gap is information loss
from eviction, full stop. Position compaction (exact RoPE re-indexing):
no recovery. Every retention knob at fixed slots: no recovery. What works:
8-bit KV quantization is accuracy-free (verified with dense-q8 == h2o-q8
controls), so at equal memory you afford 2x slots -- when the cache covers
the prompt, accuracy equals dense at ~half the memory. Under genuine
eviction the gap is real and degrades gracefully (~25% retention costs
~9pp EM on trivia).

**Memory frontier, 7B on 48GB L40S** (`docs/GRPO_CONTEXT_SCALING.md`):
dense GRPO OOMs at 24k context. Full-coverage sparse OOMs at 32k. Bounded
evicting cache (q8@8192): flat ~35.7GB from 24k to **65k** -- the only
configuration that trains at all in this regime.

## Benchmark hygiene

The HELMET dev/eval split was drawn unseeded (our port dropped the
harness-level seeding HELMET itself uses) -- found by cross-host
replication, fixed in PR #145 (merged). Cross-host splits shared only
8-10/100 questions; all campaign numbers came from one box/one cache, so
paired comparisons stand. Canonical split caches (8k campaign draw + new
seeded 32k draws) now live on S3 + a GitHub release, with SHA-256
manifests over the record IDs. Upstream HELMET has a related (smaller)
demo-sampling nondeterminism; issue filed (princeton-nlp/HELMET#43) with
a fix ready.

## Done: 7B @32k flagship -- gains where dense training cannot run

Full detail in `docs/FLAGSHIP_32K_RESULTS.md`. Qwen2.5-7B, HELMET @32k,
RLOO + em_f1, `h2o-torch-quantized8@8192` (~25-30% prompt retention), 200
steps, **3 seeds per task**. Dense GRPO OOMs at 24k on the same 48GB card,
so every run trains in a regime full attention cannot enter.

Through-cache eval (the deployment condition), EM vs base:

| task | base | trained (s0 / s1 / s2) |
|---|---:|---|
| nq | 0.52 | 0.68 / 0.56 / 0.58 |
| hotpot_qa | 0.24 | 0.36 / 0.38 / 0.36 |

**6/6 arms positive**, recovering roughly half the eviction penalty; the
dense-vs-cached gap on the same checkpoint shrinks (nq 28pp -> 12-20pp;
hotpot 16pp -> 6-14pp). Explicitly NOT claimed: sparse beating dense --
dense inference still wins on every checkpoint, as A1/A2 predicts. The nq
*dense-eval* column is mixed across seeds (+8/-12/-16) and is treated as
metric-artifact territory: generation inspection shows training compresses
verbose scaffolding to terse spans, which collides with nq's substring-EM
(targets like "in Super Bowl LII"). Caveats: n=50 (SE ~7pp), one cache
config, 200 steps.

Infra used: S3 job queue + self-provisioning workers + optional Terraform
module (`terraform/`, autoscaling across all AZs/types -- single-GPU
capacity is scarce and manual placement was costing hours).

## In flight: LongProc (the genuinely hard task)

html_to_tsv_2k passed the capability gate (base 7B greedy F1 0.157
through-cache; travel_planning scored 0.000 and is excluded like json_kv).
Two RLOO seeds are training now on `examples/grpo_longproc.py`
(rule-checkable row-F1 as the reward, 2600-token generations). Seed 0 is
past step 100 with training reward ~0.62 (vs 0.157 base) -- i.e. learning,
final evals pending.

Two findings from this regime, both new relative to the QA campaigns:

1. **Reward cold start.** At temperature 1.0 no sampled rollout satisfied
   the checker's strict output format, so every group member scored 0.0
   and RLOO had no gradient (30+ dead steps while greedy eval scored
   0.157). Fixed by adding a small format-adherence bonus
   (`extraction_rate`) to the *training* reward and lowering rollout
   temperature to 0.7; eval stays unshaped.
2. **A real bug in the chunked backward** (issue #148): the replay unpack
   in `autograd_hooks.py` assumes the final buffer for a node is at most
   one chunk ahead of the annotation being unpacked; with long *generated*
   regions spanning 3+ backward chunks it can be two ahead, and backward
   dies (`final chunk_idx = 3, must be in [1, 2]`). Sampling-dependent, so
   it looks like flaky infra. Never triggered by our 32-token-completion QA
   runs. Workaround in use: `chunk_size >= max_new_tokens`.

## Hard-task selection: status

Gate for a candidate: base capability low but nonzero, verifiable reward,
not style-gameable. Probe results so far (base 7B, through-cache):

| candidate | base score | verdict |
|---|---:|---|
| LongProc html_to_tsv_2k | F1 0.157 | **selected**, training now |
| LongProc travel_planning_2k | 0.000 | excluded (capability-null) |
| json_kv | 0.000 | excluded (capability-null) |
| nq @32k | EM 0.52 | usable but base is strong (0.80 dense) |
| hotpot_qa @32k | EM 0.24 | usable, harder than nq |

Remaining candidates, unprobed:

1. **infinite_bench_qa / narrative_qa at 64k+**: deep in the
   dense-impossible regime (we train at 65k); probes need a 48GB worker
   (the A10G OOMs at 32k dense eval).
2. **RULER multi-key needle variants**: difficulty is a dial (keys,
   depth), fully verifiable, and directly measures what eviction destroys
   -- also useful as a diagnostic axis (accuracy vs needle depth at fixed
   cache budget).
3. **ALCE citation generation**: verifiable grounding, heavier reward
   engineering.

## Positioning vs related work

**LongStraw** (arXiv:2607.14952, the RL harness Konstantinos flagged) and
**OOMB** (openreview dSa3ImCQr7): assessed in
`docs/LONGSTRAW_OOMB_ASSESSMENT.md` (branch `longstraw-assessment`).
Summary: the public LongStraw tree is review-only and GLM/Megatron/Ray-
specific, so the wrapper is not adoptable -- but its *schedule* is, and its
gradient-parity receipts validate it. We implemented the response-only-
update idea as shared-prompt prefill (`prompt-reuse`: prefill the shared
prompt once, expand the retained cache state per group member, token-exact,
1.5x generation speedup). The structural contrast with OOMB frames our
whole effort: OOMB *keeps the full KV and hides the cost* (paging, CPU
offload, O(1) activations via chunk recurrence); we *bound the state and
pay in accuracy* (eviction) -- which is why the accuracy-gap workstream
exists, and why our bounded retained state composes better with
LongStraw-style group reuse at extreme context (per-member restore is
O(cache_length), not O(context)).

**Spherical KV** (arXiv:2605.18856): angle-domain key storage +
rate-distortion retention (joint keep/drop + precision-tier decisions
under a byte budget). Our A2 decomposition -- retention determines
accuracy, 8-bit quantization is accuracy-free, so spend bytes on more
retained tokens -- is an independent confirmation of their rate-distortion
framing from the training side. Their retention policy is the natural
next candidate for the A2 sweep, and the angle-domain representation
connects to the spherical-embedding direction (A4).

**Needle-in-a-haystack / RULER**: the standard instruments for measuring
exactly what eviction destroys (retrieval of planted facts vs depth and
distractor count). RULER's multi-key variants give a difficulty dial and
fully verifiable rewards; ranked as hard-task candidate 3 above, and
independently useful as a diagnostic axis for the flagship (eviction loss
as a function of needle depth at fixed cache budget).

## Where the work lives (branches / PRs)

Upstream (awslabs/keys_values):

- **PR #142** -- the RL core (GRPO loop, cache reuse, HELMET drivers,
  results docs). Review comments addressed; awaiting merge.
- **PR #145** -- HELMET loader seed fix. Merged.
- **Issue #148** -- chunked backward fails when a generated region spans
  3+ chunks (found by the LongProc runs; workaround documented).
- #133 (vLLM) and #144 (eviction defaults) closed as discussed.

Staged on the fork (stacked on `experiments`/`grpo-upstream`, to be
rebased onto `main` and re-targeted upstream once #142 lands):

- **mmjerge#4** -- A1/A2 sparse-accuracy analysis: position compaction
  (negative), eviction-policy sweep, quantization controls, budget
  frontier.
- **mmjerge#5** -- optional Terraform module for the worker fleet
  (autoscaling across all AZs/instance types, S3 job queue,
  self-provisioning workers, collaborator read grants).
- **mmjerge#6** -- LongProc RL driver (`examples/grpo_longproc.py`):
  long procedural generation with rule-checkable outputs as verifiable
  rewards; includes `--eval-only` capability probing. Base-7B probes for
  LongProc + infinite_bench_qa/narrative_qa are queued behind the
  flagship runs.

Feature branches pending as follow-up PRs after #142: `rl-rloo`,
`sft-helmet`, `prompt-reuse` (shared-prompt prefill, token-exact, 1.5x
generation speedup).

Shared artifacts: canonical HELMET splits (8k campaign draw + seeded 32k)
at `s3://keys-values-helmet-canonical` (your account has read access) and
the `helmet-splits-v1` GitHub release; run artifacts land in
`s3://keys-values-rl-results` (read access granted as well).
