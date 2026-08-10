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

## In flight: 7B @32k flagship (first "dense can't do this" gains attempt)

Base-model probes (n=50): nq 0.80 dense-eval / 0.52 through-cache;
hotpot 0.40 / 0.24 -- a 16-28pp eviction gap for training to close, on a
model/context where dense TRAINING cannot run. Four RLOO runs (nq, hotpot
x 2 seeds; em_f1 reward; q8@8192 cache; 200 steps) are running on a
3-worker L40S fleet, ~350s/step, ETA ~2 days. Infra: S3 job queue +
self-provisioning workers + optional Terraform module (`terraform/`,
autoscaling across all AZs/types -- single-GPU capacity is scarce and
manual placement was costing hours).

## Open question: what counts as a genuinely hard task

nq at 32k is real long-context RL but the base model is already strong
(0.80 dense). Candidates for a harder flagship, to be gated by base-model
probes (capability > 0 but low; verifiable reward; not style-gameable):

1. **LongProc** (long procedural generation, HELMET add-on): models score
   low, outputs are 2-8k tokens and rule-checkable, and long *decode*
   stresses eviction differently than long prefill. Strongest candidate.
2. **infinite_bench_qa / narrative_qa at 64k+**: harder contexts, existing
   loaders; 64k+ is deep in the dense-impossible regime.
3. **RULER multi-key needle variants**: difficulty is a dial (keys,
   depth), fully verifiable, and directly measures what eviction destroys.
4. **ALCE citation generation**: verifiable citation grounding; hard, but
   reward engineering is heavier.

json_kv remains capability-null for the base models (0.00) and stays
excluded (cold-start gate).

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
