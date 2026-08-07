# LongStraw / OOMB: relation to KeysAndValues and integration assessment

Context: mseeger suggested checking whether our method can be integrated into
LongStraw (arXiv:2607.14952), which claims to integrate OOMB
(openreview dSa3ImCQr7, arXiv:2602.02108) -- "a close contender to what we've
been doing". This note maps the three systems onto each other and proposes an
integration path.

## What each system is

**OOMB** ("Out of the Memory Barrier"): a *gradient system* for long-context
training. Chunk-recurrent training with on-the-fly activation recomputation
(O(1) activation memory); the bottleneck then becomes the KV cache, which OOMB
manages with a paged memory manager (for KV *and* KV-gradients), asynchronous
CPU offloading, and page-level sparse attention. Reported: Qwen2.5-7B at 4M
tokens on a single H200; ~10 MB end-to-end memory growth per 10K tokens.

**LongStraw**: an *RL (GRPO) harness* around a long-context gradient method.
Its "transaction": capture the shared prompt **without autograd**; retain only
architecture-required state on explicitly owned pages; **restore that state for
each group member**; score old-policy/reference branches graph-free; **replay
one policy response at a time with autograd**; accumulate gradients; one
distributed finalization + optimizer step. Live training graph is bounded by
the response suffix; the expensive prompt computation is reused across the
whole GRPO group. Instantiated on hybrid architectures (GDN+attention,
MLA/DSA+MoE) with CP/EP sharding.

**KeysAndValues (this repo)**: bounded-memory long-context training/inference
via *evicting* KV caches (H2O et al.) + chunked forward (LongContextInference-
Model) + cell-checkpointed chunked backward (LongContextGradientModel, with
replay caches) + attention-weight kernels for score-based eviction + CPU
offloading of cache checkpoints.

## Key comparison: KeysAndValues vs OOMB

| axis | OOMB | KeysAndValues |
|---|---|---|
| activation memory | O(1) via chunk-recurrence | bounded via cells/chunk checkpointing |
| KV state | **full** KV, paged + offloaded => grows linearly (~10MB/10K tok) | **bounded** (evicting cache, fixed cache_length) |
| sparsity | page-level (coarse, efficiency-motivated) | token-level score-based eviction (H2O), quality-characterized |
| gradients w.r.t. cache | paged KV-gradient manager | replay-cache reconstruction (autograd hooks) |
| quality accounting | not the focus | measured (HELMET campaign, docs/GRPO_HELMET_RESULTS.md) |

The structural difference: OOMB *keeps everything and hides the cost*
(paging/offload); we *bound the state and pay in accuracy* (eviction). This is
exactly why the two compose differently with LongStraw (below), and why the
accuracy-gap workstream matters for us.

## Integration into LongStraw: mapping the transaction contract

LongStraw's contract has four slots; KeysAndValues has a natural component for
each:

| LongStraw slot | KeysAndValues component | status |
|---|---|---|
| prompt capture w/o autograd | chunked prefill (LongContextInferenceModel) | exists |
| retained state (owned pages) | evicting cache buffers + token_pos (+ H2O scores) -- **bounded**, so restore is O(cache_length), not O(context) | exists, not paged |
| graph-free old/ref scoring | rl/logprobs.compute_logprobs (LogProbsHeadModel) | exists |
| response replay with autograd | LongContextGradientModel + TrainingAttnWeightsReplayCache | exists |

Our USP inside LongStraw would be the **retained-state slot**: an
H2O-compacted cache page set is a small constant, vs OOMB's linearly growing
paged KV. At multi-million-token prompts that difference dominates the
restore/replay cost per group member.

What we do NOT have today:

1. **Prompt-state reuse across the GRPO group.** grpo_step currently expands
   the prompt to G copies and prefill runs G times (batched). LongStraw
   prefills once and restores per member. For us this is tractable
   *single-GPU*: prefill batch-1, snapshot the (bounded) cache buffers +
   token_pos + scores, restore per rollout. Estimated saving: prefill cost
   drops G-fold; at 7k-token prompts and G=8 that is the dominant generation
   cost. This is also the original "cache reuse" idea from the start of the
   RL effort, now with a systems framing and a citable reference.
2. **Paged state / virtualization** (their explicit page ownership) and
   **distributed execution** (CP/EP). Out of scope short-term.
3. Their per-architecture specializations (GDN, MLA/DSA/MoE) do not apply to
   our litgpt GPT class.

**Feasibility verdict**: integration is plausible and attractive in two
stages: (a) adopt LongStraw's *schedule* inside keys_values (single-GPU
prompt-snapshot/restore in grpo_step) -- moderate effort, big win, no external
dependency; (b) implement their transaction interface with keys_values as the
retained-state + replay provider -- depends on their code release quality;
worth a scoping pass once we have (a).

## B1 scoping result (public repo review, MindLab-Research/longstraw)

Verdict from reading `STATUS.md`, `docs/limitations.md`, `ARCHITECTURE.md`:

1. **Adopting the wrapper directly is a non-starter today.** The public tree
   is explicitly `review_only_not_runnable`: no distributable runtime, model
   snapshot, or fixtures; the CLI fail-closes before `ray.init` without their
   deployment "doctor" accepting an immutable pinned image. The
   implementation is GLM-5.2-specific (78-layer MLA/DSA + MoE), built on
   Megatron + Ray + a multi-node Tinker sampler, validated on 32x H20. The
   `integrations/huggingface` surface is a model-snapshot prep script, not a
   generic trainer API. There is no path by which a litgpt GPT on one A10G
   "uses LongStraw" as a library.
2. **The schedule is fully documented and independently validated -- take
   that.** Their "response-only update" loop (restore resident prefix
   boundary -> replay decoder + chunked head per response -> token-aligned
   log-probs -> clipped GRPO objective -> backprop -> restore boundary ->
   accumulate over G -> one optimizer step) is precisely implementable in
   keys_values, and their receipts derisk it: **gradient cosine 0.99993 /
   relative L2 ~0.0117 versus conventional full-sequence gradients at 32K**.
   That parity number validates the response-only-gradient design our loop
   already follows, and justifies building the prompt-snapshot/restore
   optimization (stage (a)) with confidence.
3. **Their architecture split is a good template.** The surface separation
   (prefix_state ownership / capture hooks / response_replay adapter /
   engine-neutral GRPO math / chunked LM head) maps directly onto keys_values
   components; stage (a) should mirror the `prefix_state` + `response_replay`
   boundary so a future stage (b) -- if they ever ship a runnable generic
   backend -- is a thin adapter.
4. **Side-finding for the PE workstream**: they hit the positional question
   at 2M and resolved it the same way we propose -- opt-in RoPE scaling
   (YaRN) for execution, and explicitly "a different position encoding is not
   required before training"; position quality is treated as the *RL training
   objective*, with LongRoPE-style changes deferred unless trained
   checkpoints expose a collapse. This supports our A3 design (train through
   the sparse layout; measure whether adaptation closes the gap) and frames
   compaction/PE changes as inference-quality levers, not training
   prerequisites.

## The accuracy-gap workstream (mseeger: "needs to be dealt with now")

Our measured sparse-vs-dense gaps (base model): ~1pp EM on single-hop QA
(nq/trivia-style), ~11pp on multi-hop (hotpot), catastrophic on synthetic
needle tasks at tight budgets; RL training *shrinks* the hotpot inference gap
to 3-5pp. Levers, in increasing order of invasiveness:

1. **Position handling for retained tokens.** After eviction, retained tokens
   keep their *original* RoPE positions => the query sees a scattered layout
   with holes, unlike anything seen in pretraining. *Position compaction*
   (renumber retained tokens contiguously; cheap with RoPE algebra -- apply a
   delta rotation to retained keys) is the standard mitigation (cf.
   StreamingLLM's within-cache positions). Experiment: A/B original vs
   compacted positions on the longqa/hotpot harness against the known gaps.
   `pos_encoding.py`'s adaptive-context-width classes are the natural hook.
2. **Eviction-policy variants** already in-repo (normalize_scores,
   keep_initial_fraction, grace, smart-lastrec) -- partially characterized in
   issue #140; a systematic sweep against the gap is cheap.
3. **Learned positional adaptation.** Since we already fine-tune with RL/SFT,
   the model can partially adapt to the evicted layout (observed: gap shrinks
   with training). A dedicated adaptation phase (SFT on evicted-cache
   forward) may close more of it.
4. **Alternative encodings (e.g. spherical KV encodings).** A wholesale PE
   replacement cannot be dropped into a pretrained RoPE checkpoint zero-shot,
   but two realistic paths: as a *re-indexing scheme* for retained tokens
   (the compaction rotation in 1 generalizes), or as a *fine-tuned
   adaptation* on top of (3). To be scoped against the sphere-attention
   work.

Recommended order: (1) compaction A/B (days, uses existing harness), (2)
policy sweep (days), then decide on (3)/(4) with data in hand.
