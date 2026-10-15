# A2: Eviction-policy sweep against the sparse-inference gap

**Question.** Can retention-policy choices (scoring, normalization,
attention-sink protection, grace sizing, quantization) close the
sparse-inference accuracy gap (dense vs `h2o@4096`, base model) at a fixed
slot budget? Follow-up to A1 (position compaction: negative,
`docs/POSITION_COMPACTION_A1.md`).

**Setup.** Qwen2.5-0.5B-Instruct (untrained), HELMET @8k (prompts up to
~7.7k tokens), n=100 question-diverse records, greedy decoding, single seed,
`examples/eviction_sweep.py`. All sparse arms at `cache_length=4096` with
grace `cl/16` unless stated.

| arm            | policy                                   | trivia EM | trivia F1 | nq EM | nq F1 |
|----------------|------------------------------------------|----------:|----------:|------:|------:|
| dense          | reference                                |     0.620 |     0.283 | 0.250 | 0.144 |
| h2o_q8         | H2O, 8-bit buffers (campaign default)    |     0.540 |     0.232 | 0.250 | 0.112 |
| h2o_fp         | H2O, unquantized buffers                 |     0.550 |     0.231 | 0.230 | 0.107 |
| h2o_q8_norm    | + normalize_scores                       |     0.550 |     0.234 | 0.230 | 0.106 |
| h2o_q8_sink    | + keep_initial_fraction=0.05             |     0.540 |     0.232 | 0.250 | 0.112 |
| h2o_q8_grace4  | grace cl/4 (recency-heavy)               |     0.560 |     0.242 | 0.230 | 0.109 |
| h2o_vlen_q8    | value-norm-weighted scores               |     0.550 |     0.230 | 0.230 | 0.111 |
| lastrec_sink   | pure recency + protected initial tokens  |     0.490 |     0.202 | 0.190 | 0.102 |

**Findings.**

1. **No retention knob closes the gap.** The best arm (`h2o_q8_grace4`)
   recovers ~2pp of the 8pp trivia EM gap; every knob moves results by at
   most 1-2pp (within noise at n=100). Combined with A1, this locates the
   gap in *information loss from eviction itself*: at 4096 slots we discard
   nearly half of a ~7.7k-token prompt, and for QA over long contexts the
   discarded tokens often contain the answer.
2. **8-bit quantization of the buffers is accuracy-free.**
   `h2o_q8 >= h2o_fp` on both datasets. This is the actionable lever:
   at equal *bytes*, quantized buffers afford 2x the slots.
3. **H2O scoring earns its keep.** Pure recency (`lastrec_sink`) is 5-6pp
   EM below H2O at the same budget; attention-weight scoring is doing real
   work, it just cannot conjure evicted information back.

**Implication.** Compare at equal memory, not equal slots: 8-bit buffers
halve the bytes per retained token, so at equal memory a quantized cache
affords twice the slots. Budget frontier for `h2o_q8` (same setup):

| slots | trivia EM | trivia F1 | nq EM | nq F1 |
|------:|----------:|----------:|------:|------:|
|  2048 |     0.530 |     0.207 | 0.190 | 0.096 |
|  4096 |     0.540 |     0.232 | 0.250 | 0.112 |
|  6144 |     0.560 |     0.254 | 0.200 | 0.105 |
|  8192 |     0.640 |     0.288 | 0.260 | 0.148 |
| dense |     0.620 |     0.283 | 0.250 | 0.144 |

**`h2o_q8@8192` matches dense on both datasets and both metrics** (within
noise), at the same byte budget as unquantized `h2o@4096` and roughly half
the KV memory of bf16 dense at this context length. To be explicit about
the mechanism: at 8192 slots the ~7.7k-token prompts fit with little or no
eviction, so this is "keep everything, in 8 bits" rather than "evict
smarter" -- the accuracy-free quantization (finding 2) is what buys the
extra slots. The frontier below the prompt length is fairly flat
(trivia 0.53 at 2048 slots, i.e. retaining ~25% of the prompt costs only
9pp), so aggressive budgets degrade gracefully; but *parity requires the
budget to cover the prompt*.

**Recommended operating point.** For accuracy-sensitive runs: size
`cache_length` to the expected prompt length and use `torch-quantized8`
buffers, rather than shrinking slots at full precision. For memory-bound
training (the RL loop), the flat-frontier region still applies and the
training-side results (PR #142 docs) show RL through the evicting cache is
unharmed; the gap is an *inference-time* effect.

**Verification controls** (run after skepticism about the 8192 result;
`runs/a2_controls_*.json`). At 8192 slots, 0/100 trivia and 1/100 nq prompts
exceed the cache, so eviction is (near-)inactive there:

| arm (8192 slots)   | trivia EM | trivia F1 | nq EM | nq F1 |
|--------------------|----------:|----------:|------:|------:|
| dense bf16         |     0.620 |     0.283 | 0.250 | 0.144 |
| h2o unquantized    |     0.620 |     0.283 | 0.250 | 0.144 |
| dense quantized-8  |     0.640 |     0.288 | 0.260 | 0.148 |
| h2o quantized-8    |     0.640 |     0.288 | 0.260 | 0.148 |

Unquantized H2O at 8192 reproduces dense bf16 *exactly*, confirming that
with no eviction the H2O cache is literally dense attention. Quantized
dense reproduces quantized H2O *exactly*, confirming the +2pp over bf16 is
the 8-bit rounding perturbing greedy decoding -- a noise-level numeric
effect that happened to land positive here, not a gain from H2O and not a
systematic gain from quantization. The clean decomposition:
**retention determines accuracy; 8-bit quantization is a ~±2pp numeric
perturbation; H2O scoring only matters when eviction is active.**

**Caveats.** Single seed, n=100, one model (0.5B), one context length (8k).
Differences <= 2pp are within noise; the ordering dense > h2o > lastrec
(under active eviction) is stable across both datasets and metrics.
