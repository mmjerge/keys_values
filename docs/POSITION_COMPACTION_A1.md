# A1: Position compaction at inference (negative result)

**Question.** Is the sparse-inference accuracy gap (evicting H2O cache vs
dense, ~8-14pp EM on HELMET QA tasks at `h2o@4096`) caused by the "holey"
RoPE position layout that eviction leaves behind (retained tokens keep
their original absolute positions, e.g. 0, 3, 47, 812, ...)?

**Method.** `keys_values/kvcache/pos_compact.py` re-expresses queries and
cached keys at *compacted* positions: each retained token's position becomes
its rank among survivors. The transform is an exact delta-rotation (RoPE
rotations compose), applied at attention time; relative order is preserved so
the causal mask is unchanged. Correctness is enforced in
`test/kvcache/test_pos_compact.py` (delta-rotation exactness, identity on
dense caches). A/B via
`examples/grpo_helmet_crosseval.py --compact-positions`, which adds an
`h2o+compact` eval arm.

**Result.** Qwen2.5-0.5B-Instruct (untrained), HELMET @8k, n=100
question-diverse records, `h2o-torch-quantized8@4096`, grace `cl/16`,
greedy decoding, single seed:

| dataset   | eval cache   |    EM |    F1 |
|-----------|--------------|------:|------:|
| nq        | dense        | 0.250 | 0.144 |
| nq        | h2o          | 0.250 | 0.112 |
| nq        | h2o+compact  | 0.240 | 0.108 |
| trivia_qa | dense        | 0.620 | 0.283 |
| trivia_qa | h2o          | 0.540 | 0.232 |
| trivia_qa | h2o+compact  | 0.520 | 0.229 |

**Conclusion.** Position compaction does **not** recover the gap; if
anything it is marginally worse (differences of 1-2pp are within noise at
n=100, but there is no recovery in either dataset on either metric). The
holey position layout is therefore unlikely to be the main cause of the
sparse-inference gap for this model/task combination. The more likely cause
is information loss from eviction itself (the needed tokens are gone, so no
re-encoding of the survivors can help). This shifts priority to A2
(eviction-policy quality: score normalization, grace sizing, retention
policies such as rate-distortion-style keep/drop) over positional re-encoding
at inference.

**Caveats.** Single seed, n=100, one model size (0.5B), one cache length.
The result rules out compaction as a *sufficient* fix at this operating
point; it does not rule out positional effects mattering at longer contexts
or tighter cache budgets, where relative-distance distortion is larger.

Reproduce:

```bash
python examples/grpo_helmet_crosseval.py --device cuda --dataset-key trivia_qa \
    --checkpoints base --h2o-cache-length 4096 --n-eval 100 \
    --disable-flashinfer --compact-positions --out runs/a1_compact_trivia.json
```
