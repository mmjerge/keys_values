#!/usr/bin/env bash
# Queue RLVR (Sparrow-protocol) jobs for the single-GPU worker fleet.
#
#   scripts/queue_rlvr_jobs.sh probe  <model> <arm>            [prefix]
#   scripts/queue_rlvr_jobs.sh train  <model> <arm> <seeds>    [prefix]
#
#   arm = h2o<K>   bounded evicting cache, K slots (e.g. h2o4096)
#       | dense    dense-default cache sized to prompt + generation
#
# Env knobs: KV_MAXNEW (8192), KV_STEPS (200), KV_GROUP (8), KV_EVAL_SETS
# (math500,aime24,aime25,amc23), KV_N_EVAL (0 = full sets), KV_EVAL_SAMPLES
# (1), KV_TRAIN (deepscaler), KV_BRANCH (rl-longproc), KV_EXTRA (appended).
#
# Jobs are claimed alphabetically: prefixes starting with "aa" jump the queue.
set -euo pipefail
MODE="${1:?probe|train}"
MODEL="${2:?HF model id, e.g. Qwen/Qwen3-1.7B}"
ARM="${3:?h2o<K>|dense}"
if [ "$MODE" = train ]; then
  SEEDS="${4:?comma-separated seeds}"; PREFIX="${5:-rlvr}"
else
  SEEDS="0"; PREFIX="${4:-rlvr}"
fi

BUCKET="s3://keys-values-rl-results"
REGION="us-east-2"
BRANCH="${KV_BRANCH:-rl-longproc}"
MAXNEW="${KV_MAXNEW:-8192}"
STEPS="${KV_STEPS:-200}"
GROUP="${KV_GROUP:-8}"
EVAL_SETS="${KV_EVAL_SETS:-math500,aime24,aime25,amc23}"
N_EVAL="${KV_N_EVAL:-0}"
EVAL_SAMPLES="${KV_EVAL_SAMPLES:-1}"
TRAIN="${KV_TRAIN:-deepscaler}"
EXTRA="${KV_EXTRA:-}"

case "$ARM" in
  h2o*)  CACHE_ARGS="--kv-cache-name h2o-torch-quantized8 --cache-length ${ARM#h2o}" ;;
  dense) # prompts are a few hundred tokens; 1024 headroom covers them
         CACHE_ARGS="--kv-cache-name dense-default --cache-length $((MAXNEW + 1024)) --dense-baseline" ;;
  *)     echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
MTAG=$(echo "$MODEL" | sed 's#.*/##; s/[^A-Za-z0-9.]/-/g' | tr 'A-Z' 'a-z')

TMP=$(mktemp -d)
for seed in ${SEEDS//,/ }; do
  if [ "$MODE" = probe ]; then
    NAME="${PREFIX}_probe_${MTAG}_${ARM}"
    RUN_ARGS="--eval-only --eval-sets ${EVAL_SETS} --n-eval ${N_EVAL} --eval-samples ${EVAL_SAMPLES}"
  else
    NAME="${PREFIX}_train_${MTAG}_${ARM}_${TRAIN}_s${seed}"
    RUN_ARGS="--train-dataset ${TRAIN} --steps ${STEPS} --group-size ${GROUP} \\
    --prompts-per-update 2 --adv-mode grpo --lr 1e-6 --optimizer paged_adamw8bit \\
    --eval-sets ${EVAL_SETS} --n-eval ${N_EVAL} --eval-samples ${EVAL_SAMPLES} \\
    --eval-every 50 --seed ${seed}"
  fi
  cat > "$TMP/$NAME.sh" <<EOF
cd ~/keys_values
git fetch -q origin && git checkout -q ${BRANCH} && git reset -q --hard origin/${BRANCH}
source ~/venv/bin/activate
pip install -q bitsandbytes==0.49.1 math-verify==0.8.0
# System CUDA ahead of torch's bundled libs makes cuDNN SDPA abort with
# "cublasLtGetVersion" (exit 134); torch finds its own libs without this.
unset LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python examples/grpo_rlvr.py --device cuda --model ${MODEL} \\
    ${CACHE_ARGS} --max-new-tokens ${MAXNEW} --chunk-size 1024 \\
    ${RUN_ARGS} --disable-flashinfer ${EXTRA} --out-dir \$OUT
EOF
  aws s3 cp "$TMP/$NAME.sh" "$BUCKET/queue/pending/$NAME.sh" \
      --region $REGION --only-show-errors
  echo "queued $NAME"
  [ "$MODE" = probe ] && break
done
rm -rf "$TMP"
echo "pending now: $(aws s3 ls $BUCKET/queue/pending/ --region $REGION | wc -l | tr -d ' ') jobs"
