#!/bin/bash
# keys_values RL worker loop: claim jobs from an S3 queue, run, upload results.
#
# Queue layout in s3://keys-values-rl-results/:
#   queue/pending/<job>.sh     one shell snippet per job (the command to run)
#   queue/claimed/<job>.sh     moved here (with instance-id suffix) on claim
#   queue/failed/<job>.sh      parked here if the job exits non-zero
#   runs/<job>/                stdout log + any files the job leaves in $OUT
#
# Three protections, each added after it cost us GPU hours:
#
# 1. One loop per instance (flock). Two loops on one box run two trainings on
#    one GPU, which OOMs both at model-load time.
# 2. Never start on a busy GPU. Belt-and-braces for (1): also covers a loop
#    started before this guard existed, which cannot hold the lock.
# 3. Resolve double claims. Claiming is copy+delete and S3 has no atomic
#    move, so two workers can claim one job. That is not harmless: they race
#    on the result upload and duplicate hours of compute. After claiming we
#    look for competing markers and back off unless we own the smallest
#    instance id.
#
# Usage:  ./worker_loop.sh            # run until queue empty, then exit
#         KV_STOP_WHEN_DONE=1 ./worker_loop.sh   # ...then stop this instance
set -uo pipefail

# --- protection 1: single loop per instance ---------------------------------
if [ "${KV_WORKER_LOCKED:-0}" != "1" ]; then
    export KV_WORKER_LOCKED=1
    exec flock -n "$HOME/.kv_worker.lock" "$0" "$@" || {
        echo "another worker loop already holds the lock on this instance"
        exit 0
    }
fi

BUCKET="s3://keys-values-rl-results"
REGION="us-east-2"
IID=$(TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60") && \
      curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-id)
cd "$HOME/keys_values"
source "$HOME/venv/bin/activate"

while true; do
    JOB=$(aws s3 ls "$BUCKET/queue/pending/" --region $REGION 2>/dev/null \
          | awk '{print $4}' | grep '\.sh$' | head -1)
    if [ -z "$JOB" ]; then
        echo "queue empty"
        break
    fi
    NAME="${JOB%.sh}"
    echo "claiming $NAME"
    # claim: copy to claimed/ then delete from pending/
    aws s3 cp "$BUCKET/queue/pending/$JOB" "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
        --region $REGION --only-show-errors || continue
    aws s3 rm "$BUCKET/queue/pending/$JOB" --region $REGION --only-show-errors

    # --- protection 3: resolve double claims -------------------------------
    # Wait out the window in which a competitor's marker may not be visible
    # yet, then let the smallest instance id win.
    sleep $(( (RANDOM % 5) + 8 ))
    OWNERS=$(aws s3 ls "$BUCKET/queue/claimed/" --region $REGION 2>/dev/null \
             | awk '{print $4}' | grep "^${NAME}\." \
             | sed "s/^${NAME}\.//; s/\.sh$//" | sort)
    WINNER=$(echo "$OWNERS" | head -1)
    if [ -n "$WINNER" ] && [ "$WINNER" != "$IID" ]; then
        echo "double claim on $NAME: $WINNER owns it, dropping our marker"
        aws s3 rm "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            --region $REGION --only-show-errors
        continue
    fi

    aws s3 cp "$BUCKET/queue/claimed/${NAME}.${IID}.sh" "/tmp/${JOB}" \
        --region $REGION --only-show-errors

    # --- protection 2: never start on a busy GPU ---------------------------
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${GPU_USED:-}" ] && [ "$GPU_USED" -gt 2000 ]; then
        echo "GPU busy (${GPU_USED} MiB used): returning $NAME to pending and exiting"
        aws s3 cp "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            "$BUCKET/queue/pending/${JOB}" --region $REGION --only-show-errors
        aws s3 rm "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            --region $REGION --only-show-errors
        break
    fi

    export OUT="$HOME/runs/$NAME"
    mkdir -p "$OUT"
    echo "=== running $NAME on $IID ($(date -u +%FT%TZ)) ==="
    bash "/tmp/${JOB}" > "$OUT/job.log" 2>&1
    STATUS=$?
    echo "exit=$STATUS" >> "$OUT/job.log"

    aws s3 sync "$OUT" "$BUCKET/runs/$NAME/" --region $REGION --only-show-errors
    # Local checkpoints are 15 GB each and are now in S3: drop them, or the
    # disk fills after ~10 jobs and every later job dies on import.
    if [ $? -eq 0 ]; then
        rm -f "$OUT"/*.pt
    fi
    if [ $STATUS -ne 0 ]; then
        # Park failed jobs visibly instead of silently draining the queue;
        # requeue after diagnosis with: aws s3 mv .../failed/X.sh .../pending/X.sh
        aws s3 mv "$BUCKET/queue/claimed/${NAME}.${IID}.sh" \
            "$BUCKET/queue/failed/${JOB}" --region $REGION --only-show-errors
        echo "$NAME FAILED (exit=$STATUS), parked in queue/failed/, log at $BUCKET/runs/$NAME/job.log"
    else
        echo "$NAME done, results at $BUCKET/runs/$NAME/"
    fi
done

if [ "${KV_STOP_WHEN_DONE:-0}" = "1" ]; then
    echo "stopping instance $IID"
    aws ec2 stop-instances --instance-ids "$IID" \
        --region "$(curl -s -H "X-aws-ec2-metadata-token: $(curl -s -X PUT \
          http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" \
          http://169.254.169.254/latest/meta-data/placement/region)"
fi
