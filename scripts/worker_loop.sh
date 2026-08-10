#!/bin/bash
# keys_values RL worker loop: claim jobs from an S3 queue, run, upload results.
#
# Queue layout in s3://keys-values-rl-results/:
#   queue/pending/<job>.sh     one shell snippet per job (the command to run)
#   queue/claimed/<job>.sh     moved here (with instance-id suffix) on claim
#   runs/<job>/                stdout log + any files the job leaves in $OUT
#
# Claiming is a copy+delete; S3 has no atomic move, so a rare double-claim is
# possible -- harmless for idempotent experiment jobs (last writer wins on
# upload, both produce the same result).
#
# Usage:  ./worker_loop.sh            # run until queue empty, then exit
#         KV_STOP_WHEN_DONE=1 ./worker_loop.sh   # ...then stop this instance
set -uo pipefail

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
    aws s3 cp "$BUCKET/queue/claimed/${NAME}.${IID}.sh" "/tmp/${JOB}" \
        --region $REGION --only-show-errors

    export OUT="$HOME/runs/$NAME"
    mkdir -p "$OUT"
    echo "=== running $NAME on $IID ($(date -u +%FT%TZ)) ==="
    bash "/tmp/${JOB}" > "$OUT/job.log" 2>&1
    STATUS=$?
    echo "exit=$STATUS" >> "$OUT/job.log"

    aws s3 sync "$OUT" "$BUCKET/runs/$NAME/" --region $REGION --only-show-errors
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
