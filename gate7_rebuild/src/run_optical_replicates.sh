#!/usr/bin/env bash
# Two matched optical-only sea-ice seeds, queued without oversubscribing GPUs that
# are already busy. Safe under SSH disconnect when launched with nohup/setsid.
set -euo pipefail
ROOT="${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT"
. .venv/bin/activate
export SEAICE_ROOT="$ROOT"
export SEAICE_CACHE="${SEAICE_CACHE:-/dev/shm/seaice_expanded}"
export SEAICE_TOKENS="${SEAICE_TOKENS:-$ROOT/work/expansion/tokens_subsample.parquet}"
export SEAICE_FEATS="${SEAICE_FEATS:-$ROOT/work/expansion/feature_cols.json}"

GPUS="${GPUS:-0 1}"
GPUS="${GPUS//,/ }"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"
MAX_BUSY_UTIL="${MAX_BUSY_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-30}"
mkdir -p audit_r2/logs runs
QUEUE="$ROOT/audit_r2/optical_queue.tsv"
QUEUE_LOCK="$ROOT/audit_r2/optical_queue.lock"
FAILED="$ROOT/audit_r2/optical_failed.tsv"
: > "$QUEUE"
: > "$FAILED"

ACQS=$(python -c "import json; d=json.load(open('audit_r2/DESIGN.json'))['sea_ice']; print(' '.join(d['acquisitions']))")
SEEDS=$(python -c "import json; d=json.load(open('audit_r2/DESIGN.json'))['sea_ice']; print(' '.join(map(str,d['optical_replication_seeds'])))")
for seed in $SEEDS; do
  for acq in $ACQS; do
    for labels in scene original; do
      name="loao_optical_${labels}_${acq}_s${seed}"
      if python audit_r2/src/validate_optical_replicates.py --run "$name" >/dev/null 2>&1; then
        continue
      fi
      printf '%s|%s|%s|%s\n' "$name" "$labels" "$acq" "$seed" >> "$QUEUE"
    done
  done
done
echo "[$(date --iso-8601=seconds)] queued $(wc -l < "$QUEUE") of 68 runs"

claim_job () {
  local line="" tmp
  exec 9>>"$QUEUE_LOCK"
  flock 9
  if [ -s "$QUEUE" ]; then
    line=$(head -n 1 "$QUEUE")
    tmp="${QUEUE}.candidate.$$"
    tail -n +2 "$QUEUE" > "$tmp"
    mv "$tmp" "$QUEUE"
  fi
  flock -u 9
  exec 9>&-
  printf '%s' "$line"
}

wait_for_gpu () {
  local gpu="$1" ready=0 free util
  while [ "$ready" -lt 2 ]; do
    IFS=',' read -r free util < <(nvidia-smi -i "$gpu" \
      --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
    free=${free//[[:space:]]/}; util=${util//[[:space:]]/}
    if [ "$free" -ge "$MIN_FREE_MB" ] && [ "$util" -le "$MAX_BUSY_UTIL" ]; then
      ready=$((ready + 1))
    else
      ready=0
      echo "[$(date --iso-8601=seconds)] gpu$gpu waiting: free=${free}MB util=${util}%"
    fi
    [ "$ready" -ge 2 ] || sleep "$POLL_SECONDS"
  done
}

worker () {
  local gpu="$1" line name labels acq seed
  while :; do
    # Wait before claiming. A busy card must not hold the final queued job while
    # the other card is free and capable of completing the whole sweep alone.
    [ -s "$QUEUE" ] || break
    wait_for_gpu "$gpu"
    line=$(claim_job)
    [ -n "$line" ] || break
    IFS='|' read -r name labels acq seed <<< "$line"
    echo "[$(date --iso-8601=seconds)] gpu$gpu starting $name"
    if CUDA_VISIBLE_DEVICES="$gpu" python audit_r2/src/train.py \
        --batch 128 --amp fp16 --lr 1e-4 --wd 1e-4 --fusion deep \
        --encoder resnet18 --loss focal --split-col split --all-train-acq \
        --photon none --labels "$labels" --holdout-acq "$acq" --seed "$seed" \
        --epochs 12 --patience 5 --name "$name" \
        > "audit_r2/logs/${name}.log" 2>&1 && \
       python audit_r2/src/validate_optical_replicates.py --run "$name" \
        >> "audit_r2/logs/${name}.log" 2>&1; then
      echo "[$(date --iso-8601=seconds)] gpu$gpu completed $name"
    else
      echo "$line" >> "$FAILED"
      echo "[$(date --iso-8601=seconds)] gpu$gpu FAILED $name"
    fi
  done
}

pids=()
for gpu in $GPUS; do worker "$gpu" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ -s "$FAILED" ]; then
  echo "FAILED RUNS:"
  cat "$FAILED"
  exit 1
fi
python audit_r2/src/validate_optical_replicates.py --complete
echo "[$(date --iso-8601=seconds)] all 68 matched optical-only runs complete"
exit "$status"
