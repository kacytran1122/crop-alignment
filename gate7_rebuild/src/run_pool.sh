#!/usr/bin/env bash
# Work pool for the 68 sea-ice models the submitted kappa result actually scores:
# 17 acquisitions x 2 label arms x 2 explicitly different input modes. Seed 42 is
# optical-only and primary; seed 7 is photon-enabled sensitivity. They are never
# averaged as repeat seeds.
set -euo pipefail
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
export SEAICE_TOKENS=$PWD/work/expansion/tokens_subsample.parquet
export SEAICE_FEATS=$PWD/work/expansion/feature_cols.json
mkdir -p logs runs/mosaic

ACQS=$(python -c "import pandas as pd;print(' '.join(sorted(pd.read_parquet('work/expansion/tokens_subsample.parquet').acq_id.unique())))")
JOBS=/tmp/seaice_jobs.txt
: > $JOBS
# These are deliberately different input-mode arms, not repeat seeds to average:
# seed 42 is the optical-only primary result; seed 7 is photon-enabled sensitivity.
for a in $ACQS; do for L in scene original; do
  echo "loao_${L}_${a}_s42|--photon none --labels $L --holdout-acq $a --seed 42 --epochs 12 --patience 5" >> $JOBS
  echo "loao_${L}_${a}_s7|--photon true --labels $L --holdout-acq $a --seed 7 --epochs 12 --patience 5" >> $JOBS
done; done

TODO=/tmp/seaice_todo.txt
: > $TODO
while IFS='|' read -r n rest; do
  if python src/validate_primary_runs.py --run "$n" >/dev/null 2>&1; then
    :
  else
    echo "$n|$rest" >> "$TODO"
  fi
done < $JOBS
TOTAL=$(wc -l < $TODO)
echo "[$(date +%H:%M:%S)] work pool: $TOTAL runs outstanding of $(wc -l < $JOBS)"

declare -A PID
FAILED=0
reap_slot () {
  local k="$1" p="${PID[$k]:-}"
  [ -n "$p" ] || return 0
  if wait "$p"; then
    :
  else
    echo "[$(date +%H:%M:%S)] worker in slot $k failed"
    FAILED=1
  fi
  unset 'PID[$k]'
}
run_one () {   # gpu, name, args
  local g="$1" n="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$g python src/train.py --batch 128 --amp fp16 \
    --lr 1e-4 --wd 1e-4 --fusion deep --encoder resnet18 --loss focal \
    --split-col split --all-train-acq --name "$n" "$@" > "logs/$n.log" 2>&1
}

i=0
exec 3< $TODO
while IFS='|' read -r n rest <&3; do
  # wait until a slot frees, then dispatch to it immediately
  while :; do
    slot=""
    for g in 0 1; do
      p="${PID[$g]:-}"
      if [ -z "$p" ]; then
        slot=$g
        break
      fi
      if ! kill -0 "$p" 2>/dev/null; then
        reap_slot "$g"
        slot=$g
        break
      fi
    done
    [ -n "$slot" ] && break
    sleep 10
  done
  i=$((i + 1))
  echo "[$(date +%H:%M:%S)] ($i/$TOTAL) gpu$slot $n"
  run_one "$slot" "$n" $rest &
  PID[$slot]=$!
done
for g in "${!PID[@]}"; do reap_slot "$g"; done
[ "$FAILED" -eq 0 ] || { echo "TRAINING SWEEP FAILED"; exit 1; }
python src/validate_primary_runs.py --complete
echo "[$(date +%H:%M:%S)] all $TOTAL runs complete; run src/run_kappa.sh next"
