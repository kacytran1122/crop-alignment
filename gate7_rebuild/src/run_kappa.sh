#!/usr/bin/env bash
# The crop-alignment measurement across every fold, with its negative control.
#
# For each held-out acquisition, two numbers on the SAME artefact pixels with the
# SAME per-crop labels, differing only in what the model was trained on:
#
#   treatment  U-Net trained on the crop-noisy labels
#   control    U-Net trained on the crop-invariant labels, which has never seen a
#              crop-dependent label and so should sit at the structural null
#
# Inference only, from checkpoints already on disk.
#
# INPUT MODES. Seed 42 checkpoints are the optical-only primary arm. Seed 7
# checkpoints use the photon branch and are scored as a separately labelled
# input-mode sensitivity check; they must never be averaged as repeat seeds.
set -euo pipefail
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
mkdir -p runs/kappa logs
python src/validate_primary_runs.py --complete

SEEDS="${1:-42 7}"
GPUS="${2:-0 1}"
# one entry per slot, so several slots can share a card; PID is keyed by slot index
SLOTGPU=(); for g in $GPUS; do SLOTGPU+=("$g"); done
NSLOT=${#SLOTGPU[@]}

ACQS=$(python -c "import pandas as pd;print(' '.join(sorted(pd.read_parquet('work/expansion/tokens_subsample.parquet').acq_id.unique())))")

JOBS=/tmp/kappa_jobs.txt
: > $JOBS
for s in $SEEDS; do
  for a in $ACQS; do
    t="loao_original_${a}_s${s}"
    c="loao_scene_${a}_s${s}"
    if ! python src/validate_primary_runs.py --score "runs/kappa/$t.json" \
        >/dev/null 2>&1; then
      echo "run:${t}|tag:" >> $JOBS
    fi
    if ! python src/validate_primary_runs.py --score \
        "runs/kappa/${c}_vsorig.json" >/dev/null 2>&1; then
      echo "run:${c}|tag:_vsorig" >> $JOBS
    fi
  done
done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] crop alignment: $TOTAL runs, seeds [$SEEDS], $NSLOT slots"

declare -A PID
FAILED=0
reap_slot () {
  local k="$1" p="${PID[$k]:-}"
  [ -n "$p" ] || return 0
  if wait "$p"; then
    :
  else
    echo "[$(date +%H:%M:%S)] scorer in slot $k failed"
    FAILED=1
  fi
  unset 'PID[$k]'
}
i=0
exec 3< $JOBS
while IFS= read -r line <&3; do
  r=$(echo "$line" | tr '|' '\n' | sed -n 's/^run://p')
  tg=$(echo "$line" | tr '|' '\n' | sed -n 's/^tag://p')
  [ -n "$r" ] || continue
  while :; do
    slot=""
    for k in $(seq 0 $((NSLOT - 1))); do
      pd="${PID[$k]:-}"
      if [ -z "$pd" ]; then
        slot=$k
        break
      fi
      if ! kill -0 "$pd" 2>/dev/null; then
        reap_slot "$k"
        slot=$k
        break
      fi
    done
    [ -n "$slot" ] && break
    sleep 5
  done
  i=$((i + 1))
  echo "[$(date +%H:%M:%S)] ($i/$TOTAL) slot$slot gpu${SLOTGPU[$slot]} ${r}${tg}"
  if [ "$tg" = "_vsorig" ]; then
    CUDA_VISIBLE_DEVICES=${SLOTGPU[$slot]} python src/crop_alignment.py --run "$r" \
        --eval-labels original --tag _vsorig --batch 96 \
        > "logs/kappa_${r}_vsorig.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=${SLOTGPU[$slot]} python src/crop_alignment.py --run "$r" \
        --batch 96 > "logs/kappa_${r}.log" 2>&1 &
  fi
  PID[$slot]=$!
done
for k in "${!PID[@]}"; do reap_slot "$k"; done
[ "$FAILED" -eq 0 ] || { echo "SCORING SWEEP FAILED"; exit 1; }

# Every queued job must have produced its file. A sweep that silently scored a
# subset would still print a summary, which is the failure this project is about.
miss=0
while IFS= read -r line; do
  r=$(echo "$line" | tr '|' '\n' | sed -n 's/^run://p')
  tg=$(echo "$line" | tr '|' '\n' | sed -n 's/^tag://p')
  [ -n "$r" ] || continue
  if [ ! -f "runs/kappa/${r}${tg}.json" ]; then
    echo "  MISSING: ${r}${tg}"
    tail -3 "logs/kappa_${r}${tg}.log" 2>/dev/null | sed 's/^/      /'
    miss=$((miss + 1))
  fi
done < $JOBS
echo "[$(date +%H:%M:%S)] crop-alignment sweep done, $miss of $TOTAL produced nothing"
[ "$miss" -gt 0 ] && { echo "SWEEP INCOMPLETE"; exit 1; }
python src/validate_primary_runs.py --scores-complete

# every output must carry the pixel-weighted statistic, or the set is mixed
np=$(grep -l kappa_pixel runs/kappa/*.json 2>/dev/null | wc -l)
na=$(ls runs/kappa/*.json 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] $np of $na outputs carry kappa_pixel"
[ "$np" -eq "$na" ] || { echo "MIXED WEIGHTING SET"; exit 1; }

python src/kappa_summary.py 2>&1 | tee runs/kappa_summary.txt
