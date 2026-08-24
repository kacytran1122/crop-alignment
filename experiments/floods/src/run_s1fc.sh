#!/usr/bin/env bash
# Dose-response for kappa: 5 settings of the dial x 11 held-out events.
#
# Usage: run_s1fc.sh GPU PART NPARTS
# Each instance takes the jobs whose index mod NPARTS equals PART, so the sweep can
# start as a single stream beside the sea-ice pool and pick up a second card later
# without restarting or double-running anything.
#
# Train and score each model back to back rather than in two phases. The kappa pass
# is the only reason the model exists, it needs the weights that were just written,
# and pairing them means a killed sweep leaves finished results rather than a pile
# of checkpoints waiting on a second stage.
#
# Ordered alpha-outward: 1.0 and 0.0 first across all events, then the interior
# points. The two ends are what carry the claim -- full closed-loop against the
# structural-null control -- so if this has to be stopped early, what survives is
# the comparison that matters and not half a curve.
set -u
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
. .venv/bin/activate
mkdir -p runs_s1fc logs

G="$1"; PART="$2"; NPARTS="$3"
EVENTS="Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA"

JOBS=/tmp/s1fc_jobs_${PART}.txt
: > $JOBS
for A in 100 000 050 025 075; do
  for e in $EVENTS; do
    echo "${A}|${e}" >> $JOBS
  done
done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] s1fc sweep: $TOTAL jobs total, this stream takes part $PART of $NPARTS on gpu$G"

i=0
while IFS='|' read -r A e; do
  i=$((i + 1))
  [ $(( (i - 1) % NPARTS )) -eq "$PART" ] || continue
  n="s1fc_a${A}_${e}_s42"
  af=$(python -c "print(${A}/100.0)")
  if [ ! -f "runs_s1fc/$n/test_metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=$G python src/s1f_crop_train.py --holdout-event "$e" \
      --alpha "$af" --seed 42 --epochs 20 --patience 6 --name "$n" \
      >> logs/s1fc_g${G}.log 2>&1
  fi
  if [ -f "runs_s1fc/$n/best.pt" ] && [ ! -f "runs_s1fc/kappa/$n.json" ]; then
    CUDA_VISIBLE_DEVICES=$G python src/s1f_crop_kappa.py --run "$n" --batch 64 \
      >> logs/s1fc_g${G}.log 2>&1
  fi
  echo "[$(date +%H:%M:%S)] ($i/$TOTAL) $n done"
done < $JOBS
echo "[$(date +%H:%M:%S)] S1FC STREAM $PART COMPLETE"
