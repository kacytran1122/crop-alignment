#!/usr/bin/env bash
# J1: rebuild the flood calibration on a defensible footing.
#
# WHAT THIS IS FOR, written before any of it has run.
#
# The calibration as it stood could not distinguish "the labels became
# crop-dependent" from "the labels became bad". Three things are fixed here and all
# three were decided before launch:
#
#   1. alpha = 0 is now the PUBLISHED per-event threshold on the 9x9 focal-mean band
#      that section 7 recovers, not a per-chip Otsu of our own. Neither end of the
#      old dial was the label set Sen1Floods11 ships.
#   2. A scramble arm preserves the alpha=1 threshold multiset within each chip
#      while randomly reassigning positions. It is a sensitivity arm, not an
#      isolating control: the mapping allows fixed points and overlapping donors,
#      and every donor comes from the same spatially correlated chip.
#   3. An offset arm varies the threshold per chip at the same 1.453 dB scale but
#      keeps it constant within a chip: crop-invariant, so kappa must stay at floor
#      whatever it does to accuracy.
#
# Hygiene, fixed in the same pass because they share a fold: validation now holds
# out a whole EVENT instead of a 15% draw of chips from every training event, and
# normalisation is computed from that fold's training chips instead of from global
# statistics that had seen the held-out event.
#
# WHAT WOULD FALSIFY THE CLAIM: kappa at perm coming back near kappa at a100. That
# would say the statistic responds to crop-varying labels as such, not to labels
# derived from the crop's own pixels, and the calibration would not support the
# reading the paper gives it. All arms are reported either way.
#
# Failure handling is explicit because the last sweep in this project printed
# COMPLETE with 22 of 55 results on disk: every job's exit status is recorded, and
# the run refuses to report success if any job produced no metrics file.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate

ARMS="a000 a025 a050 a075 a100 perm offc"
EVENTS="Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA"
SEEDS="42 7 123"
EPOCHS=30

mkdir -p logs runs_s1fc
JOBS=/tmp/j1_jobs.txt
: > $JOBS
for arm in $ARMS; do
  for ev in $EVENTS; do
    for sd in $SEEDS; do
      n="j1_${arm}_${ev}_s${sd}"
      [ -f "runs_s1fc/${n}/test_metrics.json" ] && continue
      # tag:value pairs, never bare numbers: a zero-padded bare token is what the
      # alpha sweep passed to python -c and 050 is a malformed octal literal
      echo "name:${n}|arm:${arm}|event:${ev}|seed:${sd}" >> $JOBS
    done
  done
done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] J1: $TOTAL runs outstanding"

declare -A PID
exec 3< $JOBS
i=0
while IFS= read -r line <&3; do
  n=$(echo "$line"  | tr '|' '\n' | sed -n 's/^name://p')
  arm=$(echo "$line" | tr '|' '\n' | sed -n 's/^arm://p')
  ev=$(echo "$line"  | tr '|' '\n' | sed -n 's/^event://p')
  sd=$(echo "$line"  | tr '|' '\n' | sed -n 's/^seed://p')
  if [ -z "$n" ] || [ -z "$arm" ] || [ -z "$ev" ] || [ -z "$sd" ]; then
    echo "  MALFORMED JOB LINE: $line"; continue
  fi
  while :; do
    slot=""
    for g in 0 1; do
      p="${PID[$g]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then slot=$g; break; fi
    done
    [ -n "$slot" ] && break
    sleep 5
  done
  i=$((i + 1))
  echo "[$(date +%H:%M:%S)] ($i/$TOTAL) gpu$slot $n"
  CUDA_VISIBLE_DEVICES=$slot python src/s1f_crop_train.py \
      --holdout-event "$ev" --arm "$arm" --seed "$sd" --epochs $EPOCHS \
      --name "$n" > "logs/${n}.log" 2>&1 &
  PID[$slot]=$!
done
wait
echo "[$(date +%H:%M:%S)] J1 training finished, $i launched"

# say loudly if anything produced nothing
miss=0
while IFS= read -r line; do
  n=$(echo "$line" | tr '|' '\n' | sed -n 's/^name://p')
  if [ ! -f "runs_s1fc/${n}/test_metrics.json" ]; then
    echo "  MISSING: $n   (tail of its log)"
    tail -3 "logs/${n}.log" 2>/dev/null | sed 's/^/      /'
    miss=$((miss + 1))
  fi
done < $JOBS
echo "[$(date +%H:%M:%S)] runs with no metrics file: $miss of $TOTAL"
if [ "$miss" -gt 0 ]; then
  echo "[$(date +%H:%M:%S)] J1 INCOMPLETE -- do not read the summaries as final"
  exit 1
fi
echo "[$(date +%H:%M:%S)] J1 TRAINING COMPLETE, all $TOTAL runs produced metrics"
