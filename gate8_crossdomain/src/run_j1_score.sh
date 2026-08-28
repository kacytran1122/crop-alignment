#!/usr/bin/env bash
# J1 stage 2: score every arm on the SAME artefact set.
#
# A is fixed at the a100 arm for every run, so the arms differ in what the model was
# trained on and in nothing about how it is measured. Rebuilding A per arm would
# shrink it as crop-dependence fell and empty it at a000, which would make the
# negative control unmeasurable rather than negative.
#
# Full 13x13 grid, no subsampling: kappa is grid-relative, so every arm has to be
# read on one geometry for the comparison to mean anything. One score takes about
# ten seconds, so the whole sweep is twenty minutes on two cards.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate

mkdir -p logs runs_s1fc/kappa
JOBS=/tmp/j1_score_jobs.txt
: > $JOBS
for d in runs_s1fc/j1_*_s*; do
  [ -f "$d/test_metrics.json" ] || continue
  n=$(basename "$d")
  [ -f "runs_s1fc/kappa/${n}.json" ] && continue
  echo "$n" >> $JOBS
done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] J1 scoring: $TOTAL runs"

declare -A PID
i=0
exec 3< $JOBS
while IFS= read -r n <&3; do
  [ -n "$n" ] || continue
  while :; do
    slot=""
    for g in 0 1; do
      p="${PID[$g]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then slot=$g; break; fi
    done
    [ -n "$slot" ] && break
    sleep 2
  done
  i=$((i + 1))
  [ $((i % 20)) -eq 0 ] && echo "[$(date +%H:%M:%S)] ($i/$TOTAL)"
  CUDA_VISIBLE_DEVICES=$slot python src/s1f_crop_kappa.py --run "$n" \
      --grid-step 1 --eval-arm a100 > "logs/kappa_${n}.log" 2>&1 &
  PID[$slot]=$!
done
wait

miss=0
while IFS= read -r n; do
  [ -n "$n" ] || continue
  if [ ! -f "runs_s1fc/kappa/${n}.json" ]; then
    echo "  MISSING kappa: $n"; tail -2 "logs/kappa_${n}.log" 2>/dev/null | sed 's/^/      /'
    miss=$((miss + 1))
  fi
done < $JOBS
echo "[$(date +%H:%M:%S)] scored, $miss of $TOTAL produced nothing"
if [ "$miss" -gt 0 ]; then
  echo "[$(date +%H:%M:%S)] J1 SCORING INCOMPLETE -- summaries would be partial"
  exit 1
fi
echo "[$(date +%H:%M:%S)] J1 SCORING COMPLETE"
