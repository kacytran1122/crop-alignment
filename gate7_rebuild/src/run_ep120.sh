#!/usr/bin/env bash
# Third training budget: 120 epochs, all 17 folds, both arms.
#
# WHY THIS IS RUN, stated before the numbers exist, because the reason has to be
# independent of what comes back. 21 of the 34 sixty-epoch runs reached the cap
# without early stopping ever firing -- with patience 15 and a cap of 60, any best
# epoch at or after 46 means the run was still improving when the budget ran out.
# The paper cannot describe 60 epochs as a converged budget while that is true, and
# a reviewer would be right to press it. That is a convergence criterion, not a
# result, and it would have justified this block whichever way the premium moved.
#
# What it is NOT run for: the 60-epoch premium came back at +0.0522 (t = 1.67), so
# it is still unresolved at n = 17, and d/s improved from 0.200 to 0.405 between the
# two budgets against the 0.679 that resolution at this sample size would need.
# Extending the budget until an effect crosses a threshold is the practice this
# paper audits. So the stopping rule is identical to the other two budgets -- only
# the cap changes -- and ALL THREE budgets get reported whatever this returns.
#
# The 6-fold preview of the 60-epoch block predicted the premium would sharpen
# (+0.0483, t = 2.13, spread falling by a third). At 17 folds it did not: the spread
# ROSE from 0.0910 to 0.1290 as two folds became extreme. That is a small-sample
# artefact of exactly the kind this paper is about, in our own analysis, and it is
# the reason this block runs all 17 folds and not a subset.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
export SEAICE_TOKENS=$PWD/work/expansion/tokens_subsample.parquet
export SEAICE_FEATS=$PWD/work/expansion/feature_cols.json
mkdir -p logs runs/mosaic

ACQS=$(python -c "import pandas as pd;print(' '.join(sorted(pd.read_parquet('work/expansion/tokens_subsample.parquet').acq_id.unique())))")

JOBS=/tmp/ep120_jobs.txt
: > $JOBS
for a in $ACQS; do for L in scene original; do
  n="ep120_${L}_${a}_s42"
  [ -f "runs/$n/test_metrics.json" ] && continue
  echo "$n|--labels $L --holdout-acq $a --seed 42 --epochs 120 --patience 15" >> $JOBS
done; done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] ep120: $TOTAL runs outstanding"

declare -A PID
i=0
exec 3< $JOBS
while IFS='|' read -r n rest <&3; do
  while :; do
    slot=""
    for g in 0 1; do
      p="${PID[$g]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then slot=$g; break; fi
    done
    [ -n "$slot" ] && break
    sleep 10
  done
  i=$((i + 1))
  echo "[$(date +%H:%M:%S)] ($i/$TOTAL) gpu$slot $n"
  CUDA_VISIBLE_DEVICES=$slot python src/train.py --batch 128 --amp fp16 \
    --name "$n" $rest > "logs/$n.log" 2>&1 &
  PID[$slot]=$!
done
wait
echo "[$(date +%H:%M:%S)] ep120 training complete, $i runs"

# mosaic-score two wide, and say so loudly if anything produced nothing
MOSJOBS=/tmp/ep120_mosaic.txt
: > $MOSJOBS
for d in runs/ep120_*_s42; do
  [ -d "$d" ] || continue
  r=$(basename "$d")
  [ -f "runs/mosaic/$r.json" ] && continue
  [ -f "$d/best.pt" ] || continue
  echo "$r" >> $MOSJOBS
done
echo "[$(date +%H:%M:%S)] mosaic-scoring $(wc -l < $MOSJOBS) runs, two wide"
score_half () {
  local g="$1" par="$2" k=0
  while read -r r; do
    k=$((k + 1))
    [ $((k % 2)) -eq "$par" ] || continue
    CUDA_VISIBLE_DEVICES=$g python src/mosaic_scale.py --run "$r" --batch 96 \
      >> "logs/mosaic_ep120_g$g.log" 2>&1
  done < $MOSJOBS
}
score_half 0 0 &
score_half 1 1 &
wait

nmiss=0
for r in $(cat $MOSJOBS); do
  [ -f "runs/mosaic/$r.json" ] || { echo "  MISSING mosaic: $r"; nmiss=$((nmiss+1)); }
done
echo "[$(date +%H:%M:%S)] mosaic complete, $nmiss missing"

python src/mosaic_premium.py 2>&1 | tee runs/mosaic_premium.txt
python src/power.py           2>&1 | tee runs/power.txt
echo "[$(date +%H:%M:%S)] EP120 BLOCK COMPLETE"
