#!/usr/bin/env bash
# Finish the converged-budget arm on all 17 acquisitions, not 6.
#
# Why this is worth another block of card. At 12 epochs the mosaic-scored premium
# is +0.0182 with sd 0.0859 over 17 folds, which is the number behind the papers
# own admission that it cannot resolve its own effect: required n = 238. On the 6
# folds run to 60 epochs the same mosaic-scored premium is +0.0483 with sd 0.0555.
# Both terms move the right way at once, the effect roughly doubles and the spread
# drops by a third, because the 12-epoch runs were stopping at epoch 10 while the
# long runs stop between 26 and 59. If that holds at n=17 the required sample size
# falls to roughly 11 and the premium becomes resolvable with the data we have,
# which turns a reported failure into a reported result.
#
# It is 6 folds, so it could equally be the outlier fold flattering us: that fold
# alone moved +0.0883 of the +0.0277 paired change. The only way to find out is the
# other 11 folds, and the cards are idle.
#
# Same barrier-free pool as before: each GPU pulls the next job the moment it frees,
# and the mosaic pass afterwards runs TWO WIDE rather than sequentially on GPU 0,
# which is where the last block wasted about forty minutes.
set -u
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
export SEAICE_TOKENS=$PWD/work/expansion/tokens_subsample.parquet
export SEAICE_FEATS=$PWD/work/expansion/feature_cols.json
mkdir -p logs runs/mosaic

ACQS=$(python -c "import pandas as pd;print(' '.join(sorted(pd.read_parquet('work/expansion/tokens_subsample.parquet').acq_id.unique())))")

JOBS=/tmp/ep60_rest.txt
: > $JOBS
for a in $ACQS; do for L in scene original; do
  n="ep60_${L}_${a}_s42"
  [ -f "runs/$n/test_metrics.json" ] && continue
  echo "$n|--labels $L --holdout-acq $a --seed 42 --epochs 60 --patience 15" >> $JOBS
done; done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] ep60 completion: $TOTAL runs outstanding"

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
echo "[$(date +%H:%M:%S)] ep60 training complete, $i runs"

# Mosaic-score two wide. Split the outstanding list by parity across the cards.
MOSJOBS=/tmp/ep60_mosaic.txt
: > $MOSJOBS
for d in runs/ep60_*_s42; do
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
      >> "logs/mosaic_ep60_g$g.log" 2>&1
  done < $MOSJOBS
}
score_half 0 0 &
score_half 1 1 &
wait
echo "[$(date +%H:%M:%S)] mosaic complete"

python src/mosaic_premium.py 2>&1 | tee runs/mosaic_premium.txt
python src/epoch_control.py   2>&1 | tee runs/epoch_control.txt
python src/power.py           2>&1 | tee runs/power.txt
echo "[$(date +%H:%M:%S)] EP60 BLOCK COMPLETE"
