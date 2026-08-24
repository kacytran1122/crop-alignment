#!/usr/bin/env bash
# One work pool over every remaining run, with no barriers.
#
# The chained scripts this replaces launched jobs in pairs and then waited for
# BOTH before starting the next pair, so whenever two runs finished at different
# times one GPU sat idle for the difference. Measured: GPU 0 at 0% while GPU 1
# worked. Over ~114 remaining runs that is hours of wasted card.
#
# Here each GPU is a slot that pulls the next job the moment it is free. Jobs are
# ordered by which reviewer objection they close, hardest-to-answer first, so that
# stopping early still leaves a defensible set:
#
#   1. disjoint  -- "your train and test share ground"      (34 runs)
#   2. ep60      -- "you stopped at 12 epochs, undertrained" (12 runs)
#   3. seeds     -- "every number is one seed"               (68 runs)
#   4. fusion    -- across-acquisition interval on the photon contrast (26 runs)
#
# Fusion is last because the photon null already has three seeds and five
# equivalence tests; the others have no answer at all without these runs.
set -u
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
export SEAICE_TOKENS=$PWD/work/expansion/tokens_subsample.parquet
export SEAICE_FEATS=$PWD/work/expansion/feature_cols.json
mkdir -p logs runs/mosaic

ACQS=$(python -c "import pandas as pd;print(' '.join(sorted(pd.read_parquet('work/expansion/tokens_subsample.parquet').acq_id.unique())))")
EP60_FOLDS="20191122182834_08680 20191103184432_05780 20191119194532_08230 \
20191120191952_08380 20191113191053_07310 20191127175434_09440"

JOBS=/tmp/seaice_jobs.txt
: > $JOBS
for a in $ACQS; do for L in scene original; do
  echo "dj_${L}_${a}_s42|--labels $L --holdout-acq $a --disjoint-tiles --seed 42 --epochs 12 --patience 5" >> $JOBS
done; done
for a in $EP60_FOLDS; do for L in scene original; do
  echo "ep60_${L}_${a}_s42|--labels $L --holdout-acq $a --seed 42 --epochs 60 --patience 15" >> $JOBS
done; done
for a in $ACQS; do for m in true shuffle; do
  echo "lf_${m}_${a}_s42|--photon $m --labels scene --holdout-acq $a --seed 42 --epochs 12 --patience 5" >> $JOBS
done; done
# One extra seed, not two, and last. Measured from exp_*_s{42,7,123}: seed sd is
# 0.00114 (scene) and 0.00477 (original), which propagates to 0.0049 in the
# premium against a total sd of 0.0879 -- 0.31% of the variance. Averaging three
# seeds buys a 1.001x gain in standard error. Seeds establish that the sign is not
# a seed fluke; they buy no power, and 34 runs is enough to show that.
for s in 7; do for a in $ACQS; do for L in scene original; do
  echo "loao_${L}_${a}_s${s}|--labels $L --holdout-acq $a --seed $s --epochs 12 --patience 5" >> $JOBS
done; done; done

TODO=/tmp/seaice_todo.txt
: > $TODO
while IFS='|' read -r n rest; do
  [ -f "runs/$n/test_metrics.json" ] || echo "$n|$rest" >> $TODO
done < $JOBS
TOTAL=$(wc -l < $TODO)
echo "[$(date +%H:%M:%S)] work pool: $TOTAL runs outstanding of $(wc -l < $JOBS)"

declare -A PID
run_one () {   # gpu, name, args
  local g="$1" n="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$g python src/train.py --batch 128 --amp fp16 \
    --name "$n" "$@" > "logs/$n.log" 2>&1
}

i=0
exec 3< $TODO
while IFS='|' read -r n rest <&3; do
  # wait until a slot frees, then dispatch to it immediately
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
  run_one "$slot" "$n" $rest &
  PID[$slot]=$!
done
wait
echo "[$(date +%H:%M:%S)] all $TOTAL runs complete"

echo "[$(date +%H:%M:%S)] mosaic-scoring every leave-one-acquisition-out run"
for d in runs/loao_*_s* runs/dj_*_s42 runs/ep60_*_s42; do
  [ -d "$d" ] || continue
  r=$(basename "$d")
  [ -f "runs/mosaic/$r.json" ] && continue
  [ -f "$d/best.pt" ] || continue
  CUDA_VISIBLE_DEVICES=0 python src/mosaic_scale.py --run "$r" --batch 96 \
    >> logs/mosaic_scale.log 2>&1
done

python src/mosaic_premium.py 2>&1 | tee runs/mosaic_premium.txt
python src/epoch_control.py   2>&1 | tee runs/epoch_control.txt
python src/premium_test.py    2>&1 | tee runs/premium_final.txt
python src/tost_nulls.py      2>&1 | tee runs/tost_final.txt
echo "[$(date +%H:%M:%S)] POOL COMPLETE"
