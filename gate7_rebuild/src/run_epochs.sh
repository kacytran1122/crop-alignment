#!/usr/bin/env bash
# Does the artefact premium survive a converged optimisation budget?
#
# KNOWN_ISSUES #7: every result at scale used --epochs 12 --patience 5, and
# best_epoch hit the cap of 12 in several runs, so some models were still
# improving when training stopped. A reviewer can argue the premium measures a
# difference in how fast the two label sets are fitted rather than a difference in
# what is learnable from them, and that objection cannot be answered from runs
# that all stopped early.
#
# Answer it directly. Re-run six folds at 60 epochs with patience 15, chosen to
# span the range of premia rather than to flatter it: the two largest, the two
# smallest, and the one negative fold. If the premium is a budget artefact it
# should shrink; if it is a property of the labels it should not move.
#
# Six folds is not the full 17 and the paper must say so. It is what fits before
# the deadline at five times the epoch budget, and it is a control rather than a
# replacement.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
export SEAICE_TOKENS=$PWD/work/expansion/tokens_subsample.parquet
export SEAICE_FEATS=$PWD/work/expansion/feature_cols.json

echo "[$(date +%H:%M:%S)] waiting for the robustness sweep ..."
while pgrep -f '[r]un_robustness' > /dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] starting the converged-budget control"

# largest premia, smallest premia, and the one fold that went the other way
FOLDS="20191122182834_08680 20191103184432_05780 20191119194532_08230 \
20191120191952_08380 20191113191053_07310 20191127175434_09440"

I=0
for a in $FOLDS; do
  for L in scene original; do
    n="ep60_${L}_${a}_s42"
    [ -f "runs/$n/test_metrics.json" ] && continue
    g=$((I % 2)); I=$((I + 1))
    CUDA_VISIBLE_DEVICES=$g python src/train.py --labels "$L" --holdout-acq "$a" \
      --seed 42 --epochs 60 --patience 15 --batch 128 --amp fp16 --name "$n" \
      > "logs/$n.log" 2>&1 &
    [ $((I % 2)) -eq 0 ] && wait
  done
done
wait
echo "[$(date +%H:%M:%S)] converged-budget runs done, mosaic-scoring them"

for d in runs/ep60_*_s42; do
  r=$(basename "$d")
  [ -f "runs/mosaic/$r.json" ] && continue
  [ -f "$d/best.pt" ] || continue
  CUDA_VISIBLE_DEVICES=0 python src/mosaic_scale.py --run "$r" --batch 96 \
    >> logs/mosaic_scale.log 2>&1
done

python src/epoch_control.py 2>&1 | tee runs/epoch_control.txt
echo "[$(date +%H:%M:%S)] EPOCH CONTROL COMPLETE"
