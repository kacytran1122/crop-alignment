#!/usr/bin/env bash
# Two things the adversarial reviews said the flagship result needs, in the order
# they matter.
#
# The flagship result is the ARTEFACT PREMIUM: within each held-out acquisition,
# the U-Net's advantage over a two-parameter threshold is larger on the crop-noisy
# labels than on the crop-invariant ones. Paired across 17 acquisitions it is
# +0.0488 (t=4.12, sign test 16/17, p=0.0003). Two objections remain.
#
# 1. TILE OVERLAP. LOAO holds out an acquisition, but the tile grid is fixed, so
#    the same ground can sit on both sides of the boundary on a different date --
#    T03CWP is trained on 20191122 and tested on 20191126. --disjoint-tiles drops
#    train/val patches on any tile present in the holdout. Measured cost: median
#    1.6% of training data, worst case 20.8%, so the comparison stays fair.
#
# 2. ONE SEED. Every LOAO run is s42. Seed noise inflates the fold-to-fold sd, so
#    the current premium is conservative rather than optimistic, but a reviewer is
#    entitled to see it.
#
# Disjoint first: it is a correctness question, and if the premium does not
# survive it, the seeds do not matter.
set -u
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
export SEAICE_TOKENS=$PWD/work/expansion/tokens_subsample.parquet
export SEAICE_FEATS=$PWD/work/expansion/feature_cols.json

echo "[$(date +%H:%M:%S)] waiting for the LOAO fusion sweep ..."
while pgrep -f '[r]un_loao_fusion' > /dev/null; do sleep 120; done
echo "[$(date +%H:%M:%S)] fusion sweep done, starting robustness"

ACQS=$(python -c "import pandas as pd;print(' '.join(sorted(pd.read_parquet('work/expansion/tokens_subsample.parquet').acq_id.unique())))")

launch () {  # name, then the flags
  local n="$1"; shift
  [ -f "runs/$n/test_metrics.json" ] && return
  local g=$((I % 2)); I=$((I + 1))
  CUDA_VISIBLE_DEVICES=$g python src/train.py --seed "$SEED" --epochs 12 \
    --patience 5 --batch 128 --amp fp16 --name "$n" "$@" > "logs/$n.log" 2>&1 &
  [ $((I % 2)) -eq 0 ] && wait
}

I=0
echo "[$(date +%H:%M:%S)] === stage 1: tile-disjoint LOAO, both label sets ==="
SEED=42
for a in $ACQS; do
  for L in scene original; do
    launch "dj_${L}_${a}_s42" --labels "$L" --holdout-acq "$a" --disjoint-tiles
  done
done
wait
echo "[$(date +%H:%M:%S)] stage 1 complete"

echo "[$(date +%H:%M:%S)] === stage 2: seeds 7 and 123 on standard LOAO ==="
for s in 7 123; do
  SEED=$s
  for a in $ACQS; do
    for L in scene original; do
      launch "loao_${L}_${a}_s${s}" --labels "$L" --holdout-acq "$a"
    done
  done
done
wait
echo "[$(date +%H:%M:%S)] ROBUSTNESS COMPLETE"

python src/premium_test.py 2>&1 | tee runs/premium_final.txt
