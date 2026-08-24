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
# Inference only, from checkpoints already on disk. Runs on GPU 0 beside the
# training pool; it is light enough not to starve it.
set -u
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
. .venv/bin/activate
export SEAICE_CACHE=/dev/shm/seaice_expanded
mkdir -p runs/kappa logs

ACQS=$(python -c "import pandas as pd;print(' '.join(sorted(pd.read_parquet('work/expansion/tokens_subsample.parquet').acq_id.unique())))")
n=0
for a in $ACQS; do
  t="loao_original_${a}_s42"
  c="loao_scene_${a}_s42"
  if [ -f "runs/$t/best.pt" ] && [ ! -f "runs/kappa/$t.json" ]; then
    CUDA_VISIBLE_DEVICES=0 python src/crop_alignment.py --run "$t" --batch 96 \
      >> logs/kappa.log 2>&1 && n=$((n+1))
  fi
  if [ -f "runs/$c/best.pt" ] && [ ! -f "runs/kappa/${c}_vsorig.json" ]; then
    CUDA_VISIBLE_DEVICES=0 python src/crop_alignment.py --run "$c" \
      --eval-labels original --tag _vsorig --batch 96 >> logs/kappa.log 2>&1 \
      && n=$((n+1))
  fi
done
echo "[$(date +%H:%M:%S)] crop-alignment sweep done, $n new"
python src/kappa_summary.py 2>&1 | tee runs/kappa_summary.txt
