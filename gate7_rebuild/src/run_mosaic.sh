#!/usr/bin/env bash
# Re-score every leave-one-acquisition-out fold by mosaicking, so the paper obeys
# its own Recommendation 5. Inference only: the checkpoints already exist, so this
# is cheap next to retraining, and it runs alongside the training queue on one GPU.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate
mkdir -p runs/mosaic logs
n=0
for d in runs/loao_*_s42 runs/dj_*_s42 runs/loao_*_s7 runs/loao_*_s123; do
  [ -d "$d" ] || continue
  r=$(basename "$d")
  [ -f "runs/mosaic/$r.json" ] && continue
  [ -f "$d/best.pt" ] || continue
  n=$((n + 1))
  CUDA_VISIBLE_DEVICES=0 python src/mosaic_scale.py --run "$r" --batch 96 \
    >> logs/mosaic_scale.log 2>&1
done
echo "[$(date +%H:%M:%S)] mosaic re-scoring done, $n new folds"
python src/mosaic_premium.py 2>&1 | tee runs/mosaic_premium.txt
