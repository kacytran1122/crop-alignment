#!/usr/bin/env bash
# Wait for J1 training, then score and summarise without anyone having to be awake.
#
# It refuses to proceed if training did not finish cleanly. The point of the
# explicit failure checks in run_j1.sh is lost if the next stage runs anyway on a
# partial set and prints a summary that looks complete -- which is exactly what
# happened to the alpha sweep, where a runner announced COMPLETE with 22 of 55
# results on disk.
set -euo pipefail
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

for i in $(seq 1 2000); do
  if grep -q "J1 TRAINING COMPLETE" logs/j1.log 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] training finished cleanly"
    break
  fi
  if grep -q "J1 INCOMPLETE" logs/j1.log 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] TRAINING INCOMPLETE -- not scoring a partial set"
    exit 1
  fi
  sleep 30
done

if ! grep -q "J1 TRAINING COMPLETE" logs/j1.log 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] timed out waiting for training"
  exit 1
fi

echo "[$(date +%H:%M:%S)] === stage 2: kappa on a fixed artefact set ==="
bash run_j1_score.sh || { echo "scoring failed"; exit 1; }

echo "[$(date +%H:%M:%S)] === stage 3: accuracy against expert labels ==="
. .venv/bin/activate
python src/s1fc_damage.py 'j1_*_s*' > runs/j1_damage.txt 2>&1
tail -20 runs/j1_damage.txt

echo "[$(date +%H:%M:%S)] === stage 4: descriptive scramble summary ==="
python src/s1fc_j1_summary.py > runs/j1_summary.txt 2>&1
cat runs/j1_summary.txt

echo "[$(date +%H:%M:%S)] J1 COMPLETE END TO END"
