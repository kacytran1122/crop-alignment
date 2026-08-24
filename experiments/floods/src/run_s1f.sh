#!/usr/bin/env bash
# Sen1Floods11: 11 leave-one-event-out folds x 2 label arms x 3 seeds = 66 runs.
#
# Held out by EVENT because all chips in an event share one generator threshold and
# therefore one artefact; a chip-level split would leak the artefact across the
# boundary. That also means the honest n is 11, worse than the 17 sea-ice
# acquisitions -- this buys generality and an externally validated recovery, not
# statistical resolution. Say so rather than letting a reviewer say it.
#
# Runs on GPU 0 beside the sea-ice pool. Each run is ~4.5 min and under 1 GB, so
# the contention is small and the result decides whether the cross-domain arm is
# worth writing up before the deadline.
set -u
PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_DIR"
. .venv/bin/activate
mkdir -p runs_s1f logs

EVENTS="Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA"
n=0
for s in 42 7 123; do
  for e in $EVENTS; do
    for L in ref alg; do
      name="s1f_${L}_${e}_s${s}"
      [ -f "runs_s1f/$name/test_metrics.json" ] && continue
      CUDA_VISIBLE_DEVICES=0 python src/s1f_train.py --holdout-event "$e" \
        --labels "$L" --seed "$s" --epochs 40 --patience 10 \
        >> logs/s1f.log 2>&1
      n=$((n + 1))
    done
  done
  echo "[$(date +%H:%M:%S)] seed $s complete ($n runs so far)"
done
echo "[$(date +%H:%M:%S)] SEN1FLOODS11 COMPLETE, $n runs"
python src/s1f_analyse.py 2>&1 | tee runs_s1f/summary.txt
