#!/usr/bin/env bash
# Dose-response for kappa, second attempt. Two bugs fixed, the second worse.
#
# Bug 1. The alpha was derived with  python -c "print(${A}/100.0)"  where A was a
# zero-padded tag. Python reads 050 as a malformed octal literal and refuses it,
# while 000 is a legal all-zero literal and 100 is ordinary, so exactly the two ends
# of the sweep survived and the three interior points died. Fixed by carrying the
# tag and the value as an explicit pair, so no arithmetic is done on a padded
# string at all.
#
# Bug 2, the one that matters. The loop printed "done" whether or not the run had
# produced anything, so 33 failures reported as successes and the sweep announced
# S1FC STREAM COMPLETE with 22 of 55 results on disk. A clean all-clear over a
# silent failure is the same defect this paper audits, so this version checks that
# each artefact exists and says FAILED loudly when it does not.
#
# Only the interior points are outstanding: alpha 1.0 and 0.0 completed for all 11
# events and are left alone. Ordering the sweep ends-first is why the fixed run
# costs 33 jobs and not 55, and why the claim was never actually at risk.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate
mkdir -p runs_s1fc logs

G="$1"; PART="$2"; NPARTS="$3"
EVENTS="Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA"

JOBS=/tmp/s1fc2_jobs_${PART}.txt
: > $JOBS
# tag:value pairs, so nothing ever parses a zero-padded number
for P in 050:0.5 025:0.25 075:0.75; do
  for e in $EVENTS; do
    echo "${P}|${e}" >> $JOBS
  done
done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] s1fc interior sweep: $TOTAL jobs, stream $PART of $NPARTS on gpu$G"

i=0; nfail=0
while IFS='|' read -r P e; do
  i=$((i + 1))
  [ $(( (i - 1) % NPARTS )) -eq "$PART" ] || continue
  A="${P%%:*}"; af="${P##*:}"
  n="s1fc_a${A}_${e}_s42"
  if [ ! -f "runs_s1fc/$n/test_metrics.json" ]; then
    CUDA_VISIBLE_DEVICES=$G python src/s1f_crop_train.py --holdout-event "$e" \
      --alpha "$af" --seed 42 --epochs 20 --patience 6 --name "$n" \
      >> logs/s1fc2_g${G}.log 2>&1
  fi
  if [ ! -f "runs_s1fc/$n/test_metrics.json" ]; then
    echo "[$(date +%H:%M:%S)] ($i/$TOTAL) $n TRAIN FAILED (alpha=$af)"
    nfail=$((nfail + 1)); continue
  fi
  if [ ! -f "runs_s1fc/kappa/$n.json" ]; then
    CUDA_VISIBLE_DEVICES=$G python src/s1f_crop_kappa.py --run "$n" --batch 64 \
      >> logs/s1fc2_g${G}.log 2>&1
  fi
  if [ ! -f "runs_s1fc/kappa/$n.json" ]; then
    echo "[$(date +%H:%M:%S)] ($i/$TOTAL) $n KAPPA FAILED"
    nfail=$((nfail + 1)); continue
  fi
  echo "[$(date +%H:%M:%S)] ($i/$TOTAL) $n ok"
done < $JOBS

echo "[$(date +%H:%M:%S)] stream $PART finished: $nfail failures"
[ "$nfail" -eq 0 ] || echo "[$(date +%H:%M:%S)] WARNING: $nfail jobs produced nothing"
