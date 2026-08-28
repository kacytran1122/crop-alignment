#!/usr/bin/env bash
# Rescore every model-scored kappa run with the normalisation its model was trained
# with.
#
# WHY THIS EXISTS. s1f_crop_kappa.py normalised inputs with the global cache
# statistics while s1f_crop_train.py trained each model on fold-local statistics it
# recorded per run. The two differ by about two global standard deviations in the
# mean, so every flood kappa was measured on inputs the model had never seen the
# distribution of. s1fc_damage.py already read the per-run values back; the kappa
# scorer did not. Inference only, no retraining.
#
# WHAT IS AND IS NOT RESCORED.
#   rescored:     231 J1 runs and 44 architecture runs at stride 1, plus the 22 J1
#                 stride-2 grid controls. All are model-scored, so all are affected.
#   not rescored: the four *_thr files. On the threshold path model is None and the
#                 normalised tensor is dead work, which is why omega is exactly 0.0
#                 on all four. Moved to kappa_keep/ so they survive untouched.
#   not rescored: the pre-J1 s1fc_* dial, superseded and no longer cited.
#
# The old outputs are in kappa_globalnorm/, archived rather than deleted, because the
# before/after comparison is the evidence that the fix did what it claims.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate

GPUS="${1:-0 1}"
PERGPU="${2:-5}"
# One entry per SLOT so several slots share a card. PID is keyed by slot index, not
# by gpu id: keying by id made a repeated id collapse to one key and serialise.
SLOTGPU=(); for g in $GPUS; do for _ in $(seq 1 "$PERGPU"); do SLOTGPU+=("$g"); done; done
NSLOT=${#SLOTGPU[@]}

JOBS=/tmp/rescore_jobs.txt
: > $JOBS
for d in runs_s1fc/j1_*_s*/ runs_s1fc/arch_*/; do
  [ -f "${d}test_metrics.json" ] || continue
  n=$(basename "$d")
  [ -f "runs_s1fc/kappa/${n}.json" ] || echo "step:1|name:${n}" >> $JOBS
done
# the stride-64 grid control, J1 generation only
for arm in a000 a100; do
  for d in runs_s1fc/j1_${arm}_*_s42/; do
    [ -f "${d}test_metrics.json" ] || continue
    n=$(basename "$d")
    [ -f "runs_s1fc/kappa/${n}_step2.json" ] || echo "step:2|name:${n}" >> $JOBS
  done
done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] rescore: $TOTAL jobs, $NSLOT slots over gpus [$GPUS]"

declare -A PID
exec 3< $JOBS
i=0
while IFS= read -r line <&3; do
  st=$(echo "$line" | tr '|' '\n' | sed -n 's/^step://p')
  n=$(echo "$line"  | tr '|' '\n' | sed -n 's/^name://p')
  [ -n "$st" ] && [ -n "$n" ] || { echo "  MALFORMED: $line"; continue; }
  while :; do
    slot=""
    for k in $(seq 0 $((NSLOT - 1))); do
      p="${PID[$k]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then slot=$k; break; fi
    done
    [ -n "$slot" ] && break
    sleep 3
  done
  gpu=${SLOTGPU[$slot]}
  i=$((i + 1))
  [ $((i % 25)) -eq 0 ] && echo "[$(date +%H:%M:%S)] ($i/$TOTAL)"
  CUDA_VISIBLE_DEVICES=$gpu python src/s1f_crop_kappa.py --run "$n" \
      --grid-step "$st" --eval-arm a100 > "logs/rs_${n}_s${st}.log" 2>&1 &
  PID[$slot]=$!
done
wait

miss=0
while IFS= read -r line; do
  st=$(echo "$line" | tr '|' '\n' | sed -n 's/^step://p')
  n=$(echo "$line"  | tr '|' '\n' | sed -n 's/^name://p')
  [ -n "$n" ] || continue
  tag=""; [ "$st" = "2" ] && tag="_step2"
  if [ ! -f "runs_s1fc/kappa/${n}${tag}.json" ]; then
    echo "  MISSING: ${n}${tag}"; tail -2 "logs/rs_${n}_s${st}.log" 2>/dev/null | sed 's/^/      /'
    miss=$((miss + 1))
  fi
done < $JOBS
echo "[$(date +%H:%M:%S)] $miss of $TOTAL produced nothing"
[ "$miss" -gt 0 ] && { echo "RESCORE INCOMPLETE -- summaries would be partial"; exit 1; }

# every output must say it used the training normalisation, or the fix did not take
bad=$(grep -L '"norm_source": "fold-local"' runs_s1fc/kappa/j1_*.json \
        runs_s1fc/kappa/arch_*.json 2>/dev/null | wc -l)
echo "[$(date +%H:%M:%S)] outputs not marked fold-local: $bad"
[ "$bad" -gt 0 ] && { echo "RESCORE USED THE WRONG NORMALISATION"; exit 1; }
echo "[$(date +%H:%M:%S)] rescore complete: $TOTAL scored, all fold-local"
