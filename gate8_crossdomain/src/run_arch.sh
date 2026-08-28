#!/usr/bin/env bash
# Is crop-label alignment a property of the labels, or of one architecture?
#
# WRITTEN BEFORE ANY OF IT HAS RUN.
#
# Every primary-table and primary-figure kappa comes from a U-Net with a ResNet-18
# encoder. Section 3.3
# explains why a trained network can sit above the theoretical zero in terms of padding,
# receptive fields truncated at the crop border. Those are properties of
# convolutions. That explanation is also the obvious
# attack: if kappa is reading padding rather than labels, it is a fact about U-Nets
# and not about benchmarks, and the paper's claim needs scoping to the architecture
# it was measured on.
#
# So vary the encoder and change nothing else. Same labels, same folds, same seed,
# same schedule, same epochs, same artefact set, same scorer.
#
#   efficientnet-b0   a different convolutional family, depthwise separable, so a
#                     different padding and receptive-field profile
#   mit_b0            a transformer encoder, and the one that matters, because its
#                     attention blocks carry the receptive field rather than a stack
#                     of convolutions.
#
#                     CORRECTED AFTER THE FACT, and the correction is recorded here
#                     rather than quietly dropped: this comment first claimed mit_b0
#                     has no convolutional padding in its trunk. That is false. Its
#                     overlapping patch embedding is a strided convolution with
#                     padding, 12 of the encoder's 18 convolutions are padded, and
#                     the decoder it drives here adds 10 more. So this arm never was
#                     a padding-free control and cannot be read as one. It remains a
#                     genuine architecture change, and the falsification criterion
#                     below stands untouched, because that criterion was written
#                     about the contrast and not about padding.
#
# WHAT WOULD FALSIFY THE READING: the a100 minus a000 contrast collapsing toward
# zero on either encoder. That would say kappa responds to something about ResNet
# U-Nets rather than to labels derived from the crop's own pixels, and the paper
# would have to scope every kappa it reports to one architecture. Both encoders are
# reported either way, including if they disagree with each other.
#
# Only the two ends of the dial are run. The interior alphas and the scramble arm
# established what kappa measures; this asks only whether the headline contrast
# survives a change of backbone, so it spends its runs on events rather than levels.
set -u
cd "${SEAICE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
. .venv/bin/activate

GPUS="${1:-0}"                 # gpu ids, e.g. "0" or "0 1"
PERGPU="${2:-1}"               # concurrent jobs per gpu
# One entry per SLOT, so several slots can share a gpu. PID is keyed by slot
# index, not by gpu id: keying by id made a repeated id collapse to one key.
SLOTGPU=(); for g in $GPUS; do for _ in $(seq 1 $PERGPU); do SLOTGPU+=("$g"); done; done
NSLOT=${#SLOTGPU[@]}
ENCODERS="mit_b0 efficientnet-b0"
ARMS="a000 a100"
EVENTS="Bolivia Ghana India Mekong Nigeria Pakistan Paraguay Somalia Spain Sri-Lanka USA"
SEED=42
EPOCHS=30

mkdir -p logs runs_s1fc/kappa
JOBS=/tmp/arch_jobs.txt
: > $JOBS
for enc in $ENCODERS; do
  for arm in $ARMS; do
    for ev in $EVENTS; do
      n="arch_${enc}_${arm}_${ev}_s${SEED}"
      [ -f "runs_s1fc/${n}/test_metrics.json" ] && continue
      echo "name:${n}|enc:${enc}|arm:${arm}|event:${ev}" >> $JOBS
    done
  done
done
TOTAL=$(wc -l < $JOBS)
echo "[$(date +%H:%M:%S)] arch sweep: $TOTAL runs, $NSLOT slots over gpus [$GPUS]"

declare -A PID
exec 3< $JOBS
i=0
while IFS= read -r line <&3; do
  n=$(echo "$line"   | tr '|' '\n' | sed -n 's/^name://p')
  enc=$(echo "$line" | tr '|' '\n' | sed -n 's/^enc://p')
  arm=$(echo "$line" | tr '|' '\n' | sed -n 's/^arm://p')
  ev=$(echo "$line"  | tr '|' '\n' | sed -n 's/^event://p')
  if [ -z "$n" ] || [ -z "$enc" ] || [ -z "$arm" ] || [ -z "$ev" ]; then
    echo "  MALFORMED JOB LINE: $line"; continue
  fi
  while :; do
    slot=""
    for k in $(seq 0 $((NSLOT - 1))); do
      p="${PID[$k]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then slot=$k; break; fi
    done
    [ -n "$slot" ] && break
    sleep 5
  done
  i=$((i + 1))
  gpu=${SLOTGPU[$slot]}
  echo "[$(date +%H:%M:%S)] ($i/$TOTAL) slot$slot gpu$gpu $n"
  CUDA_VISIBLE_DEVICES=$gpu python src/s1f_crop_train.py \
      --holdout-event "$ev" --arm "$arm" --seed "$SEED" --epochs $EPOCHS \
      --encoder "$enc" --name "$n" > "logs/${n}.log" 2>&1 &
  PID[$slot]=$!
done
wait

miss=0
while IFS= read -r line; do
  n=$(echo "$line" | tr '|' '\n' | sed -n 's/^name://p')
  [ -n "$n" ] || continue
  if [ ! -f "runs_s1fc/${n}/test_metrics.json" ]; then
    echo "  NO METRICS: $n"; tail -3 "logs/${n}.log" 2>/dev/null | sed 's/^/      /'
    miss=$((miss + 1))
  fi
done < $JOBS
echo "[$(date +%H:%M:%S)] trained, $miss of $TOTAL produced nothing"
[ "$miss" -gt 0 ] && { echo "ARCH SWEEP INCOMPLETE -- not scoring a partial set"; exit 1; }

# score kappa on the same artefact set as every other arm in the paper
SJOBS=/tmp/arch_score.txt
: > $SJOBS
for d in runs_s1fc/arch_*_s${SEED}; do
  [ -f "$d/test_metrics.json" ] || continue
  n=$(basename "$d")
  [ -f "runs_s1fc/kappa/${n}.json" ] && continue
  echo "$n" >> $SJOBS
done
STOTAL=$(wc -l < $SJOBS)
echo "[$(date +%H:%M:%S)] scoring $STOTAL runs"
i=0
exec 4< $SJOBS
while IFS= read -r n <&4; do
  [ -n "$n" ] || continue
  while :; do
    slot=""
    for k in $(seq 0 $((NSLOT - 1))); do
      p="${PID[$k]:-}"
      if [ -z "$p" ] || ! kill -0 "$p" 2>/dev/null; then slot=$k; break; fi
    done
    [ -n "$slot" ] && break
    sleep 2
  done
  i=$((i + 1))
  CUDA_VISIBLE_DEVICES=${SLOTGPU[$slot]} python src/s1f_crop_kappa.py --run "$n" \
      --grid-step 1 --eval-arm a100 > "logs/kappa_${n}.log" 2>&1 &
  PID[$slot]=$!
done
wait

smiss=0
while IFS= read -r n; do
  [ -n "$n" ] || continue
  if [ ! -f "runs_s1fc/kappa/${n}.json" ]; then
    echo "  MISSING kappa: $n"; tail -2 "logs/kappa_${n}.log" 2>/dev/null | sed 's/^/      /'
    smiss=$((smiss + 1))
  fi
done < $SJOBS
echo "[$(date +%H:%M:%S)] scored, $smiss of $STOTAL produced nothing"
[ "$smiss" -gt 0 ] && { echo "SCORING INCOMPLETE"; exit 1; }
echo "[$(date +%H:%M:%S)] arch sweep complete: $TOTAL trained, $STOTAL scored"
