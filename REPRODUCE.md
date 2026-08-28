# Reproducing the results in `paper/main.tex`

The repository includes earlier gates that the rebuild superseded. This document
lists only what produces a claim in the current paper, in the order it has to run,
with the output file each claim is read from.

The last section is the useful one if you only want to check the paper rather than
rerun it: release `verify.py` reaggregates the primary flood and sea-ice kappa tables
from exact per-run design cells. `verify_draft_numbers.py` separately checks stored
values against the submitted sources without calling summary-only quantities
reproduced.

---

## 1. Environment

Exact versions from the machine the results were produced on. Python 3.12.3.

```
torch                        2.11.0+cu128       cuda 12.8
torchvision                  0.26.0+cu128
segmentation_models_pytorch  0.5.0
numpy                        2.5.2
scipy                        1.18.0
pandas                       3.0.5
pyarrow                      25.0.1
rasterio                     1.5.1
scikit-learn                 1.9.0
matplotlib                   3.11.1
opencv-python-headless       5.0.0.93
```

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Hardware.** Two NVIDIA RTX A6000 (48 GB each), power-capped at 300 W. Both cards
run at roughly 80% of their 2100 MHz maximum under full load because of that cap,
which is why timings below are what they are.

`cropalign.py`, the released test, needs **numpy only**. This was verified in a clean venv
containing nothing else.

---

## 2. Data, and what it costs to obtain

| dataset | size | source | access |
|:--|--:|:--|:--|
| Sentinel-2 L1C tiles | 7.8 GB | Copernicus Open Access Hub | free, no token |
| ICESat-2 ATL03 granules | 72 GB | NASA CMR / NSIDC (`cmr.earthdata.nasa.gov`) | **Earthdata bearer token required** |
| Sen1Floods11 | 1.6 GB | `storage.googleapis.com/sen1floods11` | public bucket, no token |

**The Earthdata token is not in this repository and must not be.** `fetch_atl03.py`
reads it from `.edl_token` (mode 600, gitignored) and never echoes it. Get your own
from an Earthdata Login profile. If you are reusing a token that has ever appeared
in a terminal, a log, or a chat, rotate it.

Sen1Floods11 is the cheap entry point: it is 1.6 GB, needs no credentials, and
carries the calibration (§5), the expert-label association (§6), the generator recovery
(§6.2) and the transfer collapse (§6.3), which form most of the paper's spine. The sea-ice
half needs the full 80 GB and an Earthdata account.

---

## 3. Run order

### Sea ice

```bash
python src/cmr_scope.py                 # which S2 scenes have an ATL03 granule within 2 h
python src/fetch_atl03.py               # needs .edl_token
python src/atl03_extract.py             # photon segments -> features
python src/build_scene_masks_all.py     # the REPAIRED, scene-level label set
python src/build_expanded_cache.py      # /dev/shm cache + token table
bash src/run_pool.sh                    # 68 models: seed-42 optical-only primary and
                                        # seed-7 photon-enabled sensitivity
bash src/run_kappa.sh                   # crop alignment + its control
```

### Floods

```bash
python src/fetch_sen1floods11.py        # public bucket
python src/s1f_cache.py                 # 446 chips -> images/y_ref/y_alg .npy
python src/recover_vh_smooth.py         # generator recovery, the 9x9 sweep
bash src/run_s1f.sh                     # 66 whole-chip runs -> transfer collapse
python src/s1f_crop_labels.py           # per-crop Otsu thresholds: the dial
```

### Analysis

```bash
python gate7_rebuild/src/kappa_summary.py
python gate7_rebuild/src/verify_cropalign.py
python gate7_rebuild/src/coloc_offsets.py
python gate7_rebuild/src/drift_test.py
python gate8_crossdomain/src/s1fc_j1_summary.py
```

The repository retains superseded analysis scripts for provenance. In particular,
`gate8_crossdomain/src/kappa_summary.py`, `s1fc_summary.py`, and `figures_v3.py`
implement withdrawn pre-J1 or fold-inferential analyses; they are not release inputs
and must not be used for the current paper. The old sea-ice premium, budget and
fold-inference scripts are likewise historical and excluded from the release; the
current sea-ice path ends with `run_kappa.sh` and `kappa_summary.py`.

---

### Where the scripts live

Paths below are written as `src/<name>` because that is the layout on the machine the
runs were executed on. In this repository the same files sit under the gate that
produced them: `gate5_expansion/src/`, `gate7_rebuild/src/`, `gate8_crossdomain/src/`.
Run a command from the gate directory that contains its `src/` and `runs/` folders.
`SEAICE_ROOT` redirects the training and scorer pipelines, but several survey fetchers
use paths relative to the current working directory; it is not a universal path
resolver.

### J1: the rebuilt flood calibration

J1 replaces an earlier, superseded dial whose zero was a per-chip Otsu of our own
rather than the label set Sen1Floods11 ships, and repairs model selection at the
event level. It adds one
within-chip threshold scramble that preserves each chip's threshold multiset while
attenuating, but not eliminating, its correspondence with recipient pixels.

```bash
python src/s1fc_arms.py 42     # build every arm as an explicit threshold field
bash run_j1.sh                 # 231 runs: 7 arms x 11 events x 3 seeds
bash run_j1_score.sh           # kappa, artefact set fixed at the a100 arm
python src/s1fc_damage.py 'j1_*_s*'  # all 231 models; mosaic-scored expert accuracy
python src/s1fc_j1_summary.py  # descriptive dial and scramble summaries
```

### The four label surveys, which train nothing

Each measures `P(A)`, the share of multiply-covered pixels whose covering crops
disagree, which is a property of a labelling rule and a crop grid and needs no model.
Three run on CPU in minutes, two of them on Sen1Floods11 and one on CloudSEN12+.
The fourth calls a segmentation model and needs a GPU.

```bash
# six labelling rules used in SAR water mapping, on the 446 released SAR chips
python src/s1_labeller_survey.py

# the published sea-ice labeller itself, run inside every crop of the optical chips,
# with each quantity it fits moved from the crop to the whole scene in turn
python src/fetch_s2hand.py          # 446 Sentinel-2 chips, about 1 GB
python src/s2_published_survey.py
python src/s2_mechanism.py          # why the two min-max steps are inert as published

# a third dataset, a different task (cloud masking), and an accuracy column
python src/fetch_cloudsen12.py      # 1.84 GB, CC0, records revision + sha256
python src/s3_cloudsen12_probe.py   # band identity and padding, from the pixels
python src/s3_cloudsen12_survey.py --workers 16

# how much of the magnitude is our reflectance ceiling rather than the rule
for s in 2000 3000 5000; do
  python src/s2_published_survey.py --scale $s --out runs/s2_survey_s$s.json
done

# a foundation-model labeller with a FIXED semantic rule: SAM proposes the regions,
# one global threshold classifies them. Needs a GPU, about 45 min on one A6000.
python src/s1_sam_survey.py --chips 40
python src/s1_sam_determinism.py    # the control: same crop twice, same masks?
```

The SAM arm is the only survey that needs a GPU, and the only one where a control on
the labeller itself is required: if the mask generator were not deterministic, two
runs on the same pixels would disagree on their own and the survey would be measuring
that rather than crop-dependence. `s1_sam_determinism.py` checks it and writes the
result to a file the number checker reads.

The retained SAM output does not record the source commit or checkpoint digest.
Consequently the release can reaggregate its stored per-chip survey values but does
not claim byte-level reproduction of SAM inference. The SAM code is therefore not
included in the pinned core environment; install Meta's `segment-anything` package
and obtain the named ViT-B checkpoint before running this optional OOD stress test.

`s2_published_survey.py` reproduces `shadow_cloud_removal` and `color_segmentation`
from the published pipeline, including its channel order: the pipeline is fed by
`cv2.imread`, which returns BGR, and then calls `COLOR_RGB2HSV` on it, so bands are
stacked 2, 3, 4 rather than 4, 3, 2. Stage 1 does not care, since the value channel is
a max over the three, but stage 2 takes a weighted grey and does.

Input scaling is one global constant for every chip and every crop, so no per-crop
statistic enters before the rule under test and any `P(A)` above zero is the rule's.
That fixes crop-invariance, not magnitude, which is what the `--scale` sweep is for.

## 4. Claim to output map

Each row is a claim in the paper, the script that produces it, and the file to read.
Paths are relative to the working directory on the compute machine: `runs/`,
`runs_s1f/`, `runs_s1fc/`, `work/` and `data/` are outputs and inputs there, not
files in this repository, which carries the code and the write-ups.

Section numbers are those of the built `paper/main.pdf`.

| § | claim | script | output |
|:--|:--|:--|:--|
| 4.1 | estimator returns \|kappa\| < 1.4e-15 | `s1f_crop_kappa.py --predictor threshold` | `runs_s1fc/kappa/*_thr.json` |
| 4.2 | six labelling rules, fitted per crop 12–18%, fixed and per-scene 0.00% | `s1_labeller_survey.py` | `runs/labeller_survey.json` |
| 4.2 | cloud masking, three fitted rules 26–35%, three fixed/per-scene exactly 0.00%, with cloud IoU | `s3_cloudsen12_survey.py` | `runs/s3_cloudsen12_survey.json` |
| 4.3 | grid dependence, 1.72×, ranking Spearman r = +0.918 | `s1f_crop_kappa.py --grid-step 2` | `runs_s1fc/kappa/*_step2.json` |
| 5 | dose response −0.0008 → +0.2605; one scramble retains 94% of the endpoint mIoU difference and 26% of the alignment | `s1fc_arms.py`, `s1fc_j1_summary.py` | `runs_s1fc/j1_summary.json` |
| 6 | descriptive −0.29 mIoU endpoint difference across the constructed dial | `s1fc_damage.py` | `runs_s1fc/damage_summary.json` |
| 8.1 | 9×9 kernel, 11/11 within 0.25 dB | `recover_vh_smooth.py` | `data/sen1floods11/vh_recovery_smooth.json` |
| 8.1 | published labeller per crop 37.04%, stage 1 alone 0.00%, all fitted per scene 0.00% | `fetch_s2hand.py`, `s2_published_survey.py` | `runs/s2_published_survey.json` |
| 8.2 | optical-only primary `kappa` +0.1518 vs +0.0097, contrast +0.1421; photon-enabled sensitivity +0.1454 | `crop_alignment.py`, `kappa_summary.py` | `runs/kappa_summary.json` |
| 4.2 | SAM per crop nonzero on all 37 chips, both controls exactly 0.00% | `s1_sam_survey.py`, `s1_sam_determinism.py` | `runs/s1_sam_survey.json` |
| supp | why the min-max steps are inert until the others are repaired | `s2_mechanism.py` | `runs/s2_mechanism.json` |
| supp | transfer collapse −0.0454 (the r = +0.719 mechanism claim is withdrawn) | `s1f_mechanism.py` | `runs_s1f/mechanism.json` |
| n/a | released tool agrees with the flood pipeline to 5.6e-17 on `kappa_pixel` and `kappa_crop_read`, using that run's fold-local normalisation | `verify_cropalign.py` | stdout |

The superseded pre-J1 dial (+0.0569 → +0.2476) and its cost (−0.1056) are no longer
reported anywhere in the paper. `runs_s1fc/summary.json` still holds them.

---

## 5. Checking the paper without rerunning it

If the stored outputs are present, this reports the live number of stored-value
checks and asserts that each expected formatted value appears in the submitted
sources:

```bash
python gate7_rebuild/src/verify_draft_numbers.py
```

With no argument it reads `paper/sec/*.tex`, `paper/main.tex` and `paper/supp.tex`,
which are the files that become the PDF. It exits non-zero if any check was skipped
or any input was missing, because a run that checks nothing is not a run that passed.

Point it at `PAPER_DRAFT_V2.md`, a superseded draft kept for this purpose, and it
returns 44 mismatches instead; a checker that passes everything is not evidence of
anything until you have watched it fail.

This is a token-level consistency audit, not occurrence-level proof: a value repeated
in two legitimate places is not tied to one prose sentence. The release verifier
adds exact design-cell reaggregation for the primary kappa tables.

The anonymous archive has a separate final preflight. Review builds may retain the
conference placeholder; the final command refuses it:

```bash
python make_release.py --check          # anonymous clean-room review build
python make_release.py --final          # after inserting the WACV paper ID
```

And the released test verifies itself against six cases with known answers, one of
which pins a magnitude rather than a sign:

```bash
python cropalign.py
```

---

## 6. Cost

| stage | GPU-hours | note |
|:--|--:|:--|
| sea-ice primary + input-mode sensitivity (68 runs) | not metered separately | no total extrapolated |
| crop alignment (68 passes: 17 acquisitions x 2 arms x 2 input modes) | ~12 | |
| floods, whole chip (66 runs) | ~3 | 2.9 min per run |
| floods, J1 crop sweep (231 training runs + kappa) | ~16 | ~4 min per training run |
| J1 damage scoring (231 models, no training) | ~1.3 | inference only |

The rows above are the runs needed for the results retained in the paper;
exploratory budget runs and withdrawn analyses are omitted.

---

## 7. What will not reproduce exactly

- **GPU nondeterminism.** cuDNN autotuning and atomics make per-run mIoU vary by
  about 0.0023 between nominally identical configurations. The paper therefore shows
  training-seed sensitivity for the retained calibration contrasts.
- **The Sentinel-2 and ATL03 fetches** depend on upstream availability. The
  co-location scoping found 52/52 scenes with a granule within 2 hours at the time
  it ran; a later run could differ.
The primary flood and sea-ice kappa tables are reaggregated exactly from stored
per-run scorer outputs. Expert-label mIoU and the 94% scramble sensitivity quantity
are re-read from stored summaries; inference and training are not reproduced by the
archive.

The historical sea-ice `test_metrics.json` files predate complete optimiser and
data fingerprints. Their exact bytes and SHA-256 values are preserved in
`gate7_rebuild/results/seaice_training_metadata.json`; the export script requires
the complete 17 x 2 x 2 design and checks each recorded seed, arm, acquisition and
photon mode before writing it. The records authenticate those fields, the encoder
and loss, while the remaining setup-table settings are reconstructed from the
frozen launcher. The current launcher therefore refuses those legacy directories as
a training cache. A fresh run records and validates the full optimiser/schedule,
batch, AMP and stopping rule plus a content manifest over its code, imagery,
arm-specific labels, token table and feature schema. Each new score is also bound
to its checkpoint and training metadata, scorer/model code, evaluation labels and
scene-valid masks. The standalone summary command enforces those bindings for live
runs; the archive verifier separately checks the disclosed legacy outputs.
