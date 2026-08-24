# Reproducibility guide

## Evidence levels

1. **Machine-readable and recomputable:** sea-ice crop alignment and matched control
   in `experiments/sea_ice/results/kappa_summary.json`; flood transfer mechanism in
   `experiments/floods/results/mechanism.json`.
2. **Recomputable from rounded event rows in committed logs:** flood calibration and
   expert-label cost. Their aggregate arithmetic is checkable, but raw predictions
   and checkpoints are absent.
3. **Committed aggregate logs only:** the longer-budget sea-ice sensitivity results.
   Per-run JSON and checkpoints are absent.
4. **External data required:** model retraining, crop generation, and checkpoint
   scoring.

Run the evidence verifier from the repository root:

```bash
python experiments/sea_ice/src/verify_paper_numbers.py
```

It reports recomputed sea-ice statistics, parses the eleven-event flood tables to
check their aggregate arithmetic, and emits explicit provenance warnings.

## Data placement

Large datasets are intentionally ignored by Git. Experiment scripts resolve paths
relative to their experiment directory.

```text
experiments/sea_ice/data/          source sea-ice imagery and derived products
experiments/sea_ice/work/          generated manifests and feature tables
experiments/sea_ice/runs/          checkpoints and scored folds
experiments/floods/data/sen1floods11/
experiments/floods/runs_s1f/
experiments/floods/runs_s1fc/
```

The Sen1Floods11 subset can be fetched with:

```bash
python experiments/floods/src/fetch_sen1floods11.py
python experiments/floods/src/s1f_cache.py
```

The sea-ice source data are not redistributed here. Reproducing those experiments
requires obtaining the benchmark inputs described and cited in the paper, then
creating the token and feature files expected by `experiments/sea_ice/src/train.py`.

## RTX A6000 environment

On a CUDA host with one or two RTX A6000 GPUs:

```bash
python -m venv experiments/sea_ice/.venv
source experiments/sea_ice/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_a6000.txt
nvidia-smi
```

The long sea-ice budget is launched with:

```bash
bash experiments/sea_ice/src/run_epoch_120.sh
```

The script schedules treatment/control folds across GPU 0 and GPU 1, then performs
mosaic scoring. Run directories are resumable: completed `test_metrics.json` files
and mosaics are skipped. The paper reports 49--149 minutes per 120-epoch fold pair
on the original two-A6000 system, but timing depends on storage and preprocessing.

Flood calibration and transfer experiments use:

```bash
bash experiments/floods/src/run_s1f.sh
bash experiments/floods/src/run_s1fc2.sh 0 0 1
```

## Regenerating paper artifacts

```bash
python paper/figures/generate_figures.py
cd paper
tectonic -X compile crop_alignment_wacv_2027.tex
tectonic -X compile crop_alignment_wacv_2027_supplement.tex
python validate_layout.py
python validate_submission.py
```

The review build deliberately hides the public repository URL. Change the WACV
style from `review` only for the camera-ready build, after adding the final author
list and assigned paper ID.
