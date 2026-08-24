# Crop Alignment

**A Structural-Null Test for Closed-Loop Labels in Segmentation Benchmarks**

[![Quality checks](https://github.com/kacytran1122/crop-alignment/actions/workflows/quality.yml/badge.svg)](https://github.com/kacytran1122/crop-alignment/actions/workflows/quality.yml)

[Paper](paper/crop_alignment_wacv_2027.pdf) ·
[Supplement](paper/crop_alignment_wacv_2027_supplement.pdf) ·
[Reproducibility guide](docs/reproducibility_guide.md) ·
[Artifact audit](docs/artifact_audit.md)

Many segmentation benchmarks use labels produced automatically from the same
imagery given to a model. If the labeller is applied separately to each crop, two
overlapping crops can assign different labels to the same physical pixel. A model
can then improve its benchmark score by learning this crop-dependent labelling rule
rather than the underlying phenomenon.

This repository introduces **crop alignment** ($\kappa$), a structural-null test
for that crop-dependent shortcut. It is not a general test for every kind of
closed-loop or input-derived label.

![Model architecture and crop-alignment audit](paper/figures/crop_alignment_architecture.png)

## Main result

On the audited sea-ice benchmark, the 16 folds with committed machine-readable
evidence give:

| Measurement | Result |
|---|---:|
| Crop alignment, benchmark-label training | `+0.1230` |
| Crop alignment, matched control | `+0.0157` |
| Paired crop-alignment difference | **`+0.1073`** |
| Paired t statistic | **`12.88`** |
| Positive paired folds | **`16 / 16`** |

For any crop-invariant predictor, $\kappa=0$ exactly by construction. On a
per-pixel threshold that cannot observe crop context, the implementation returns
zero to floating-point precision. The theorem requires a nonempty set of pixels on
which overlapping crops disagree. If that set is empty, $\kappa$ is not estimable;
this is not evidence that the labels are shortcut-free.

## Contributions

- A crop-alignment estimand with an exact structural null.
- A standalone NumPy implementation in [`crop_alignment.py`](crop_alignment.py).
- Controlled calibration on a crop-dependence dial.
- Evaluation against expert flood labels to measure the cost of the shortcut.
- Machine-readable verification of every supported headline statistic.
- WACV-formatted paper source, supplementary material, and code-faithful vector
  architecture diagram.

## Quick start

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/kacytran1122/crop-alignment.git
cd crop-alignment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the standalone self-test and verify the paper statistics:

```bash
python crop_alignment.py
python experiments/sea_ice/src/verify_paper_numbers.py
```

The self-test should end with `all checks passed`. The paper verifier reports
`pass_with_warnings`: the warnings deliberately expose aggregate results whose
underlying per-run artifacts are not present rather than treating them as fully
reproducible evidence.

## Repository structure

```text
crop_alignment.py           standalone crop-alignment implementation
experiments/
  sea_ice/                   training, audit code, and compact result artifacts
  floods/                    calibration and cross-domain mechanism checks
paper/                       WACV source, figures, review PDF, and supplement
docs/
  artifact_audit.md          corrected claims and artifact evidence levels
  reproducibility_guide.md   data preparation and RTX A6000 instructions
```

Raw imagery, generated caches, model checkpoints, and local training directories
are excluded from Git. The compact release verifies the primary machine-readable
result without pretending to reproduce runs whose required artifacts are absent.

## RTX A6000 retraining

Training requires the source datasets and a CUDA host. After preparing the data
layout documented in [`docs/reproducibility_guide.md`](docs/reproducibility_guide.md), install
the GPU environment and launch the long sea-ice sweep:

```bash
python -m pip install -r requirements_a6000.txt
bash experiments/sea_ice/src/run_epoch_120.sh
```

The runner is resumable and schedules treatment/control jobs across GPU 0 and GPU
1. All released scripts use repository-relative paths.

## Building the paper

With [Tectonic](https://tectonic-typesetting.github.io/) installed:

```bash
cd paper
tectonic -X compile crop_alignment_wacv_2027.tex
tectonic -X compile crop_alignment_wacv_2027_supplement.tex
python validate_layout.py
python validate_submission.py
```

The committed review PDF contains eight pages of main content, with references
beginning on page 9. The public repository URL is present in the LaTeX source but
is hidden in review mode for double-blind submission and enabled in camera-ready
mode.

## Artifact integrity

An earlier draft reported a 17-fold aggregate (`+0.1106`, `t=13.01`) from a text
summary, while the committed JSON contains only 16 paired folds. This release uses
the reproducible 16-fold result (`+0.1073`, `t=12.88`) in the abstract and
conclusion and discloses the unmatched text-only row.

The 60- and 120-epoch aggregates are retained as sensitivity evidence, not as
independently reproducible headline claims, because their per-run JSON and
checkpoints are not included. Flood calibration/cost rows are likewise available as
summary logs. Unpublished model-comparison effects used in an earlier draft were
removed because this repository has neither a citable public source nor the
fold-level evidence needed to support them. The compact release verifies supported
aggregate arithmetic; it does not reproduce training end to end. Full details are in the
[`artifact audit`](docs/artifact_audit.md).

## Citation

The repository is currently anonymized for double-blind review. Use the paper title
to refer to this artifact during review; an author-complete BibTeX entry will replace
this block for the camera-ready release.

```bibtex
@misc{crop_alignment_2027,
  title  = {Crop Alignment: A Structural-Null Test for Closed-Loop Labels in
            Segmentation Benchmarks},
  author = {Anonymous Authors},
  year   = {2027},
  note   = {WACV submission and reproducibility artifact}
}
```
