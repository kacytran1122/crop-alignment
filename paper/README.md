# WACV paper

This directory contains the WACV 2027 Datasets Track manuscript, supplementary
material, figures, bibliography, and official author-kit style files.

The review build is double-blind. The source contains the public repository URL,
but `crop_alignment_wacv_2027.tex` reveals it only when the WACV style is switched from `review` to a
camera-ready build. The public artifact is:

<https://github.com/kacytran1122/crop-alignment>

## Build

From this directory, run:

```bash
tectonic -X compile crop_alignment_wacv_2027.tex
tectonic -X compile crop_alignment_wacv_2027_supplement.tex
python validate_layout.py
python validate_submission.py
```

The committed `crop_alignment_wacv_2027.pdf` has eight pages of main content;
references begin on page 9. Before submission, replace the `*****` paper-ID
placeholder in the manuscript and supplementary source with the assigned WACV
paper ID.

The architecture, power, and crop-alignment figures are regenerated from committed
source or artifacts with:

```bash
python figures/generate_figures.py
```

The architecture diagram is vector PDF and mirrors the released model in
`experiments/sea_ice/src/train.py`.
