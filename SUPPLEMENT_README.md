# WACV 2027 Supplementary Material - Paper 3638

This ZIP contains the supplementary document, anonymous code, stored scorer
outputs, and verification tools for the submission.

## Start here

1. Read `paper/supp.pdf` for the additional experiments, controls, protocols,
   limitations, and release inventory.
2. Install the lightweight verification dependencies with
   `pip install -r requirements.txt`.
3. Run `python verify.py` from the extracted archive root.

The verifier checks the estimator controls, lifecycle regressions, stored-value
claims, exact experimental design cells, rendered layout, presentation checks,
and the SHA-256 manifest. It reports which quantities are recomputed, re-read,
or unavailable because imagery and checkpoints are intentionally omitted.

## Important file distinction

`paper/main.pdf` is an exact copy of the separately submitted manuscript. It is
included only because the verifier checks printed claims and rendered layout. It
is not a revised or corrected manuscript and must not replace the primary paper
uploaded in OpenReview.

`paper/supp.pdf` is the supplementary document intended for reviewer reading.

## Scope

The archive does not contain imagery, model checkpoints, per-crop threshold
fields, or per-run human-label damage scores. See `REPRODUCE.md` and the release
inventory in `paper/supp.pdf` for the resulting verification boundaries.

All paths and payloads are anonymous. `MANIFEST.json` records every shipped file,
its size, and its SHA-256 digest.
