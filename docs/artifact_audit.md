# Publication audit

This report distinguishes claims supported by committed machine-readable evidence
from values that appear only in terminal-style text summaries.

## Publication blockers in the audited draft

The earlier paper was not publication-ready for four independent reasons.

### 1. The main statistic mixed incompatible artifact states

The draft's lead sea-ice claim combined incompatible artifact states:

| source | folds | paired difference | paired sd | t statistic |
|---|---:|---:|---:|---:|
| `experiments/sea_ice/results/kappa_summary.json` | 16 | +0.1073 | 0.0333 | 12.88 |
| `experiments/sea_ice/results/kappa_summary.txt` | 17 | +0.1106 | 0.0351 | 13.01 |

The text file contains a rounded seventeenth row, but the corresponding per-fold
JSON is absent. A reviewer cannot reconstruct or validate that row from the release.
Using the larger 17-fold value as the headline therefore violates the paper's own
reproducibility standard and makes the draft unpublishable as written.

### 2. The method's stated scope was mathematically too broad

Crop alignment is an expectation over pixels where overlapping crops receive
different labels. When that disagreement set is empty, there are no observations:
the estimand is not defined. The earlier wording could be read as assigning a
reassuring zero in that case and as diagnosing closed-loop labels generally. It can
do neither. A crop-invariant pseudo-labeller may still encode an input-derived
shortcut while producing no crop disagreements, and non-geometric shortcuts also
fall outside the method.

### 3. Unpublished comparisons were used as evidence

The power section and abstract relied on model-comparison effects from an
unpublished manuscript. Although those values appeared in local summaries, they
had no public, citable source and no fold-level evidence in this release. They could
not support a WACV claim. The revised paper removes them from the abstract, power
figure, analysis, and conclusion.

### 4. Arithmetic verification was described as end-to-end reproduction

The release can recompute the 16-fold sea-ice aggregate from committed arrays and
the flood calibration/cost aggregates from rounded event rows in committed logs.
It cannot regenerate those rows because raw predictions, checkpoints, and source
imagery are absent. Longer-budget sea-ice results are aggregate-log evidence only.
Calling the package fully reproducible overstated what a reviewer can verify.

## Corrections made

- Recomputed the primary result directly from the 16 aligned arrays in JSON.
- Replaced `+0.1106`, `t=13.01`, `17/17`, and control `+0.0154` throughout the main
  paper with the supported values `+0.1073`, `t=12.88`, `16/16`, and `+0.0157`.
- Added a prominent disclosure of the unmatched text-only fold.
- Added an executable verifier that fails when paper numbers drift from committed
  JSON.
- Restricted the theorem and claims to a nonempty disagreement set and explicitly
  excluded crop-invariant and non-geometric shortcuts from the diagnostic scope.
- Recast a positive treatment--control contrast as targeted evidence of
  crop-dependent behaviour, not unique causal proof that the network copied the
  labeller.
- Removed all effect sizes derived from the unpublished model-comparison manuscript.
- Distinguished aggregate arithmetic verification from end-to-end training
  reproduction in the abstract, limitations, supplement, and README.
- Downgraded 60/120-epoch aggregates to sensitivity evidence. Their summaries are
  committed, but their per-run JSON and checkpoints are not.
- Replaced private absolute filesystem paths in released Python and shell scripts
  with repository-relative paths.
- Rebuilt the paper in WACV Datasets Track format, restored double-blind handling,
  and added a vector architecture/audit diagram that matches the released model.

## Remaining limitations

- Raw sea-ice imagery, model checkpoints, and large generated caches are not in Git.
- The 120-epoch sweep cannot be independently recomputed from the compact release.
- Several flood aggregates are committed as logs rather than per-run JSON.
- The sea-ice benchmark has no human reference labels; the flood experiment provides
  the reference-label arm.
- External validity is limited to two remote-sensing domains; the flood artifact is
  deliberately constructed rather than naturally present in the published dataset.
- Crop alignment is grid-relative and does not directly estimate score inflation.
- The assigned WACV paper ID is still unknown and remains `*****` in the source.

These limitations do not invalidate the structural-null proof or the 16-fold
machine-readable crop-alignment aggregate. They substantially narrow the claim: the
paper provides evidence for one crop-dependent shortcut in the studied settings,
not a universal test for closed-loop labelling. The revised paper now states that
boundary directly.
