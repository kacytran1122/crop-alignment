# Statistical rebuild: what survives, with intervals

Re-derived after four adversarial reviews. Every number here is paired within
held-out acquisition, so scene difficulty cancels, and every claim carries an
interval. Code in `src/`.

> **Read section 8 first.** Sections 1 to 7 were written against patch-pixel
> scoring. Section 8 re-derives the headline under the mosaic protocol the paper
> itself recommends, and the headline does not survive it at conventional
> significance. Section 1 is retained because the contrast between the two is now
> the result.

## 1. The result under patch scoring (superseded by section 8)

The **artefact premium** — how much of a deep model's advantage is paid for
reproducing the label generator rather than the phenomenon — tested as a
difference-in-differences within each held-out acquisition:

```
premium(a) = [U-Net(a,crop-noisy)   - threshold(a,crop-noisy)]
           - [U-Net(a,crop-invariant) - threshold(a,crop-invariant)]
```

| test | result |
|:--|:--|
| paired mean over 17 acquisitions | **+0.0488 mIoU** |
| t | 4.12, 95% CI [+0.0237, +0.0740] |
| bootstrap over acquisitions (20k) | 95% CI [+0.0241, +0.0694], P(≤0) = 0.0002 |
| sign test | 16/17 positive, p = 0.00027 |
| ratio | **1.66×**, bootstrap CI [1.23×, 2.44×] |
| dropping the collapsing fold | +0.0490, t = 3.88, 1.64× |

Three tests that assume different things all agree, and it survives the outlier.
This is the paper.

~~It is also *conservative*: it is one seed, and seed noise inflates the fold-to-fold
sd that sits in the denominator.~~ **Wrong, by two orders of magnitude.** Measured
from `exp_*_s{42,7,123}`: seed sd is 0.00114 (scene) and 0.00477 (original), which
propagates to 0.0049 in the premium against a total sd of 0.0879 — **0.31% of the
variance**. Averaging three seeds buys a 1.001× gain in standard error. Extra seeds
establish that the sign is not a seed fluke; they buy no power, and any claim that
they do is checkable against the repository in ten minutes.

## 2. The nulls are now equivalence results, not absences of evidence

TOST against ±0.0306, the fusion gain the audited work reports. Margin fixed in
advance from the literature, not from our data.

| contrast (17 acquisitions) | difference | 90% CI | verdict |
|:--|--:|:--|:--|
| real photons − optical only | +0.0025 | [−0.0026, +0.0076] | **equivalent** |
| shuffled − optical only | +0.0019 | [−0.0001, +0.0039] | **equivalent** |
| Gaussian noise − optical only | +0.0043 | [−0.0021, +0.0107] | **equivalent** |
| constant − optical only | +0.0017 | [−0.0019, +0.0054] | **equivalent** |
| zeros − optical only | +0.0027 | [−0.0007, +0.0061] | **equivalent** |

All five sit inside the margin. The claim is now "we can rule out an effect as
large as the one reported", which is falsifiable, rather than "we found no
difference", which is not.

At 2 acquisitions the same tests are **inconclusive** (3 of 3). That is the
honest reading: small data cannot support the claim in either direction, which is
itself an argument for testing at scale.

## 3. The 0.1164 belongs to the threshold, not the U-Net

Not previously recorded. This is a tenth finding.

| model | labels | mean | sd | min | max |
|:--|:--|--:|--:|--:|--:|
| threshold | scene | 0.7877 | **0.1164** | **0.5050** | **0.9063** |
| U-Net | scene | 0.8613 | 0.0934 | 0.5362 | 0.9439 |
| threshold | original | 0.7208 | 0.0916 | 0.5311 | 0.8102 |
| U-Net | original | 0.8432 | 0.0721 | 0.6641 | 0.9077 |

`PAPER_DRAFT.md`'s abstract says *"a single fixed model scores between 0.505 and
0.907 mIoU depending only on which acquisition it is evaluated on."* That is
literally true of the two-parameter threshold and false of the U-Net. In a paper
comparing the two, every reader will attach it to the deep model. Quoting the
weakest arm's spread as the benchmark's noise floor inflates it.

## 4. The spread has a mechanism: class prevalence, not model instability

`diagnose_folds.py`. Correlation between fold mIoU and log₁₀ of the rarest present
class prevalence is **+0.814** (Pearson) on scene labels.

| macro-IoU over classes with prevalence ≥ | mean | sd |
|--:|--:|--:|
| 0.00% (as reported) | 0.8613 | **0.0934** |
| 0.10% | 0.8886 | **0.0502** |
| 1.00% | 0.8922 | 0.0496 |

The worst fold (`20191111200211_07010`) is **99.99% thick ice**; its rarest class
is 0.008% of pixels. Macro-averaging IoU over an essentially absent class is an
arithmetic artefact, not a measurement of the model.

This *replaces* a raw spread with a diagnosed one, and sharpens Recommendation 7:
report class prevalence beside macro-IoU, and do not macro-average over
near-absent classes.

## 5. The paired advantage, correctly computed

| contrast | mean | 95% CI | t | folds won | MDE n=1 | MDE n=17 |
|:--|--:|:--|--:|:--|--:|--:|
| U-Net − threshold, scene | +0.0736 | [+0.0382, +0.1089] | 4.41 | 16/17 | 0.1458 | 0.0354 |
| U-Net − threshold, original | +0.1224 | [+0.1017, +0.1431] | 12.54 | 17/17 | 0.0853 | 0.0207 |

## 6. Tile overlap is real but small

`tile_overlap.py`. 8 of 39 tiles appear on more than one acquisition. Removing
every training patch that shares a tile with the holdout costs a median of 1.6% of
training data and at worst 20.8%, and no fold drops below 79.2%. So a tile-disjoint
LOAO is a fair comparison rather than a confound with training-set size, and
`train.py --disjoint-tiles` now runs it. The gap between the two designs *is* the
leak, measured instead of argued.

## 7. A sixth correction to our own analysis, caught before it was reported

The first mosaic run returned a premium of **−0.2105** (t = −5.63, 1/17 folds
positive) — a clean reversal of the +0.0488 above, with a tight interval and a
decisive-looking sign test. It was wrong.

`mosaic_scale.py` scored both label sets against the crop-invariant scene canvas.
But the per-crop scheme *has no scene canvas by construction*: a pixel covered by
several crops can carry several labels, which is the phenomenon under study. So
the crop-noisy arm was being graded against the wrong answer key, which destroyed
its advantage and flipped the difference.

The repair is to build the ground truth by vote from the same cached patch labels
the model trained on, and to report how often the crops agree. That yields a
verification that could not be faked:

| label scheme | crop unanimity |
|:--|--:|
| crop-invariant (scene) | **1.0000** |
| per-crop (original) | 0.9760 |

The repaired scheme is exactly crop-invariant, to the last pixel, which is what it
was built to be. After the fix the premium on that fold is +0.0507, against
+0.0488 from patch scoring.

This belongs in the paper. It is the sixth error we have made that produced a
plausible, well-formed, *wrong* number rather than a crash — and it would have
reversed the headline had we trusted it. That is the failure mode the paper
documents, encountered again while documenting it.

## 8. The headline does not survive the paper's own scoring protocol

All 17 folds re-scored by mosaicking: one decision per source pixel, both arms on
an identical pixel set, one empty-class convention. Mean patch-pixel reuse
eliminated: **33.09×**.

| scoring | premium | t | 95% CI | sign test |
|:--|--:|--:|:--|:--|
| patch pixels (as previously reported) | +0.0488 | 4.12 | [+0.0237, +0.0740] | 16/17, p = 0.0003 |
| **mosaic, one decision per pixel** | **+0.0182** | **0.82** | **[−0.0286, +0.0650]** | 14/17, p = 0.0127 |

The effect shrinks 2.7× and the interval crosses zero. The **direction** still
replicates — 14 of 17 folds positive, p = 0.013 by sign test — but the magnitude
is not resolvable with 17 acquisitions once each pixel is counted once.

**This is not a bug, and we checked.** Building the ground truth by majority vote
could have repaired part of the crop-dependence before measuring it, which would
bias the premium down. So both conventions were computed:

- **vote** — majority label over the crops covering a pixel
- **draw** — one covering crop sampled per pixel, in proportion to its votes,
  which preserves crop-dependence in expectation

For the crop-invariant scheme the two are bit-identical, which is a free check on
the sampler. For the crop-noisy scheme they differ by about 0.003. The convention
is not what shrank the effect. Pseudo-replication was inflating the significance.

### What this actually means

The paper's Recommendation 5 says score each source pixel once. Applied to our own
headline finding, it cost us that finding. That is worth more than the finding was:

> We applied our own recommended protocol to our own strongest result and it
> shrank by 2.7× and lost significance. This is the size of the error that
> pseudo-replicated scoring introduces, measured on a real claim rather than
> argued in the abstract.

The honest claim set is now:

1. **Crop-dependence is exactly measurable and exactly repairable.** Crop unanimity
   is **1.0000** on all 17 folds for the repaired scheme and **0.9678** (range
   0.9319–0.9908) for the original. That is a clean, verifiable, first-of-its-kind
   measurement, and it does not depend on any model.
2. **The photon branch is inert**, with five equivalence results at a margin taken
   from the audited work.
3. **The artefact premium is directionally consistent but below this benchmark's
   resolution**, which is the paper's own thesis applied to itself.

Claim 3 is weaker than what was reported an hour ago, and the paper must say so
plainly rather than lead with the patch-scored number.

## 9. The mechanism, measured directly instead of inferred from aggregates

The premium asks a fourth-order question about four aggregate mIoU numbers, and its
noise is dominated by how well a brightness threshold happens to fit each scene —
a quantity with nothing to do with the claim. The claim itself is about a
per-pixel object that is directly observable.

Crops overlap ~33×, so a source pixel `p` is seen inside many crops `c`. Under the
per-crop scheme those crops can disagree about `p`. Let `A` be the pixels where
they do. For `p ∈ A`:

```
kappa = P(pred(p | crop c) = label_c(p))        the crop the model is reading
      - P(pred(p | crop c) = label_c'(p))       a different crop, same pixel
```

The physical state of `p` does not depend on which crop it is viewed through, so
**κ ≡ 0 exactly for any crop-invariant predictor.** This is a structural null, not
an estimated one.

On acquisition `20191103184432_05780`, `A` = 1,409,471 px (7.36% of covered):

| model trained on | P(own crop) | P(other crop) | **κ** |
|:--|--:|--:|--:|
| crop-noisy labels | 0.7849 | 0.6096 | **+0.1753** |
| crop-invariant labels (control) | 0.6236 | 0.6121 | **+0.0115** |

Same pixels, same labels, same metric. The only difference is what the model was
trained on, and the model that never saw a crop-dependent label sits at the
structural null. **15× separation.**

Supporting, and free: `Ω`, the model's own crop-dependence, is 0.0185 for the
crop-invariant model (the padding and boundary floor, measured rather than assumed)
and 0.0573 for the crop-noisy one. For the two-parameter threshold `Ω = 0` exactly,
since V is a per-pixel function and the thresholds are per-fold constants — a free
correctness check on the whole pipeline. And on the crop-invariant label set `A` is
**empty to the last pixel**, which is the same verification from the other side.

This is decisive in both directions, which the mIoU premium is not: at 17
acquisitions the premium is guaranteed inconclusive, whereas κ either separates
from its exact null or falsifies the mechanism cleanly. Sweep across all 17 folds
is running.

## 10. Three corrections a reviewer would have found in ten minutes

**Seed variance is 0.31% of the premium variance,** not a conservative inflation of
it. Section 1 has been struck and corrected.

**The difference-in-differences over-subtracts.** It fixes the threshold arm's
coefficient at 1. Estimating it: β̂ = 0.711 (se 0.175), and the test of β = 1 gives
t = −1.65, p = 0.121 — so the DiD is defensible, but the ANCOVA form cuts the
patch-scored premium from +0.0488 (t = 4.12) to **+0.0295 (t = 1.81)**. Reported
here because a hostile reviewer can run it from this repository in ten minutes, and
finding it ourselves is worth more than defending it later.

**Macro-averaging is cancelling the effect.** Per class:

| class | mean ΔU | sd | t | |
|:--|--:|--:|--:|:--|
| thick | −0.0894 | 0.1312 | **−2.81** | |
| thin | −0.0446 | 0.0701 | **−2.62** | |
| water | +0.0796 | 0.2423 | +1.35 | threshold IoU = 1.0 **by construction** |

Thick and thin both carry the effect. Water is degenerate — water *is* `V ≤ 30`, so
the baseline is tautologically perfect — and its 0.2423 sd drags the macro-average
to t = −1.15. Per-class reporting with Holm correction is both more honest and
better powered than any single aggregate.

Related: prevalence-flooring helps the *levels* (sd 0.0934 → 0.0502, section 4) but
makes the *contrast* 2–4× worse, because it drops different classes from the two
label sets on the same acquisition. On `20191111200211_07010` the scene labels have
2 classes present and the original labels have 3, so that fold's premium subtracts
a 2-class macro-average from a 3-class one. That mismatch needs flagging regardless
of what else changes.

## Still open

- seeds 7 and 123 on standard LOAO (running)
- tile-disjoint LOAO, both label sets (running)
- converged epoch budget, 6 folds at 60 epochs (queued)
- the 7.2× cross-section contradiction (issue 4) — a writing fix
- Gate 3 run-level significance (issue 3) — claims to be withdrawn
