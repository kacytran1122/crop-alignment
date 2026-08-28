# The mechanism, measured — and the same generator found in a second domain

Two results. The first replaces the paper's failed headline with a decisive one.
The second takes it out of sea ice.

---

## 1. The model reproduces the label of the crop it is reading

Crops overlap ~33×, so a source pixel `p` is seen inside many crops `c`. Under the
per-crop labelling scheme those crops can disagree about `p`. Let `A` be the pixels
where they do. For `p` in `A`:

```
kappa = P(pred(p | crop c) = label_c(p))     the crop the model is reading
      - P(pred(p | crop c) = label_c'(p))    a different crop, same physical pixel
```

The ice does not change according to which crop you view it through, so
**kappa ≡ 0 exactly for any crop-invariant predictor.** A structural null, not an
estimated one.

Across 16 leave-one-acquisition-out folds, with `A` averaging **13.58%** of covered
pixels and **24.4M** (pixel, crop) instances per fold:

| | mean | sd | t | folds | 97.9% distribution-free CI |
|:--|--:|--:|--:|:--|:--|
| trained on crop-noisy labels | **+0.1230** | 0.0361 | **13.62** | 16/16 | [+0.0977, +0.1590] |
| trained on crop-invariant labels (control) | +0.0157 | 0.0100 | 6.24 | 15/16 | [+0.0071, +0.0233] |
| **difference** | **+0.1073** | 0.0333 | **12.88** | **16/16** | **[+0.0789, +0.1292]** |

Sign test on the difference: p = 0.00003.

**Required n at 80% power: 1.** The artefact premium needed 238 and we have 17.
This is the same claim measured on the object it is actually about, and the change
of estimand is worth two orders of magnitude in sample size.

The control is not assumed to be zero, it is measured. A crop-invariant-trained
model still shows +0.0157, because its prediction at a pixel depends on that crop's
padding and receptive field, and crop-dependent labels correlate with
crop-dependent image statistics. That floor is exactly why the paired difference,
not the raw level, is the estimate we report.

Free checks that came with it:

- on the crop-invariant label set, `A` is **empty to the last pixel**
- `Omega`, the model's own crop-dependence, is **0 exactly** for the two-parameter
  threshold, since V is a per-pixel function and the thresholds are per-fold
  constants. Median ratio of crop-noisy to crop-invariant `Omega` is **1.75×**.
  The mean of 344× is meaningless: one fold is 99.99% thick ice, so its control
  `Omega` is near zero and the ratio explodes. Report the median.

---

## 2. The same generator, in floods, with the constant published

Sen1Floods11 ships, for the same 446 chips, an expert label (`LabelHand`) and an
algorithmic one (`S1OtsuLabelHand`). The generator is a structural clone of the
sea-ice one: Otsu on a single channel, threshold refit per event from selected
sub-windows, applied to a normalised version of the channel the model reads.

Unlike sea ice, **the dataset publishes the constant its authors used**, so the
recovery can be validated from outside instead of closing on itself.

Recovering the threshold from the raw VH band gives 94.9% pixel agreement but sits
1.19 dB low on **all ten** events. A bias of one sign everywhere is a signature,
not noise: the published procedure applies the threshold to a *focal-mean-smoothed*
band. Sweeping the radius:

| focal mean | mean abs difference | bias | agreement |
|:--|--:|--:|--:|
| 1×1 (raw) | 1.187 dB | −1.187 | 0.9491 |
| 5×5 | 0.316 | −0.316 | 0.9815 |
| 7×7 | 0.143 | −0.143 | 0.9892 |
| **9×9** | **0.040** | **−0.033** | **0.9905** |
| 11×11 | 0.074 | +0.033 | 0.9866 |
| 15×15 | 0.154 | +0.101 | 0.9776 |

The bias crosses zero at a 9×9 kernel, and there **10 of 10 events land within
0.25 dB** of the published constant:

| event | published | recovered | diff | agreement |
|:--|--:|--:|--:|--:|
| bolivia | −20.44 | −20.42 | +0.02 | 0.9936 |
| ghana | −22.81 | −22.86 | −0.05 | 0.9978 |
| india | −21.56 | −21.54 | +0.02 | 0.9949 |
| nigeria | −21.94 | −22.03 | −0.09 | 0.9920 |
| pakistan | −19.56 | −19.60 | −0.04 | 0.9755 |
| paraguay | −19.94 | −19.99 | −0.05 | 0.9901 |
| somalia | −21.06 | −21.10 | −0.04 | 0.9768 |
| spain | −25.13 | −25.14 | −0.01 | 0.9948 |
| sri-lanka | −21.69 | −21.69 | −0.00 | 0.9944 |
| usa | −22.62 | −22.70 | −0.08 | 0.9954 |

We recovered the generator's **constant and its preprocessing kernel** from nothing
but the shipped rasters. The label set is reconstructible from the model's own
input at 99.05% agreement using one scalar per event.

The constant moves **5.57 dB** across events (−25.13 to −19.56). No global
threshold can express that. A model that sees the whole chip can infer which event
it is in and adapt — the sea-ice mechanism, in a second domain.

This matters beyond replication. Sen1Floods11 is a current evaluation benchmark for
the Prithvi-EO geospatial foundation models, and its own paper claims that models
trained on its automatically generated labels *outperform* those trained on its
hand labels. That claim is the artefact premium, asserted by the dataset authors.

---

## Still to run

- **Sen1Floods11 training**: 11 leave-one-event-out folds × 2 label arms × 3 seeds,
  roughly 6 GPU-hours. Yields the **transfer collapse** — mIoU(trained on
  algorithmic, scored against expert) minus mIoU(trained on expert, scored against
  expert) — which sea ice cannot support because it has no reference labels.
- Model input must be **Sentinel-1**. The hand labels were produced by analysts
  correcting a Sentinel-2 index classification, so the reference arm is open-loop
  only if the model never reads S2.
- Power caveat: the honest cluster is the **event**, because all chips in an event
  share one threshold and therefore one artefact. That is n = 11, worse than the
  17 acquisitions in sea ice. Sen1Floods11 buys generality and external validation,
  not resolution. Do not sell it as the power fix.
- **CloudSEN12+** (~2,000 ROIs, 8 shipped generators of graded receptive field,
  CC0) is the properly powered version at 8–14 days. Post-deadline work.
