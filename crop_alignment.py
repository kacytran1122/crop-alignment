"""Crop Alignment: measure whether a segmentation model learned its label generator.

Auto-labelled benchmarks are often built with the generator reading the same pixels
the model reads. When that happens the labels stop being a fixed property of the
ground and become a function of the window they were computed in: two overlapping
crops can assign the same source pixel different classes. A model can then score
well by reproducing the generator's window-dependence rather than the phenomenon,
and patch-wise evaluation will pay it for doing so.

This file measures that directly. For a source pixel p covered by several crops that
disagree about its label, crop alignment asks:

    kappa = P(pred = label of the crop being read | A)
          - P(pred = label of another covering crop | A)

where A is the set of pixels the covering crops disagree about.

WHY THE NULL IS TRUSTWORTHY. kappa is exactly 0 for any predictor whose output does
not depend on which crop it reads -- not on average, exactly, for every dataset.
Proof: fix p in A with m covering crops, let y be the (crop-independent) prediction
and n_y the number of covering crops assigning p class y. Summing over crops, the
own-term is n_y, and the other-term is 1/(m-1) * n_y * (m-1) = n_y, because each
other crop is counted exactly m-1 times. The difference vanishes pixel by pixel.

So there is no control arm to train and no sampling error on the null. Run it on
your own benchmark; if kappa is meaningfully above zero, the model's output depends
on the crop in the direction of the generator.

WHAT TO COMPARE AGAINST. A convolutional network reading crops is not crop-invariant
-- zero padding, truncated receptive fields and crop-local normalisation all make its
output depend on the window near crop borders. So a real model has a small positive
kappa even when trained on window-independent labels. Report the CONTRAST between
your model and a matched control trained on labels computed once per scene, and use
Omega (returned alongside) to see how much of the residual is the architecture.

KAPPA IS RELATIVE TO YOUR CROP GRID. This is a real limit, not a caveat to skim.
kappa is measured on pixels where covering crops disagree, so how densely you sample
crops changes both the artefact set and the number. Measured on identical models and
labels, thinning the grid from 16 covering crops per pixel to 4 shrank the artefact
set from 24.7% to 16.3% of covered pixels and RAISED the contrast from +0.191 to
+0.326 -- a factor of 1.71. A sparser grid keeps only the pixels the crops disagree
about most, and averages the other-term over fewer, more distant crops.

What survived that change was the ranking: the two geometries ordered eleven events
almost identically, r = +0.984. So use kappa to compare arms, models or label sets
evaluated on the SAME grid, which is what it is for. Do not compare a raw kappa
against one from a paper that tiled differently, and always report your crop size
and stride beside the number -- without them it cannot be interpreted.

USAGE

    from crop_alignment import CropAlignment

    ca = CropAlignment(n_classes=3, shape=(H, W), valid=valid_mask)

    for box, labels in crops:              # pass 1: labels only
        ca.add_labels(box, labels)
    ca.finalize_labels()

    for box, labels, preds in crops:       # pass 2: predictions
        ca.add_predictions(box, labels, preds)

    print(ca.result())

`box` is (row0, col0) of the crop in source coordinates; `labels` and `preds` are
2-D integer arrays of the crop's shape. Two passes are required, not a stylistic
choice: the artefact set is not known until every crop has voted.

Accumulate across as many scenes as you like by calling `add_scene` between them;
totals carry over and `result()` reports the pooled estimate.

Requires NumPy only. Run `python crop_alignment.py` to execute the self-test.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CropAlignment"]


class CropAlignment:
    """Accumulates crop alignment (kappa) and crop dependence (Omega)."""

    def __init__(self, n_classes: int, shape=None, valid=None):
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.nc = int(n_classes)
        self._own = 0.0
        self._other = 0.0
        self._inst = 0
        self._pairs_tot = 0.0
        self._pairs_dis = 0.0
        self._n_A = 0
        self._n_cov = 0
        self._scene = False
        if shape is not None:
            self.add_scene(shape, valid)

    # -- scene lifecycle ---------------------------------------------------

    def add_scene(self, shape, valid=None):
        """Begin a new scene. Flushes the previous one into the running totals."""
        if self._scene:
            self._flush_scene()
        h, w = int(shape[0]), int(shape[1])
        self.valid = (np.ones((h, w), bool) if valid is None
                      else np.asarray(valid, bool))
        if self.valid.shape != (h, w):
            raise ValueError("valid mask shape does not match scene shape")
        self.votes = np.zeros((self.nc, h, w), np.int32)
        self.pvotes = np.zeros((self.nc, h, w), np.int32)
        self.cnt = None
        self.split = None
        self._scene = True

    def _slice(self, box, arr):
        r0, c0 = int(box[0]), int(box[1])
        a = np.asarray(arr)
        if a.ndim != 2:
            raise ValueError("labels and preds must be 2-D")
        h, w = a.shape
        H, W = self.valid.shape
        if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
            raise ValueError("crop at {} of size {} falls outside the scene {}"
                             .format((r0, c0), (h, w), (H, W)))
        return r0, r0 + h, c0, c0 + w, a

    # -- pass 1 ------------------------------------------------------------

    def add_labels(self, box, labels):
        """Record one crop's labels. Call for every crop before finalize_labels."""
        if not self._scene:
            raise RuntimeError("call add_scene first")
        if self.cnt is not None:
            raise RuntimeError("labels are finalized; start a new scene to re-add")
        r0, r1, c0, c1, lab = self._slice(box, labels)
        m = self.valid[r0:r1, c0:c1]
        for k in range(self.nc):
            self.votes[k, r0:r1, c0:c1] += ((lab == k) & m)

    def finalize_labels(self):
        """Freeze the label votes and derive the artefact set A."""
        if not self._scene:
            raise RuntimeError("call add_scene first")
        self.cnt = self.votes.sum(0)
        covered = (self.cnt > 0) & self.valid
        # A: covered, and no single class holds every vote
        self.split = covered & (self.votes.max(0) < self.cnt)
        self._n_A += int(self.split.sum())
        self._n_cov += int(covered.sum())

    # -- pass 2 ------------------------------------------------------------

    def add_predictions(self, box, labels, preds):
        """Score one crop's predictions against its own and the other crops' labels."""
        if self.cnt is None:
            raise RuntimeError("call finalize_labels between the two passes")
        r0, r1, c0, c1, pred = self._slice(box, preds)
        lab = np.asarray(labels)
        if lab.shape != pred.shape:
            raise ValueError("labels and preds must have the same shape")
        vm = self.valid[r0:r1, c0:c1]
        for k in range(self.nc):
            self.pvotes[k, r0:r1, c0:c1] += ((pred == k) & vm)

        m = self.split[r0:r1, c0:c1]
        if not m.any():
            return
        yh = pred[m].astype(np.intp)
        yc = lab[m]
        nm = self.cnt[r0:r1, c0:c1][m]
        # how many covering crops assign the class this crop's prediction chose
        vsel = np.take_along_axis(
            self.votes[:, r0:r1, c0:c1][:, m], yh[None, :], axis=0)[0].astype(np.float64)
        hit_own = (yh == yc).astype(np.float64)
        # subtract this crop's own vote, then average over the other m-1 crops
        self._own += float(hit_own.sum())
        self._other += float(((vsel - hit_own)
                              / np.maximum(nm - 1, 1).astype(np.float64)).sum())
        self._inst += int(hit_own.size)

    # -- results -----------------------------------------------------------

    def _flush_scene(self):
        pc = self.pvotes[:, self.valid & (self.cnt > 0)].astype(np.float64) \
            if self.cnt is not None else None
        if pc is not None and pc.size:
            n_p = pc.sum(0)
            ok = n_p > 1
            if ok.any():
                agree = (pc[:, ok] * (pc[:, ok] - 1)).sum(0)
                tot = n_p[ok] * (n_p[ok] - 1)
                self._pairs_tot += float(tot.sum())
                self._pairs_dis += float((tot - agree).sum())
        self._scene = False

    def result(self) -> dict:
        """Pooled kappa, Omega and the artefact-set size across all scenes."""
        if self._scene:
            self._flush_scene()
        if self._inst == 0:
            raise RuntimeError(
                "no scored instances: the artefact set is empty, so no two crops "
                "ever disagreed about a pixel. Either the labels are already "
                "crop-invariant -- in which case kappa is 0 by the proposition and "
                "there is nothing to measure -- or the crops do not overlap.")
        p_own = self._own / self._inst
        p_other = self._other / self._inst
        return {
            "kappa": p_own - p_other,
            "p_own": p_own,
            "p_other": p_other,
            "omega": (self._pairs_dis / self._pairs_tot
                      if self._pairs_tot else 0.0),
            "frac_artefact": self._n_A / self._n_cov if self._n_cov else 0.0,
            "n_instances": self._inst,
            "n_artefact": self._n_A,
            "n_covered": self._n_cov,
        }


# ---------------------------------------------------------------------------


def _selftest(seed: int = 0, verbose: bool = True) -> None:
    """Three predictors with known answers, on synthetic crop-dependent labels.

    The point of the first case is that it is not an approximation. A crop-invariant
    predictor must return exactly 0, so anything above floating-point roundoff is a
    bug in this file rather than a property of the data.
    """
    rng = np.random.RandomState(seed)
    H = W = 96
    CROP, STRIDE, NC = 32, 8, 2
    starts = list(range(0, H - CROP + 1, STRIDE))

    field = rng.rand(H, W).astype(np.float64)
    valid = np.ones((H, W), bool)
    # each crop thresholds on its own local median: a closed-loop generator
    boxes, labels = [], []
    for r0 in starts:
        for c0 in starts:
            sub = field[r0:r0 + CROP, c0:c0 + CROP]
            boxes.append((r0, c0))
            labels.append((sub > np.median(sub)).astype(np.int64))

    def run(predict, name, expect=None):
        ca = CropAlignment(NC, (H, W), valid)
        for b, lab in zip(boxes, labels):
            ca.add_labels(b, lab)
        ca.finalize_labels()
        for b, lab in zip(boxes, labels):
            ca.add_predictions(b, lab, predict(b, lab))
        r = ca.result()
        if verbose:
            print("  {:34s} kappa {:+.6f}   Omega {:.4f}   A {:.1%}".format(
                name, r["kappa"], r["omega"], r["frac_artefact"]))
        if expect == "zero":
            assert abs(r["kappa"]) < 1e-12, "structural null violated: " + str(r["kappa"])
            assert r["omega"] == 0.0, "crop-invariant predictor has Omega > 0"
        if expect == "positive":
            assert r["kappa"] > 0.05, "generator-tracking predictor read as null"
        if expect == "negative":
            assert r["kappa"] < -0.05, "anti-tracking predictor did not read negative"
        return r

    if verbose:
        print("crop_alignment self-test")
    gthr = np.median(field)

    # 1. crop-invariant: one global threshold, same answer whatever window it sees
    run(lambda b, lab: (field[b[0]:b[0] + CROP, b[1]:b[1] + CROP] > gthr).astype(np.int64),
        "crop-invariant (must be exactly 0)", "zero")

    # 2. perfectly tracks the generator: returns the reading crop's own label
    run(lambda b, lab: lab.copy(),
        "reproduces the reading crop's label", "positive")

    # 3. tracks it backwards, to show the sign is meaningful and not an artefact
    run(lambda b, lab: 1 - lab,
        "reproduces its complement", "negative")

    # 4. empty artefact set is an error, not a silent zero
    ca = CropAlignment(NC, (H, W), valid)
    const = np.zeros((CROP, CROP), np.int64)
    for b in boxes:
        ca.add_labels(b, const)
    ca.finalize_labels()
    for b in boxes:
        ca.add_predictions(b, const, const)
    try:
        ca.result()
    except RuntimeError:
        if verbose:
            print("  {:34s} raises, as it should".format("crop-invariant LABELS"))
    else:
        raise AssertionError("empty artefact set returned a number instead of raising")

    if verbose:
        print("all checks passed")


if __name__ == "__main__":
    _selftest()
