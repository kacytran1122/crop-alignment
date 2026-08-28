"""cropalign -- measure directional alignment with crop-specific labels.

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

TWO WEIGHTINGS, AND WHICH ONE THIS RETURNS. That difference has to be averaged over
something, and the choice matters when coverage is uneven. `kappa_pixel` gives every
source pixel in A equal weight:

    kappa_pixel = (1/|A|) * sum_{p in A} (1/m(p)) * sum_{c in C(p)} [own - oth]

`kappa_crop_read` averages over (pixel, crop) instances instead, so a pixel covered
by m(p) crops counts m(p) times. On a benchmark where coverage runs from 1 to over
100 crops per pixel the two differ substantially. This file returns both, and
`kappa` is an alias for `kappa_pixel`, which is the one to report. It returned the
crop-read quantity alone for a whole revision while the paper reported the other,
and no test caught it, because every exact-value control here happened to have one
artefact pixel or uniform coverage. There is now one that separates them.

WHY THE NULL IS TRUSTWORTHY. kappa is exactly 0 for any predictor whose output does
not depend on which crop it reads -- not on average, exactly, for every dataset.
Proof: fix p in A with m covering crops, let y be the (crop-independent) prediction
and n_y the number of covering crops assigning p class y. Summing over crops, the
own-term is n_y, and the other-term is 1/(m-1) * n_y * (m-1) = n_y, because each
other crop is counted exactly m-1 times. The difference vanishes pixel by pixel.

So no control arm or simulated null calibration is needed for fixed evaluated
predictions. Run it on
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
set from 24.7% to 16.3% of covered pixels and RAISED the contrast from +0.2666 to
+0.4587 -- a factor of 1.72. A sparser grid keeps only the pixels the crops disagree
about most, and averages the other-term over fewer, more distant crops.

What survived that change was the ranking: the two geometries ordered eleven events
almost identically, Spearman r = +0.918. So use kappa to compare arms, models or label sets
evaluated on the SAME grid, which is what it is for. Do not compare a raw kappa
against one from a paper that tiled differently, and always report your crop size
and stride beside the number -- without them it cannot be interpreted.

USAGE

    from cropalign import CropAlignment

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

Requires numpy only. Run `python cropalign.py` to execute the self-test.
"""
from __future__ import annotations

import numpy as np

__all__ = ["CropAlignment"]


def _caller():
    """The public method that raised, for a state error that names itself."""
    import inspect
    return inspect.stack()[2].function + "()"


def _key(box, arr):
    """A crop's identity: its origin AND its size.

    The origin alone is not an identity. Two crops of different sizes sharing an
    origin collided on it, so pass two could substitute one for the other and the
    label-signature check was the only thing that noticed, by accident.
    """
    a = np.asarray(arr)
    return (int(box[0]), int(box[1]), int(a.shape[0]), int(a.shape[1]))


def _infer_stride(offsets):
    """Smallest non-zero step between crop origins, per axis.

    Reported beside kappa because the statistic is measured only on pixels where
    covering crops disagree, so a denser grid changes both the artefact set and the
    number. Returns None on an axis with fewer than two distinct origins, which is
    the honest answer for a single crop or a one-dimensional sweep.
    """
    if not offsets:
        return None
    out = []
    for axis in (0, 1):
        vals = sorted({o[axis] for o in offsets})
        steps = {b - a for a, b in zip(vals, vals[1:]) if b > a}
        out.append(min(steps) if steps else None)
    return tuple(out)


# Scene states. A guard that checks one condition at the point of use lets every
# other order through: the previous version accepted a duplicate crop in pass one, a
# second finalize_labels(), and a scene closed with no predictions at all, each of
# which returned a plausible number. The lifecycle is now explicit and every entry
# point asserts the state it requires.
_CLOSED, _LABELS_OPEN, _LABELS_FINAL, _PRED_OPEN = 0, 1, 2, 3
_STATE_NAME = {_CLOSED: "no open scene", _LABELS_OPEN: "collecting labels",
               _LABELS_FINAL: "labels finalised", _PRED_OPEN: "collecting predictions"}


class CropAlignment:
    """Accumulates crop alignment (kappa) and crop dependence (Omega)."""

    def __init__(self, n_classes: int, shape=None, valid=None):
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        self.nc = int(n_classes)
        self._own = 0.0
        self._other = 0.0
        self._inst = 0
        # kappa weighted uniformly over SOURCE PIXELS: each pixel in the artefact set
        # contributes the mean of own-minus-other over its covering crops, once.
        # The instance sums above weight a pixel by how many crops cover it, which is
        # the crop-read quantity and is reported beside this one.
        self._pix_num = 0.0
        self._pix_den = 0
        self._pairs_tot = 0.0
        self._pairs_dis = 0.0
        self._n_A = 0
        self._n_cov = 0
        self._geom = set()      # (crop_h, crop_w) seen, and the offsets they sat at
        self._offs = set()
        self._state = _CLOSED
        self._lab_sig = {}
        self._pred_seen = set()
        self._boxes = []
        if shape is not None:
            self.add_scene(shape, valid)

    def _require(self, *states):
        if self._state not in states:
            raise RuntimeError(
                "{} cannot be called while {}; the order is add_scene, add_labels, "
                "finalize_labels, add_predictions, result".format(
                    _caller(), _STATE_NAME[self._state]))

    # -- scene lifecycle ---------------------------------------------------

    def add_scene(self, shape, valid=None):
        """Begin a new scene. Flushes the previous one into the running totals.

        A previous scene is closed here, and closing it enforces that its two passes
        matched. Starting a new scene therefore cannot be used to abandon one whose
        predictions were never supplied.
        """
        if self._state != _CLOSED:
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
        # per-scene bookkeeping: a hash of each crop's pass-1 labels, so pass 2 can
        # refuse a different label field, and the crop geometry, because the module
        # docstring says a kappa reported without its crop size and stride cannot be
        # interpreted and the result should therefore carry them
        self._acc = np.zeros((h, w), np.float64)   # per-pixel (own - other)
        self._lab_sig = {}
        self._pred_seen = set()
        self._boxes = []
        self._state = _LABELS_OPEN

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
        self._require(_LABELS_OPEN)
        k = _key(box, labels)
        # A crop offered twice in pass one votes twice, which moves the artefact set
        # and the answer: on a two-crop scene whose true kappa is +1.0000, one
        # duplicate returned +0.7500 and raised nothing.
        if k in self._lab_sig:
            raise ValueError(
                "crop {} was given labels twice in pass 1; each crop votes once, "
                "and a repeat changes the artefact set".format(k))
        r0, r1, c0, c1, lab = self._slice(box, labels)
        import hashlib
        self._lab_sig[k] = hashlib.sha1(
            np.ascontiguousarray(labels)).hexdigest()
        self._boxes.append((int(r0), int(c0), int(r1 - r0), int(c1 - c0)))
        self._geom.add((int(r1 - r0), int(c1 - c0)))
        self._offs.add((int(r0), int(c0)))
        m = self.valid[r0:r1, c0:c1]
        for k in range(self.nc):
            self.votes[k, r0:r1, c0:c1] += ((lab == k) & m)

    def finalize_labels(self):
        """Freeze the label votes and derive the artefact set A."""
        self._require(_LABELS_OPEN)
        self.cnt = self.votes.sum(0)
        covered = (self.cnt > 0) & self.valid
        # A: covered, and no single class holds every vote
        self.split = covered & (self.votes.max(0) < self.cnt)
        self._n_A += int(self.split.sum())
        self._n_cov += int(covered.sum())
        self._state = _LABELS_FINAL

    # -- pass 2 ------------------------------------------------------------

    def add_predictions(self, box, labels, preds):
        """Score one crop's predictions against its own and the other crops' labels."""
        self._require(_LABELS_FINAL, _PRED_OPEN)
        self._state = _PRED_OPEN
        # Every one of these used to be accepted silently, and each returned a
        # plausible number rather than failing: a missing crop scored a subset of
        # the artefact set, a duplicate counted its pixels twice, and a crop pass 1
        # never saw contributed predictions to Omega. All three produced kappa =
        # +1.0000 on a case whose true value is +1.0000, so nothing looked wrong.
        key = _key(box, labels)
        if key not in self._lab_sig:
            raise ValueError(
                "crop {} was not given labels in pass 1; kappa is defined over the "
                "crops that built the artefact set, so a crop the label pass never "
                "saw cannot be scored".format(key))
        if key in self._pred_seen:
            raise ValueError(
                "crop {} was submitted twice in pass 2; each crop is one instance "
                "per covered pixel and a repeat would count its pixels twice"
                .format(key))
        import hashlib
        sig = hashlib.sha1(np.ascontiguousarray(labels)).hexdigest()
        if sig != self._lab_sig[key]:
            raise ValueError(
                "crop {} was given different labels in pass 1 and pass 2; "
                "kappa is defined against one label field".format(key))
        self._pred_seen.add(key)
        r0, r1, c0, c1, pred = self._slice(box, preds)
        lab = np.asarray(labels)
        if lab.shape != pred.shape:
            raise ValueError("labels and preds must have the same shape")
        # A prediction outside [0, nc) would index the vote array by wrapping and
        # return a number instead of failing: an all -1 array used to give -1.0.
        if pred.size and (pred.min() < 0 or pred.max() >= self.nc):
            raise ValueError(
                "predictions must lie in [0, {}); got [{}, {}]".format(
                    self.nc, int(pred.min()), int(pred.max())))
        vm = self.valid[r0:r1, c0:c1]
        for k in range(self.nc):
            self.pvotes[k, r0:r1, c0:c1] += ((pred == k) & vm)

        # A crop is scored only where it contributed a label vote in pass 1. Labels
        # outside [0, nc) -- sensor nodata, an ignore index -- are counted by
        # nobody, so scoring them would put an instance in the numerator whose crop
        # is missing from the denominator, and the structural null would not hold.
        own_in = (lab >= 0) & (lab < self.nc)
        m = self.split[r0:r1, c0:c1] & own_in
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
        denom = np.maximum(nm - 1, 1).astype(np.float64)
        self._own += float(hit_own.sum())
        self._other += float(((vsel - hit_own) / denom).sum())
        self._inst += int(hit_own.size)
        # basic slicing gives a view, so the masked add writes through to _acc
        self._acc[r0:r1, c0:c1][m] += hit_own - (vsel - hit_own) / denom

    # -- results -----------------------------------------------------------

    def _flush_scene(self):
        """Close the scene, and refuse to close one whose passes do not match.

        The set of crops in pass two must equal the set in pass one, with no
        exceptions. The previous version guarded this with `if missing and
        self._pred_seen`, so a scene with NO predictions at all skipped the check
        entirely and was discarded in silence when the next scene opened.
        """
        missing = set(self._lab_sig) - self._pred_seen
        if missing:
            raise RuntimeError(
                "{} of {} crop(s) had labels in pass 1 but no predictions in pass 2, "
                "for example {}; kappa would be averaged over a subset of the "
                "artefact set".format(len(missing), len(self._lab_sig),
                                      sorted(missing)[0]))
        if self._lab_sig and self._state == _LABELS_FINAL:
            raise RuntimeError(
                "the scene was closed with labels finalised and no predictions "
                "supplied; call add_predictions for every crop, or do not open it")

        # one contribution per source pixel in A, not one per (pixel, crop)
        if self.split is not None and self.split.any():
            self._pix_num += float(
                (self._acc[self.split]
                 / self.cnt[self.split].astype(np.float64)).sum())
            self._pix_den += int(self.split.sum())
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
        self._state = _CLOSED

    def result(self) -> dict:
        """Pooled kappa, Omega and the artefact-set size across all scenes."""
        if self._state != _CLOSED:
            self._flush_scene()
        if self._inst == 0:
            raise RuntimeError(
                "no scored instances: the artefact set is empty, so no two crops "
                "ever disagreed about a pixel. kappa is UNDEFINED there, not zero: "
                "it averages over the instances in A and there are none. Either the "
                "labels are already crop-invariant, in which case there is nothing "
                "to measure, or the crops do not overlap.")
        p_own = self._own / self._inst
        p_other = self._other / self._inst
        # The two weightings of the same quantity. kappa_pixel gives every source
        # pixel in the artefact set equal weight; kappa_crop_read gives a pixel
        # weight in proportion to how many crops cover it, which is what this file
        # used to return alone while the paper reported the other one. `kappa` is
        # kept as an alias for the primary so a caller reading it gets the reported
        # statistic rather than silently getting the superseded one.
        k_pixel = self._pix_num / self._pix_den if self._pix_den else float("nan")
        return {
            "kappa": k_pixel,
            "kappa_pixel": k_pixel,
            "kappa_crop_read": p_own - p_other,
            "n_pixels_A": self._pix_den,
            "p_own": p_own,
            "p_other": p_other,
            "omega": (self._pairs_dis / self._pairs_tot
                      if self._pairs_tot else 0.0),
            # kappa is measured on the pixels where covering crops disagree, so how
            # densely crops are sampled changes both the artefact set and the
            # number. The docstring says a kappa without its geometry cannot be
            # interpreted; it now travels with the result rather than beside it.
            "crop_size": (sorted(self._geom)[0] if len(self._geom) == 1
                          else sorted(self._geom)),
            "stride": _infer_stride(self._offs),
            "n_crops": len(self._offs),
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
        print("cropalign self-test")
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

    # 5. an exact value, which is the only case here that pins a magnitude.
    #
    # Cases 1 to 3 pin a zero and two signs. None of them pins a number, so any bug
    # that scales kappa by a constant passes all four checks above: normalising by
    # covered pixels rather than by scored instances is one, and it leaves the null
    # at zero and both signs intact while returning 0.25 here instead of 2/3.
    # Verified by making that edit and watching this case, and only this case, fail.
    #
    # One pixel is covered by exactly three crops, which label it 0, 1 and 1. Every
    # other pixel is unanimous, so the artefact set is that pixel alone. The
    # predictor returns the reading crop's own label, so the first term is 1. For the
    # second, the reader labelled 0 sees two others that both say 1 and matches
    # neither; each reader labelled 1 sees one other that says 1 and one that says 0.
    #     kappa = 1 - (0 + 1/2 + 1/2)/3 = 2/3
    # Dividing by m rather than m-1 would give 1 - (0 + 1/3 + 1/3)/3 = 7/9 instead.
    H2 = W2 = 4
    boxes3 = [(0, 0), (0, 1), (1, 0)]        # all three contain pixel (1, 1)
    local = [(1, 1), (1, 0), (0, 1)]         # where that pixel sits inside each
    vals = [0, 1, 1]
    ca = CropAlignment(NC, (H2, W2), np.ones((H2, W2), bool))
    labs = []
    for (lr, lc), v in zip(local, vals):
        lab = np.zeros((2, 2), np.int64)
        lab[lr, lc] = v
        labs.append(lab)
    for b, lab in zip(boxes3, labs):
        ca.add_labels(b, lab)
    ca.finalize_labels()
    for b, lab in zip(boxes3, labs):
        ca.add_predictions(b, lab, lab.copy())
    r = ca.result()
    if verbose:
        print("  {:34s} kappa {:+.6f}   expected {:+.6f}".format(
            "exact value, three crops disagree", r["kappa"], 2.0 / 3.0))
    assert abs(r["kappa"] - 2.0 / 3.0) < 1e-12, (
        "expected exactly 2/3, got {!r}; an m normaliser would give 7/9".format(
            r["kappa"]))

    # ---- 6: the two weightings must differ, and both must hit their hand value.
    # Every earlier control has one artefact pixel or uniform coverage, so the
    # pixel-weighted and crop-read statistics agree on all of them. That is exactly
    # why this file could return the wrong one for a whole revision without any test
    # noticing. This case separates them.
    #
    # Region 1, cols 0-2: two width-2 crops at 0 and 1. Col 1 is covered by both,
    # they disagree, and each crop predicts its OWN label, so each contributes +1.
    # Region 2, cols 5-11: four width-4 crops at 5,6,7,8 labelled 0,1,0,1, with a
    # CROP-INVARIANT predictor. By the proposition every one of those pixels
    # contributes exactly 0, but they still enter A and the instance count.
    #   A = {col 1} + {cols 6..10}, coverage 2 and 2,3,4,3,2 -> 6 pixels, 16 instances
    #   kappa_pixel     = (1 + 0*5) / 6      = 1/6
    #   kappa_crop_read = (2*1 + 0) / (2+14) = 1/8
    ca = CropAlignment(2)
    ca.add_scene((1, 12))
    trio = [((0, 0), np.array([[0, 0]], np.int64), np.array([[0, 0]], np.int64)),
            ((0, 1), np.array([[1, 1]], np.int64), np.array([[1, 1]], np.int64))]
    for off, lab in ((5, 0), (6, 1), (7, 0), (8, 1)):
        trio.append(((0, off), np.full((1, 4), lab, np.int64),
                     np.zeros((1, 4), np.int64)))
    for box, lab, _ in trio:
        ca.add_labels(box, lab)
    ca.finalize_labels()
    for box, lab, pr in trio:
        ca.add_predictions(box, lab, pr)
    r = ca.result()
    if verbose:
        print("  {:34s} kappa {:+.6f}   crop-read {:+.6f}".format(
            "two weightings, different answers", r["kappa_pixel"],
            r["kappa_crop_read"]))
    assert abs(r["kappa_pixel"] - 1.0 / 6.0) < 1e-12, (
        "pixel-weighted kappa should be exactly 1/6, got {!r}".format(
            r["kappa_pixel"]))
    assert abs(r["kappa_crop_read"] - 0.125) < 1e-12, (
        "crop-read kappa should be exactly 1/8, got {!r}".format(
            r["kappa_crop_read"]))
    assert r["n_pixels_A"] == 6 and r["n_instances"] == 16

    if verbose:
        print("all checks passed")


if __name__ == "__main__":
    _selftest()
