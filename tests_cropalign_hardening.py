r"""Negative controls for the released estimator, plus exact-value checks.

Each of these returned a plausible number before its fix rather than raising, which
is why none was caught by the self-test in `cropalign.py`.

The last group is the two-pass lifecycle. An external audit found that pass two did
not fail closed: a missing crop, a duplicated crop and a crop pass one never saw were
all accepted, and each returned kappa = +1.0000 on a case whose true value is
+1.0000, so nothing looked wrong. A malformed submission scoring a subset of the
artefact set, or counting one crop's pixels twice, is exactly the silent-wrong-number
failure this project is about, in the one file other people are meant to run.

These tests `assert` and return None. They used to return their message, which pytest
flags (PytestReturnNotNoneWarning) because a returned value is easy to mistake for a
pass/fail signal and a test returning something falsy still passes. Messages are now
recorded out of band by `detail()`.
"""
import inspect
import itertools
import sys

import numpy as np

import os
sys.path.insert(0, os.environ.get("PROJECT_ROOT",
                                  os.path.dirname(os.path.abspath(__file__))))
import cropalign  # noqa: E402

_DETAIL = {}


def detail(msg):
    """Record a human-readable result for the runner without returning it."""
    _DETAIL[inspect.stack()[1].function] = msg


def geometry(H=64, CROP=32, STRIDE=16):
    starts = list(range(0, H - CROP + 1, STRIDE))
    return [(r, c) for r in starts for c in starts], CROP


def _two_crop_scene(finalize=True):
    """One row, two crops that disagree wherever they overlap.

    An own-label predictor scores exactly +1.0 here, which makes it a sharp probe:
    every lifecycle fault below also produced +1.0, so the value alone cannot
    distinguish a correct run from a malformed one. Only failing closed can.
    """
    ca = cropalign.CropAlignment(n_classes=2)
    ca.add_scene((1, 6))
    a = ((0, 0), np.array([[0, 0, 0]], np.int64))
    b = ((0, 1), np.array([[1, 1, 1]], np.int64))
    if finalize:
        ca.add_labels(a[0], a[1])
        ca.add_labels(b[0], b[1])
        ca.finalize_labels()
    return ca, a, b


def test_nodata_preserves_the_null():
    """A crop-invariant predictor must give kappa 0 even when a crop has nodata."""
    boxes, CROP = geometry()
    H = 64
    rng = np.random.default_rng(3)
    base = rng.integers(0, 2, (H, H)).astype(np.int64)
    ca = cropalign.CropAlignment(2, (H, H), np.ones((H, H), bool))
    labs = {}
    for i, (r, c) in enumerate(boxes):
        lab = base[r:r + CROP, c:c + CROP].copy()
        # make the labels genuinely crop-dependent so A is non-empty ...
        lab[0, 0] = 1 - lab[0, 0]
        if i % 3 == 0:                       # ... and mark some crops nodata
            lab[:4, :4] = -1
        labs[(r, c)] = lab
        ca.add_labels((r, c), lab)
    ca.finalize_labels()
    for (r, c) in boxes:
        pred = base[r:r + CROP, c:c + CROP].copy()   # crop-INVARIANT
        ca.add_predictions((r, c), labs[(r, c)], pred)
    k = ca.result()["kappa"]
    assert abs(k) < 1e-12, "nodata broke the structural null: kappa = {:+.6f}".format(k)
    detail("kappa = {:+.2e} with nodata labels present".format(k))


def test_negative_prediction_raises():
    boxes, CROP = geometry()
    H = 64
    rng = np.random.default_rng(4)
    ca = cropalign.CropAlignment(2, (H, H), np.ones((H, H), bool))
    labs = {}
    for (r, c) in boxes:
        lab = rng.integers(0, 2, (CROP, CROP)).astype(np.int64)
        labs[(r, c)] = lab
        ca.add_labels((r, c), lab)
    ca.finalize_labels()
    r, c = boxes[0]
    try:
        ca.add_predictions((r, c), labs[(r, c)], np.full((CROP, CROP), -1, np.int64))
    except ValueError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("an all -1 prediction was accepted")


def test_changed_labels_raise():
    boxes, CROP = geometry()
    H = 64
    rng = np.random.default_rng(5)
    ca = cropalign.CropAlignment(2, (H, H), np.ones((H, H), bool))
    labs = {}
    for (r, c) in boxes:
        lab = rng.integers(0, 2, (CROP, CROP)).astype(np.int64)
        labs[(r, c)] = lab
        ca.add_labels((r, c), lab)
    ca.finalize_labels()
    r, c = boxes[0]
    other = 1 - labs[(r, c)]
    try:
        ca.add_predictions((r, c), other, labs[(r, c)])
    except ValueError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("a different pass-2 label field was accepted")


def test_exact_value_two_crops():
    """m = 2 everywhere on A: an own-label predictor must give exactly 1.0.

    This is the case that separates the (m-1) normaliser from an m normaliser; the
    old self-test passed under both.
    """
    H, CROP = 8, 6
    boxes = [(0, 0), (0, 2)]
    ca = cropalign.CropAlignment(2, (H, H), np.ones((H, H), bool))
    labs = {}
    for i, (r, c) in enumerate(boxes):
        lab = np.full((CROP, CROP), i, np.int64)   # the two crops disagree wherever
        labs[(r, c)] = lab                          # they overlap
        ca.add_labels((r, c), lab)
    ca.finalize_labels()
    for (r, c) in boxes:
        ca.add_predictions((r, c), labs[(r, c)], labs[(r, c)])
    res = ca.result()
    k = res["kappa"]
    assert abs(k - 1.0) < 1e-12, \
        "own-label predictor on m=2 gave {:+.6f}, not 1.0".format(k)
    detail("kappa = {:+.6f}, crop_size {}, stride {}".format(
        k, res["crop_size"], res["stride"]))


def test_geometry_reported():
    boxes, CROP = geometry(H=64, CROP=32, STRIDE=16)
    H = 64
    rng = np.random.default_rng(6)
    ca = cropalign.CropAlignment(2, (H, H), np.ones((H, H), bool))
    labs = {}
    for (r, c) in boxes:
        lab = rng.integers(0, 2, (CROP, CROP)).astype(np.int64)
        labs[(r, c)] = lab
        ca.add_labels((r, c), lab)
    ca.finalize_labels()
    for (r, c) in boxes:
        ca.add_predictions((r, c), labs[(r, c)], labs[(r, c)])
    res = ca.result()
    assert res["crop_size"] == (32, 32), res["crop_size"]
    assert res["stride"] == (16, 16), res["stride"]
    detail("crop_size {} stride {} n_crops {}".format(
        res["crop_size"], res["stride"], res["n_crops"]))


# -- the two-pass lifecycle must fail closed -------------------------------

def test_missing_prediction_crop_raises():
    """Pass two omitting a crop scored a subset of A and returned +1.0000."""
    ca, a, _ = _two_crop_scene()
    ca.add_predictions(a[0], a[1], a[1])
    try:
        ca.result()
    except RuntimeError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("a pass 2 missing a crop was accepted")


def test_duplicate_prediction_crop_raises():
    """A repeated crop counted its pixels twice and still returned +1.0000."""
    ca, a, _ = _two_crop_scene()
    ca.add_predictions(a[0], a[1], a[1])
    try:
        ca.add_predictions(a[0], a[1], a[1])
    except ValueError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("a duplicated pass-2 crop was accepted")


def test_extra_prediction_crop_raises():
    """A crop pass one never saw contributed predictions to Omega regardless."""
    ca, a, b = _two_crop_scene()
    ca.add_predictions(a[0], a[1], a[1])
    ca.add_predictions(b[0], b[1], b[1])
    lab = np.array([[0, 0, 0]], np.int64)
    try:
        ca.add_predictions((0, 3), lab, lab)
    except ValueError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("a crop absent from pass 1 was accepted")


def test_same_offset_different_size_raises():
    """A crop's identity is its origin AND its size; the origin alone collided."""
    ca, a, b = _two_crop_scene()
    ca.add_predictions(a[0], a[1], a[1])
    ca.add_predictions(b[0], b[1], b[1])
    lab = np.array([[0, 0, 0, 0]], np.int64)      # same origin as a, wider
    try:
        ca.add_predictions((0, 0), lab, lab)
    except ValueError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("a different-sized crop at a used origin was accepted")


def test_reordered_predictions_are_accepted():
    """Order is not part of the contract: only the SET of crops is."""
    ca, a, b = _two_crop_scene()
    ca.add_predictions(a[0], a[1], a[1])
    ca.add_predictions(b[0], b[1], b[1])
    forward = ca.result()["kappa"]
    ca2, a2, b2 = _two_crop_scene()
    ca2.add_predictions(b2[0], b2[1], b2[1])      # reversed
    ca2.add_predictions(a2[0], a2[1], a2[1])
    reverse = ca2.result()["kappa"]
    assert abs(forward - reverse) < 1e-12, \
        "order changed the answer: {:+.6f} vs {:+.6f}".format(forward, reverse)
    detail("kappa = {:+.6f} either order".format(forward))


def test_empty_artefact_set_says_undefined():
    """The message must not claim kappa is 0 there. It is undefined."""
    ca = cropalign.CropAlignment(n_classes=2)
    ca.add_scene((1, 6))
    lab = np.array([[0, 0, 0]], np.int64)
    for box in ((0, 0), (0, 1)):                  # identical labels: A is empty
        ca.add_labels(box, lab)
    ca.finalize_labels()
    for box in ((0, 0), (0, 1)):
        ca.add_predictions(box, lab, lab)
    try:
        ca.result()
    except RuntimeError as e:
        msg = str(e)
        assert "UNDEFINED" in msg, "empty A did not say undefined: " + msg[:70]
        assert "0 by the proposition" not in msg, \
            "empty A still claims the proposition gives 0"
        detail("raised, and says undefined")
        return
    raise AssertionError("an empty artefact set returned a result")


# -- the lifecycle is a state machine, not a guard --------------------------
#
# A second audit reproduced three more ways in. Each returned a plausible number
# rather than raising, and the third was a hole opened by the previous fix: the
# guard read `if missing and self._pred_seen`, so a pass two that supplied NOTHING
# skipped the check entirely.

def test_duplicate_label_crop_raises():
    """A crop offered twice in pass one votes twice and moves the artefact set.

    On the two-crop scene whose true value is +1.0000, one duplicate returned
    +0.7500 and raised nothing.
    """
    ca, a, b = _two_crop_scene(finalize=False)
    ca.add_labels(a[0], a[1])
    try:
        ca.add_labels(a[0], a[1])
    except ValueError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("a duplicated pass-1 crop was accepted")


def test_double_finalize_raises():
    ca, a, b = _two_crop_scene()
    try:
        ca.finalize_labels()
    except RuntimeError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("finalize_labels() was accepted twice")


def test_scene_with_no_predictions_raises():
    """Opening the next scene must not quietly discard an unscored one."""
    ca, a, b = _two_crop_scene()
    try:
        ca.add_scene((1, 6))          # no predictions were ever supplied
    except RuntimeError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("a scene with zero predictions was discarded silently")


def test_out_of_order_calls_raise():
    ca, a, b = _two_crop_scene(finalize=False)
    ca.add_labels(a[0], a[1])
    ca.add_labels(b[0], b[1])
    try:
        ca.add_predictions(a[0], a[1], a[1])       # before finalize_labels
    except RuntimeError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("add_predictions was accepted before finalize_labels")


def test_labels_after_finalize_raise():
    ca, a, b = _two_crop_scene()
    try:
        ca.add_labels(b[0], b[1])
    except RuntimeError as e:
        detail("raised: {}".format(str(e)[:56]))
        return
    raise AssertionError("add_labels was accepted after finalize_labels")


def test_two_weightings_differ_on_a_known_case():
    """The released file must return BOTH statistics, with the reported one primary.

    Every other exact-value test here has one artefact pixel or uniform coverage,
    so the two weightings agree and none of them could tell that this file returned
    the superseded one for a whole revision. Hand values: 1/6 and 1/8.
    """
    ca = cropalign.CropAlignment(2)
    ca.add_scene((1, 12))
    crops = [((0, 0), np.array([[0, 0]], np.int64), np.array([[0, 0]], np.int64)),
             ((0, 1), np.array([[1, 1]], np.int64), np.array([[1, 1]], np.int64))]
    for off, lab in ((5, 0), (6, 1), (7, 0), (8, 1)):
        crops.append(((0, off), np.full((1, 4), lab, np.int64),
                      np.zeros((1, 4), np.int64)))
    for box, lab, _ in crops:
        ca.add_labels(box, lab)
    ca.finalize_labels()
    for box, lab, pr in crops:
        ca.add_predictions(box, lab, pr)
    r = ca.result()
    assert abs(r["kappa_pixel"] - 1.0 / 6.0) < 1e-12, r["kappa_pixel"]
    assert abs(r["kappa_crop_read"] - 0.125) < 1e-12, r["kappa_crop_read"]
    assert r["kappa"] == r["kappa_pixel"], "kappa must alias the reported statistic"
    assert abs(r["kappa_pixel"] - r["kappa_crop_read"]) > 1e-9,         "this case is supposed to separate the two weightings"
    detail("pixel {:+.6f} vs crop-read {:+.6f}".format(
        r["kappa_pixel"], r["kappa_crop_read"]))


def test_omega_matches_its_printed_equation():
    """Omega as the paper prints it, recomputed by brute force over crop pairs.

    Omega was carried in four printed values and defined only in words, so nothing
    tied the number to a formula a reader could check. The equation says: ordered
    pairs of distinct crops covering p that disagree, summed over pixels with
    m(p) >= 2, divided by the sum of m(p)(m(p)-1) over the same pixels. That is
    computed here from first principles and compared with the estimator, so a change
    to either one that separates them fails.
    """
    rng = np.random.default_rng(0)
    H = W = 24
    S = 16
    boxes = [(0, 0), (0, 8), (8, 0), (8, 8), (4, 4)]
    labels = {b: rng.integers(0, 2, (S, S)).astype(np.uint8) for b in boxes}
    preds = {b: rng.integers(0, 2, (S, S)).astype(np.uint8) for b in boxes}

    est = cropalign.CropAlignment(n_classes=2)
    est.add_scene((H, W))
    for b in boxes:
        est.add_labels(b, labels[b])
    est.finalize_labels()
    for b in boxes:
        est.add_predictions(b, labels[b], preds[b])
    got = est.result()["omega"]

    num = den = 0
    for r in range(H):
        for c in range(W):
            cov = [b for b in boxes if b[0] <= r < b[0] + S and b[1] <= c < b[1] + S]
            m = len(cov)
            if m < 2:
                continue
            den += m * (m - 1)
            for a, bb in itertools.permutations(cov, 2):
                num += int(preds[a][r - a[0], c - a[1]]
                           != preds[bb][r - bb[0], c - bb[1]])
    assert den > 0, "the case must have overlapping crops or it tests nothing"
    assert abs(got - num / den) < 1e-12, (
        "estimator {:.12f} against the printed equation {:.12f}".format(
            got, num / den))
    detail("omega {:+.6f} == equation, {} ordered pairs".format(got, den))


TESTS = [test_nodata_preserves_the_null, test_negative_prediction_raises,
         test_changed_labels_raise, test_exact_value_two_crops,
         test_geometry_reported,
         test_missing_prediction_crop_raises,
         test_duplicate_prediction_crop_raises,
         test_extra_prediction_crop_raises,
         test_same_offset_different_size_raises,
         test_reordered_predictions_are_accepted,
         test_empty_artefact_set_says_undefined,
         test_duplicate_label_crop_raises,
         test_double_finalize_raises,
         test_scene_with_no_predictions_raises,
         test_out_of_order_calls_raise,
         test_labels_after_finalize_raise,
         test_two_weightings_differ_on_a_known_case,
         test_omega_matches_its_printed_equation]


if __name__ == "__main__":
    bad = 0
    for t in TESTS:
        try:
            t()
            print("  PASS  {:38s} {}".format(t.__name__, _DETAIL.get(t.__name__, "")))
        except AssertionError as e:
            print("  FAIL  {:38s} {}".format(t.__name__, e))
            bad += 1
    print("\n{} of {} hardening checks passed".format(len(TESTS) - bad, len(TESTS)))
    sys.exit(1 if bad else 0)
