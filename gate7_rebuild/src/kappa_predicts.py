"""Does kappa PREDICT the inflation, or does it only diagnose it?

Everything in the paper so far measures the artefact two ways and shows both are
nonzero. That is a diagnosis. A theory has to say how big, in advance, from a
quantity measured somewhere else, and then be wrong if the data disagrees.

The derivation. Take a source pixel p covered by overlapping crops. Patch scoring
evaluates the model once per (crop, pixel) instance against L_c(p), the label as
the generator computed it FOR THAT CROP. On the artefact set A -- the pixels where
overlapping crops disagree about the label -- a crop-dependent model matches the
label of the crop it is reading with probability p_own and some other crop with
p_other, and kappa is defined as exactly that difference. Mosaic scoring gives each
source pixel one decision and one label, so nothing about which crop was read can
be tracked, and the p_own term is unavailable.

The gain therefore lives entirely on A and is worth kappa per pixel there:

    predicted (patch accuracy - mosaic accuracy)  =  P(A) * kappa

Both factors are measured by crop_alignment.py, on a completely different pass over
the data than the one that produced the accuracies. Nothing here is fitted.

Two nuisances make the raw gap a bad test, and one control removes both. Patch
scoring weights a pixel by how many crops cover it (mean reuse is over 100x) while
mosaic weights every source pixel once, and the two conventions handle absent
classes differently. Both nuisances are properties of the SCORING, not of the
model, so they hit the crop-invariant arm equally. Differencing the two arms
removes them and leaves the prediction:

    measured   = (patch - mosaic)_noisy - (patch - mosaic)_control
    predicted  = P(A) * (kappa_noisy - kappa_control)

The control arm is the scene-trained model evaluated on the ORIGINAL label set, so
both arms are scored against the same artefact set A and P(A) is common.

Known first-order approximations, stated because they set which way it should miss.
The mosaic label is a VOTE over the covering crops, not an arbitrary other crop, so
mosaic accuracy on A sits above p_other and the measured gap should come in BELOW
P(A)*kappa. Accuracy is also not mIoU. So the honest prediction is a slope in
(0, 1] against an intercept of zero, and the falsifiable content is that the slope
is positive, well under control, and that folds with more artefact and more kappa
show more inflation -- not that the slope is exactly 1.
"""
import glob
import json
from pathlib import Path

import numpy as np

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
K = W / "runs" / "kappa"
M = W / "runs" / "mosaic"
R = W / "runs"


def load_kappa():
    """Per acquisition: kappa for the crop-noisy arm, for the control, and P(A)."""
    noisy, control, frac = {}, {}, {}
    for f in glob.glob(str(K / "loao_original_*_s42.json")):
        if "_vsorig" in f:
            continue
        d = json.loads(Path(f).read_text())
        noisy[d["acq"]] = d["kappa"]
        frac[d["acq"]] = d["frac_artefact"]
    for f in glob.glob(str(K / "loao_scene_*_s42_vsorig.json")):
        d = json.loads(Path(f).read_text())
        control[d["acq"]] = d["kappa"]
        # both arms must be scored on the same artefact set or P(A) is not common
        if d["acq"] in frac and abs(d["frac_artefact"] - frac[d["acq"]]) > 1e-9:
            raise SystemExit(
                "artefact set differs between arms for {}: {:.6f} vs {:.6f}".format(
                    d["acq"], frac[d["acq"]], d["frac_artefact"]))
    return noisy, control, frac


def acc_gap(labels, acq):
    """patch pixel accuracy minus mosaic pixel accuracy, same run."""
    p = R / "loao_{}_{}_s42".format(labels, acq) / "test_metrics.json"
    m = M / "loao_{}_{}_s42.json".format(labels, acq)
    if not (p.exists() and m.exists()):
        return None
    return (json.loads(p.read_text())["pix_acc"]
            - json.loads(m.read_text())["unet"]["pix_acc"])


def main():
    noisy, control, frac = load_kappa()
    acqs = sorted(set(noisy) & set(control) & set(frac))

    rows = []
    for a in acqs:
        gn, gc = acc_gap("original", a), acc_gap("scene", a)
        if gn is None or gc is None:
            continue
        rows.append((a, frac[a], noisy[a] - control[a],
                     frac[a] * (noisy[a] - control[a]), gn - gc))
    if len(rows) < 3:
        raise SystemExit("only {} complete folds".format(len(rows)))

    print("=" * 100)
    print("DOES KAPPA PREDICT THE INFLATION?")
    print("  predicted = P(artefact set) x (kappa_noisy - kappa_control)")
    print("  measured  = (patch - mosaic) accuracy gap, crop-noisy arm minus control arm")
    print("  nothing below is fitted: both factors come from a separate pass over the data")
    print("=" * 100)
    print("{:22s} {:>8s} {:>10s} {:>11s} {:>11s} {:>9s}".format(
        "held-out acquisition", "P(A)", "d_kappa", "predicted", "measured", "ratio"))
    for a, fa, dk, pred, meas in rows:
        print("{:22s} {:8.4f} {:10.4f} {:+11.4f} {:+11.4f} {:9.2f}".format(
            a[:20], fa, dk, pred, meas, meas / pred if pred else float("nan")))

    pred = np.array([r[3] for r in rows])
    meas = np.array([r[4] for r in rows])
    n = len(pred)

    print("\n" + "-" * 100)
    print("  mean predicted {:+.4f}   mean measured {:+.4f}   ratio {:.2f}".format(
        pred.mean(), meas.mean(), meas.mean() / pred.mean()))

    # unfitted point prediction: is the residual centred on zero?
    d = meas - pred
    se = d.std(ddof=1) / np.sqrt(n)
    print("  residual (measured - predicted) {:+.4f}  sd {:.4f}  t {:+.2f}".format(
        d.mean(), d.std(ddof=1), d.mean() / se if se else float("nan")))

    # does it track fold to fold, which is the part coincidence cannot supply
    r = float(np.corrcoef(pred, meas)[0, 1])
    A = np.column_stack([np.ones(n), pred])
    beta, *_ = np.linalg.lstsq(A, meas, rcond=None)
    resid = meas - A @ beta
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    print("  correlation across folds  r = {:+.3f}   r^2 = {:.3f}".format(r, r * r))
    print("  slope {:+.3f} +- {:.3f}  (t = {:+.2f} vs 0)".format(
        beta[1], np.sqrt(cov[1, 1]), beta[1] / np.sqrt(cov[1, 1])))
    print("  intercept {:+.5f} +- {:.5f}  (t = {:+.2f} vs 0)".format(
        beta[0], np.sqrt(cov[0, 0]), beta[0] / np.sqrt(cov[0, 0])))
    print("  sign agreement: {}/{} folds".format(
        int((np.sign(pred) == np.sign(meas)).sum()), n))

    print("\n" + "=" * 100)
    if r > 0.5 and beta[1] > 0 and beta[1] / np.sqrt(cov[1, 1]) > 2:
        print("  The artefact is not merely present, it is the size the mechanism says.")
        print("  A quantity measured per pixel on one pass predicts, fold by fold, an")
        print("  aggregate gap measured on another. That is the difference between")
        print("  reporting a defect and explaining one.")
    else:
        print("  The prediction does NOT track fold to fold. kappa stays a valid")
        print("  diagnostic with a structural null, but the paper must not claim it")
        print("  explains the magnitude of the inflation. Report this as it stands.")
    print("=" * 100)

    (R / "kappa_predicts.json").write_text(json.dumps(
        {"acqs": [r[0] for r in rows], "frac_artefact": [r[1] for r in rows],
         "d_kappa": [r[2] for r in rows], "predicted": pred.tolist(),
         "measured": meas.tolist(), "r": r, "slope": float(beta[1]),
         "slope_se": float(np.sqrt(cov[1, 1])), "intercept": float(beta[0]),
         "intercept_se": float(np.sqrt(cov[0, 0]))}, indent=1))
    print("wrote " + str(R / "kappa_predicts.json"))


if __name__ == "__main__":
    main()
