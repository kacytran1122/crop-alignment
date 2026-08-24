"""Crop alignment across folds: the treated model against its negative control.

kappa is a one-sample test against an EXACT zero, not a comparison of two noisy
aggregates, so the paired contrast against the control is the conservative form
and the raw level is the structural one. Both are reported, with a
distribution-free interval alongside the t interval because n is 17 and nothing
here guarantees symmetry.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
K = W / "runs" / "kappa"


def load():
    treat, ctrl = {}, {}
    for f in K.glob("*.json"):
        r = json.loads(f.read_text())
        # eval_labels was added to crop_alignment.py partway through the work, so
        # files written before that change do not carry the key. A run scored on
        # its own label set has eval_labels == labels by definition, and that is
        # the right default: .get(...) with no default silently dropped a fold.
        if r.get("eval_labels", r["labels"]) != "original":
            continue
        (ctrl if r["labels"] == "scene" else treat)[r["acq"]] = r
    return treat, ctrl


def dist_free_ci(x, target=0.95):
    n, xs = len(x), np.sort(x)
    best = None
    for k in range(0, n // 2):
        cov = 1 - 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        if cov >= target:
            best = (xs[k], xs[n - 1 - k], cov)
    return best


def describe(v, name):
    n = len(v)
    if n < 3:
        print("  {:30s} n={} too few".format(name, n))
        return
    se = v.std(ddof=1) / np.sqrt(n)
    k = int((v > 0).sum())
    p = min(2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, 1.0)
    ci = dist_free_ci(v)
    print("  {:30s} n={:2d}  mean {:+.4f}  sd {:.4f}  t {:7.2f}".format(
        name, n, v.mean(), v.std(ddof=1), v.mean() / se if se else float("inf")))
    print("  {:30s}       median {:+.4f}  {:.1f}% dist-free CI [{:+.4f},{:+.4f}]  "
          "sign {}/{} p={:.5f}".format("", np.median(v), 100 * ci[2], ci[0], ci[1],
                                       k, n, p))


def main():
    treat, ctrl = load()
    acqs = sorted(set(treat) & set(ctrl))
    if not acqs:
        raise SystemExit("no paired folds in " + str(K))

    print("=" * 92)
    print("CROP ALIGNMENT: does the model reproduce the label of the crop it reads?")
    print("  kappa = P(pred = own crop label | A) - P(pred = other crop label | A)")
    print("  structural null: kappa == 0 exactly for any crop-invariant predictor")
    print("=" * 92)
    print("  {:24s} {:>8s} {:>9s} {:>9s} {:>9s} {:>9s}".format(
        "held-out acquisition", "A frac", "k_noisy", "k_clean", "diff", "Omega_r"))

    kt, kc, om_t, om_c = [], [], [], []
    for a in acqs:
        t, c = treat[a], ctrl[a]
        kt.append(t["kappa"]); kc.append(c["kappa"])
        om_t.append(t["omega"]); om_c.append(c["omega"])
        print("  {:24s} {:8.2%} {:+9.4f} {:+9.4f} {:+9.4f} {:9.2f}".format(
            a, t["frac_artefact"], t["kappa"], c["kappa"],
            t["kappa"] - c["kappa"],
            t["omega"] / c["omega"] if c["omega"] else float("nan")))

    kt, kc = np.array(kt), np.array(kc)
    om_t, om_c = np.array(om_t), np.array(om_c)
    print()
    describe(kt, "kappa, crop-noisy training")
    describe(kc, "kappa, crop-invariant (control)")
    describe(kt - kc, "difference (treated - control)")
    print()
    describe(om_t, "Omega, crop-noisy training")
    describe(om_c, "Omega, crop-invariant (control)")
    print("\n  Omega ratio: mean {:.2f}x".format(float(np.mean(om_t / om_c))))
    print("  Omega is 0 exactly for the two-parameter threshold, by construction.")

    d = kt - kc
    s = d.std(ddof=1)
    print("\n  required n at 80% power for the difference: {:.0f}   (have {})".format(
        np.ceil((2.80 * s / abs(d.mean())) ** 2), len(d)))
    frac = np.array([treat[a]["frac_artefact"] for a in acqs])
    print("  artefact set A: {:.2%} of covered pixels on average, "
          "{:,} instances per fold".format(
              frac.mean(), int(np.mean([treat[a]["n_instances"] for a in acqs]))))

    (W / "runs" / "kappa_summary.json").write_text(json.dumps(
        {"acqs": acqs, "kappa_noisy": kt.tolist(), "kappa_clean": kc.tolist(),
         "omega_noisy": om_t.tolist(), "omega_clean": om_c.tolist(),
         "frac_artefact": frac.tolist()}, indent=1))


if __name__ == "__main__":
    main()
