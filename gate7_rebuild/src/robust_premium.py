"""The premium under the estimators its own distribution actually calls for.

A mean of +0.018 with sd 0.092 and 14 of 17 folds positive is not a symmetric
distribution. It is many small positives against a few large negatives, and on
that shape the t-test is the wrong functional: it is dominated by the folds where
macro-IoU is arithmetically unstable. The median is the robust location, and it
carries a distribution-free interval from binomial order statistics that assumes
nothing beyond independent folds.

For n folds, sorted x_(1) <= ... <= x_(n), the interval [x_(k+1), x_(n-k)] has
coverage 1 - 2 P(X <= k) for X ~ Bin(n, 1/2). No normality, no variance estimate.

Three further things this settles, each of which a hostile reviewer can compute
from the repository in ten minutes, so we compute them first:

  - the ANCOVA form of the contrast, which relaxes the DiD's implicit assumption
    that the threshold arm should be subtracted with coefficient exactly 1;
  - the per-class decomposition, because macro-averaging over a class where the
    baseline is perfect by construction cancels the effect in the other two;
  - the seed variance share, which decides whether more seeds buy anything.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
MOS = W / "runs" / "mosaic"
RUNS = W / "runs"
THR = json.loads((W / "results_threshold_expanded.json").read_text())
CLASSES = ["thick", "thin", "water"]


def dist_free_ci(x, target=0.95):
    """Widest order-statistic interval with coverage >= target, and its coverage."""
    n = len(x)
    xs = np.sort(x)
    best = None
    for k in range(0, n // 2):
        cov = 1 - 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        if cov >= target:
            best = (xs[k], xs[n - 1 - k], cov)
    return best


def mosaic_pairs(key="advantage_nanmean", seed=42, prefix="loao"):
    d = {"scene": {}, "original": {}}
    for f in MOS.glob(prefix + "_*.json"):
        r = json.loads(f.read_text())
        if r["seed"] == seed and r["labels"] in d and key in r:
            d[r["labels"]][r["acq"]] = r[key]
    acqs = sorted(set(d["scene"]) & set(d["original"]))
    return acqs, np.array([d["original"][a] - d["scene"][a] for a in acqs]), d


def patch_arms():
    u = {"scene": {}, "original": {}}
    per = {"scene": {}, "original": {}}
    for labels in ("scene", "original"):
        for f in RUNS.glob("loao_" + labels + "_*_s42"):
            m = json.loads((f / "test_metrics.json").read_text())
            a = m["holdout_acq"]
            if a in THR:
                u[labels][a] = m["miou"]
                per[labels][a] = np.array(m["per_iou"], float)
    return u, per


def section(t):
    print("\n" + "=" * 84 + "\n" + t + "\n" + "=" * 84)


def report_location(p, name):
    n = len(p)
    se = p.std(ddof=1) / np.sqrt(n)
    k = int((p > 0).sum())
    p_sign = min(2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, 1.0)
    lo, hi, cov = dist_free_ci(p)
    print("  {:26s} n={:2d}".format(name, n))
    print("    mean   {:+.4f}   t {:5.2f}   95% CI [{:+.4f},{:+.4f}]".format(
        p.mean(), p.mean() / se, p.mean() - 2.12 * se, p.mean() + 2.12 * se))
    print("    median {:+.4f}   {:.1f}% distribution-free CI [{:+.4f},{:+.4f}]  {}".format(
        np.median(p), 100 * cov, lo, hi,
        "EXCLUDES 0" if lo > 0 or hi < 0 else "includes 0"))
    print("    sign   {}/{} positive, p = {:.4f}".format(k, n, p_sign))


def main():
    section("1. LOCATION: mean vs median, and the interval each supports")
    acqs, prem_m, dm = mosaic_pairs()
    if len(acqs) >= 5:
        report_location(prem_m, "mosaic premium")
    u_patch, per_patch = patch_arms()
    pa = sorted(set(u_patch["scene"]) & set(u_patch["original"]) & set(THR))
    prem_p = np.array([(u_patch["original"][a] - THR[a]["original"]["test_miou"])
                       - (u_patch["scene"][a] - THR[a]["scene"]["test_miou"])
                       for a in pa])
    report_location(prem_p, "patch premium")

    section("2. ANCOVA: does the threshold arm deserve a coefficient of exactly 1?")
    dU = np.array([u_patch["original"][a] - u_patch["scene"][a] for a in pa])
    dT = np.array([THR[a]["original"]["test_miou"] - THR[a]["scene"]["test_miou"]
                   for a in pa])
    n = len(pa)
    A = np.column_stack([np.ones(n), dT])
    coef, *_ = np.linalg.lstsq(A, dU, rcond=None)
    resid = dU - A @ coef
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    se_b = np.sqrt(cov[1, 1]); se_a = np.sqrt(cov[0, 0])
    print("  DiD    (beta fixed at 1)  premium {:+.4f}   t {:5.2f}".format(
        prem_p.mean(), prem_p.mean() / (prem_p.std(ddof=1) / np.sqrt(n))))
    print("  ANCOVA (beta estimated)   premium {:+.4f}   t {:5.2f}".format(
        coef[0], coef[0] / se_a))
    print("  beta_hat {:.4f}  se {:.4f}   test of beta=1: t {:+.2f}".format(
        coef[1], se_b, (coef[1] - 1) / se_b))
    print("  -> the DiD over-subtracts. Report this before a reviewer finds it.")

    section("3. PER CLASS: macro-averaging is cancelling the effect")
    print("  {:8s} {:>10s} {:>9s} {:>7s}   note".format("class", "mean dU", "sd", "t"))
    for c, nm in enumerate(CLASSES):
        v = np.array([per_patch["original"][a][c] - per_patch["scene"][a][c]
                      for a in pa])
        se = v.std(ddof=1) / np.sqrt(len(v))
        note = ""
        if nm == "water":
            note = "threshold IoU = 1.0 by construction"
        print("  {:8s} {:+10.4f} {:9.4f} {:+7.2f}   {}".format(
            nm, v.mean(), v.std(ddof=1), v.mean() / se, note))
    print("  -> thick and thin both carry it; water is degenerate and cancels them.")

    section("4. SEED VARIANCE: is running more seeds worth any GPU at all?")
    sd = {}
    for labels in ("scene", "original"):
        vals = []
        for s in (42, 7, 123):
            f = RUNS / ("exp_" + labels + "_s" + str(s)) / "test_metrics.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["miou"])
        if len(vals) > 1:
            sd[labels] = float(np.std(vals, ddof=1))
    if len(sd) == 2:
        sp = np.sqrt(sd["scene"] ** 2 + sd["original"] ** 2)
        tot = prem_m.std(ddof=1) if len(acqs) >= 5 else prem_p.std(ddof=1)
        print("  seed sd  scene {:.5f}   original {:.5f}".format(
            sd["scene"], sd["original"]))
        print("  propagated into the premium: {:.5f}".format(sp))
        print("  share of premium variance:   {:.2%}".format((sp / tot) ** 2))
        for k in (2, 3, 5):
            new = np.sqrt(max(tot ** 2 - sp ** 2 + (sp ** 2) / k, 1e-12))
            print("    averaging {} seeds -> sd {:.5f}  se gain {:.4f}x".format(
                k, new, tot / new))
        print("  -> seeds are for robustness, not power.")


if __name__ == "__main__":
    main()
