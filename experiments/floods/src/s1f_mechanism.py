"""Transfer collapse, with a correct two-sided sign test and a mechanism check.

Two things the first pass got wrong or left undone.

The sign test in s1f_analyse.py counts positives and tests the upper tail, which
is right for the sea-ice premium but wrong here: the collapse is NEGATIVE, so
counting 2 positives out of 11 and testing upward returns p = 1.0 and understates
a result that is actually 9/11 in the expected direction. The two-sided test uses
whichever tail is larger.

The mechanism check is the more important addition. If the collapse is caused by
the model learning a generator rather than the phenomenon, then it should be
LARGEST where the generator disagrees most with the human -- events where the
algorithmic and expert labels part company are events where reproducing the
generator costs the most against truth. That is a prediction the data can refuse,
and it is much harder to explain away than the collapse alone.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
C = W / "data" / "sen1floods11" / "cache"
R = W / "runs_s1f"
IGNORE = 255


def two_sided_sign(v):
    n = len(v)
    pos = int((v > 0).sum())
    k = max(pos, n - pos)
    p = min(2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, 1.0)
    return pos, n - pos, p


def main():
    runs = {}
    for d in sorted(R.glob("s1f_*_s*")):
        f = d / "test_metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            runs[(m["labels"], m["holdout_event"], m["seed"])] = m
    events = sorted({k[1] for k in runs})
    seeds = sorted({k[2] for k in runs})

    # per-event agreement between the two label sets, from the cache
    ev = np.load(C / "events.npy")
    y_ref = np.load(C / "y_ref.npy")
    y_alg = np.load(C / "y_alg.npy")
    agree = {}
    for e in events:
        k = ev == e
        sc = y_ref[k] != IGNORE
        agree[e] = float(((y_ref[k] == y_alg[k]) & sc).sum() / max(sc.sum(), 1))

    coll, ag, prem = [], [], []
    print("{:12s} {:>10s} {:>11s} {:>11s}".format(
        "event", "agreement", "collapse", "premium"))
    for e in events:
        r = np.mean([runs[("ref", e, s)]["unet_vs_ref"]["miou"]
                     for s in seeds if ("ref", e, s) in runs])
        a = np.mean([runs[("alg", e, s)]["unet_vs_ref"]["miou"]
                     for s in seeds if ("alg", e, s) in runs])
        ua = np.mean([runs[("alg", e, s)]["unet_vs_alg"]["miou"]
                      for s in seeds if ("alg", e, s) in runs])
        ta = np.mean([runs[("alg", e, s)]["thr_vs_alg"]["miou"]
                      for s in seeds if ("alg", e, s) in runs])
        tr = np.mean([runs[("ref", e, s)]["thr_vs_ref"]["miou"]
                      for s in seeds if ("ref", e, s) in runs])
        coll.append(a - r); ag.append(agree[e]); prem.append((ua - ta) - (r - tr))
        print("{:12s} {:10.4f} {:+11.4f} {:+11.4f}".format(
            e, agree[e], a - r, (ua - ta) - (r - tr)))

    coll, ag, prem = np.array(coll), np.array(ag), np.array(prem)
    n = len(coll)

    print("\n" + "=" * 78)
    print("TRANSFER COLLAPSE, corrected two-sided test")
    print("=" * 78)
    se = coll.std(ddof=1) / np.sqrt(n)
    pos, neg, p = two_sided_sign(coll)
    print("  mean {:+.4f}   sd {:.4f}   t {:+.2f}   95% CI [{:+.4f},{:+.4f}]".format(
        coll.mean(), coll.std(ddof=1), coll.mean() / se,
        coll.mean() - 2.23 * se, coll.mean() + 2.23 * se))
    print("  {} of {} events negative, two-sided sign test p = {:.4f}".format(
        neg, n, p))

    print("\n" + "=" * 78)
    print("MECHANISM: is the collapse largest where the generator and the human")
    print("           disagree most?")
    print("=" * 78)
    r_p = float(np.corrcoef(ag, coll)[0, 1])
    order = np.argsort(np.argsort(ag)), np.argsort(np.argsort(coll))
    r_s = float(np.corrcoef(*order)[0, 1])
    A = np.column_stack([np.ones(n), ag])
    beta, *_ = np.linalg.lstsq(A, coll, rcond=None)
    resid = coll - A @ beta
    seb = np.sqrt((resid @ resid / (n - 2)) * np.linalg.inv(A.T @ A)[1, 1])
    print("  corr(label agreement, collapse)  pearson {:+.3f}   spearman {:+.3f}".format(
        r_p, r_s))
    print("  slope {:+.4f} per unit agreement   se {:.4f}   t {:+.2f}".format(
        beta[1], seb, beta[1] / seb))
    lo = coll[ag <= np.median(ag)]
    hi = coll[ag > np.median(ag)]
    print("  events where the labels agree LEAST: collapse {:+.4f} (n={})".format(
        lo.mean(), len(lo)))
    print("  events where the labels agree MOST:  collapse {:+.4f} (n={})".format(
        hi.mean(), len(hi)))
    print("  ratio {:.1f}x".format(lo.mean() / hi.mean() if hi.mean() else float("nan")))
    if beta[1] / seb > 2:
        print("\n  -> the collapse scales with generator/human disagreement, which is")
        print("     what the mechanism predicts and coincidence does not.")

    print("\n" + "=" * 78)
    print("PREMIUM, for comparison with sea ice")
    print("=" * 78)
    sep = prem.std(ddof=1) / np.sqrt(n)
    pos, neg, p = two_sided_sign(prem)
    print("  mean {:+.4f}   sd {:.4f}   t {:+.2f}   {} of {} positive, p = {:.4f}".format(
        prem.mean(), prem.std(ddof=1), prem.mean() / sep, pos, n, p))
    print("  sea ice, mosaic scoring: +0.0182, t = 0.82. Both domains agree the")
    print("  aggregate premium is small and unresolvable at these sample sizes.")

    (R / "mechanism.json").write_text(json.dumps(
        {"events": events, "agreement": ag.tolist(), "collapse": coll.tolist(),
         "premium": prem.tolist(), "slope": float(beta[1]),
         "slope_t": float(beta[1] / seb)}, indent=1))


if __name__ == "__main__":
    main()
