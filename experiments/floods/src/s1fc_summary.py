"""Does kappa track the dial?

Three questions, in the order that decides whether the estimand is trustworthy.

1. Is the control null? At alpha = 0 the generator uses one threshold per chip, so
   the labels carry no crop information at all. A predictor that ignores which crop
   it is reading must score 0. A CNN on crops is not such a predictor -- padding,
   receptive-field truncation and batch statistics all make its output depend on
   the window -- so the empirical control sits a little above 0 and its size is the
   floor beneath which nothing in this design can be resolved.

2. Does kappa rise with alpha? This is the part coincidence cannot supply. Any
   statistic that happens to be positive on closed-loop labels can be positive by
   accident; one that reproduces a known ORDERING across five settings, per event,
   is measuring the thing that was varied.

3. Is the rise proportional? A dial that doubles the generator's crop-dependence
   should roughly double the alignment. Slope and linearity are reported, but the
   claim rests on ordering, which is assumption-free, not on linearity, which is
   not.

Paired throughout by event. Events differ enormously in flooding, terrain and
speckle, so the between-event spread swamps the effect of alpha unless each event
is its own block.
"""
import glob
import json
from math import comb
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
K = W / "runs_s1fc" / "kappa"


def sign_p(pos, n):
    if n == 0:
        return 1.0
    k = max(pos, n - pos)
    return min(2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, 1.0)


def main():
    rows = {}
    for f in glob.glob(str(K / "*.json")):
        d = json.loads(Path(f).read_text())
        rows[(round(d["alpha"], 2), d["event"])] = d
    if not rows:
        raise SystemExit("no kappa results yet in " + str(K))

    alphas = sorted({k[0] for k in rows})
    events = sorted({k[1] for k in rows})
    full = [e for e in events if all((a, e) in rows for a in alphas)]

    print("=" * 92)
    print("KAPPA AGAINST A GENERATOR WHOSE CROP-DEPENDENCE WE SET")
    print("  alpha = 0: one Otsu threshold per chip (crop-invariant control)")
    print("  alpha = 1: threshold recomputed inside every crop (fully closed-loop)")
    print("  artefact set held fixed at the alpha=1 labels, so all arms score the")
    print("  same pixels; only the labels the model TRAINED on differ")
    print("=" * 92)
    print("{:12s}".format("event") + "".join(
        "{:>11s}".format("a=" + format(a, ".2f")) for a in alphas))
    for e in events:
        line = "{:12s}".format(e[:12])
        for a in alphas:
            d = rows.get((a, e))
            line += "{:>11s}".format("{:+.4f}".format(d["kappa"]) if d else "-")
        print(line)

    print("-" * 92)
    means = []
    for a in alphas:
        v = np.array([rows[(a, e)]["kappa"] for e in full])
        means.append(v.mean())
    print("{:12s}".format("mean") + "".join(
        "{:>11s}".format("{:+.4f}".format(m)) for m in means))
    sds = [np.std([rows[(a, e)]["kappa"] for e in full], ddof=1) for a in alphas]
    print("{:12s}".format("sd") + "".join(
        "{:>11s}".format("{:.4f}".format(s)) for s in sds))
    fa = np.mean([rows[(alphas[-1], e)]["frac_artefact"] for e in full])
    om = [np.mean([rows[(a, e)]["omega"] for e in full]) for a in alphas]
    print("{:12s}".format("Omega") + "".join(
        "{:>11s}".format("{:.4f}".format(o)) for o in om))
    print("\n  {} events complete of {};  artefact set = {:.2%} of covered pixels".format(
        len(full), len(events), fa))

    if len(full) < 3:
        print("\n  too few complete events for the tests; rerun when the sweep fills in")
        return

    lo, hi = alphas[0], alphas[-1]
    print("\n" + "=" * 92)
    print("1. IS THE CONTROL NULL?")
    print("=" * 92)
    c = np.array([rows[(lo, e)]["kappa"] for e in full])
    se = c.std(ddof=1) / np.sqrt(len(c))
    print("  kappa at alpha={:.2f}: mean {:+.4f}  sd {:.4f}  t {:+.2f}  range [{:+.4f},{:+.4f}]".format(
        lo, c.mean(), c.std(ddof=1), c.mean() / se if se else float("nan"),
        c.min(), c.max()))
    print("  It is not exactly 0, and it should not be: Omega = {:.4f} at alpha=0,".format(om[0]))
    print("  so the network's own output already depends on the crop through padding")
    print("  and receptive-field truncation. This is the resolution floor, and the")
    print("  estimand the paper reports is the CONTRAST against it, never raw kappa.")

    print("\n" + "=" * 92)
    print("2. DOES KAPPA RISE WITH ALPHA?")
    print("=" * 92)
    d = np.array([rows[(hi, e)]["kappa"] - rows[(lo, e)]["kappa"] for e in full])
    se = d.std(ddof=1) / np.sqrt(len(d))
    pos = int((d > 0).sum())
    print("  contrast alpha={:.2f} minus alpha={:.2f}".format(hi, lo))
    print("    mean {:+.4f}  sd {:.4f}  t {:+.2f}  {}/{} events positive  sign p = {:.5f}".format(
        d.mean(), d.std(ddof=1), d.mean() / se if se else float("nan"),
        pos, len(d), sign_p(pos, len(d))))

    # per-event rank agreement with the known ordering of alpha
    rhos = []
    for e in full:
        v = np.array([rows[(a, e)]["kappa"] for a in alphas])
        ra, rv = np.argsort(np.argsort(alphas)), np.argsort(np.argsort(v))
        rhos.append(float(np.corrcoef(ra, rv)[0, 1]))
    rhos = np.array(rhos)
    perfect = int((rhos > 0.999).sum())
    print("\n  per-event rank correlation between alpha and kappa")
    print("    mean rho {:+.3f}   perfectly ordered in {}/{} events   min {:+.3f}".format(
        rhos.mean(), perfect, len(rhos), rhos.min()))
    if len(alphas) >= 3:
        # probability a single event orders all levels correctly by chance
        import math
        print("    a single event orders {} levels correctly by chance with p = {:.4f}".format(
            len(alphas), 1.0 / math.factorial(len(alphas))))

    print("\n" + "=" * 92)
    print("3. IS THE RISE PROPORTIONAL?")
    print("=" * 92)
    xs = np.repeat(alphas, len(full))
    ys = np.concatenate([[rows[(a, e)]["kappa"] for e in full] for a in alphas])
    A = np.column_stack([np.ones(len(xs)), xs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ beta
    s2 = resid @ resid / (len(xs) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    r = float(np.corrcoef(xs, ys)[0, 1])
    print("  kappa = {:+.4f} + {:+.4f} * alpha     r = {:+.3f}   r^2 = {:.3f}".format(
        beta[0], beta[1], r, r * r))
    print("  slope se {:.4f}  t {:+.2f}      intercept se {:.4f}  t {:+.2f}".format(
        np.sqrt(cov[1, 1]), beta[1] / np.sqrt(cov[1, 1]),
        np.sqrt(cov[0, 0]), beta[0] / np.sqrt(cov[0, 0])))

    (W / "runs_s1fc" / "summary.json").write_text(json.dumps(
        {"alphas": alphas, "events": full,
         "kappa": {str(a): [rows[(a, e)]["kappa"] for e in full] for a in alphas},
         "omega": {str(a): [rows[(a, e)]["omega"] for e in full] for a in alphas},
         "frac_artefact": float(fa), "contrast_mean": float(d.mean()),
         "contrast_t": float(d.mean() / se) if se else None,
         "rank_rho_mean": float(rhos.mean()), "n_perfect": perfect,
         "slope": float(beta[1]), "intercept": float(beta[0])}, indent=1))
    print("\nwrote " + str(W / "runs_s1fc" / "summary.json"))


if __name__ == "__main__":
    main()
