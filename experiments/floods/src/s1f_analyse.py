"""Sen1Floods11: the transfer collapse, and the premium for continuity.

TRANSFER COLLAPSE is the quantity sea ice could not support, because sea ice has no
reference labels:

    collapse(e) = mIoU(trained on algorithmic, scored vs expert)
                - mIoU(trained on expert,      scored vs expert)

Both terms are scored against the same expert labels on the same held-out event, so
this is a like-for-like comparison of two training regimes against one truth. A
negative value is how much of the model is lost when the labels it learned from are
generated rather than observed. It needs no baseline-family argument.

ARTEFACT PREMIUM, kept so the two domains speak in the same units:

    premium(e) = [UNet_alg - thr_alg](scored vs algorithmic labels)
               - [UNet_ref - thr_ref](scored vs expert labels)

Each arm scored against the labels it was trained on, with the context-free
threshold refit on that same label set, so neither arm is handicapped.

Reported with mean, median and a distribution-free interval, since n = 11 events
and nothing guarantees symmetry.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
R = W / "runs_s1f"


def load():
    out = {}
    for d in sorted(R.glob("s1f_*_s*")):
        f = d / "test_metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        out[(m["labels"], m["holdout_event"], m["seed"])] = m
    return out


def dfci(x, target=0.95):
    n, xs = len(x), np.sort(x)
    best = None
    for k in range(0, n // 2):
        cov = 1 - 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
        if cov >= target:
            best = (xs[k], xs[n - 1 - k], cov)
    return best or (xs[0], xs[-1], 1.0)


def describe(v, name):
    n = len(v)
    if n < 3:
        print("  {:34s} n={} too few".format(name, n))
        return
    se = v.std(ddof=1) / np.sqrt(n)
    # two-sided: use whichever tail is larger, or a negative effect reads as p=1
    pos = int((v > 0).sum())
    k = max(pos, n - pos)
    p = min(2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, 1.0)
    k = pos
    lo, hi, cov = dfci(v)
    print("  {:34s} n={:2d}  mean {:+.4f}  sd {:.4f}  t {:+6.2f}".format(
        name, n, v.mean(), v.std(ddof=1), v.mean() / se if se else float("nan")))
    print("  {:34s}       median {:+.4f}  {:.1f}% CI [{:+.4f},{:+.4f}]  "
          "sign {}/{} p={:.4f}".format("", np.median(v), 100 * cov, lo, hi, k, n, p))


def main():
    runs = load()
    if not runs:
        raise SystemExit("no completed runs in " + str(R))
    events = sorted({k[1] for k in runs})
    seeds = sorted({k[2] for k in runs})
    print("{} runs, {} events, seeds {}".format(len(runs), len(events), seeds))

    print("\n" + "=" * 96)
    print("TRANSFER COLLAPSE: how much survives contact with a human")
    print("  both terms scored against the SAME expert labels on the same event")
    print("=" * 96)
    print("  {:12s} {:>10s} {:>10s} {:>11s} {:>10s} {:>10s}".format(
        "event", "train=ref", "train=alg", "collapse", "thr vs ref", "agree"))

    coll, prem = [], []
    for e in events:
        rr = [runs[("ref", e, s)]["unet_vs_ref"]["miou"]
              for s in seeds if ("ref", e, s) in runs]
        aa = [runs[("alg", e, s)]["unet_vs_ref"]["miou"]
              for s in seeds if ("alg", e, s) in runs]
        if not rr or not aa:
            continue
        r_, a_ = float(np.mean(rr)), float(np.mean(aa))
        tr = float(np.mean([runs[("ref", e, s)]["thr_vs_ref"]["miou"]
                            for s in seeds if ("ref", e, s) in runs]))
        cm = np.array(runs[("ref", e, seeds[0])]["unet_vs_ref"]["cm"])
        coll.append(a_ - r_)
        print("  {:12s} {:10.4f} {:10.4f} {:+11.4f} {:10.4f} {:10s}".format(
            e, r_, a_, a_ - r_, tr, ""))

        # premium: each arm against its own labels, threshold refit on that arm
        pu_a = np.mean([runs[("alg", e, s)]["unet_vs_alg"]["miou"]
                        for s in seeds if ("alg", e, s) in runs])
        pt_a = np.mean([runs[("alg", e, s)]["thr_vs_alg"]["miou"]
                        for s in seeds if ("alg", e, s) in runs])
        pu_r = np.mean([runs[("ref", e, s)]["unet_vs_ref"]["miou"]
                        for s in seeds if ("ref", e, s) in runs])
        pt_r = np.mean([runs[("ref", e, s)]["thr_vs_ref"]["miou"]
                        for s in seeds if ("ref", e, s) in runs])
        prem.append((pu_a - pt_a) - (pu_r - pt_r))

    coll, prem = np.array(coll), np.array(prem)
    print()
    describe(coll, "transfer collapse")
    print()
    print("=" * 96)
    print("ARTEFACT PREMIUM, same units as the sea-ice paper")
    print("=" * 96)
    describe(prem, "premium (alg arm - ref arm)")

    thr = [runs[k]["thr_vh"] for k in runs if k[0] == "alg"]
    if thr:
        print("\n  recovered VH threshold across folds: {:.2f} to {:.2f} "
              "(spread {:.2f} dB)".format(min(thr), max(thr), max(thr) - min(thr)))

    (R / "summary.json").write_text(json.dumps(
        {"events": events, "collapse": coll.tolist(), "premium": prem.tolist()},
        indent=1))
    print("\n  n = {} events. All chips in an event share one generator threshold,".format(
        len(coll)))
    print("  so the event is the independent unit and this design buys generality,")
    print("  not resolution. State that rather than letting a reviewer state it.")


if __name__ == "__main__":
    main()
