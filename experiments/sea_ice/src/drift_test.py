"""Is the photon null explained by co-location drift rather than by an inert branch?

The measured offsets between the Sentinel-2 and ICESat-2 halves of each pair have a
median of 35.9 minutes and a maximum of 52.4; only 2 of 17 sit within 15 minutes.
At the western Ross Sea drift rate of 12.03 cm/s (Farooq et al. 2022) that is about
26 pixels of displacement at the median and 38 at the worst pair. So a reviewer can
reasonably propose that the descriptors are attached to ground that has moved, and
that the null is about our matching rather than about altimetry.

Two things bear on it, and only the second is a test.

The weaker argument, worth stating but not sufficient: the descriptors are
broadcast uniformly over a 128x128 patch, so 26 pixels is a fifth of the patch
width and the token still describes ground largely inside the same patch. A reader
is entitled to find that unpersuasive.

The test: if drift explains the null, then the fusion contrast must depend on the
offset. Pairs 8 minutes apart should show a real benefit that pairs 52 minutes
apart have lost. Regress the paired per-acquisition fusion contrast on the absolute
offset and look at the slope. A flat line across a sevenfold range of offsets says
drift is not what is producing the null.

The shuffle arm makes it sharper still. Shuffling destroys the patch-to-photon
correspondence completely, whereas drift merely degrades it. If drift were the
mechanism, real descriptors should still beat shuffled ones at the short offsets.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[1]
RUNS = W / "runs"


def miou(name):
    f = RUNS / name / "test_metrics.json"
    if f.exists():
        return json.loads(f.read_text())["miou"]
    return None


def main():
    # NOT pd.read_json: it coerces acq_id to int64 and silently eats the
    # underscore, turning 20191103184432_05780 into 2019110318443205780, which
    # matches no run directory and reports zero complete folds instead of failing.
    recs = json.loads((W / "work" / "coloc_offsets.json").read_text())
    omap = {}
    for r in recs:
        a = str(r["acq_id"])
        omap[a] = max(omap.get(a, 0.0), abs(float(r["offset_s"])) / 60.0)

    rows = []
    for acq, mins in sorted(omap.items(), key=lambda kv: kv[1]):
        none_ = miou("loao_scene_{}_s42".format(acq))
        true_ = miou("lf_true_{}_s42".format(acq))
        shuf_ = miou("lf_shuffle_{}_s42".format(acq))
        if None in (none_, true_, shuf_):
            continue
        rows.append({"acq": acq, "offset_min": mins, "none": none_,
                     "true": true_, "shuffle": shuf_,
                     "true_none": true_ - none_, "true_shuffle": true_ - shuf_})

    if len(rows) < 5:
        print("only {} complete fusion folds; the sweep is still running".format(
            len(rows)))
        for r in rows:
            print("  {acq}  offset {offset_min:5.1f} m  "
                  "true-none {true_none:+.4f}".format(**r))
        return

    d = pd.DataFrame(rows)
    print("{:24s} {:>8s} {:>10s} {:>12s}".format(
        "acquisition", "offset", "true-none", "true-shuffle"))
    for r in d.itertuples():
        print("{:24s} {:7.1f}m {:+10.4f} {:+12.4f}".format(
            r.acq, r.offset_min, r.true_none, r.true_shuffle))

    for col in ("true_none", "true_shuffle"):
        y = d[col].to_numpy()
        x = d["offset_min"].to_numpy()
        n = len(y)
        A = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        s2 = resid @ resid / (n - 2)
        se = np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
        r = float(np.corrcoef(x, y)[0, 1])
        print("\n  {}".format(col))
        print("    mean {:+.4f}   sd {:.4f}".format(y.mean(), y.std(ddof=1)))
        print("    slope on offset {:+.6f} per minute   se {:.6f}   t {:+.2f}".format(
            coef[1], se, coef[1] / se if se else float("nan")))
        print("    correlation with offset {:+.3f}".format(r))
        lo = y[x <= np.median(x)]
        hi = y[x > np.median(x)]
        print("    short-offset folds {:+.4f} (n={})   long-offset folds {:+.4f} (n={})".format(
            lo.mean(), len(lo), hi.mean(), len(hi)))
        print("    predicted at 0 offset: {:+.4f}".format(coef[0]))

    print("\n  A slope indistinguishable from zero across a {:.0f}x range of offsets"
          "\n  ({:.1f} to {:.1f} minutes) means drift is not what produces the null."
          .format(d.offset_min.max() / max(d.offset_min.min(), 1e-9),
                  d.offset_min.min(), d.offset_min.max()))
    d.to_json(RUNS / "drift_test.json", orient="records", indent=1)


if __name__ == "__main__":
    main()
