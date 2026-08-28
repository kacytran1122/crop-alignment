"""
Equivalence tests for the nulls.

KNOWN_ISSUES #9: three claims in the draft accept the null hypothesis ("no
difference", "eliminated", "inert"). A non-significant difference is not evidence
of no difference, and at n=3 seeds it is very weak evidence of anything. The
standard repair is TOST: declare equivalence only if the whole interval for the
difference lies inside a margin fixed in advance.

The margin must be argued rather than chosen after seeing the data. We use the
fusion gain the audited work reports, +0.0306. It is the smallest effect anyone
has claimed matters on this benchmark, so the resulting statement -- "we can rule
out an effect as large as the one reported" -- is both the one a reader wants and
falsifiable.
"""
import json
from pathlib import Path

import numpy as np

RUNS = Path("runs")
MARGIN = 0.0306
NAN = float("nan")
T90 = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132}
ORDER = ["none", "true", "shuffle", "noise", "dummy", "zero"]


def load(prefix):
    rows = {}
    for d in sorted(RUNS.glob(prefix + "*")):
        f = d / "test_metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            rows.setdefault(m["photon"], {})[m["seed"]] = m["miou"]
    return rows


def report(rows, title):
    if not rows:
        print("no runs for " + title)
        return
    seeds = sorted(set().union(*[set(v) for v in rows.values()]))
    print("\n" + "=" * 74)
    print(title + "   " + str(len(rows)) + " conditions, seeds " + str(seeds))
    print("=" * 74)
    head = "".join("s" + str(s) + "      " for s in seeds)
    print("condition    " + head + " mean      sd")
    for c in ORDER:
        if c not in rows:
            continue
        v = np.array([rows[c][s] for s in seeds if s in rows[c]])
        cells = "".join("{:<8.4f}".format(rows[c].get(s, NAN)) for s in seeds)
        sd = v.std(ddof=1) if len(v) > 1 else NAN
        print("{:12s} {} {:.4f}  {:.4f}".format(c, cells, v.mean(), sd))

    print("\nTOST vs optical-only, margin +/-{:.4f}".format(MARGIN))
    print("{:22s} {:>8s} {:>7s} {:>19s}  {}".format(
        "contrast", "diff", "sd", "90% CI", "verdict"))
    for c in ORDER[1:]:
        if c not in rows or "none" not in rows:
            continue
        d = np.array([rows[c][s] - rows["none"][s]
                      for s in seeds if s in rows[c] and s in rows["none"]])
        if len(d) < 2:
            continue
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = T90.get(len(d), 1.96)
        lo, hi = d.mean() - t * se, d.mean() + t * se
        ok = lo > -MARGIN and hi < MARGIN
        print("{:22s} {:+8.4f} {:7.4f} [{:+.4f},{:+.4f}]  {}".format(
            c + " - none", d.mean(), d.std(ddof=1), lo, hi,
            "EQUIVALENT" if ok else "inconclusive"))

    means = np.array([np.mean([rows[c][s] for s in seeds if s in rows[c]])
                      for c in rows])
    sds = [np.std([rows[c][s] for s in seeds if s in rows[c]], ddof=1)
           for c in rows if len(rows[c]) > 1]
    print("\nspread of condition means      {:.4f}".format(means.max() - means.min()))
    print("pooled within-condition seed sd {:.4f}".format(np.mean(sds)))
    print("smallest effect this design could detect at 80% power, n={}: {:.4f}".format(
        len(seeds), 2.9 * np.mean(sds) / np.sqrt(len(seeds))))


report(load("xg3_"), "17 acquisitions (scale)")
report(load("mech_"), "2 acquisitions, original labels")
report(load("rev_"), "2 acquisitions, reversal check")
