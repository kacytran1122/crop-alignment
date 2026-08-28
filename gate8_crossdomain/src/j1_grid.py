r"""Section 4.2's grid comparison, recomputed on the J1 arms.

The table in the paper (+0.1907 at stride 32 against +0.3260 at stride 64, a factor
of 1.71) was measured on the pre-J1 dial, whose alpha = 1 arm is not the alpha = 1 arm
of the rebuilt calibration. Same models it is not, so the same numbers it should not
be.

The claim being tested is unchanged: kappa is measured on the pixels where covering
crops disagree, so thinning the grid changes both the artefact set and the number,
and what has to survive is the RANKING of events rather than the magnitude.
"""
import glob
import json
import os

import numpy as np
from scipy import stats as st

import os
W = os.environ.get("SEAICE_ROOT",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K = os.path.join(W, "runs_s1fc", "kappa")


def grab(suffix, arm="a100"):
    out = {}
    for f in glob.glob(os.path.join(K, "j1_" + arm + "_*_s42" + suffix + ".json")):
        r = json.loads(open(f).read())
        out[r["event"]] = r
    return out


d32 = grab("")
d64 = grab("_step2")
z32 = {e: r["kappa_pixel"] for e, r in grab("", "a000").items()}
z64 = {e: r["kappa_pixel"] for e, r in grab("_step2", "a000").items()}
ev = sorted(set(d32) & set(d64))
print("events on both grids:", len(ev))

# the pixel-weighted statistic throughout, matching the rest of the paper
K = "kappa_pixel"
for _d in (d32, d64):
    for _e, _r in _d.items():
        if K not in _r:
            raise SystemExit("{} has no {}: rescore first".format(_e, K))
k32 = np.array([d32[e][K] for e in ev])
k64 = np.array([d64[e][K] for e in ev])
a32 = np.array([d32[e].get("frac_artefact", np.nan) for e in ev])
a64 = np.array([d64[e].get("frac_artefact", np.nan) for e in ev])
c32 = d32[ev[0]].get("n_crops_per_chip")
c64 = d64[ev[0]].get("n_crops_per_chip")

print()
print("{:10s} {:>12s} {:>12s}".format("", "stride 32", "stride 64"))
print("{:10s} {:>12d} {:>12d}".format("crops/chip", c32 or -1, c64 or -1))
if np.isfinite(a32).all():
    print("{:10s} {:>11.2f}% {:>11.2f}%".format("A frac", 100 * a32.mean(),
                                                100 * a64.mean()))
# both terms must be the SAME estimand: mixing the crop-read a100 with the
# pixel-weighted a000 gave a contrast that reconciled with neither column
c32v = np.array([d32[e][K] - z32[e] for e in ev])
c64v = np.array([d64[e][K] - z64[e] for e in ev])
print("{:10s} {:>+12.4f} {:>+12.4f}".format("kappa a100", k32.mean(), k64.mean()))
print("{:10s} {:>+12.4f} {:>+12.4f}".format("kappa a000", np.mean(list(z32.values())), np.mean(list(z64.values()))))
print("{:10s} {:>+12.4f} {:>+12.4f}".format("contrast", c32v.mean(), c64v.mean()))
print()
print("contrast ratio: {:.2f}x".format(c64v.mean() / c32v.mean()))
r = st.spearmanr(c32v, c64v)
print("event ranking preserved: Spearman r = {:+.3f}".format(r.statistic))
rp = st.pearsonr(c32v, c64v)
print("                         Pearson  r = {:+.3f}".format(rp.statistic))

# A machine-readable copy beside the printed table. The printed table alone let the
# supplement quote grid numbers that no artefact in the repository could confirm:
# the file on disk was still the pre-rescore one and nothing compared the two.
out = {
    "events": ev,
    "n_events": len(ev),
    "crops_per_chip": {"stride32": c32, "stride64": c64},
    "a_frac": {"stride32": float(a32.mean()), "stride64": float(a64.mean())},
    "kappa_a100": {"stride32": float(k32.mean()), "stride64": float(k64.mean())},
    "kappa_a000": {"stride32": float(np.mean(list(z32.values()))),
                   "stride64": float(np.mean(list(z64.values())))},
    "contrast": {"stride32": float(c32v.mean()), "stride64": float(c64v.mean())},
    "ratio": float(c64v.mean() / c32v.mean()),
    "spearman": float(r.statistic),
    "pearson": float(rp.statistic),
}
p = os.path.join(W, "runs", "j1_grid.json")
with open(p, "w") as fh:
    json.dump(out, fh, indent=1)
print()
print("wrote " + p)
