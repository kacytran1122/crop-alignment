r"""Descriptive kappa/expert-label associations across the rebuilt J1 dial.

The dial moves kappa and expert-label performance together, so their pooled
within-event association is a joint dose response. Fixed-alpha correlations remove
the dial level but are heterogeneous and do not identify a mechanism.

The correlations use the rebuilt arms, whose zero is the published threshold.
Leave-one-event-out models share training data, so this script emits descriptive
quantities only.
"""
import json
import sys

import numpy as np

import os
W = os.environ.get("SEAICE_ROOT",
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d = json.loads(open(W + "/runs/j1_summary.json").read())
k, g = d["kappa_by_event"], d["damage_by_event"]
DIAL = ["a000", "a025", "a050", "a075", "a100"]
ev = sorted(set(k["a000"]) & set(g["a000"]))
n = len(ev)
print("events:", n)

K = np.array([[k[a][e] for e in ev] for a in DIAL])
M = np.array([[g[a][e] for e in ev] for a in DIAL])

print("\nfixed-alpha correlation between kappa and mIoU-vs-expert, across events")
rs = []
for i, a in enumerate(DIAL):
    r = float(np.corrcoef(K[i], M[i])[0, 1])
    rs.append(r)
    print("  alpha {:<5} r = {:+.3f}".format(a[1:], r))
print("  mean r = {:+.3f}".format(float(np.mean(rs))))
print("  all five negative: {}".format(all(r < 0 for r in rs)))

# within-event: remove each event's mean so between-event difficulty cannot drive it
Kc = K - K.mean(0)
Mc = M - M.mean(0)
r_within = float(np.corrcoef(Kc.ravel(), Mc.ravel())[0, 1])
print("\nwithin-event, event means removed: r = {:+.3f}".format(r_within))

slope = float(np.polyfit(Kc.ravel(), Mc.ravel(), 1)[0])
print("  slope {:+.4f} mIoU per unit kappa".format(slope))

# the cost itself, event-level
c = M[-1] - M[0]
print("\nendpoint expert-label difference (a100 - a000; not causal)")
print("  mean {:+.4f}  descriptive sd {:.4f}  worse in {}/{} events".format(
    c.mean(), c.std(ddof=1), int((c < 0).sum()), n))

# A machine-readable copy, for the same reason as j1_grid.py: the printed table was
# the only record, so Section 6 quoted a within-event r and slope from a scoring the
# repository had since replaced, and no check compared the two.
out = {
    "n_events": int(n),
    "fixed_alpha_r": [float(x) for x in rs],
    "fixed_alpha_mean_r": float(np.mean(rs)),
    "all_five_negative": bool(all(x < 0 for x in rs)),
    "r_within": r_within,
    "slope_within": slope,
    "cost_mean": float(c.mean()),
    "cost_sd": float(c.std(ddof=1)),
    "cost_worse_events": int((c < 0).sum()),
}
p = os.path.join(W, "runs", "j1_fixed_alpha.json")
with open(p, "w") as fh:
    json.dump(out, fh, indent=1)
print()
print("wrote " + p)
