"""Figure 4, rebuilt on J1: the dial and one threshold-scramble sensitivity arm.

The old figure read runs_s1fc/summary.json, which is the pre-J1 dial whose zero was a
per-chip Otsu of ours. Every value in it moved when the zero became the published
per-event threshold, so redrawing from the same file would have put a stale picture
beside a corrected table.

The right panel compares the dial endpoint with one within-chip threshold scramble.
It preserves each chip's threshold multiset and retained similar aggregate mIoU, but
the unconstrained mapping permits fixed points, overlap and same-chip dependence.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, BLUE, RED, GREY = "#16222E", "#2E6E8E", "#B4261E", "#8A99A6"
W = os.environ.get("SEAICE_ROOT", ".")
src = os.path.join(W, "runs", "j1_summary.json")
if not os.path.exists(src):
    src = os.path.join(W, "results", "j1_summary.json")
d = json.loads(open(src).read())
k, g = d["kappa_by_event"], d["damage_by_event"]
DIAL = ["a000", "a025", "a050", "a075", "a100"]
A = [0.0, 0.25, 0.50, 0.75, 1.0]
ev = sorted(set(k["a000"]) & set(g["a000"]))

K = np.array([[k[a][e] for e in ev] for a in DIAL])
M = np.array([[g[a][e] for e in ev] for a in DIAL])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55))

# ---- left: the dose-response, one grey line per event ----------------------
for j in range(len(ev)):
    ax1.plot(A, K[:, j], color=GREY, lw=0.8, alpha=0.75)
ax1.plot(A, K.mean(1), color=RED, lw=2.0, marker="o", ms=4.5, zorder=5)
ax1.axhline(0, color=INK, lw=1.0, ls="--")
ax1.set_xlabel(r"$\alpha$   (0 = published, 1 = predominantly per crop)", fontsize=8)
ax1.set_ylabel(r"crop-label alignment $\kappa$", fontsize=8)
ax1.set_title("$\\kappa$ tracks how much of the threshold is per crop",
              fontsize=8.5, color=INK)
ax1.tick_params(labelsize=7.5)

# ---- right: one within-chip threshold scramble -----------------------------
arms = ["a000", "a100", "perm"]
names = ["published\n$\\alpha=0$", "predom. crop\n$\\alpha=1$", "scrambled\nthresholds"]
kv = [np.mean([k[a][e] for e in ev]) for a in arms]
mv = [np.mean([g[a][e] for e in ev]) for a in arms]
x = np.arange(3)
b1 = ax2.bar(x - 0.19, kv, 0.36, color=RED, label=r"$\kappa$")
ax2.set_ylabel(r"$\kappa$", color=RED, fontsize=8)
ax2.tick_params(axis="y", labelcolor=RED, labelsize=7.5)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7.5)
ax2.axhline(0, color=INK, lw=0.9)

ax3 = ax2.twinx()
ax3.bar(x + 0.19, mv, 0.36, color=BLUE, label="mIoU")
ax3.set_ylabel("mIoU vs expert", color=BLUE, fontsize=8)
ax3.tick_params(axis="y", labelcolor=BLUE, labelsize=7.5)
ax3.set_ylim(0, 0.85)

for xi, v in zip(x, kv):
    ax2.annotate(("{:+.3f}".format(v) if abs(v) >= 5e-4 else "0.000"), (xi - 0.19, v), ha="center", va="bottom",
                 fontsize=6.8, color=RED)
for xi, v in zip(x, mv):
    ax3.annotate("{:.2f}".format(v), (xi + 0.19, v), ha="center", va="bottom",
                 fontsize=6.8, color=BLUE)
ax2.set_title("scramble control", fontsize=8.5, color=INK)

fig.tight_layout(pad=0.6)
out = sys.argv[1] if len(sys.argv) > 1 else "fig4_calibration.png"
fig.savefig(out, dpi=400)
print("wrote {}: kappa {:+.4f} -> {:+.4f}, perm {:+.4f}; mIoU {:.3f} -> {:.3f}, "
      "perm {:.3f}".format(out, kv[0], kv[1], kv[2], mv[0], mv[1], mv[2]))
