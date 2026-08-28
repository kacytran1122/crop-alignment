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
if not os.path.exists(src):
    src = os.path.join(W, "gate8_crossdomain", "results", "j1_summary.json")
d = json.loads(open(src).read())
k, g = d["kappa_by_event"], d["damage_by_event"]
DIAL = ["a000", "a025", "a050", "a075", "a100"]
A = [0.0, 0.25, 0.50, 0.75, 1.0]
ev = sorted(set(k["a000"]) & set(g["a000"]))

K = np.array([[k[a][e] for e in ev] for a in DIAL])
M = np.array([[g[a][e] for e in ev] for a in DIAL])

# SIZING. The figure is placed at \linewidth = 236pt = 3.28in. Drawn at 7in it gets
# shrunk by 0.47 on the page, so the 8pt labels this script used to set arrived at
# about 3.8pt against 9.96pt body text: present, but not readable. Drawing it at
# 3.28in instead made the sizes honest and left no room, so the labels collided.
# The canvas therefore stays wide and every size below is multiplied by SC, the
# reciprocal of the shrink, so PT below is the size on the page. The wordy titles
# and axis labels are cut down to fit at that size; the caption carries what they
# used to say.
FIGW = 6.6
SC = FIGW / 3.278
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGW, FIGW * 0.364))

# ---- left: the dose-response, one grey line per event ----------------------
for j in range(len(ev)):
    ax1.plot(A, K[:, j], color=GREY, lw=0.8, alpha=0.75)
ax1.plot(A, K.mean(1), color=RED, lw=2.0, marker="o", ms=4.5, zorder=5)
ax1.axhline(0, color=INK, lw=1.0, ls="--")
ax1.set_xlabel(r"$\alpha$", fontsize=7.2 * SC)
ax1.set_ylabel(r"$\kappa$", fontsize=7.2 * SC)
ax1.set_title("the dial",  fontsize=7.6 * SC, color=INK)
ax1.tick_params(labelsize=6.8 * SC)

# ---- right: one within-chip threshold scramble -----------------------------
arms = ["a000", "a100", "perm"]
# short enough not to collide at three categories in half a column; the caption
# says what each one is
names = [r"$\alpha$=0", r"$\alpha$=1", "perm"]
kv = [np.mean([k[a][e] for e in ev]) for a in arms]
mv = [np.mean([g[a][e] for e in ev]) for a in arms]
x = np.arange(3)
b1 = ax2.bar(x - 0.19, kv, 0.36, color=RED, label=r"$\kappa$")
ax2.set_ylabel(r"$\kappa$", color=RED, fontsize=7.2 * SC)
ax2.tick_params(axis="y", labelcolor=RED, labelsize=6.8 * SC)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=6.4 * SC)
ax2.set_xlim(-0.62, 2.62)
ax2.set_ylim(0, 0.315)          # headroom for the value labels above the bars
ax2.axhline(0, color=INK, lw=0.9)

ax3 = ax2.twinx()
ax3.bar(x + 0.19, mv, 0.36, color=BLUE, label="mIoU")
ax3.set_ylabel("mIoU", color=BLUE, fontsize=7.2 * SC)
ax3.tick_params(axis="y", labelcolor=BLUE, labelsize=6.8 * SC)
ax3.set_ylim(0, 1.02)

# twinx() puts ax3 above ax2, so ax2's value labels were drawn UNDER ax3's bars
# and the last digit of the permutation label sat behind the blue bar: it read
# "+0.0" for +0.07. The bars do not overlap horizontally, so lifting ax2 above
# ax3 is safe as long as its background is made transparent.
ax2.set_zorder(ax3.get_zorder() + 1)
ax2.patch.set_visible(False)

for xi, v in zip(x, kv):
    # a value at the floor sits below the axis, where the label would be clipped
    ax2.annotate(("{:+.2f}".format(v) if abs(v) >= 5e-3 else "0.00"),
                 (xi - 0.19, max(v, 0.0)), ha="center", va="bottom",
                 fontsize=5.6 * SC, color=RED)
# Both sets of labels are drawn on ax2, the upper axes. Left on ax3 they were
# painted under ax2's bars, which is the same overlap in the other direction.
# ax3's values are converted into ax2's scale to place them.
SCALE = 0.315 / 1.02
for xi, v in zip(x, mv):
    ax2.annotate("{:.2f}".format(v), (xi + 0.19, v * SCALE), ha="center",
                 va="bottom", fontsize=5.6 * SC, color=BLUE)
ax2.set_title("scramble control", fontsize=7.6 * SC, color=INK)

fig.tight_layout(pad=0.6)
out = sys.argv[1] if len(sys.argv) > 1 else "fig4_calibration.png"
fig.savefig(out, dpi=400)
print("wrote {}: kappa {:+.4f} -> {:+.4f}, perm {:+.4f}; mIoU {:.3f} -> {:.3f}, "
      "perm {:.3f}".format(out, kv[0], kv[1], kv[2], mv[0], mv[1], mv[2]))
