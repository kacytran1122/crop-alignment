"""Figure 1: two crops, one pixel, two labels.

Nothing in the paper drew the mechanism it is about. A reader met the artefact set,
the own-term and the other-term as algebra on page 3, and the first picture of any
kind arrived on page 4. This puts the idea on page 1.

Single column on purpose. A full-width version cost about four tenths of a page in a
paper with no slack, and its histogram panel showed in a picture what one clause of
the caption can say in words. What a caption cannot do is show two windows
overlapping on one pixel, so that is what the figure keeps.

The class each crop assigns is COMPUTED from its own threshold rather than written in
by hand. A first draft asserted the two labels and got them the wrong way round,
which is a small instance of this paper's subject: a value stated rather than derived.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

INK, BLUE, RED, GREY = "#16222E", "#2E6E8E", "#B4261E", "#7A8A98"

rng = np.random.default_rng(7)
H = W = 64
yy, xx = np.mgrid[0:H, 0:W]
field = np.clip(-26 + 10 * (xx / W) + 6 * (yy / H) + rng.normal(0, 0.6, (H, W)),
                -27, -13)

CROP = 40
boxes = [(0, 0), (H - CROP, W - CROP)]
names, cols = ["A", "B"], [BLUE, RED]
th = [float(np.percentile(field[r:r + CROP, c:c + CROP], 45)) for r, c in boxes]

lo, hi = min(th), max(th)
inside = np.ones((H, W), bool)
for r, c in boxes:
    m = np.zeros((H, W), bool)
    m[r:r + CROP, c:c + CROP] = True
    inside &= m
ys, xs = np.where(inside & (field > lo) & (field < hi))
py, px = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
val = float(field[py, px])
says = ["water" if val <= t else "land" for t in th]
assert says[0] != says[1], "the chosen pixel does not actually disagree"

fig = plt.figure(figsize=(3.32, 1.60))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.30], wspace=0.10,
                      left=0.015, right=0.985, top=0.99, bottom=0.02)

ax = fig.add_subplot(gs[0])
ax.imshow(field, cmap="Greys_r", vmin=-27, vmax=-13, interpolation="nearest")
for (r, c), col, nm in zip(boxes, cols, names):
    ax.add_patch(Rectangle((c - .5, r - .5), CROP, CROP, fill=False, lw=1.5,
                           edgecolor=col))
    ax.text(c + 2, r + 5 if r == 0 else r + CROP - 3, nm, color=col, fontsize=7.5,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.7, alpha=0.85))
ax.plot(px, py, "o", ms=6.5, mfc="none", mec="#000", mew=1.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim(-.5, W - .5); ax.set_ylim(H - .5, -.5)

ax = fig.add_subplot(gs[1])
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for i, (nm, col, sy, face) in enumerate(
        zip(names, cols, says, ("#DCEEF4", "#F9E7E4"))):
    ax.add_patch(Rectangle((0.02, 0.78 - i * 0.21), 0.96, 0.18, facecolor=face,
                           edgecolor=col, lw=1.1))
    ax.text(0.5, 0.87 - i * 0.21, "crop {} says {}".format(nm, sy), ha="center",
            va="center", fontsize=7.6, color=col, fontweight="bold")
ax.text(0.5, 0.485, "one pixel, two labels", ha="center", fontsize=7.3, color=GREY)
ax.text(0.5, 0.285,
        "$\\kappa$ = P(match the crop read)\n$-$ P(match another crop)",
        ha="center", va="center", fontsize=7.3, color=INK)
ax.text(0.5, 0.045, "exactly 0 if the model ignores the crop",
        ha="center", va="center", fontsize=6.9, color=INK)

fig.savefig("fig0_mechanism.png", dpi=460)
print("wrote fig0_mechanism.png: A says {}, B says {}; pixel {:.2f} dB between "
      "thresholds {:.2f} and {:.2f}".format(says[0], says[1], val, th[0], th[1]))
