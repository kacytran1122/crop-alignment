"""The two sea-ice figures retained by the submitted paper.

Fig 2  crop alignment against its control. The null is structural, so the figure
       has a real zero line rather than an estimated one.
Fig 3  recovering the flood generator preprocessing. The bias crosses zero at a
       9x9 focal mean, which is how we know we found the kernel and not just a
       number that fits.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
OUT = W / "figures" / "v2"
OUT.mkdir(parents=True, exist_ok=True)
INK = "#1a1a1a"
RED = "#c0392b"
BLUE = "#2c6fbb"
GREY = "#9aa0a6"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def fig2():
    f = W / "runs" / "kappa_summary.json"
    if not f.exists():
        f = W / "results" / "kappa_summary.json"
    if not f.exists():
        print("  fig2 skipped: no kappa summary")
        return
    d = json.loads(f.read_text())
    kn = np.array(d["kappa_noisy"]); kc = np.array(d["kappa_clean"])
    n = len(kn)
    # Placed at \linewidth = 236pt = 3.28in, so a 7.4in canvas is shrunk by 0.44 on
    # the page and the sizes below arrived at 2.9 to 4.4pt against 9.96pt body text.
    # SC is the reciprocal of that shrink, so the numbers below are page points.
    FIGW = 7.4
    SC = FIGW / 3.278
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(FIGW, FIGW * 0.375),
                                  gridspec_kw={"width_ratios": [2.0, 1.0]})
    x = np.arange(n)
    ax.axhline(0, color=INK, lw=1.4)
    top = max(kn.max(), kc.max()) * 1.12
    # the in-plot null annotation is gone: at this aspect it ran through the control
    # points, and the caption already says the zero line is the exact null rather
    # than an estimated baseline. Two places to state one fact is one too many.
    ax.set_ylim(-0.012, top)
    for i in x:
        ax.plot([i, i], [kc[i], kn[i]], color=GREY, lw=0.9, zorder=1)
    # no legend: at this width its box spans the axes whatever loc is asked for, and
    # it sat on the data. The caption carries the colour key instead.
    ax.scatter(x, kn, s=26, color=RED, zorder=3)
    ax.scatter(x, kc, s=26, color=BLUE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([a[4:8] for a in d["acqs"]], rotation=90,
                       fontsize=5.4 * SC)
    ax.set_xlabel("held-out acquisition", fontsize=6.6 * SC)
    ax.set_ylabel("crop alignment  $\\kappa$", fontsize=6.6 * SC)
    ax.tick_params(axis="y", labelsize=6.2 * SC)
    ax.set_title("predicts the crop's own label",
                 fontsize=6.4 * SC, loc="left", pad=8)

    diff = kn - kc
    ax2.axhline(0, color=INK, lw=1.4)
    parts = ax2.violinplot([kn, kc, diff], showmeans=True, widths=0.75)
    for b, c in zip(parts["bodies"], (RED, BLUE, "#4c9a2a")):
        b.set_facecolor(c); b.set_alpha(0.45)
    for k in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[k].set_color(INK); parts[k].set_linewidth(1.0)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["noisy", "ctrl", "diff"], fontsize=6.2 * SC)
    ax2.tick_params(axis="y", labelsize=6.2 * SC)
    ax2.set_ylabel("$\\kappa$", fontsize=6.6 * SC)
    se = diff.std(ddof=1) / np.sqrt(n)
    # the t is gone from the figure: the folds are not independent replicates, and a
    # statistic printed on a figure travels without the caveat that qualifies it
    ax2.set_title("{:+.4f},  {}/{} folds".format(
        diff.mean(), int((diff > 0).sum()), n),
        fontsize=6.4 * SC, loc="left", pad=8)
    fig.savefig(OUT / "fig2_crop_alignment.png")
    plt.close(fig)
    print("fig2_crop_alignment.png")


def fig3():
    f = W / "data" / "sen1floods11" / "vh_recovery_smooth.json"
    if not f.exists():
        print("  fig3 skipped: no recovery sweep")
        return
    tab = json.loads(f.read_text())
    radii = sorted(int(k) for k in tab)
    bias, mad = [], []
    for r in radii:
        d = np.array([v[2] for v in tab[str(r)].values()])
        bias.append(d.mean()); mad.append(np.abs(d).mean())
    ks = [2 * r + 1 for r in radii]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0),
                                  gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax.axhline(0, color=INK, lw=1.2)
    ax.plot(ks, bias, "o-", color=RED, lw=1.6, ms=5, label="mean bias")
    ax.plot(ks, mad, "s--", color=BLUE, lw=1.3, ms=4, label="mean |difference|")
    best = ks[int(np.argmin(mad))]
    ax.axvline(best, color="#4c9a2a", lw=1.2, ls=":")
    ax.text(best, max(mad) * 0.92, "  {}x{}".format(best, best), fontsize=8,
            color="#4c9a2a")
    ax.set_xlabel("focal-mean kernel applied before thresholding")
    ax.set_ylabel("recovered $-$ published  (dB)")
    ax.legend(frameon=False, fontsize=7.5)
    # titles removed: they collided on the page and the caption says this


    r = radii[int(np.argmin(mad))]
    ev = sorted(tab[str(r)])
    pub = np.array([tab[str(r)][e][0] for e in ev])
    rec = np.array([tab[str(r)][e][1] for e in ev])
    lim = [min(pub.min(), rec.min()) - 0.6, max(pub.max(), rec.max()) + 0.6]
    ax2.plot(lim, lim, color=GREY, lw=1.0, ls="--")
    ax2.scatter(pub, rec, s=30, color=RED, zorder=3)
    # point labels removed: they overprinted between -22 and -21.5 dB

    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("published VH threshold (dB)")
    ax2.set_ylabel("recovered (dB)")

    fig.savefig(OUT / "fig3_generator_recovery.png")
    plt.close(fig)
    print("fig3_generator_recovery.png")


if __name__ == "__main__":
    fig2(); fig3()
    print("-> " + str(OUT))
