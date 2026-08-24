"""The three figures that carry the rebuilt paper.

Fig 1  benchmark resolution as a required sample size. The point is not that the
       spread is large, it is that every effect reported on this benchmark needs
       more acquisitions than anyone used -- including ours.
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

W = Path(__file__).resolve().parents[1]
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


def fig1():
    """Required n at 80% power, ours and theirs, on a log axis.

    Every number here is read from power.json rather than typed in. An earlier
    version carried 238, 65, 38 and 74,922 as literals; the calculation was later
    rerun on the completed fold set and they became 197, 70, 41 and 80,215, and the
    figure went on rendering the old ones without complaint.
    """
    pw = json.loads((W / "runs" / "power.json").read_text())
    # the audited work's effects, sized against the same paired spread
    s_paired = pw["mosaic_vote"]["s"]
    need = lambda d: int(np.ceil((2.80 * s_paired / d) ** 2))
    kap = json.loads((W / "runs" / "kappa_summary.json").read_text())
    kd = np.array(kap["kappa_noisy"]) - np.array(kap["kappa_clean"])

    # the converged-budget premium, read from the mosaic results rather than
    # typed in, for the reason in this function's docstring
    def budget_premium(prefix):
        arms = {"scene": {}, "original": {}}
        for f in (W / "runs" / "mosaic").glob(prefix + "_*.json"):
            r = json.loads(f.read_text())
            if r["seed"] == 42 and r["labels"] in arms:
                arms[r["labels"]][r["acq"]] = r["advantage_nanmean"]
        acq = sorted(set(arms["scene"]) & set(arms["original"]))
        v = np.array([arms["original"][a] - arms["scene"][a] for a in acq])
        return v.mean(), int(np.ceil((2.80 * v.std(ddof=1) / v.mean()) ** 2))

    d120, n120 = budget_premium("ep120")
    items = [
        ("test-set leak\n0.0009", need(0.0009), RED, "audited"),
        ("our premium, mosaic,\n12 epochs\n{:+.4f}".format(pw["mosaic_vote"]["d"]),
         int(pw["mosaic_vote"]["n_power80"]), BLUE, "ours"),
        ("our premium, mosaic,\n120 epochs\n{:+.4f}".format(d120), n120, BLUE, "ours"),
        ("fusion gain, unpublished\n+0.0306", need(0.0306), RED, "audited"),
        ("fusion-strategy spread\n0.0401", need(0.0401), RED, "audited"),
        ("our premium,\npatch scoring\n{:+.4f}".format(pw["patch"]["d"]),
         int(pw["patch"]["n_power80"]), BLUE, "ours"),
        # the text declines to quote this: the formula returns under 2, and a
        # paired test cannot estimate a variance from one fold. Draw the bound.
        ("crop alignment\n{:+.4f}".format(kd.mean()), 2, BLUE, "ours"),
    ]
    items.sort(key=lambda it: -it[1])
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    y = np.arange(len(items))
    for i, (lab, n, c, _) in enumerate(items):
        ax.barh(i, n, color=c, alpha=0.85, height=0.6)
        lab = "< 2" if (n == 2 and "crop alignment" in items[i][0]) else "{:,}".format(n)
        ax.text(n * 1.25, i, lab, va="center", fontsize=8.5, color=c)
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 3e5)
    ax.set_xlabel("acquisitions required to detect the effect at 80% power")
    for v, lab, st in ((2, "reported from 2", "-"),
                       (17, "we have 17", "--")):
        ax.axvline(v, color=INK, ls=st, lw=1.1, alpha=0.75)
        ax.text(v, len(items) - 0.42, "  " + lab, fontsize=7.5, color=INK,
                rotation=90, va="top", ha="left", zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                          alpha=0.85))
    ax.set_title("Every effect the audited work reports needs more acquisitions "
                 "than it used",
                 fontsize=10, loc="left", pad=10)
    # the label carries newlines and a trailing effect size; keep the name only
    # footnote removed: 4pt grey is unreadable in print and the caption says it
    fig.savefig(OUT / "required_sample_size.png")
    plt.close(fig)
    print("required_sample_size.png")


def fig2():
    f = W / "runs" / "kappa_summary.json"
    if not f.exists():
        print("  fig2 skipped: no kappa summary")
        return
    d = json.loads(f.read_text())
    kn = np.array(d["kappa_noisy"]); kc = np.array(d["kappa_clean"])
    n = len(kn)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                                  gridspec_kw={"width_ratios": [2.0, 1.0]})
    x = np.arange(n)
    ax.axhline(0, color=INK, lw=1.4)
    top = max(kn.max(), kc.max()) * 1.12
    ax.set_ylim(-0.024, top)
    ax.text(n - 0.4, -0.019, "structural null: 0 for any crop-invariant predictor",
            fontsize=7.2, ha="right", color=INK)
    for i in x:
        ax.plot([i, i], [kc[i], kn[i]], color=GREY, lw=0.9, zorder=1)
    ax.scatter(x, kn, s=26, color=RED, zorder=3,
               label="trained on crop-noisy labels")
    ax.scatter(x, kc, s=26, color=BLUE, zorder=3,
               label="trained on repaired labels (control)")
    ax.set_xticks(x)
    ax.set_xticklabels([a[4:8] for a in d["acqs"]], rotation=90, fontsize=6.5)
    ax.set_xlabel("held-out acquisition")
    ax.set_ylabel("crop alignment  $\\kappa$")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title("The model predicts the label of the crop it is reading",
                 fontsize=10, loc="left", pad=8)

    diff = kn - kc
    ax2.axhline(0, color=INK, lw=1.4)
    parts = ax2.violinplot([kn, kc, diff], showmeans=True, widths=0.75)
    for b, c in zip(parts["bodies"], (RED, BLUE, "#4c9a2a")):
        b.set_facecolor(c); b.set_alpha(0.45)
    for k in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[k].set_color(INK); parts[k].set_linewidth(1.0)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["noisy", "control", "difference"], fontsize=8)
    ax2.set_ylabel("$\\kappa$")
    se = diff.std(ddof=1) / np.sqrt(n)
    ax2.set_title("difference {:+.4f},  t = {:.1f},  {}/{} folds".format(
        diff.mean(), diff.mean() / se, int((diff > 0).sum()), n),
        fontsize=8.5, loc="left", pad=8)
    fig.savefig(OUT / "crop_alignment_results.png")
    plt.close(fig)
    print("crop_alignment_results.png")


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

    fig.savefig(OUT / "label_generator_recovery.png")
    plt.close(fig)
    print("label_generator_recovery.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("-> " + str(OUT))
