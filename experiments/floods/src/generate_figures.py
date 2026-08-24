"""The two figures the reframed paper needs and does not have.

Fig 4  the calibration. A statistic that is positive on closed-loop labels can be
       positive by accident; one that reproduces a known ordering across five
       settings of a dial, per event, is measuring the thing that was varied. The
       figure has to show the per-event lines, not just the mean, because the
       per-event ordering is the evidence and an averaged curve hides it.

Fig 5  the cost. kappa on one axis, mIoU against human truth on the other, with
       event means removed so the between-event spread cannot manufacture the
       slope. This is the figure that answers "so what", and it is the one a
       reader will remember.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = Path(__file__).resolve().parents[1]
OUT = W / "figures" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
INK, RED, BLUE, GREY = "#1a1a1a", "#c0392b", "#2c6fbb", "#9aa0a6"
GREEN = "#4c9a2a"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})


def fig4():
    d = json.loads((W / "runs_s1fc" / "summary.json").read_text())
    alphas = [float(a) for a in d["alphas"]]
    events = d["events"]
    K = np.array([d["kappa"][str(a)] for a in d["alphas"]])      # (n_alpha, n_event)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.3),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    for j, e in enumerate(events):
        ax.plot(alphas, K[:, j], "-", color=GREY, lw=0.9, alpha=0.85, zorder=1)
        ax.plot(alphas, K[:, j], "o", color=GREY, ms=2.6, zorder=1)
    ax.plot(alphas, K.mean(1), "-o", color=RED, lw=2.0, ms=5.5, zorder=3,
            label="mean over 11 events")
    ax.axhline(0, color=INK, lw=1.2)
    ax.set_xlabel("$\\alpha$   (0 = one threshold per chip,  1 = one per crop)")
    ax.set_ylabel("crop alignment  $\\kappa$")
    ax.set_xticks(alphas)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("$\\kappa$ tracks a generator's crop-dependence",
                 fontsize=10, loc="left", pad=8)
    ax.text(0.02, K.mean(1)[0] - 0.028,
            "control: not 0, because a CNN\nreading crops is not crop-invariant",
            fontsize=6.8, color=INK, va="top")

    # per-event rank agreement with the known ordering
    rho = []
    for j in range(K.shape[1]):
        ra = np.argsort(np.argsort(alphas))
        rv = np.argsort(np.argsort(K[:, j]))
        rho.append(np.corrcoef(ra, rv)[0, 1])
    rho = np.array(rho)
    order = np.argsort(rho)
    ax2.barh(np.arange(len(events)), rho[order], color=BLUE, alpha=0.85, height=0.62)
    ax2.axvline(1.0, color=GREEN, lw=1.4, ls="--")
    ax2.set_yticks(np.arange(len(events)))
    ax2.set_yticklabels([events[i][:10] for i in order], fontsize=7)
    ax2.set_xlim(0.55, 1.03)
    ax2.set_xlabel("rank correlation of $\\kappa$ with $\\alpha$")
    ax2.set_title("perfect in {}/{} events    ($p=0.008$ each, by chance)".format(
        int((rho > 0.999).sum()), len(events)), fontsize=8.5, loc="left", pad=8)
    fig.savefig(OUT / "controlled_calibration.png")
    plt.close(fig)
    print("controlled_calibration.png")


def fig5():
    dk = json.loads((W / "runs_s1fc" / "summary.json").read_text())
    dd = json.loads((W / "runs_s1fc" / "damage_summary.json").read_text())
    alphas = dk["alphas"]
    events = [e for e in dk["events"] if e in dd["events"]]
    ik = [dk["events"].index(e) for e in events]
    idd = [dd["events"].index(e) for e in events]
    K = np.array([[dk["kappa"][str(a)][j] for j in ik] for a in alphas])
    M = np.array([[dd["miou_vs_expert"][str(a)][j] for j in idd] for a in alphas])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.3),
                                  gridspec_kw={"width_ratios": [1.0, 1.15],
                                               "wspace": 0.42})

    ax.plot([float(a) for a in alphas], M.mean(1), "-o", color=RED, lw=2.0, ms=5.5)
    for j in range(M.shape[1]):
        ax.plot([float(a) for a in alphas], M[:, j], "-", color=GREY, lw=0.8,
                alpha=0.8, zorder=1)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("mIoU against expert labels")
    ax.set_xticks([float(a) for a in alphas])
    ax.set_title("and it costs accuracy against humans",
                 fontsize=10, loc="left", pad=8)
    ax.text(0.5, M.mean(1)[0] + 0.005,
            "{:+.4f} mIoU from $\\alpha$=0 to $\\alpha$=1\n$t$ = {:.2f}, {} of {} events worse".format(
                dd["cost_mean"], dd["cost_t"], 9, len(events)),
            fontsize=7.4, color=INK, ha="center", va="bottom")

    # within-event: remove each event's mean so between-event spread cannot drive it
    kc = (K - K.mean(0)).ravel()
    mc = (M - M.mean(0)).ravel()
    ax2.axhline(0, color=GREY, lw=0.8)
    ax2.axvline(0, color=GREY, lw=0.8)
    ax2.scatter(kc, mc, s=22, color=BLUE, alpha=0.8, zorder=3)
    A = np.column_stack([np.ones(len(kc)), kc])
    beta, *_ = np.linalg.lstsq(A, mc, rcond=None)
    xs = np.linspace(kc.min(), kc.max(), 20)
    ax2.plot(xs, beta[0] + beta[1] * xs, color=RED, lw=1.8)
    r = float(np.corrcoef(kc, mc)[0, 1])
    ax2.set_xlabel("$\\kappa$, event mean removed")
    ax2.set_ylabel("mIoU vs expert (centred)")
    ax2.set_title("$r$ = {:+.3f},   slope {:+.3f} mIoU per unit $\\kappa$".format(
        r, beta[1]), fontsize=8.5, loc="left", pad=8)
    fig.text(0.005, -0.04,
             "Each event is its own block, so the correlation is within-event and "
             "cannot come from events differing in difficulty.",
             fontsize=7, color=GREY)
    fig.savefig(OUT / "shortcut_cost.png")
    plt.close(fig)
    print("shortcut_cost.png")


if __name__ == "__main__":
    fig4(); fig5()
    print("-> " + str(OUT))
