"""Generate vector paper figures that depend only on committed source/results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

INK = "#24303b"
BLUE = "#dcecff"
BLUE_EDGE = "#3b78b5"
GREEN = "#dff2e3"
GREEN_EDGE = "#4f9660"
ORANGE = "#ffe9cd"
ORANGE_EDGE = "#c37b21"
RED = "#f8dcdc"
RED_EDGE = "#b85252"
GREY = "#eef1f4"


def box(ax, xy, width, height, text, face, edge, fontsize=8.5, lw=1.15):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, label=None, bend=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.1,
        color=INK,
        connectionstyle=f"arc3,rad={bend}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    if label:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=7.3,
            color=INK,
        )


def architecture():
    fig, ax = plt.subplots(figsize=(11.2, 5.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Two restrained panels keep the architecture and the post-training audit
    # visually distinct while preserving a single full-width WACV figure.
    for y, h in ((0.425, 0.555), (0.025, 0.36)):
        ax.add_patch(
            FancyBboxPatch(
                (0.012, y),
                0.976,
                h,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor="#fbfcfd",
                edgecolor="#c8d0d7",
                linewidth=0.85,
            )
        )

    ax.text(0.027, 0.944, "(a)  Multimodal segmentation model", fontsize=11.2, weight="bold", color=INK)
    ax.text(0.027, 0.866, "IMAGE BRANCH", fontsize=7.4, weight="bold", color=BLUE_EDGE)
    ax.text(0.027, 0.598, "PHOTON BRANCH", fontsize=7.4, weight="bold", color=GREEN_EDGE)

    # Sentinel-2 crop: a small image-like tile is more immediate than a text box.
    tile_x, tile_y, tile_w, tile_h = 0.044, 0.682, 0.074, 0.142
    ax.add_patch(Rectangle((tile_x - 0.010, tile_y + 0.010), tile_w, tile_h, facecolor="#f3f6f8", edgecolor="#aebac4", lw=0.7))
    tile_colors = np.array(
        [
            ["#83a7bb", "#6f9fa9", "#d6e6e7", "#cfddd5", "#7d9ca7", "#aac6c5"],
            ["#7b9dab", "#426c7d", "#9fc0c3", "#e2ece8", "#a9c4bd", "#7eaaa6"],
            ["#d2e1dc", "#9fbbb4", "#577f89", "#789ba1", "#dce8e4", "#bed0c8"],
            ["#dbe7e1", "#c4d6d0", "#84a6a2", "#557d86", "#85a6a8", "#d6e3de"],
            ["#9fbab7", "#d4e2dc", "#c1d2cd", "#71959a", "#4b7280", "#92b0b1"],
            ["#759ba2", "#aec5c1", "#e1eae5", "#b6cbc5", "#7b9fa3", "#537884"],
        ]
    )
    for row in range(6):
        for col in range(6):
            ax.add_patch(
                Rectangle(
                    (tile_x + col * tile_w / 6, tile_y + (5 - row) * tile_h / 6),
                    tile_w / 6,
                    tile_h / 6,
                    facecolor=tile_colors[row, col],
                    edgecolor="none",
                )
            )
    ax.add_patch(Rectangle((tile_x, tile_y), tile_w, tile_h, facecolor="none", edgecolor=BLUE_EDGE, lw=1.2))
    ax.text(tile_x + tile_w / 2, tile_y - 0.016, "Sentinel-2 RGB · 128×128×3", ha="center", va="top", fontsize=6.8, color=INK)

    # Compact encoder-decoder schematic with explicit skip connections.
    ux = [0.167, 0.215, 0.263, 0.311, 0.359, 0.407, 0.455]
    uy = [0.755, 0.715, 0.675, 0.645, 0.675, 0.715, 0.755]
    uh = [0.105, 0.125, 0.145, 0.165, 0.145, 0.125, 0.105]
    for i, (x, y, h) in enumerate(zip(ux, uy, uh)):
        color = "#cfe3fa" if i < 3 else ("#b9d5f2" if i == 3 else "#dcecff")
        ax.add_patch(Rectangle((x, y - h / 2), 0.025, h, facecolor=color, edgecolor=BLUE_EDGE, lw=1.0))
        if i < len(ux) - 1:
            arrow(ax, (x + 0.026, y), (ux[i + 1] - 0.002, uy[i + 1]))
    for left, right, height in ((0, 6, 0.838), (1, 5, 0.817), (2, 4, 0.797)):
        left_x = ux[left] + 0.0125
        right_x = ux[right] + 0.0125
        left_y = uy[left] + uh[left] / 2
        right_y = uy[right] + uh[right] / 2
        ax.plot([left_x, left_x, right_x], [left_y, height, height], color="#7299bf", lw=0.8)
        arrow(ax, (right_x, height), (right_x, right_y))
    ax.text(0.318, 0.858, "ResNet-18 U-Net · ImageNet init.", ha="center", va="bottom", fontsize=8.6, weight="bold", color=INK)
    arrow(ax, (tile_x + tile_w, tile_y + tile_h / 2), (ux[0] - 0.004, uy[0]))
    box(ax, (0.493, 0.695), 0.095, 0.122, "image map\n128×128×16", BLUE, BLUE_EDGE, fontsize=7.6)
    arrow(ax, (ux[-1] + 0.026, uy[-1]), (0.493, 0.756))

    # Ten scalar photon features pass through the actual 10→96→16→16 MLP.
    for row in range(5):
        for col in range(2):
            ax.add_patch(Circle((0.061 + col * 0.018, 0.535 - row * 0.021), 0.0052, facecolor=GREEN, edgecolor=GREEN_EDGE, lw=0.75))
    ax.text(0.070, 0.558, "10 ICESat-2 features", ha="center", va="bottom", fontsize=7.1, color=INK)
    arrow(ax, (0.092, 0.493), (0.142, 0.493))
    mlp_x = [0.146, 0.205, 0.264, 0.323]
    mlp_labels = ["10", "96", "16", "16"]
    mlp_heights = [0.105, 0.148, 0.126, 0.126]
    for i, (x, label, height) in enumerate(zip(mlp_x, mlp_labels, mlp_heights)):
        ax.add_patch(Rectangle((x, 0.493 - height / 2), 0.029, height, facecolor=GREEN, edgecolor=GREEN_EDGE, lw=1.0))
        ax.text(x + 0.0145, 0.493, label, ha="center", va="center", fontsize=7.2, weight="bold", color=INK)
        if i < len(mlp_x) - 1:
            arrow(ax, (x + 0.030, 0.493), (mlp_x[i + 1] - 0.002, 0.493))
    ax.text(0.252, 0.566, "MLP · ELU + dropout", ha="center", va="bottom", fontsize=8.2, weight="bold", color=INK)
    box(ax, (0.383, 0.432), 0.096, 0.122, "broadcast\n128×128×16", GREEN, GREEN_EDGE, fontsize=7.5)
    arrow(ax, (mlp_x[-1] + 0.030, 0.493), (0.383, 0.493))

    # Fusion and segmentation head.
    box(ax, (0.626, 0.623), 0.087, 0.132, "concatenate\n32 channels", ORANGE, ORANGE_EDGE, fontsize=7.5)
    arrow(ax, (0.588, 0.756), (0.626, 0.705), bend=0.08)
    arrow(ax, (0.479, 0.493), (0.626, 0.656), bend=-0.08)
    box(ax, (0.746, 0.623), 0.079, 0.132, "squeeze–\nexcitation", ORANGE, ORANGE_EDGE, fontsize=7.5)
    arrow(ax, (0.713, 0.689), (0.746, 0.689))
    box(ax, (0.858, 0.607), 0.070, 0.164, "3×3 conv\nBN + ReLU\ndropout\n1×1 conv", RED, RED_EDGE, fontsize=7.0)
    arrow(ax, (0.825, 0.689), (0.858, 0.689))

    mask_x, mask_y, mask_w, mask_h = 0.951, 0.626, 0.031, 0.126
    mask_colors = [["#93bad3", "#93bad3", "#f0cf8b"], ["#93bad3", "#d47a72", "#f0cf8b"], ["#d47a72", "#d47a72", "#93bad3"]]
    for row in range(3):
        for col in range(3):
            ax.add_patch(Rectangle((mask_x + col * mask_w / 3, mask_y + (2 - row) * mask_h / 3), mask_w / 3, mask_h / 3, facecolor=mask_colors[row][col], edgecolor="white", lw=0.35))
    ax.add_patch(Rectangle((mask_x, mask_y), mask_w, mask_h, facecolor="none", edgecolor=RED_EDGE, lw=1.0))
    arrow(ax, (0.928, 0.689), (0.949, 0.689))
    ax.text(mask_x + mask_w / 2, 0.596, "3-class mask", ha="center", va="top", fontsize=7.2, color=INK)

    # Post-training audit: the same physical pixel is observed through overlapping crops.
    ax.text(0.027, 0.348, "(b)  Crop-alignment audit", fontsize=11.2, weight="bold", color=INK)
    ax.text(0.983, 0.349, "shared weights · no retraining", ha="right", fontsize=7.3, color="#596874")

    scene_x, scene_y, scene_w, scene_h = 0.045, 0.090, 0.150, 0.192
    ax.add_patch(Rectangle((scene_x, scene_y), scene_w, scene_h, facecolor="#f2f4f5", edgecolor="#aeb8c1", lw=0.8))
    for k in range(1, 6):
        ax.plot([scene_x + scene_w * k / 6] * 2, [scene_y, scene_y + scene_h], color="white", lw=0.65)
        ax.plot([scene_x, scene_x + scene_w], [scene_y + scene_h * k / 6] * 2, color="white", lw=0.65)
    ax.add_patch(Rectangle((scene_x + 0.011, scene_y + 0.052), 0.098, 0.112, facecolor=BLUE, edgecolor=BLUE_EDGE, lw=1.5, alpha=0.68))
    ax.add_patch(Rectangle((scene_x + 0.042, scene_y + 0.025), 0.096, 0.118, facecolor=ORANGE, edgecolor=ORANGE_EDGE, lw=1.5, alpha=0.58))
    pixel = (scene_x + 0.079, scene_y + 0.097)
    ax.add_patch(Circle(pixel, 0.009, facecolor=RED_EDGE, edgecolor="white", lw=1.0, zorder=6))
    ax.text(pixel[0] + 0.012, pixel[1] + 0.012, "$p$", fontsize=8.3, weight="bold", color=RED_EDGE)
    ax.text(scene_x + scene_w / 2, 0.058, "overlapping crops $c,c'$", ha="center", va="top", fontsize=7.6, color=INK)

    box(ax, (0.247, 0.119), 0.111, 0.125, "same trained\nmodel  $f$", BLUE, BLUE_EDGE, fontsize=8.0)
    arrow(ax, (scene_x + scene_w, 0.186), (0.247, 0.186))
    box(ax, (0.409, 0.119), 0.125, 0.125, "prediction at $p$\n" r"$f(p\mid c)$", BLUE, BLUE_EDGE, fontsize=8.0)
    arrow(ax, (0.358, 0.181), (0.409, 0.181))

    box(ax, (0.590, 0.210), 0.130, 0.087, "own label  $L_c(p)$", GREEN, GREEN_EDGE, fontsize=7.8)
    box(ax, (0.590, 0.073), 0.130, 0.087, "other labels  $L_{c'}(p)$", ORANGE, ORANGE_EDGE, fontsize=7.6)
    arrow(ax, (0.534, 0.191), (0.590, 0.253), bend=-0.08)
    arrow(ax, (0.534, 0.168), (0.590, 0.117), bend=0.08)
    box(
        ax,
        (0.774, 0.087),
        0.190,
        0.184,
        "$\\kappa = \\mathbb{E}[\\,\\mathrm{own}-\\mathrm{other}\\,]$\n\n$\\kappa=0$ if predictions are\ncrop-invariant",
        RED,
        RED_EDGE,
        fontsize=8.2,
        lw=1.25,
    )
    arrow(ax, (0.720, 0.253), (0.774, 0.215), bend=0.03)
    arrow(ax, (0.720, 0.117), (0.774, 0.143), bend=-0.03)

    fig.savefig(OUT / "crop_alignment_architecture.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT / "crop_alignment_architecture.png", dpi=240, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def committed_crop_alignment():
    path = ROOT / "experiments" / "sea_ice" / "results" / "kappa_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    noisy = np.asarray(data["kappa_noisy"], dtype=float)
    clean = np.asarray(data["kappa_clean"], dtype=float)
    order = np.argsort(noisy - clean)
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(x, noisy[order], "o-", color=BLUE_EDGE, lw=1.5, ms=4.3, label="crop-noisy training")
    ax.plot(x, clean[order], "s-", color=GREEN_EDGE, lw=1.25, ms=3.8, label="crop-invariant control")
    ax.fill_between(x, clean[order], noisy[order], color="#dce6ef", alpha=0.55)
    ax.set_xlabel("held-out acquisition (ordered by paired difference)")
    ax.set_ylabel("crop alignment κ")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(x))], fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax.text(
        0.99,
        0.05,
        f"machine-readable folds: n={len(x)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#5e6871",
    )
    fig.tight_layout()
    fig.savefig(OUT / "crop_alignment_results.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def required_sample_size():
    """Redraw the power figure from values stated in the committed manuscript/logs."""
    labels = [
        "mosaic, draw · 12 ep",
        "mosaic, vote · 12 ep",
        "mosaic, vote · 60 ep",
        "patch premium · 12 ep",
        "mosaic, vote · 120 ep",
    ]
    required_n = np.asarray([542, 197, 48, 8, 7], dtype=float)
    effects = ["+0.0109", "+0.0182", "+0.0522", "+0.0488", "+0.0796"]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    colors = [BLUE_EDGE, "#8e9ba6", GREEN_EDGE, BLUE_EDGE, GREEN_EDGE]
    ax.barh(y, required_n, color=colors, height=0.63, alpha=0.88)
    ax.set_xscale("log")
    ax.axvline(17, color=INK, ls=":", lw=1.3, label="available n=17")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("independent acquisitions required for 80% power")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for yi, n, effect in zip(y, required_n, effects):
        ax.text(n * 1.12, yi, f"n={int(n):,}  ({effect})", va="center", fontsize=7.5, color=INK)
    ax.legend(frameon=False, fontsize=7.8, loc="lower right")
    ax.grid(axis="x", which="both", color="#d7dde2", lw=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "required_sample_size.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    required_sample_size()
    committed_crop_alignment()
    print(f"wrote figures to {OUT}")
