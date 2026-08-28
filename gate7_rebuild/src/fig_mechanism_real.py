"""The artefact, on real imagery, produced by the published labeller itself.

Figure 1 is a schematic. A reviewer is entitled to ask whether the thing it draws
actually happens, so this draws the same picture from a Sentinel-2 scene using
`shadow_cloud` from the published survey: the same function, in the same two modes,
that produced the numbers in the paper.

Five panels over one region:

  1. the source imagery, with two overlapping crops marked
  2. crop A's labels, as the labeller produces them inside crop A
  3. crop B's labels, from the same rule applied inside crop B
  4. the artefact set: source pixels the two crops label differently
  5. the same region labelled once per scene, where the artefact set is empty

Nothing is trained and nothing is fitted by us. The only choice is which region to
show, and `--scan` reports the artefact fraction for many candidates so that choice
is made in the open rather than by hunting for a flattering one.

    python fig_mechanism_real.py --scan
    python fig_mechanism_real.py --scene 0 --row 7000 --col 7000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gate8_crossdomain" / "src"))

import cv2  # noqa: E402
import rasterio  # noqa: E402
from s2_published_survey import scene_context, shadow_cloud  # noqa: E402

TIF_DIR = REPO / "S2_tiff" / "S2_tiff"
OUT = REPO / "paper" / "figures"
CROP = 128
SHIFT = 32          # how far crop B is offset from crop A
REGION = CROP + SHIFT


def read_region(tif, r0, c0, size):
    """One RGB window, in the band order the labeller expects."""
    with rasterio.open(tif) as src:
        w = rasterio.windows.Window(c0, r0, size, size)
        a = src.read(window=w)
    return np.transpose(a, (1, 2, 0)).astype(np.uint8)


def classes(free):
    """The labeller returns a colour image; collapse it to class indices.

    water = 0 (black), thin ice = 1 (grey 155), ice = 2 (white). Anything else is
    the untouched background, which is class 3 and is not scored.
    """
    g = free[:, :, 0]
    out = np.full(g.shape, 3, np.uint8)
    out[g == 0] = 0
    out[g == 155] = 1
    out[g == 255] = 2
    return out


def _ctx_for(ctx, r, c):
    """The scene context as one crop sees it.

    The background estimate is an image, so the scene-fitted arm hands each crop the
    matching window of the scene's background, exactly as the survey does. The
    threshold and the two normalisation ranges are scalars and pass through. Getting
    this wrong is the difference between a scene-fitted arm and a crash.
    """
    if ctx is None:
        return None
    out = {k: ctx[k] for k in ("otsu", "n1", "n2")}
    out["bg"] = ctx["bg"][r:r + CROP, c:c + CROP]
    return out


def two_crops(region, ctx=None, scene=()):
    """Label the region's two overlapping crops, each inside its own window."""
    a = region[0:CROP, 0:CROP]
    b = region[SHIFT:SHIFT + CROP, SHIFT:SHIFT + CROP]
    la = classes(shadow_cloud(a, _ctx_for(ctx, 0, 0), scene))
    lb = classes(shadow_cloud(b, _ctx_for(ctx, SHIFT, SHIFT), scene))
    return la, lb


def overlap_disagreement(la, lb):
    """Where the two crops label the same source pixel differently.

    The overlap is the last CROP-SHIFT rows and columns of crop A and the first of
    crop B, which is the only place both crops have an opinion.
    """
    k = CROP - SHIFT
    oa = la[SHIFT:, SHIFT:]
    ob = lb[:k, :k]
    return oa, ob, oa != ob


def artefact_fraction(region, ctx=None, scene=()):
    la, lb = two_crops(region, ctx, scene)
    _, _, dis = overlap_disagreement(la, lb)
    return float(dis.mean()), la, lb, dis


def scan(n_scenes, n_sites, seed=0):
    """Artefact fraction at many sites, reported before any figure is drawn."""
    rng = np.random.default_rng(seed)
    tifs = sorted(TIF_DIR.glob("*.tif"))[:n_scenes]
    rows = []
    for t in tifs:
        with rasterio.open(t) as src:
            H, W = src.height, src.width
        for _ in range(n_sites):
            r0 = int(rng.integers(0, H - REGION))
            c0 = int(rng.integers(0, W - REGION))
            reg = read_region(t, r0, c0, REGION)
            if reg.max() == 0:
                continue                      # off the swath
            f, _, _, _ = artefact_fraction(reg)
            rows.append((f, t.name, r0, c0))
    rows.sort(reverse=True)
    print("  {} sites scanned over {} scenes".format(len(rows), len(tifs)))
    if rows:
        fr = np.array([r[0] for r in rows])
        print("  artefact fraction on the overlap: median {:.2%}  mean {:.2%}  "
              "max {:.2%}".format(np.median(fr), fr.mean(), fr.max()))
        print("  {} of {} sites have a nonzero artefact set".format(
            int((fr > 0).sum()), len(fr)))
        print("  top sites:")
        for f, name, r0, c0 in rows[:8]:
            print("    {:6.2%}  {}  row {} col {}".format(f, name, r0, c0))
    return rows


def draw(tif, r0, c0, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    region = read_region(tif, r0, c0, REGION)
    f_crop, la, lb, dis = artefact_fraction(region)
    ctx = scene_context(region)
    f_scene, sa, sb, dis_s = artefact_fraction(
        region, ctx, ("bg", "otsu", "n1", "n2"))
    print("  per-crop labelling : {:.2%} of the overlap disagrees".format(f_crop))
    print("  once per scene     : {:.2%}".format(f_scene))

    # water black, thin ice mid grey, ice white, background a flat blue-grey so it
    # cannot be confused with a class
    cmap = matplotlib.colors.ListedColormap(
        ["#101418", "#8a8f96", "#f2f4f6", "#4a5a72"])

    fig, ax = plt.subplots(1, 5, figsize=(7.4, 1.72))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.6); sp.set_color("#333333")

    # A fixed gain blew the ice out to flat white, which hid the texture the crops
    # are disagreeing about. Stretch between percentiles of this region instead, and
    # say so: it is a display choice and changes no number on the page.
    lo, hi = np.percentile(region, (2, 98))
    disp = np.clip((region.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1)),
                   0, 255).astype(np.uint8)
    ax[0].imshow(disp)
    ax[0].add_patch(Rectangle((0, 0), CROP, CROP, fill=False, lw=1.3,
                              edgecolor="#1f6fb4"))
    ax[0].add_patch(Rectangle((SHIFT, SHIFT), CROP, CROP, fill=False, lw=1.3,
                              edgecolor="#c0392b"))
    ax[0].set_title("two crops, one scene", fontsize=6.4, loc="left", pad=3)

    ax[1].imshow(la, cmap=cmap, vmin=0, vmax=3)
    ax[1].set_title("crop A's labels", fontsize=6.4, loc="left", pad=3,
                    color="#1f6fb4")
    ax[2].imshow(lb, cmap=cmap, vmin=0, vmax=3)
    ax[2].set_title("crop B's labels", fontsize=6.4, loc="left", pad=3,
                    color="#c0392b")

    k = CROP - SHIFT
    over = np.clip(disp[SHIFT:SHIFT + k, SHIFT:SHIFT + k].astype(np.float32) * 0.72
                   + 18, 0, 255).astype(np.uint8)
    ax[3].imshow(over)
    m = np.zeros(dis.shape + (4,), np.float32)
    m[dis] = (1.0, 0.35, 0.0, 0.95)
    ax[3].imshow(m)
    ax[3].set_title("$\\mathcal{{A}}$: {:.1%} of the overlap".format(f_crop),
                    fontsize=6.4, loc="left", pad=3)

    ax[4].imshow(over)
    m2 = np.zeros(dis_s.shape + (4,), np.float32)
    m2[dis_s] = (1.0, 0.35, 0.0, 0.95)
    ax[4].imshow(m2)
    ax[4].set_title("labelled once per scene: {:.1%}".format(f_scene),
                    fontsize=6.4, loc="left", pad=3)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.86, bottom=0.02, wspace=0.05)
    fig.savefig(out_path, dpi=300)
    print("  wrote {}".format(out_path))
    return f_crop, f_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    # Defaults ARE the published invocation. They used to be different numbers, so
    # running the script bare produced a survey that did not match the caption and
    # nothing said so. The figure's site is also the default, for the same reason.
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--sites", type=int, default=10)
    ap.add_argument("--scene", type=int, default=4)
    ap.add_argument("--row", type=int, default=3985)
    ap.add_argument("--col", type=int, default=7906)
    ap.add_argument("--survey-out", default=str(
        REPO / "gate7_rebuild" / "results" / "mechanism_survey.json"))
    args = ap.parse_args()

    if not TIF_DIR.exists():
        print("no imagery at {}; nothing to draw".format(TIF_DIR))
        return 1
    if args.scan:
        rows = scan(args.scenes, args.sites)
        fr = [r[0] for r in rows]
        out = {
            "invocation": "fig_mechanism_real.py --scan --scenes {} --sites {}".format(
                args.scenes, args.sites),
            "seed": 0,
            "n_sites": len(rows),
            "n_nonempty": int(sum(1 for f in fr if f > 0)),
            "median": float(np.median(fr)) if fr else 0.0,
            "mean": float(np.mean(fr)) if fr else 0.0,
            "max": float(max(fr)) if fr else 0.0,
            "figure_site": {"scene": 4, "row": 3985, "col": 7906},
        }
        Path(args.survey_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.survey_out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("  wrote {}".format(args.survey_out))
        return 0

    tifs = sorted(TIF_DIR.glob("*.tif"))
    tif = tifs[args.scene]
    OUT.mkdir(parents=True, exist_ok=True)
    draw(tif, args.row, args.col, OUT / "fig5_mechanism_real.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
