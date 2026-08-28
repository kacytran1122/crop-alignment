"""Cache Sen1Floods11 for GPU-resident training.

446 chips of 512x512, two Sentinel-1 bands, and two label rasters per chip: an
expert one and an algorithmically generated one. The whole thing is under 1 GB in
float16, so it goes on the card once and never touches a DataLoader, which is what
made the sea-ice runs 20x faster.

Two details that matter for correctness rather than speed.

Sentinel-1 chips carry NaN where the swath does not cover the chip. Those pixels
must become an explicit ignore label, not a filled value: the sea-ice audit found
that silently labelling nodata as a class was one of the larger defects in the work
being audited, and repeating it here would be indefensible.

The expert labels use -1 for unlabelled, and the unlabelled region is not the same
as the nodata region. A pixel is scored only where BOTH label sets are defined and
the imagery is valid, so the two arms are compared on identical pixels. This is the
matched-pixel-set discipline the sea-ice mosaic evaluator had to be rewritten to
enforce; here it is built in from the start.
"""
import json
from collections import defaultdict
from pathlib import Path

import os
_ROOT = Path(os.environ.get("SEAICE_ROOT",
                            Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio

D = (_ROOT / "data/sen1floods11")
OUT = D / "cache"
IGNORE = 255


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name.replace("_S1Hand.tif", "")
                   for p in (D / "S1Hand").glob("*.tif"))
    keep = []
    for s in stems:
        if (D / "LabelHand" / (s + "_LabelHand.tif")).exists() and \
           (D / "S1OtsuLabelHand" / (s + "_S1OtsuLabelHand.tif")).exists():
            keep.append(s)
    n = len(keep)
    print("{} chips with all three rasters".format(n))

    img = np.zeros((n, 512, 512, 2), np.float32)
    y_ref = np.full((n, 512, 512), IGNORE, np.uint8)
    y_alg = np.full((n, 512, 512), IGNORE, np.uint8)
    events, stats = [], defaultdict(lambda: [0, 0, 0])

    for i, s in enumerate(keep):
        with rasterio.open(D / "S1Hand" / (s + "_S1Hand.tif")) as f:
            a = f.read().astype(np.float32)                    # (2, H, W) VV, VH
        with rasterio.open(D / "LabelHand" / (s + "_LabelHand.tif")) as f:
            lr = f.read(1)
        with rasterio.open(D / "S1OtsuLabelHand" / (s + "_S1OtsuLabelHand.tif")) as f:
            la = f.read(1)

        finite = np.isfinite(a).all(axis=0)
        # score only where the imagery is valid AND both label sets are defined,
        # so the two arms are never compared on different pixels
        m = finite & (lr >= 0) & (la >= 0)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        img[i] = np.transpose(a, (1, 2, 0))
        y_ref[i][m] = lr[m].astype(np.uint8)
        y_alg[i][m] = la[m].astype(np.uint8)

        ev = s.split("_")[0]
        events.append(ev)
        st = stats[ev]
        st[0] += 1
        st[1] += int(m.sum())
        st[2] += int((lr[m] == 1).sum())

    ev_arr = np.array(events)
    np.save(OUT / "images.npy", img.astype(np.float16))
    np.save(OUT / "y_ref.npy", y_ref)
    np.save(OUT / "y_alg.npy", y_alg)
    np.save(OUT / "events.npy", ev_arr)

    scored = (y_ref != IGNORE)
    agree = (y_ref == y_alg) & scored
    print("\n{:12s} {:>6s} {:>14s} {:>9s} {:>10s}".format(
        "event", "chips", "scored px", "water%", "algo agree"))
    for ev in sorted(stats):
        k = ev_arr == ev
        sc = scored[k].sum()
        ag = agree[k].sum() / max(sc, 1)
        c, px, w = stats[ev]
        print("{:12s} {:6d} {:14,d} {:8.2f}% {:10.4f}".format(
            ev, c, px, 100.0 * w / max(px, 1), ag))

    tot_sc = int(scored.sum())
    print("\n  {:,} scored pixels of {:,} ({:.1%})".format(
        tot_sc, scored.size, tot_sc / scored.size))
    print("  overall expert/algorithmic label agreement: {:.4f}".format(
        agree.sum() / max(tot_sc, 1)))
    print("  water prevalence, expert {:.2%}   algorithmic {:.2%}".format(
        (y_ref[scored] == 1).mean(), (y_alg[scored] == 1).mean()))

    vv = img[..., 0][scored]
    vh = img[..., 1][scored]
    meta = {"n": n, "events": sorted(stats),
            "mean": [float(vv.mean()), float(vh.mean())],
            "std": [float(vv.std()), float(vh.std())],
            "agreement": float(agree.sum() / max(tot_sc, 1))}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("\n  VV {:.2f} +- {:.2f}   VH {:.2f} +- {:.2f}".format(
        meta["mean"][0], meta["std"][0], meta["mean"][1], meta["std"][1]))
    print("  wrote {}".format(OUT))


if __name__ == "__main__":
    main()
