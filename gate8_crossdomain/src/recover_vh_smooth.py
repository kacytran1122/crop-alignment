"""Recover the generator constant AND its preprocessing.

Recovering the threshold from the raw VH band reproduces the shipped labels at
94.9% pixel agreement, but the recovered constant sits 1.19 dB below the published
one on all ten events with a published value. A bias of one sign on every event is
a signature, not noise.

The published procedure says the threshold is applied "across a focal mean smoothed
VH band". Smoothing pulls isolated low values up toward their neighbourhood, so to
capture the same pixels in the RAW band the recovered cut must sit lower -- which
is the sign and roughly the size observed.

So sweep the smoothing radius. If the offset collapses at some radius, we have
recovered not just the generator's constant but its preprocessing, from nothing but
the shipped rasters. That is a considerably stronger statement than agreement
alone: it means the label is reconstructible from the model's own input.
"""
import json
from collections import defaultdict
from pathlib import Path

import os
_ROOT = Path(os.environ.get("SEAICE_ROOT",
                            Path(__file__).resolve().parents[1]))

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

D = (_ROOT / "data/sen1floods11")
NODATA, VH_BAND = -1, 2
RADII = [0, 1, 2, 3, 4, 5, 7]


def published():
    g = json.loads((D / "Sen1Floods11_Metadata.geojson").read_text())
    out = {}
    for f in g.get("features", []):
        p = f.get("properties", {})
        for k in p:
            if "thresh" in k.lower() and "vh" in k.lower():
                out[str(p.get("location", p.get("Location"))).lower()] = float(p[k])
    return out


def events():
    by = defaultdict(list)
    for f in sorted((D / "S1OtsuLabelHand").glob("*.tif")):
        stem = f.name.replace("_S1OtsuLabelHand.tif", "")
        img = D / "S1Hand" / (stem + "_S1Hand.tif")
        if img.exists():
            by[stem.split("_")[0].lower()].append((img, f))
    return by


def best_threshold(vh, lab):
    order = np.argsort(vh)
    v, y = vh[order], (lab[order] == 1)
    cw = np.cumsum(y)
    tot_w, n = int(y.sum()), len(v)
    fp = np.arange(n) - cw
    fn = tot_w - cw
    agree = (n - fp - fn) / n
    i = int(np.argmax(agree))
    return float(v[i]), float(agree[i])


def main():
    from recover_vh import published_for, assert_all_published
    pub, ev = published(), events()
    assert_all_published(pub, ev)
    print("{:8s} {:>8s} {:>10s} {:>9s} {:>9s}".format(
        "radius", "kernel", "mean|diff|", "bias", "agreement"))
    best = None
    table = {}
    for r in RADII:
        diffs, ags = [], []
        per_event = {}
        for name, items in sorted(ev.items()):
            vhs, labs = [], []
            for img_p, lab_p in items:
                with rasterio.open(img_p) as s:
                    vh = s.read(VH_BAND).astype(np.float32)
                with rasterio.open(lab_p) as s:
                    lab = s.read(1)
                if r > 0:
                    vh = uniform_filter(vh, size=2 * r + 1, mode="nearest")
                m = (lab != NODATA) & np.isfinite(vh)
                if m.sum() == 0:
                    continue
                a, b = vh[m], lab[m]
                k = min(len(a), 150_000)
                sel = np.random.RandomState(0).choice(len(a), k, replace=False)
                vhs.append(a[sel]); labs.append(b[sel])
            if not vhs:
                continue
            t, ag = best_threshold(np.concatenate(vhs), np.concatenate(labs))
            per_event[name] = (published_for(pub, name), t, t - published_for(pub, name), ag)
            diffs.append(t - published_for(pub, name)); ags.append(ag)
        diffs, ags = np.array(diffs), np.array(ags)
        table[r] = per_event
        print("{:8d} {:8s} {:10.3f} {:+9.3f} {:9.4f}".format(
            r, "{}x{}".format(2 * r + 1, 2 * r + 1),
            float(np.abs(diffs).mean()), float(diffs.mean()), float(ags.mean())))
        if best is None or np.abs(diffs).mean() < best[1]:
            best = (r, float(np.abs(diffs).mean()), float(ags.mean()))

    r, md, ag = best
    print("\n  best radius {} ({}x{} focal mean): mean |difference| {:.3f} dB, "
          "agreement {:.4f}".format(r, 2 * r + 1, 2 * r + 1, md, ag))
    print("\n  per event at the best radius:")
    print("  {:12s} {:>10s} {:>10s} {:>8s} {:>9s}".format(
        "event", "published", "recovered", "diff", "agree"))
    within = 0
    for name, (p, t, d, a) in sorted(table[r].items()):
        within += abs(d) <= 0.25
        print("  {:12s} {:10.2f} {:10.2f} {:+8.2f} {:9.4f}".format(name, p, t, d, a))
    print("\n  within 0.25 dB of the published constant: {}/{} events".format(
        within, len(table[r])))
    (D / "vh_recovery_smooth.json").write_text(json.dumps(
        {str(k): {n: list(v) for n, v in t.items()} for k, t in table.items()},
        indent=1))


if __name__ == "__main__":
    main()
