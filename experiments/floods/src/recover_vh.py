"""Recover the flood generator constant, and check it against the published value.

In the sea-ice audit a two-parameter grid search recovered the label generator's
water constant, t_water = 30, in 17 of 17 folds. That is persuasive only up to a
point, because the constant was recovered from labels we had regenerated
ourselves: the check closes on itself.

Sen1Floods11 lets us close it from outside. The dataset ships the algorithmic
labels (S1OtsuLabelHand), the imagery they were derived from (S1Hand), AND the
per-event threshold the authors actually used, printed in the metadata. So we can
run the same recovery against a number we did not produce.

If a single scalar on the VH band reproduces the shipped algorithmic labels, then
that label set is a one-parameter function of the model's own input, and the
closed-loop critique transfers to a benchmark that IBM and NASA currently use to
evaluate the Prithvi geospatial foundation models.

Reported per event, since the generator refits per event; the spread of the
recovered constants is itself the point.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio

D = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"
META = D / "Sen1Floods11_Metadata.geojson"
NODATA = -1          # LabelHand uses -1 for unlabelled
VH_BAND = 2          # S1Hand is VV, VH


def published():
    g = json.loads(META.read_text())
    out = {}
    for f in g.get("features", []):
        p = f.get("properties", {})
        loc = p.get("location", p.get("Location"))
        for k in p:
            if "thresh" in k.lower() and "vh" in k.lower():
                out[str(loc).lower()] = float(p[k])
    return out


def chips():
    by_event = defaultdict(list)
    for f in sorted((D / "S1OtsuLabelHand").glob("*.tif")):
        stem = f.name.replace("_S1OtsuLabelHand.tif", "")
        ev = stem.split("_")[0].lower()
        img = D / "S1Hand" / (stem + "_S1Hand.tif")
        hand = D / "LabelHand" / (stem + "_LabelHand.tif")
        if img.exists():
            by_event[ev].append((img, f, hand if hand.exists() else None))
    return by_event


def load_pair(img_p, lab_p):
    with rasterio.open(img_p) as s:
        vh = s.read(VH_BAND).astype(np.float32)
    with rasterio.open(lab_p) as s:
        lab = s.read(1)
    m = (lab != NODATA) & np.isfinite(vh)
    return vh[m], lab[m]


def recover(vh, lab, lo=-30.0, hi=-14.0, step=0.01):
    """The threshold reproducing the shipped labels best, by pixel agreement."""
    ts = np.arange(lo, hi + step, step)
    best, bt = -1.0, None
    order = np.argsort(vh)
    v, y = vh[order], lab[order]
    水 = np.cumsum(y == 1)                      # cumulative water below each cut
    tot_w = int((y == 1).sum())
    n = len(v)
    idx = np.searchsorted(v, ts)
    for t, i in zip(ts, idx):
        tp = 水[i - 1] if i > 0 else 0          # labelled water and below t
        fp = i - tp
        fn = tot_w - tp
        agree = (n - fp - fn) / n
        if agree > best:
            best, bt = agree, t
    return bt, best


def main():
    pub = published()
    ev = chips()
    print("{:12s} {:>6s} {:>10s} {:>10s} {:>8s} {:>9s}".format(
        "event", "chips", "published", "recovered", "diff", "agree"))
    rows = []
    for name in sorted(ev):
        items = ev[name]
        vhs, labs = [], []
        for img_p, lab_p, _ in items:
            a, b = load_pair(img_p, lab_p)
            if len(a):
                # subsample to keep the sort cheap; the threshold is a scalar
                k = min(len(a), 200_000)
                sel = np.random.RandomState(0).choice(len(a), k, replace=False)
                vhs.append(a[sel]); labs.append(b[sel])
        if not vhs:
            continue
        vh = np.concatenate(vhs); lab = np.concatenate(labs)
        t, agree = recover(vh, lab)
        p = pub.get(name)
        d = (t - p) if p is not None else float("nan")
        rows.append((name, len(items), p, t, d, agree))
        print("{:12s} {:6d} {:10.2f} {:10.2f} {:+8.2f} {:9.4f}".format(
            name, len(items), p if p is not None else float("nan"), t, d, agree))

    good = [r for r in rows if r[2] is not None]
    if good:
        diffs = np.array([r[4] for r in good])
        ag = np.array([r[5] for r in good])
        within = int((np.abs(diffs) <= 0.10).sum())
        print("\n  recovered within 0.10 dB of the published value: "
              "{}/{} events".format(within, len(good)))
        print("  mean |difference| {:.3f} dB   mean pixel agreement {:.4f}".format(
            float(np.abs(diffs).mean()), float(ag.mean())))
        pubv = np.array([r[2] for r in good])
        print("  spread of the generator constant across events: "
              "{:.2f} dB  ({:.2f} to {:.2f})".format(
                  pubv.max() - pubv.min(), pubv.min(), pubv.max()))
        print("\n  A single global threshold cannot express a constant that moves "
              "{:.2f} dB\n  between events. A model that sees the whole chip can "
              "infer which event\n  it is in and adapt. That is the sea-ice "
              "mechanism, in a second domain.".format(pubv.max() - pubv.min()))

    (D / "vh_recovery.json").write_text(json.dumps(
        [{"event": r[0], "n_chips": r[1], "published": r[2],
          "recovered": r[3], "diff": r[4], "agreement": r[5]} for r in rows],
        indent=1))


if __name__ == "__main__":
    main()
