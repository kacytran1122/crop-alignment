r"""S1: which labelling practices are crop-safe, measured without training anything.

The paper's claim narrowed once we read the published segmentation code. Labelling
per crop is not the hazard: the released colour thresholds are fixed constants, and a
fixed threshold is a per-pixel function, so running it inside a window returns the
label it would return on the whole scene. What creates the artefact is a rule with
something FITTED to the crop -- a threshold, a normalisation, a background estimate.

That claim is about labelling practices, not about our benchmark, so it should be
testable on someone else's imagery with no model involved. This does that. For each
chip it lays a crop grid, runs each labelling rule separately inside every crop, and
asks a question with no learning in it at all:

    of the source pixels covered by two or more crops, on what fraction do the
    covering crops DISAGREE about the label?

That fraction is P(A), the artefact set of the paper's definition. It is a property
of the labeller and the grid alone. A model trained on labels with P(A) = 0 cannot
have a nonzero kappa, by the proposition; a labeller with P(A) large is one where a
model can be rewarded for reading the window.

Every rule here is one somebody actually uses on SAR water mapping. The point is not
that Otsu is bad -- it is a good threshold -- but that WHERE you compute it decides
whether the labels are a property of the ground or of the framing.

Imagery: Sen1Floods11 S1Hand, 446 chips over 11 flood events, publicly released.
Nothing is trained, nothing is downloaded, and the whole survey runs on CPU.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

W = Path(os.environ.get("SEAICE_ROOT", Path(__file__).resolve().parents[2]))
D = W / "data" / "sen1floods11"
NODATA = -9999.0


# ---------------------------------------------------------------- labellers --
def otsu(v):
    """Otsu's threshold on a 1-D array, 256 bins. NaN if degenerate."""
    v = v[np.isfinite(v)]
    if v.size < 32:
        return np.nan
    lo, hi = np.percentile(v, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    hist, edges = np.histogram(np.clip(v, lo, hi), bins=256, range=(lo, hi))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2
    mu = np.cumsum(p * mids)
    mu_t = mu[-1]
    denom = w0 * (1 - w0)
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (mu_t * w0 - mu) ** 2 / np.where(denom > 0, denom, np.nan)
    k = int(np.nanargmax(between))
    return float(mids[k])


def kmeans2(v, iters=25):
    """Two-cluster 1-D k-means; threshold is the midpoint of the two centres."""
    v = v[np.isfinite(v)]
    if v.size < 32:
        return np.nan
    a, b = np.percentile(v, [20, 80])
    if b <= a:
        return np.nan
    for _ in range(iters):
        m = (a + b) / 2
        lo, hi = v[v <= m], v[v > m]
        if lo.size == 0 or hi.size == 0:
            break
        a2, b2 = lo.mean(), hi.mean()
        if abs(a2 - a) < 1e-6 and abs(b2 - b) < 1e-6:
            a, b = a2, b2
            break
        a, b = a2, b2
    return float((a + b) / 2)


# Each rule maps (crop pixels, whole-chip pixels) -> a per-crop threshold in dB.
# "fitted" says whether the rule reads the crop it is labelling.
RULES = {
    "fixed global constant":      (lambda c, s: -21.8, False),
    "fixed, per scene (Otsu)":    (lambda c, s: otsu(s), False),
    "per-crop Otsu":              (lambda c, s: otsu(c), True),
    "per-crop k-means":           (lambda c, s: kmeans2(c), True),
    "per-crop 45th percentile":   (lambda c, s: float(np.nanpercentile(c, 45))
                                   if np.isfinite(c).sum() > 32 else np.nan, True),
    "per-crop min-max, then fixed": (
        lambda c, s: (float(np.nanmin(c) + 0.45 * (np.nanmax(c) - np.nanmin(c)))
                      if np.isfinite(c).sum() > 32 else np.nan), True),
}


def survey_chip(vh, crop, stride):
    """P(A) per rule for one chip: the fraction of multiply-covered pixels whose
    covering crops disagree."""
    H, Wd = vh.shape
    starts_r = list(range(0, H - crop + 1, stride))
    starts_c = list(range(0, Wd - crop + 1, stride))
    valid = np.isfinite(vh)
    scene = vh[valid]
    out = {}
    for name, (fn, _) in RULES.items():
        # votes[k] counts crops labelling a pixel class k; cnt counts coverage
        votes = np.zeros((2,) + vh.shape, np.int32)
        for r in starts_r:
            for c in starts_c:
                sub = vh[r:r + crop, c:c + crop]
                m = valid[r:r + crop, c:c + crop]
                if m.sum() < 32:
                    continue
                t = fn(sub[m], scene)
                if not np.isfinite(t):
                    continue
                lab = (sub <= t).astype(np.int8)      # water where backscatter is low
                votes[1, r:r + crop, c:c + crop] += (lab == 1) & m
                votes[0, r:r + crop, c:c + crop] += (lab == 0) & m
        cnt = votes.sum(0)
        multi = cnt >= 2
        if not multi.any():
            out[name] = float("nan")
            continue
        # a pixel is in A when no single class holds every vote it received
        disagree = votes.max(0)[multi] < cnt[multi]
        out[name] = float(disagree.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="0 = every chip")
    ap.add_argument("--out", default="runs/labeller_survey.json")
    args = ap.parse_args()

    import rasterio
    files = sorted((D / "S1Hand").glob("*.tif"))
    if args.limit:
        files = files[:args.limit]
    print("chips: {}   crop {}   stride {}   crops/chip {}".format(
        len(files), args.crop, args.stride,
        ((512 - args.crop) // args.stride + 1) ** 2))

    per_chip = {k: [] for k in RULES}
    for i, f in enumerate(files):
        with rasterio.open(f) as s:
            vh = s.read(2).astype(np.float32)          # band 2 is VH
        vh[vh <= NODATA + 1] = np.nan
        if not np.isfinite(vh).any():
            continue
        r = survey_chip(vh, args.crop, args.stride)
        for k, v in r.items():
            if np.isfinite(v):
                per_chip[k].append(v)
        if (i + 1) % 50 == 0:
            print("  {}/{}".format(i + 1, len(files)), flush=True)

    print()
    print("=" * 74)
    print("WHICH LABELLING PRACTICES ARE CROP-SAFE")
    print("  P(A) = share of multiply-covered pixels whose covering crops disagree")
    print("=" * 74)
    print("{:32s} {:>8s} {:>10s} {:>10s}".format("rule", "in crop?", "mean P(A)", "max P(A)"))
    res = {}
    for k, (_, fitted) in RULES.items():
        v = np.array(per_chip[k])
        if not v.size:
            continue
        res[k] = {"fitted": fitted, "mean": float(v.mean()),
                  "max": float(v.max()), "n_chips": int(v.size)}
        print("{:32s} {:>8s} {:>9.2%} {:>10.2%}".format(
            k, "yes" if fitted else "no", v.mean(), v.max()))
    print()
    print("A labeller with P(A) = 0 has an empty artefact set, so kappa is zero for")
    print("any model by the proposition and there is nothing for a model to learn.")

    p = W / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"crop": args.crop, "stride": args.stride,
                             "n_chips": len(files), "rules": res}, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
