"""A generator whose crop-dependence we set by hand, on real imagery.

Sea ice is the defect in the wild: the pipeline really did recompute labels from
the pixels inside each crop, and kappa is positive there. What sea ice
cannot supply is a check that kappa measures what it is claimed to measure, because
the true crop-dependence of that pipeline is not known -- it is the thing being
estimated. A structural null proves kappa is 0 when there is nothing to find. It
says nothing about whether kappa is RIGHT when there is.

So this builds the crop-fitted endpoint used by a generator with a dial.
Sen1Floods11 publishes one Otsu-on-VH threshold per flood event, and we recovered
its preprocessing earlier: a 9x9 focal mean. Keeping that smoothing fixed at the
chip level and recomputing only the OTSU THRESHOLD inside each crop reproduces the
relevant sea-ice mechanism -- the generator reading the same pixels the model
reads -- with one scalar per crop as the entire source of crop-dependence.

The dial mixes the two thresholds:

    th_c(alpha) = (1 - alpha) * th_event  +  alpha * th_crop

At alpha = 0 the labels use the published event threshold and are crop-invariant.
At alpha = 1 the generator is predominantly crop-fitted, with failed crop fits
falling back to a finite chip threshold where one exists; chips with no usable
threshold are excluded consistently downstream.

To be explicit about what this is and is not: Sen1Floods11 as published uses one
threshold per event, so the artefact is NOT present in that benchmark and this
experiment does not claim to have found it there. The imagery and the generator are
real; the crop-dependence is ours, which is the whole point of a calibration.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
C = W / "data" / "sen1floods11" / "cache"
OUT = C / "crop"
IGNORE = 255
CROP, STRIDE, RADIUS = 128, 32, 4          # 9x9 focal mean = radius 4
STARTS = list(range(0, 512 - CROP + 1, STRIDE))   # 13 positions per axis


def otsu(v):
    """Otsu threshold on a 1-D float array, 256 bins. Returns nan if degenerate."""
    if v.size < 64:
        return np.nan
    lo, hi = float(v.min()), float(v.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.nan
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    w = np.cumsum(hist)
    tot = w[-1]
    if tot <= 0:
        return np.nan
    centres = (edges[:-1] + edges[1:]) / 2.0
    csum = np.cumsum(hist * centres)
    mu_t = csum[-1] / tot
    w0 = w / tot
    # guard the empty-class ends so the variance is defined
    ok = (w0 > 1e-6) & (w0 < 1 - 1e-6)
    if not ok.any():
        return np.nan
    mu0 = np.divide(csum, w, out=np.zeros_like(csum), where=w > 0)
    between = w0 * (1 - w0) * (mu0 - (mu_t - w0 * mu0) / np.maximum(1 - w0, 1e-9)) ** 2
    between = np.where(ok, between, -np.inf)
    return float(centres[int(np.argmax(between))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    imgs = np.load(C / "images.npy", mmap_mode="r")
    y_ref = np.load(C / "y_ref.npy")
    n = imgs.shape[0]
    ns = len(STARTS)
    print("{} chips, crop {} stride {} -> {}x{} = {} crops per chip".format(
        n, CROP, STRIDE, ns, ns, ns * ns))

    vh_s = np.zeros((n, 512, 512), np.float32)
    th_chip = np.full(n, np.nan, np.float64)
    th_crop = np.full((n, ns, ns), np.nan, np.float64)

    for i in range(n):
        vh = np.asarray(imgs[i][..., 1], np.float32)
        # smoothing stays at the chip level: the only thing that varies by crop is
        # the threshold, so the mechanism has exactly one degree of freedom
        s = uniform_filter(vh, size=2 * RADIUS + 1, mode="nearest")
        vh_s[i] = s
        valid = y_ref[i] != IGNORE
        if valid.sum() >= 64:
            th_chip[i] = otsu(s[valid])
        for a, r0 in enumerate(STARTS):
            for b, c0 in enumerate(STARTS):
                m = valid[r0:r0 + CROP, c0:c0 + CROP]
                if m.sum() >= 64:
                    th_crop[i, a, b] = otsu(s[r0:r0 + CROP, c0:c0 + CROP][m])
        if (i + 1) % 100 == 0:
            print("  {}/{}".format(i + 1, n), flush=True)

    # a crop with no usable histogram falls back to its chip threshold, which is
    # the conservative choice: it removes crop-dependence rather than inventing it
    fill = np.broadcast_to(th_chip[:, None, None], th_crop.shape)
    nbad = int(np.isnan(th_crop).sum())
    th_crop = np.where(np.isnan(th_crop), fill, th_crop)

    np.save(OUT / "vh_smooth.npy", vh_s.astype(np.float16))
    np.save(OUT / "th_chip.npy", th_chip)
    np.save(OUT / "th_crop.npy", th_crop)
    (OUT / "meta.json").write_text(json.dumps(
        {"crop": CROP, "stride": STRIDE, "radius": RADIUS, "starts": STARTS,
         "n_chips": int(n), "n_crops_per_chip": int(ns * ns),
         "n_crop_threshold_fallbacks": nbad}, indent=1))

    sp = th_crop.reshape(n, -1)
    rng = np.nanmax(sp, axis=1) - np.nanmin(sp, axis=1)
    print("\nchip threshold   mean {:+.2f} dB   sd {:.2f}".format(
        np.nanmean(th_chip), np.nanstd(th_chip)))
    print("within-chip crop threshold spread (max - min)")
    print("  mean {:.2f} dB   median {:.2f}   p90 {:.2f}   max {:.2f}".format(
        np.nanmean(rng), np.nanmedian(rng), np.nanpercentile(rng, 90),
        np.nanmax(rng)))
    print("  crops falling back to the chip threshold: {} of {}".format(
        nbad, th_crop.size))
    print("\nA spread of this size is what the dial has to work with: at alpha=1")
    print("two crops covering the same pixel can disagree by that many dB of")
    print("threshold, and every pixel between them flips class depending on which")
    print("crop the label was computed in.")
    print("wrote " + str(OUT))


if __name__ == "__main__":
    main()
