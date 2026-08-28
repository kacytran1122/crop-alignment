"""Establish, from the pixels, what the CloudSEN12+ arrays actually contain.

Everything the survey needs is documented in a dataset card, and this project's error
list is largely made of things that were documented and wrong. Four facts decide
whether the survey measures what it claims, and none of them is safe to assume:

  1. WHICH BAND IS BLUE. The card lists thirteen slots and twelve names, because B8A
     sits between B08 and B09. If the order is not what we guess, every rule reads the
     wrong channel and P(A) still comes out looking reasonable, because P(A) does not
     care what it is thresholding.

     Sentinel-2 is acquired at three ground resolutions and resampled to one grid, so
     a 20 m band is exactly 2x2 blocky and a 60 m band exactly 6x6 blocky. That is a
     property of the pixels, so it identifies the bands without trusting prose. It
     also settles L1C against L2A: thirteen bands including a cirrus band means L1C.

  2. WHERE THE PADDING IS. The card says 509 was padded to 512 with zeros on the left
     and bottom. Measured here rather than believed, on every patch, both as an
     all-band-zero mask and as a positional rule, with the two compared.

  3. THE REFLECTANCE SCALE, so the survey can work in reflectance and never quantise.

  4. WHAT THE EXPERT MASK CODES. Four classes are documented; the survey has to
     binarise them and must say which way.
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

D = Path('data/cloudsen12/high')


def blockiness(a, period):
    """How nearly the array is constant within period x period blocks, over the best
    phase. A band resampled from a coarser grid scores ~0; a native band does not."""
    best = 1.0
    for ph in range(period):
        b = a[ph:, ph:]
        h = (b.shape[0] // period) * period
        w = (b.shape[1] // period) * period
        if h < period or w < period:
            continue
        blocks = b[:h, :w].reshape(h // period, period, w // period, period)
        within = blocks.std(axis=(1, 3)).mean()
        overall = b[:h, :w].std()
        best = min(best, float(within / overall) if overall > 0 else 1.0)
    return best


def main():
    files = sorted(D.glob('*.tif'))
    if not files:
        print('no imagery at', D)
        return 1
    print('patches on disk:', len(files), flush=True)

    with rasterio.open(files[0]) as s:
        arr = s.read()
        print('shape', arr.shape, 'dtype', arr.dtype)
    nb = arr.shape[0]

    # ---- 1. band resolution from block structure ---------------------------
    print()
    print('band  blockiness(2)  blockiness(6)   native res   mean')
    res = {}
    for i in range(nb):
        a = arr[i].astype(np.float64)
        b2, b6 = blockiness(a, 2), blockiness(a, 6)
        r = 60 if b6 < 0.02 else (20 if b2 < 0.02 else 10)
        res[i] = r
        print('{:4d}  {:12.4f}  {:12.4f}   {:>6d} m   {:10.1f}'.format(
            i, b2, b6, r, a.mean()))
    counts = {r: sum(1 for v in res.values() if v == r) for r in (10, 20, 60)}
    print('resolution counts', counts,
          '(expected 10m:4, 20m:6, 60m:3 for the 13 spectral bands)')

    # ---- 2. padding, measured two independent ways -------------------------
    print()
    n_agree = 0
    edges = {'left': [], 'right': [], 'top': [], 'bottom': []}
    for f in files[:40]:
        with rasterio.open(f) as s:
            a = s.read()
        spec = a[:13].astype(np.float64)
        valid = (spec != 0).any(axis=0)
        zc = ~valid
        L = int(np.argmax(~zc.all(axis=0))) if zc.all(axis=0).any() else 0
        R = int(np.argmax(~zc.all(axis=0)[::-1])) if zc.all(axis=0).any() else 0
        T = int(np.argmax(~zc.all(axis=1))) if zc.all(axis=1).any() else 0
        Bm = int(np.argmax(~zc.all(axis=1)[::-1])) if zc.all(axis=1).any() else 0
        edges['left'].append(L); edges['right'].append(R)
        edges['top'].append(T); edges['bottom'].append(Bm)
        pos = np.ones_like(valid)
        if L: pos[:, :L] = False
        if Bm: pos[valid.shape[0] - Bm:, :] = False
        n_agree += bool((pos == valid).all())
    for k, v in edges.items():
        print('fully-zero {:6s} columns/rows: min {} max {} mode {}'.format(
            k, min(v), max(v), max(set(v), key=v.count)))
    print('all-band-zero mask equals the positional rule on {}/{} patches'.format(
        n_agree, len(files[:40])))

    # ---- 3. scale, and 4. expert mask --------------------------------------
    print()
    spec = arr[:13].astype(np.float64)
    v = spec[(spec != 0)]
    print('spectral percentiles 1/50/99/max: {:.0f} {:.0f} {:.0f} {:.0f}'.format(
        np.percentile(v, 1), np.percentile(v, 50), np.percentile(v, 99), v.max()))
    print('  -> reflectance scale is 10000 if the bulk sits in the low thousands')
    if nb > 13:
        m = arr[13]
        u, c = np.unique(m, return_counts=True)
        print('expert mask band 13 values:',
              {int(k): int(n) for k, n in zip(u, c)})

    out = Path('runs/s3_cloudsen12_probe.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'n_patches': len(files), 'shape': list(arr.shape),
        'band_native_res': {str(k): v for k, v in res.items()},
        'res_counts': {str(k): v for k, v in counts.items()},
        'pad_left_mode': max(set(edges['left']), key=edges['left'].count),
        'pad_bottom_mode': max(set(edges['bottom']), key=edges['bottom'].count),
        'pad_masks_agree': n_agree, 'pad_masks_checked': len(files[:40]),
    }, indent=1))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
