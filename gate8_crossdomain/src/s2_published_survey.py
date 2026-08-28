"""Run the published labeller itself, per crop, on public Sentinel-2 imagery.

The S1 survey asked which labelling practices can produce the artefact, using rules
standard in SAR water mapping but nobody's code in particular. This asks the narrower
and more pointed question: does the artefact appear when the actual published pipeline
is applied the way it is applied, to imagery its authors never touched? Nothing is
trained here either. P(A) is a property of a labeller and a grid.

The pipeline has two stages. The first assigns classes from fixed HSV constants and
reads one pixel at a time, so it is crop-invariant by inspection and must give
P(A) = 0. The second, the shadow and cloud removal that runs before it, fits four
quantities to whatever array it is handed: a median background with a 155-pixel
kernel, an Otsu threshold, and two min-max normalisations. It is handed a 128x128
patch. The kernel is wider than the patch.

The ablation moves those four, one at a time, from the crop to the whole scene,
leaving everything else alone. Whatever the difference is, it is that step.

CHANNEL ORDER. The pipeline is fed by cv2.imread, which returns BGR, and then calls
COLOR_RGB2HSV on it, so its notional red channel receives blue. Stage 1 does not care,
since the value channel is a max over the three. Stage 2 does: it takes a weighted
grey, 0.299/0.587/0.114, so feeding it true RGB changes the background, the Otsu
threshold and every label downstream. We therefore stack bands 2, 3, 4 in that order,
which is what imread would hand it for a true-colour image.

SCALING. Reflectance is scaled by one global constant for every chip and every crop,
so that no per-crop statistic enters before the rule under test. That makes any
P(A) > 0 the rule's rather than our preparation's. It does not make the magnitude
independent of the constant, which is why the clipped fraction is recorded and why
--scale exists: see the sensitivity table in the paper's supplement.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

D = Path('data/sen1floods11')
ICE, TICE, WATER, UNLAB = 0, 1, 2, 255

LO_ICE, HI_ICE = (0, 0, 205), (185, 255, 255)
LO_TICE, HI_TICE = (0, 0, 31), (185, 255, 204)
LO_WATER, HI_WATER = (0, 0, 0), (185, 255, 30)


def to_bgr8(a, scale):
    """Bands 2, 3, 4 as an 8-bit array in the order cv2.imread would produce.

    Returns the array and the fraction of samples the ceiling clipped, which bounds
    how much of the result is a property of `scale` rather than of the imagery.
    """
    x = np.stack([a[1], a[2], a[3]], -1) / scale
    clipped = float((x > 1.0).mean())
    return (np.clip(x, 0, 1) * 255).astype(np.uint8), clipped


def _u8(x):
    """Round-half-up and saturate, which is what cv2's saturate_cast does. Truncating
    instead would make the substituted branch differ from the branch it replaces by
    more than the quantity being substituted."""
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)


def _minmax(a, lo, hi):
    """cv2.normalize(NORM_MINMAX) with the range supplied rather than measured. On a
    constant array cv2 returns zeros, so this does too."""
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return _u8((a.astype(np.float32) - lo) * (255.0 / (hi - lo)))


def classes(bgr):
    """color_segmentation, returning indices instead of colours. Same masks, same
    precedence: ice, then thin ice over it, then water over both."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    out = np.full(bgr.shape[:2], UNLAB, np.uint8)
    out[cv2.inRange(hsv, LO_ICE, HI_ICE) == 255] = ICE
    out[cv2.inRange(hsv, LO_TICE, HI_TICE) == 255] = TICE
    out[cv2.inRange(hsv, LO_WATER, HI_WATER) == 255] = WATER
    return out


def shadow_cloud(bgr, ctx=None, scene=()):
    """shadow_cloud_removal. Any step named in `scene` takes its fitted value from
    ctx, computed on the whole scene, instead of from this array."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    mask_water = cv2.inRange(hsv, LO_WATER, HI_WATER)
    no_water = bgr.copy()
    no_water[mask_water == 255] = [255, 255, 255]
    img = cv2.cvtColor(no_water, cv2.COLOR_RGB2GRAY)

    if 'bg' in scene:
        bg = ctx['bg']
    else:
        bg = cv2.medianBlur(cv2.dilate(img, np.ones((7, 7), np.uint8)), 155)
    diff = 255 - cv2.absdiff(img, bg)

    if 'otsu' in scene:
        _, keep = cv2.threshold(diff, ctx['otsu'], 255, cv2.THRESH_BINARY)
    else:
        _, keep = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    d2 = cv2.bitwise_and(diff, keep)

    if 'n1' in scene:
        norm = _minmax(d2, *ctx['n1'])
    else:
        norm = cv2.normalize(d2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    _, thr = cv2.threshold(norm, 235, 0, cv2.THRESH_TRUNC)
    if 'n2' in scene:
        thr = _minmax(thr, *ctx['n2'])
    else:
        thr = cv2.normalize(thr, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    old_thin = cv2.cvtColor(thr, cv2.COLOR_GRAY2RGB)
    hsv2 = cv2.cvtColor(old_thin, cv2.COLOR_RGB2HSV)
    m_tice = cv2.inRange(hsv2, (0, 0, 0), HI_TICE)
    m_ice = cv2.bitwise_xor(mask_water, cv2.inRange(hsv2, LO_ICE, HI_ICE))
    free = old_thin.copy()
    free[m_ice == 255] = [255, 255, 255]
    free[m_tice == 255] = [155, 155, 155]
    free[mask_water == 255] = [0, 0, 0]
    return free


def scene_context(bgr):
    """The same four quantities, computed once over the entire 512x512 scene."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)
    no_water = bgr.copy()
    no_water[cv2.inRange(hsv, LO_WATER, HI_WATER) == 255] = [255, 255, 255]
    img = cv2.cvtColor(no_water, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(cv2.dilate(img, np.ones((7, 7), np.uint8)), 155)
    diff = 255 - cv2.absdiff(img, bg)
    t, keep = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    d2 = cv2.bitwise_and(diff, keep)
    norm = cv2.normalize(d2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    _, thr = cv2.threshold(norm, 235, 0, cv2.THRESH_TRUNC)
    return {'bg': bg, 'otsu': float(t),
            'n1': (float(d2.min()), float(d2.max())),
            'n2': (float(thr.min()), float(thr.max()))}


# name -> (needs stage 2, quantities taken from the scene rather than the crop)
ARMS = {
    'stage 1 only, fixed HSV constants': (False, ()),
    'full pipeline, fitted per crop': (True, ()),
    'scene background': (True, ('bg',)),
    'scene Otsu': (True, ('otsu',)),
    'scene first normalisation': (True, ('n1',)),
    'scene second normalisation': (True, ('n2',)),
    'scene background and Otsu': (True, ('bg', 'otsu')),
    'everything fitted per scene': (True, ('bg', 'otsu', 'n1', 'n2')),
}


def survey_chip(bgr, crop, stride):
    H, W = bgr.shape[:2]
    starts = [(r, c)
              for r in range(0, H - crop + 1, stride)
              for c in range(0, W - crop + 1, stride)]
    ctx = scene_context(bgr)
    base = {k: ctx[k] for k in ('otsu', 'n1', 'n2')}
    out, unlab = {}, 0
    for name, (use_s2, scene) in ARMS.items():
        votes = np.zeros((3, H, W), np.int32)
        cov = np.zeros((H, W), np.int32)
        for r, c in starts:
            sub = bgr[r:r + crop, c:c + crop]
            if use_s2:
                cc = dict(base)
                cc['bg'] = ctx['bg'][r:r + crop, c:c + crop]
                sub = shadow_cloud(sub, cc, scene)
            lab = classes(sub)
            unlab += int((lab == UNLAB).sum())
            cov[r:r + crop, c:c + crop] += 1
            for k in (ICE, TICE, WATER):
                votes[k, r:r + crop, c:c + crop] += lab == k
        # A pixel counts only if every covering crop assigned it a class. The three
        # value ranges tile 0-255 and hue never reaches 185, so nothing abstains in
        # practice; `unlab` is returned so that stays a measured fact and not an
        # assumption this restriction quietly depends on.
        sel = (cov >= 2) & (votes.sum(0) == cov)
        out[name] = (float((votes.max(0)[sel] < cov[sel]).mean())
                     if sel.any() else float('nan'))
    return out, unlab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--scale', type=float, default=3000.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='runs/s2_published_survey.json')
    a = ap.parse_args()

    import rasterio
    files = sorted((D / 'S2Hand').glob('*.tif'))
    if a.limit:
        files = files[:a.limit]
    print('chips {}  crop {}  stride {}  scale {:g}'.format(
        len(files), a.crop, a.stride, a.scale), flush=True)

    per = {k: [] for k in ARMS}
    clip, unlab = [], 0
    for i, f in enumerate(files):
        with rasterio.open(f) as s:
            arr = s.read().astype(np.float32)
        bgr, cl = to_bgr8(arr, a.scale)
        clip.append(cl)
        res, nu = survey_chip(bgr, a.crop, a.stride)
        unlab += nu
        for k, v in res.items():
            if np.isfinite(v):
                per[k].append(v)
        if (i + 1) % 25 == 0:
            print('  {}/{}'.format(i + 1, len(files)), flush=True)

    print()
    print('=' * 76)
    print('THE PUBLISHED LABELLER, RUN PER CROP ON PUBLIC SENTINEL-2 IMAGERY')
    print('scale {:g}   clipped {:.2%} of samples   unlabelled {} pixels'.format(
        a.scale, float(np.mean(clip)), unlab))
    print('=' * 76)
    print('{:36s} {:>10s} {:>10s} {:>7s}'.format(
        'fitted per crop except', 'mean P(A)', 'max P(A)', 'chips'))
    res = {}
    for k in ARMS:
        v = np.array(per[k])
        if not v.size:
            continue
        res[k] = {'mean': float(v.mean()), 'max': float(v.max()),
                  'n_chips': int(v.size)}
        print('{:36s} {:>9.2%} {:>10.2%} {:>7d}'.format(
            k, v.mean(), v.max(), v.size))

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {'crop': a.crop, 'stride': a.stride, 'scale': a.scale,
         'n_chips': len(files), 'clipped_fraction': float(np.mean(clip)),
         'unlabelled_pixels': unlab, 'arms': res}, indent=1))
    print('wrote', p)


if __name__ == '__main__':
    main()
