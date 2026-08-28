"""Why the two min-max normalisations are inert in the pipeline as published.

The survey measures that they are: moving either one from the crop to the whole scene
leaves P(A) unchanged while the background and the Otsu threshold are still fitted per
crop. Two things have to hold for the obvious explanation, and both are claims about
arrays rather than about labels, so both are measured here rather than argued.

  1. The swap has something to do. If the crop range always equalled the scene range
     the substitution would be vacuous and the finding would be about our code.
  2. It still reaches no label, because a min-max is the identity on an array that
     already spans the range it is being mapped onto.

Both normalisations are measured, not just the first. The array reaching the second
is the output of a THRESH_TRUNC at 235, so the range it is compared against is the
scene's own second range rather than 0-255, and treating the two as interchangeable is
how a partial explanation would look complete.

Writes JSON so the supplement's numbers are checkable against a file.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, 'src')
from s2_published_survey import (LO_WATER, HI_WATER, classes, scene_context,
                                 shadow_cloud, to_bgr8)


def stages(sub, ctx, r, c, crop):
    """The arrays reaching each normalisation, with the background and the Otsu
    threshold fitted per crop exactly as the published pipeline leaves them."""
    hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
    nw = sub.copy()
    nw[cv2.inRange(hsv, LO_WATER, HI_WATER) == 255] = [255, 255, 255]
    img = cv2.cvtColor(nw, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(cv2.dilate(img, np.ones((7, 7), np.uint8)), 155)
    d = 255 - cv2.absdiff(img, bg)
    _, keep = cv2.threshold(d, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    d2 = cv2.bitwise_and(d, keep)
    norm = cv2.normalize(d2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    _, thr = cv2.threshold(norm, 235, 0, cv2.THRESH_TRUNC)
    return d2, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--scale', type=float, default=3000.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='runs/s2_mechanism.json')
    a = ap.parse_args()

    import rasterio
    files = sorted(Path('data/sen1floods11/S2Hand').glob('*.tif'))
    if a.limit:
        files = files[:a.limit]
    print('chips {}  crop {}  stride {}'.format(len(files), a.crop, a.stride),
          flush=True)

    n = 0
    diff1 = diff2 = 0            # crops where the crop range differs from the scene's
    ident1 = ident2 = 0          # crops where the normalisation is already the identity
    flat1 = flat2 = 0            # constant arrays, where cv2 returns zeros
    chg1 = chg2 = 0              # pixels whose class changes when the range is swapped

    for i, f in enumerate(files):
        with rasterio.open(f) as s:
            bgr, _ = to_bgr8(s.read().astype(np.float32), a.scale)
        ctx = scene_context(bgr)
        base = {k: ctx[k] for k in ('otsu', 'n1', 'n2')}
        H, W = bgr.shape[:2]
        for r in range(0, H - a.crop + 1, a.stride):
            for c in range(0, W - a.crop + 1, a.stride):
                sub = bgr[r:r + a.crop, c:c + a.crop]
                cc = dict(base)
                cc['bg'] = ctx['bg'][r:r + a.crop, c:c + a.crop]
                n += 1

                d2, thr = stages(sub, ctx, r, c, a.crop)
                r1 = (float(d2.min()), float(d2.max()))
                r2 = (float(thr.min()), float(thr.max()))
                flat1 += r1[0] == r1[1]
                flat2 += r2[0] == r2[1]
                # the identity means the array already spans what it is mapped onto
                ident1 += r1 == (0.0, 255.0)
                ident2 += r2 == tuple(ctx['n2'])

                if r1 != tuple(ctx['n1']):
                    diff1 += 1
                    chg1 += int((classes(shadow_cloud(sub, cc, ()))
                                 != classes(shadow_cloud(sub, cc, ('n1',)))).sum())
                if r2 != tuple(ctx['n2']):
                    diff2 += 1
                    chg2 += int((classes(shadow_cloud(sub, cc, ()))
                                 != classes(shadow_cloud(sub, cc, ('n2',)))).sum())
        if (i + 1) % 50 == 0:
            print('  {}/{}'.format(i + 1, len(files)), flush=True)

    res = {
        'n_chips': len(files), 'crop': a.crop, 'stride': a.stride, 'scale': a.scale,
        'n_crops': n,
        'first': {'range_differs': diff1, 'already_identity': ident1,
                  'constant_array': flat1, 'pixels_changed': chg1,
                  'identity_fraction': ident1 / n},
        'second': {'range_differs': diff2, 'already_identity': ident2,
                   'constant_array': flat2, 'pixels_changed': chg2,
                   'identity_fraction': ident2 / n},
    }
    print()
    print('=' * 72)
    print('crops {}'.format(n))
    for k in ('first', 'second'):
        v = res[k]
        print('{} normalisation'.format(k))
        print('   range differs from the scene   {:>8d}'.format(v['range_differs']))
        print('   already the identity           {:>8d}  ({:.1%})'.format(
            v['already_identity'], v['identity_fraction']))
        print('   constant array                 {:>8d}'.format(v['constant_array']))
        print('   pixels whose class changes     {:>8d}'.format(v['pixels_changed']))

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print('wrote', p)


if __name__ == '__main__':
    main()
