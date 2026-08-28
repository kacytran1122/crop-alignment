"""A foundation-model labeller, run per crop, with a fixed semantic rule.

The paper's introduction says that benchmarks built from foundation-model masks
inherit the property it tests for. That is currently a citation, not a measurement.
This measures it, and it is the sharpest form of the paper's claim because the
semantic rule here is a global constant. Nothing about the threshold is fitted to
anything. What is fitted to the crop is the segmentation itself.

The labeller is the one people actually build: SAM proposes regions, and each region
takes a class from a fixed rule, here the region's mean VH against one global
threshold in dB. Run it on the whole scene and every pixel's region, hence its class,
is a property of the scene. Run it inside a crop, as a patch-wise pipeline does, and
the region a pixel falls into depends on the window, so its mean can fall on either
side of a threshold that never moved.

Arms:
  fixed threshold, per pixel   no region proposal at all, crop-invariant by
                               construction, and the control on this code
  SAM per scene                regions computed once on the 512x512 chip, then read
                               out inside each crop
  SAM per crop                 regions recomputed inside every crop, which is what a
                               patch-wise pipeline does

dB conversion uses one global range for every chip and every crop, so no per-crop
statistic enters before the rule under test.
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch

D = Path('data/sen1floods11')
NODATA = -9999.0
DB_LO, DB_HI = -30.0, 0.0      # fixed 8-bit range, identical for every crop
THRESH_DB = -17.5              # one global water threshold, never refitted


def to_rgb8(vh):
    """VH in dB to 8-bit, on a fixed range. SAM wants three channels."""
    x = (np.clip(vh, DB_LO, DB_HI) - DB_LO) / (DB_HI - DB_LO)
    g = (x * 255).astype(np.uint8)
    return np.repeat(g[:, :, None], 3, axis=2)


def label_from_masks(vh, masks, valid):
    """Each SAM region takes the class its own mean VH implies under the fixed
    threshold. Pixels no region claims fall back to the same threshold per pixel, so
    the fallback is crop-invariant and cannot manufacture disagreement."""
    lab = (vh <= THRESH_DB).astype(np.int8)      # fixed per-pixel fallback
    for m in masks:
        seg = m['segmentation'] & valid
        if seg.sum() < 4:
            continue
        lab[seg] = 1 if float(vh[seg].mean()) <= THRESH_DB else 0
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--chips', type=int, default=24)
    ap.add_argument('--points', type=int, default=16)
    # Split across adjacent string literals, this path was invisible to every
    # contiguous-path scan, including the repository's own anonymiser.
    ap.add_argument('--ckpt', default=os.environ.get(
        'SAM_CKPT', 'data/sam_vit_b_01ec64.pth'))
    ap.add_argument('--out', default='runs/s1_sam_survey.json')
    a = ap.parse_args()

    import rasterio
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry['vit_b'](checkpoint=a.ckpt).to('cuda').eval()
    amg = SamAutomaticMaskGenerator(sam, points_per_side=a.points,
                                    pred_iou_thresh=0.85,
                                    stability_score_thresh=0.90)

    files = sorted((D / 'S1Hand').glob('*.tif'))[:a.chips]
    print('chips {}  crop {}  stride {}  points_per_side {}'.format(
        len(files), a.crop, a.stride, a.points), flush=True)

    arms = ['fixed threshold, per pixel', 'SAM per scene', 'SAM per crop']
    per = {k: [] for k in arms}

    for i, f in enumerate(files):
        with rasterio.open(f) as s:
            vh = s.read(2).astype(np.float32)
        valid = vh > NODATA + 1
        if valid.sum() < vh.size // 2:
            continue
        vh = np.where(valid, vh, DB_LO)
        H, W = vh.shape
        starts = [(r, c)
                  for r in range(0, H - a.crop + 1, a.stride)
                  for c in range(0, W - a.crop + 1, a.stride)]

        # regions computed once on the whole chip, read out per crop afterwards
        with torch.no_grad():
            scene_masks = amg.generate(to_rgb8(vh))
        scene_lab = label_from_masks(vh, scene_masks, valid)

        votes = {k: np.zeros((2, H, W), np.int32) for k in arms}
        cov = np.zeros((H, W), np.int32)
        for r, c in starts:
            sl = (slice(r, r + a.crop), slice(c, c + a.crop))
            sub, vm = vh[sl], valid[sl]
            if vm.sum() < 32:
                continue
            cov[sl] += vm
            lab = {
                'fixed threshold, per pixel': (sub <= THRESH_DB).astype(np.int8),
                'SAM per scene': scene_lab[sl],
            }
            with torch.no_grad():
                lab['SAM per crop'] = label_from_masks(sub, amg.generate(to_rgb8(sub)),
                                                       vm)
            for k in arms:
                votes[k][1][sl] += (lab[k] == 1) & vm
                votes[k][0][sl] += (lab[k] == 0) & vm

        multi = cov >= 2
        if not multi.any():
            continue
        for k in arms:
            v = votes[k]
            tot = v.sum(0)[multi]
            per[k].append(float((v.max(0)[multi] < tot).mean()))
        print('  {}/{}  '.format(i + 1, len(files))
              + '  '.join('{} {:.2%}'.format(k.split(',')[0], per[k][-1])
                          for k in arms), flush=True)

    print()
    print('=' * 72)
    print('A FOUNDATION-MODEL LABELLER WITH A FIXED SEMANTIC RULE')
    print('threshold {:g} dB, never refitted; dB range {:g} to {:g}, global'.format(
        THRESH_DB, DB_LO, DB_HI))
    print('=' * 72)
    print('{:32s} {:>10s} {:>10s} {:>7s}'.format('arm', 'mean P(A)', 'max', 'chips'))
    res = {}
    for k in arms:
        v = np.array(per[k])
        if not v.size:
            continue
        # per-chip values, not just the summary: P(A) varies by more than an order
        # of magnitude between chips here, so a mean alone hides how many chips are
        # actually nonzero, which is the claim the exact null makes checkable
        res[k] = {'mean': float(v.mean()), 'max': float(v.max()),
                  'n_chips': int(v.size), 'n_nonzero': int((v > 0).sum()),
                  'per_chip': [float(x) for x in v]}
        print('{:32s} {:>9.2%} {:>10.2%} {:>7d}'.format(k, v.mean(), v.max(), v.size))

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'crop': a.crop, 'stride': a.stride,
                             'points_per_side': a.points, 'threshold_db': THRESH_DB,
                             'db_range': [DB_LO, DB_HI], 'arms': res}, indent=1))
    print('wrote', p)


if __name__ == '__main__':
    main()
