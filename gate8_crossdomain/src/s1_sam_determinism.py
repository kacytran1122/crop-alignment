"""Is the SAM arm's artefact set crop-dependence, or just run-to-run variation?

The survey attributes a nonzero P(A) to the segmentation being recomputed inside each
crop. There is a cheaper explanation that would produce the same number: if the mask
generator is not deterministic, two runs on the same pixels would disagree on their
own, and the survey would be measuring that instead. The two are indistinguishable
from the survey's output alone.

So run it twice on identical input and compare. Both the mask sets and the labels
derived from them have to come back identical for the survey's number to mean what it
is offered as meaning.
"""
import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch

sys.path.insert(0, 'src')
from s1_sam_survey import label_from_masks, to_rgb8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--points', type=int, default=16)
    # Split across adjacent string literals, this path was invisible to every
    # contiguous-path scan, including the repository's own anonymiser.
    ap.add_argument('--ckpt', default=os.environ.get(
        'SAM_CKPT', 'data/sam_vit_b_01ec64.pth'))
    ap.add_argument('--out', default='runs/s1_sam_determinism.json')
    a = ap.parse_args()

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry['vit_b'](checkpoint=a.ckpt).to('cuda').eval()
    amg = SamAutomaticMaskGenerator(sam, points_per_side=a.points,
                                    pred_iou_thresh=0.85,
                                    stability_score_thresh=0.90)

    files = sorted(Path('data/sen1floods11/S1Hand').glob('*.tif'))
    same_masks = same_labels = n = 0
    for f in files:
        if n >= a.n:
            break
        with rasterio.open(f) as s:
            vh = s.read(2).astype(np.float32)
        valid = vh > -9998.0
        if valid.mean() < 0.5:
            continue
        for i in range(4):
            if n >= a.n:
                break
            r, c = (i * 97) % 385, (i * 61) % 385
            sub, vm = vh[r:r + 128, c:c + 128], valid[r:r + 128, c:c + 128]
            img = to_rgb8(sub)
            with torch.no_grad():
                m1 = amg.generate(img)
                m2 = amg.generate(img)
            same_masks += (len(m1) == len(m2) and all(
                np.array_equal(x['segmentation'], y['segmentation'])
                for x, y in zip(m1, m2)))
            same_labels += np.array_equal(label_from_masks(sub, m1, vm),
                                          label_from_masks(sub, m2, vm))
            n += 1

    res = {'pairs': n, 'identical_mask_sets': int(same_masks),
           'identical_labels': int(same_labels),
           'deterministic': bool(same_labels == n and n > 0)}
    print('repeat runs on identical crops, n =', n)
    print('  identical mask sets :', same_masks, '/', n)
    print('  identical labels    :', same_labels, '/', n)
    print('DETERMINISTIC' if res['deterministic'] else
          'NONDETERMINISTIC: the survey number would be confounded')
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print('wrote', p)


if __name__ == '__main__':
    main()
