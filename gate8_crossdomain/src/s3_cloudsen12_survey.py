"""Cloud-masking rules, run per crop, on a third public dataset.

The two earlier surveys measure P(A) on water labelling over Sen1Floods11. Both are
ours to the extent that we chose the imagery. This is a third dataset we did not
assemble, a different task, and a different sensor product, and like the others it
trains nothing: P(A) is a property of a labelling rule and a crop grid.

WHAT P(A) CANNOT SEE, AND WHAT THIS SURVEY DOES ABOUT IT.

P(A) counts whether the crops covering a pixel agree, not what they agree on. It is
therefore invariant to swapping the class labels, to thresholding the wrong band, and
to whether the rule is any good at all: a rule that calls every pixel cloud scores a
perfect P(A) = 0. Controls built only from P(A) cannot fail on a semantic error, and
this project's error list is mostly semantic errors that produced plausible numbers.

So every rule also carries an accuracy column, its cloud IoU against the expert mask,
computed with the rule applied ONCE PER SCENE so that it is a property of the rule
rather than of the grid. A polarity inversion, a wrong band index or a degenerate
constant rule collapses that column while leaving P(A) untouched. CloudSEN12+ is the
only dataset in this paper that ships expert labels beside the imagery, which is what
makes the column possible and is the reason to use it.

The artefact is also reported against expert cloud fraction. If P(A) were a mere
consequence of a rule being fitted, it would not vary with what the scene contains.

DATA, ESTABLISHED FROM THE PIXELS BY s3_cloudsen12_probe.py, NOT FROM THE CARD.
  342 patches, 509 x 509, uint16, 16 bands.
  Bands 0..12 are spectral in L1C order, so index 1 is B02, blue. Established by
  resampling structure: exactly four bands are natively 10 m (1, 2, 3, 7 = B02, B03,
  B04, B08), which pins blue under either the L1C or the L2A ordering. Band 10 is
  an order of magnitude darker than the rest, which is what a cirrus band looks
  like and means this is L1C; the block-structure test does not separate the 60 m
  bands cleanly enough to carry that on its own.
  Band 13 is the expert mask, 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.
  Bands 14 and 15 are two published model predictions and are not used here.
  There is NO padding in this subset: zero fully-zero rows or columns on every edge
  of every patch checked. The card describes padding to 512 for a different variant.

Reflectance is used directly as DN / 10000. Nothing is quantised to 8 bits, so no
preparation constant can degenerate a rule, and the clipped fraction is zero by
construction rather than by choice.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

D = Path('data/cloudsen12/high')
PROV = Path('data/cloudsen12/provenance.json')
BLUE, GREEN, RED, NIR = 1, 2, 3, 7        # verified from resampling structure
EXPERT = 13
SCALE = 10000.0
CLOUD = (1, 2)                            # thick and thin; shadow counts as clear


# ---- the rules -------------------------------------------------------------
# Each takes the fitting sample and returns a per-pixel cloud mask for `img`.
# `img` is a dict of reflectance planes so a rule may be multi-band.

def _hot(img):
    """Haze Optimized Transform, Zhang et al. 2002. Fixed coefficients."""
    return img['blue'] - 0.5 * img['red'] - 0.08


def _whiteness(img):
    vis = (img['blue'] + img['green'] + img['red']) / 3.0
    with np.errstate(divide='ignore', invalid='ignore'):
        w = (np.abs(img['blue'] - vis) + np.abs(img['green'] - vis)
             + np.abs(img['red'] - vis)) / np.where(vis > 0, vis, np.nan)
    return np.nan_to_num(w, nan=10.0)


def r_fixed_blue(img, fit):
    """Sen2Cor's first cloud test is a fixed threshold on B02. Per-pixel."""
    return img['blue'] > 0.25


def r_fixed_fmask(img, fit):
    """Fmask's whiteness and brightness tests with fixed constants. Per-pixel,
    multi-band, so it cannot be dismissed as a one-band strawman."""
    return (_whiteness(img) < 0.7) & (img['blue'] > 0.18)


def _pct_rule(img, fit, q):
    h = _hot(img)
    f = _hot(fit)
    f = f[np.isfinite(f)]
    if f.size < 64:
        return None
    return h > np.percentile(f, q)


def r_pct_scene(img, fit):
    """Fmask and ACCA both set their final threshold from a percentile of the scene.
    Computed over the whole patch, so it is one constant and stays per-pixel."""
    return _pct_rule(img, fit, 82.5)


def r_pct_crop(img, fit):
    """The same estimator, fitted to the crop. Only the support changes."""
    return _pct_rule(img, fit, 82.5)


def r_otsu_crop(img, fit):
    """Otsu on HOT, fitted to the crop, over fixed global bins so that the binning
    support is not a second thing that differs between arms."""
    h, f = _hot(img), _hot(fit)
    f = f[np.isfinite(f)]
    if f.size < 64:
        return None
    lo, hi = -0.2, 1.0                     # fixed, global
    cnt, edges = np.histogram(np.clip(f, lo, hi), bins=256, range=(lo, hi))
    p = cnt.astype(np.float64) / max(cnt.sum(), 1)
    w0 = np.cumsum(p)
    mids = (edges[:-1] + edges[1:]) / 2
    m0 = np.cumsum(p * mids)
    mt = m0[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        var = (mt * w0 - m0) ** 2 / (w0 * (1 - w0))
    if not np.isfinite(var).any():
        return None                        # degenerate crop: abstain, do not raise
    k = int(np.nanargmax(var))
    return h > mids[k]


def r_minmax_crop(img, fit):
    """Contrast-stretch inside the crop, then a fixed threshold. This is the shape of
    the step that carries the sea-ice result."""
    f = fit['blue']
    f = f[np.isfinite(f)]
    if f.size < 64:
        return None
    lo, hi = float(f.min()), float(f.max())
    if hi <= lo:
        return None
    return (img['blue'] - lo) / (hi - lo) > 0.5


RULES = {
    'fixed B02 constant': (r_fixed_blue, False),
    'fixed whiteness and brightness': (r_fixed_fmask, False),
    'clear-sky percentile, per scene': (r_pct_scene, False),
    'clear-sky percentile, per crop': (r_pct_crop, True),
    'Otsu on HOT, per crop': (r_otsu_crop, True),
    'min-max per crop, then fixed': (r_minmax_crop, True),
}
FITTED_PER_CROP = {k for k, (_, f) in RULES.items() if f}


def planes(a):
    return {'blue': a[BLUE], 'green': a[GREEN], 'red': a[RED], 'nir': a[NIR]}


def one_patch(path, crop=128, stride=32):
    """P(A) per rule for one patch, plus the scene-level accuracy of each rule and
    the patch's expert cloud fraction. No shared state, so this is the parallel unit."""
    import rasterio
    with rasterio.open(path) as s:
        a = s.read()
    spec = a[:13].astype(np.float64) / SCALE
    expert = a[EXPERT]
    truth = np.isin(expert, CLOUD)
    H, W = spec.shape[1], spec.shape[2]

    scene = planes(spec)
    valid = (spec != 0).any(axis=0)

    starts = [(r, c)
              for r in range(0, H - crop + 1, stride)
              for c in range(0, W - crop + 1, stride)]

    pa, denom, iou, abstain = {}, {}, {}, {}
    for name, (fn, _) in RULES.items():
        votes = np.zeros((2, H, W), np.int32)
        cov = np.zeros((H, W), np.int32)
        nab = 0
        for r, c in starts:
            sl = (slice(r, r + crop), slice(c, c + crop))
            sub = {k: v[sl] for k, v in scene.items()}
            vm = valid[sl]
            # the fitted sample excludes invalid pixels; masking only the vote
            # accumulation would let a nodata pixel move a threshold and so
            # change the labels of valid ones
            fit = ({k: v[vm] for k, v in sub.items()} if name in FITTED_PER_CROP
                   else {k: v[valid] for k, v in scene.items()})
            lab = fn(sub, fit)
            if lab is None:
                nab += 1
                continue
            cov[sl] += vm
            votes[1][sl] += lab & vm
            votes[0][sl] += (~lab) & vm
        sel = (cov >= 2) & (votes.sum(0) == cov)
        abstain[name] = nab
        denom[name] = int(sel.sum())
        pa[name] = (float((votes.max(0)[sel] < cov[sel]).mean())
                    if sel.any() else float('nan'))

        # How good are the labels this rule actually produced? Scored the way this
        # paper scores everywhere else: one decision per source pixel, by majority
        # vote over its covering crops. Balanced accuracy rather than IoU, because
        # IoU's null is the cloud fraction, so on cloudy imagery a rule that calls
        # every pixel cloud beats every real rule and the column would certify a
        # degenerate labeller. Balanced accuracy is 0.5 for ANY constant rule and
        # below 0.5 for an inverted one, which is what the column has to detect.
        pred = votes[1] > votes[0]
        tp = float((pred & truth & sel).sum())
        fneg = float((~pred & truth & sel).sum())
        tn = float((~pred & ~truth & sel).sum())
        fp = float((pred & ~truth & sel).sum())
        sens = tp / (tp + fneg) if (tp + fneg) else float('nan')
        spec = tn / (tn + fp) if (tn + fp) else float('nan')
        iou[name] = 0.5 * (sens + spec)

    return {'file': Path(path).name, 'pa': pa, 'denom': denom, 'iou': iou,
            'abstain': abstain, 'cloud_fraction': float(truth.mean()),
            'pid': os.getpid()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--stride', type=int, default=32)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=1,
                    help='1 takes a genuinely serial path, no executor at all')
    ap.add_argument('--out', default='runs/s3_cloudsen12_survey.json')
    a = ap.parse_args()

    files = sorted(D.glob('*.tif'))
    if a.limit:
        files = files[:a.limit]
    print('patches {}  crop {}  stride {}  workers {}'.format(
        len(files), a.crop, a.stride, a.workers), flush=True)

    if a.workers <= 1:
        rows = [one_patch(f, a.crop, a.stride) for f in files]
    else:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        job = partial(one_patch, crop=a.crop, stride=a.stride)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(job, files))

    # a pool that silently ran serial would pass an identity test trivially
    pids = sorted({r['pid'] for r in rows})
    assert len(rows) == len(files), 'lost patches: {} of {}'.format(
        len(rows), len(files))

    # Every rule must be averaged over the same pixels or the columns are not
    # comparable. Here that holds by construction, since coverage is a property of
    # the grid and no rule abstains at this crop size, so this is a guard against a
    # future rule that does abstain rather than a measurement of anything now.
    bad_denoms = [r['file'] for r in rows
                  if len({r['denom'][k] for k in RULES}) != 1]
    total_abstain = {k: sum(r['abstain'][k] for r in rows) for k in RULES}

    res = {}
    for k in RULES:
        v = np.array([r['pa'][k] for r in rows], float)
        v = v[np.isfinite(v)]
        ious = np.array([r['iou'][k] for r in rows], float)
        ious = ious[np.isfinite(ious)]
        res[k] = {
            'fitted_per_crop': k in FITTED_PER_CROP,
            'mean': float(v.mean()), 'max': float(v.max()),
            'median': float(np.median(v)), 'n_patches': int(v.size),
            'n_nonzero': int((v > 0).sum()),
            'balanced_accuracy': float(ious.mean()) if ious.size else float('nan'),
            'abstained_crops': total_abstain[k],
            'per_patch': {r['file']: r['pa'][k] for r in rows},
        }

    # does the artefact depend on what the scene contains, or only on the rule's form?
    cf = np.array([r['cloud_fraction'] for r in rows])
    bins = [(0.0, 0.1), (0.1, 0.4), (0.4, 0.7), (0.7, 1.01)]
    dose = {}
    for lo, hi in bins:
        m = (cf >= lo) & (cf < hi)
        if not m.any():
            continue
        dose['{:.1f}-{:.1f}'.format(lo, min(hi, 1.0))] = {
            'n': int(m.sum()),
            **{k: float(np.nanmean([r['pa'][k] for r, keep in zip(rows, m) if keep]))
               for k in FITTED_PER_CROP}}

    prov = json.loads(PROV.read_text()) if PROV.exists() else {}
    out = {'crop': a.crop, 'stride': a.stride, 'n_patches': len(files),
           'workers': a.workers, 'distinct_worker_pids': len(pids),
           'reflectance_scale': SCALE, 'blue_band_index': BLUE,
           'expert_band_index': EXPERT, 'cloud_classes': list(CLOUD),
           'denominators_agree': not bad_denoms,
           'patches_with_unequal_denominators': bad_denoms,
           'source': {k: prov.get(k) for k in ('repo', 'file', 'revision',
                                               'sha256', 'licence')},
           'rules': res, 'by_cloud_fraction': dose}

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, sort_keys=True))

    print()
    print('=' * 86)
    print('CLOUD MASKING RULES, PER CROP, ON 342 PUBLIC CLOUDSEN12+ PATCHES')
    print('=' * 86)
    print('{:34s} {:>7s} {:>10s} {:>9s} {:>9s} {:>9s}'.format(
        'rule', 'in crop?', 'mean P(A)', 'median', 'max', 'bal acc'))
    for k in RULES:
        r = res[k]
        print('{:34s} {:>7s} {:>9.2%} {:>9.2%} {:>9.2%} {:>9.3f}'.format(
            k, 'yes' if r['fitted_per_crop'] else 'no',
            r['mean'], r['median'], r['max'], r['balanced_accuracy']))
    print()
    print('denominators identical across rules:', not bad_denoms,
          '| abstained crops:', sum(total_abstain.values()),
          '| worker pids:', len(pids))
    if dose:
        print()
        print('P(A) by expert cloud fraction (fitted rules only):')
        for b, d in dose.items():
            print('  {:9s} n={:3d}  '.format(b, d['n'])
                  + '  '.join('{} {:.2%}'.format(k.split(',')[0], d[k])
                              for k in sorted(FITTED_PER_CROP)))
    print('wrote', p)
    return 0


if __name__ == '__main__':
    sys.exit(main())
