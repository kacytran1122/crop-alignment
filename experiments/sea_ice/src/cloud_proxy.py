"""Is cloud contamination driving any of this?

Published Sentinel-2 cloud masks fail over sea ice by OMISSION: the L1C mask
separates snow from cloud on SWIR contrast, and that leverage collapses over a
bright surface. Petty et al. (2021) kept 2 of 14 scenes that passed a 10% nominal
cloud filter, one of which was half cloud-contaminated. Buckley et al. (2023)
required >90% cloud-free by the shipped mask and still found that 75% of anomalous
tiles carried cloud that had evaded it.

This matters here because a brightness threshold labels cloud top as thick ice, and
a network trained on those labels learns to do the same. If our scenes carry cloud,
that is not a nuisance, it is a mechanism a reviewer can point at.

We have no Sentinel-2 metadata: the rasters came from the audited work. So test it
from the imagery, with a proxy chosen for being defensible rather than clever.

Sea ice at 10 m carries high-frequency structure -- floe edges, leads, ridges.
Cloud over ice is bright and SMOOTH at the same scale. So among bright pixels, low
gradient energy is cloud-like. The proxy is the fraction of bright pixels whose
local gradient sits below a low percentile of the scene's own bright-pixel gradient
distribution, which makes it scale-free and avoids a hand-set radiometric cut.

The proxy is weak in absolute terms and we do not claim otherwise. What it can do
is answer the question that matters: does kappa depend on it? If the folds with the
most cloud-like signal do not have systematically different kappa, cloud is not
what produces the crop-alignment result.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

W = Path(__file__).resolve().parents[1]
TIF = W / "data" / "tif"
BRIGHT = 140          # the thick-ice side of the recovered generator
SMOOTH_PCT = 20       # "smooth" = below this percentile of bright-pixel gradient
STRIDE = 4            # subsample for speed; the statistic is a fraction


def scene_proxy(path):
    with rasterio.open(path) as s:
        a = s.read(out_shape=(s.count, s.height // STRIDE, s.width // STRIDE))
    img = np.transpose(a, (1, 2, 0))[..., :3].astype(np.float32)
    valid = img.sum(axis=2) > 0
    if valid.sum() < 10000:
        return None
    v = img.max(axis=2)
    gy, gx = np.gradient(v)
    grad = np.hypot(gy, gx)
    # local mean gradient, so single noisy pixels do not read as texture
    grad = uniform_filter(grad, size=5, mode="nearest")

    bright = valid & (v >= BRIGHT)
    if bright.sum() < 5000:
        return {"bright_frac": float(bright.sum() / valid.sum()),
                "smooth_bright_frac": 0.0, "median_grad_bright": None}
    gb = grad[bright]
    cut = np.percentile(grad[valid], SMOOTH_PCT)
    return {"bright_frac": float(bright.sum() / valid.sum()),
            "smooth_bright_frac": float((gb <= cut).mean()),
            "median_grad_bright": float(np.median(gb)),
            "median_grad_all": float(np.median(grad[valid]))}


def main():
    recs = json.loads((W / "data" / "gate5_expansion_tiles.json").read_text())
    by_acq = {}
    for r in recs:
        p = TIF / r["file"]
        if not p.exists():
            continue
        s = scene_proxy(p)
        if s is None:
            continue
        by_acq.setdefault(r["datetime"], []).append(s)

    rows = []
    for dt, items in sorted(by_acq.items()):
        rows.append({
            "s2_datetime": dt,
            "n_scenes": len(items),
            "bright_frac": float(np.mean([i["bright_frac"] for i in items])),
            "smooth_bright_frac": float(np.mean([i["smooth_bright_frac"]
                                                 for i in items])),
        })

    print("{:18s} {:>7s} {:>13s} {:>20s}".format(
        "S2 datetime", "scenes", "bright frac", "smooth-and-bright"))
    for r in sorted(rows, key=lambda x: -x["smooth_bright_frac"]):
        print("{:18s} {:7d} {:13.4f} {:20.4f}".format(
            r["s2_datetime"], r["n_scenes"], r["bright_frac"],
            r["smooth_bright_frac"]))

    sb = np.array([r["smooth_bright_frac"] for r in rows])
    print("\n  smooth-and-bright fraction: min {:.4f}  median {:.4f}  max {:.4f}".format(
        sb.min(), np.median(sb), sb.max()))

    # does kappa depend on it?
    ks = W / "runs" / "kappa_summary.json"
    if ks.exists():
        k = json.loads(ks.read_text())
        # map acquisition -> s2 datetime through the token table
        import pandas as pd
        tok = pd.read_parquet(W / "work" / "expansion" / "tokens_subsample.parquet")
        m = tok.groupby("acq_id")["s2_datetime"].first().astype(str).to_dict()
        prox = {r["s2_datetime"]: r["smooth_bright_frac"] for r in rows}
        x, y = [], []
        for a, kn, kc in zip(k["acqs"], k["kappa_noisy"], k["kappa_clean"]):
            p = prox.get(str(m.get(a, "")))
            if p is not None:
                x.append(p); y.append(kn - kc)
        if len(x) >= 5:
            x, y = np.array(x), np.array(y)
            r = float(np.corrcoef(x, y)[0, 1])
            n = len(x)
            A = np.column_stack([np.ones(n), x])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ coef
            se = np.sqrt((resid @ resid / (n - 2)) * np.linalg.inv(A.T @ A)[1, 1])
            print("\n  kappa (treated - control) vs the cloud proxy, n={}".format(n))
            print("    correlation {:+.3f}   slope {:+.4f}   se {:.4f}   t {:+.2f}".format(
                r, coef[1], se, coef[1] / se if se else float("nan")))
            if abs(coef[1] / se if se else 0) < 2:
                print("    -> no dependence detected: cloud-likeness does not explain kappa")
            else:
                print("    -> DEPENDENCE DETECTED: investigate before publishing kappa")
        else:
            print("\n  too few matched folds to test the kappa dependence")

    (W / "work" / "cloud_proxy.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
