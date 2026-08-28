"""The artefact premium, with every robustness check the reviews asked for.

The claim: when a benchmark's labels are computed from the same pixels the model
reads, extra capacity is partly paid to reproduce the label generator rather than
the phenomenon. Measured as a difference-in-differences within each held-out
acquisition, so scene difficulty cancels twice:

    premium(a) = [U-Net(a, crop-noisy)  - threshold(a, crop-noisy)]
               - [U-Net(a, crop-invariant) - threshold(a, crop-invariant)]

Reported three ways -- t, bootstrap over acquisitions, and a sign test -- because
n=17 and no one of them alone should carry a headline. Then repeated under a
tile-disjoint split and over three seeds, which are the two objections raised
against it.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
RUNS = W / "runs"
THR = json.loads((W / "results_threshold_expanded.json").read_text()) \
    if (W / "results_threshold_expanded.json").exists() else None


def thresholds():
    for c in [W / "results_threshold_expanded.json",
              W / "work" / "threshold_expanded.json",
              W / "runs" / "threshold_expanded.json"]:
        if c.exists():
            return json.loads(c.read_text())
    raise SystemExit("threshold_expanded.json not found on the remote; "
                     "copy it from the repo's gate6_scale/results/")


def unet(prefix, labels, seed):
    out = {}
    for d in RUNS.glob(f"{prefix}_{labels}_*_s{seed}"):
        f = d / "test_metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        if m.get("holdout_acq"):
            out[m["holdout_acq"]] = m["miou"]
    return out


def stats(prem, label):
    n = len(prem)
    if n < 3:
        print(f"  {label}: only {n} folds, skipped")
        return
    se = prem.std(ddof=1) / np.sqrt(n)
    tcrit = 2.12 if n >= 17 else 2.26
    rng = np.random.RandomState(0)
    bs = np.array([prem[rng.randint(0, n, n)].mean() for _ in range(20000)])
    k = int((prem > 0).sum())
    p_sign = min(2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, 1.0)
    print(f"  {label:28s} n={n:2d}  premium {prem.mean():+.4f}  t={prem.mean()/se:5.2f}  "
          f"CI [{prem.mean()-tcrit*se:+.4f},{prem.mean()+tcrit*se:+.4f}]  "
          f"boot [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  "
          f"sign {k}/{n} p={p_sign:.4f}")


def run(prefix, seed, thr, title):
    us, uo = unet(prefix, "scene", seed), unet(prefix, "original", seed)
    acqs = sorted(set(thr) & set(us) & set(uo))
    if len(acqs) < 3:
        print(f"  {title}: {len(acqs)} complete folds, skipped")
        return None
    adv_s = np.array([us[a] - thr[a]["scene"]["test_miou"] for a in acqs])
    adv_o = np.array([uo[a] - thr[a]["original"]["test_miou"] for a in acqs])
    prem = adv_o - adv_s
    stats(prem, title)
    return {"acqs": acqs, "adv_scene": adv_s.tolist(),
            "adv_original": adv_o.tolist(), "premium": prem.tolist(),
            "ratio": float(adv_o.mean() / adv_s.mean())}


def main():
    thr = thresholds()
    print("=" * 108)
    print("THE ARTEFACT PREMIUM")
    print("=" * 108)
    out = {}
    print("\nstandard LOAO (tiles may recur across the boundary):")
    for s in (42, 7, 123):
        r = run("loao", s, thr, f"seed {s}")
        if r:
            out[f"loao_s{s}"] = r
    print("\ntile-disjoint LOAO (no shared ground):")
    r = run("dj", 42, thr, "seed 42, disjoint")
    if r:
        out["disjoint_s42"] = r

    seeds = [out[k]["premium"] for k in out if k.startswith("loao_s")]
    if len(seeds) > 1:
        m = np.array(seeds)
        print(f"\nacross {len(seeds)} seeds: premium per seed "
              f"{', '.join(f'{x:+.4f}' for x in m.mean(axis=1))}")
        stats(m.mean(axis=0), "seed-averaged")
    for k, v in out.items():
        print(f"  ratio {k:18s} {v['ratio']:.2f}x")
    (RUNS / "premium_final.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {RUNS / 'premium_final.json'}")


if __name__ == "__main__":
    main()
