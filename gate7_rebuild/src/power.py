"""How large would this benchmark have to be to resolve the effects claimed on it?

The paper argues that published claims on auto-labelled sea-ice benchmarks sit
below the benchmark's own resolution. That argument is much stronger stated as a
required sample size than as a spread, because a required sample size is a number
a reader can compare against what the audited work actually used.

For a paired contrast with mean d and paired sd s over n units, the two-sided
test at alpha rejects when |d| > t * s / sqrt(N), so

    N_significance = (1.96 * s / d) ** 2
    N_80_power     = (2.80 * s / d) ** 2       # 1.96 + 0.84

and the smallest effect detectable at 80% power with the n in hand is

    MDE(n) = 2.80 * s / sqrt(n)

Reported for the premium under both scoring protocols, so the cost of
pseudo-replication is visible as a sample-size claim rather than a p-value.
"""
import json
from pathlib import Path

import numpy as np

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
MOS = W / "runs" / "mosaic"
THR = json.loads((W / "results_threshold_expanded.json").read_text())


def mosaic_adv(seed=42, prefix="loao", key="advantage_nanmean"):
    out = {"scene": {}, "original": {}}
    for f in MOS.glob(prefix + "_*.json"):
        r = json.loads(f.read_text())
        if r["seed"] != seed or r["labels"] not in out:
            continue
        out[r["labels"]][r["acq"]] = r[key]
    return out


def patch_adv():
    out = {"scene": {}, "original": {}}
    for labels in ("scene", "original"):
        for f in (W / "runs").glob("loao_" + labels + "_*_s42"):
            m = json.loads((f / "test_metrics.json").read_text())
            a = m["holdout_acq"]
            if a in THR:
                out[labels][a] = m["miou"] - THR[a][labels]["test_miou"]
    return out


def power(d, s, n, name):
    if d == 0:
        return
    n_sig = (1.96 * s / abs(d)) ** 2
    n_80 = (2.80 * s / abs(d)) ** 2
    mde = 2.80 * s / np.sqrt(n)
    print("  {:34s} d={:+.4f}  s={:.4f}  n={:2d}".format(name, d, s, n))
    print("  {:34s} MDE at n={:<3d} {:.4f}   {}".format(
        "", n, mde, "RESOLVED" if abs(d) > mde else "below the floor"))
    print("  {:34s} N for significance {:6.0f}   N for 80% power {:6.0f}".format(
        "", np.ceil(n_sig), np.ceil(n_80)))
    return {"d": d, "s": s, "n": n, "mde": mde,
            "n_significance": float(np.ceil(n_sig)), "n_power80": float(np.ceil(n_80))}


def premium_from(adv, name):
    acqs = sorted(set(adv["scene"]) & set(adv["original"]))
    if len(acqs) < 3:
        print("  {}: only {} folds".format(name, len(acqs)))
        return None
    p = np.array([adv["original"][a] - adv["scene"][a] for a in acqs])
    return power(p.mean(), p.std(ddof=1), len(p), name)


def main():
    print("=" * 86)
    print("REQUIRED SAMPLE SIZE FOR THE ARTEFACT PREMIUM")
    print("=" * 86)
    out = {}
    out["patch"] = premium_from(patch_adv(), "premium, patch-pixel scoring")
    print()
    out["mosaic_vote"] = premium_from(mosaic_adv(), "premium, mosaic (vote)")
    print()
    d = mosaic_adv(key="advantage_draw")
    if d["scene"] and "advantage_draw" in json.loads(
            next(MOS.glob("loao_*.json")).read_text()):
        out["mosaic_draw"] = premium_from(d, "premium, mosaic (draw)")
        print()

    print("=" * 86)
    print("THE SAME QUESTION FOR THE EFFECTS THE AUDITED WORK REPORTS")
    print("=" * 86)
    adv = mosaic_adv()
    acqs = sorted(set(adv["scene"]) & set(adv["original"]))
    s_prem = np.array([adv["original"][a] - adv["scene"][a] for a in acqs]).std(ddof=1)
    for nm, claimed in (("published fusion gain", 0.0306),
                        ("fusion-strategy spread", 0.0401),
                        ("test-set leak", 0.0009)):
        n_80 = (2.80 * s_prem / claimed) ** 2
        print("  {:34s} claimed {:.4f}   N for 80% power {:7.0f}".format(
            nm, claimed, np.ceil(n_80)))

    print("\n  the audited work used 2 acquisitions.")
    (W / "runs" / "power.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
