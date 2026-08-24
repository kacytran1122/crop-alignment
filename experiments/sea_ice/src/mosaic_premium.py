"""The artefact premium, recomputed under the protocol the paper recommends.

Same difference-in-differences as premium_test.py, but every number now comes from
mosaic scoring with both arms on an identical pixel set and one empty-class
convention. If the premium holds here, the paper is internally consistent: it
follows the rules it asks of the work it audits. If it moves, that is the more
important result and it goes in the paper.

Both mIoU conventions are carried through so the reader can see that the choice
does not drive the conclusion.
"""
import json
from math import comb
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
MOS = W / "runs" / "mosaic"


def load(prefix, seed):
    out = {"scene": {}, "original": {}}
    for f in MOS.glob("*.json"):
        r = json.loads(f.read_text())
        stem = f.stem
        if not stem.startswith(prefix + "_"):
            continue
        if r["seed"] != seed:
            continue
        if r["labels"] in out:
            out[r["labels"]][r["acq"]] = r
    return out


def stats(prem, label, conv):
    n = len(prem)
    if n < 3:
        print("  {:34s} only {} folds, skipped".format(label, n))
        return None
    se = prem.std(ddof=1) / np.sqrt(n)
    tcrit = 2.12 if n >= 17 else 2.31
    rng = np.random.RandomState(0)
    bs = np.array([prem[rng.randint(0, n, n)].mean() for _ in range(20000)])
    k = int((prem > 0).sum())
    p_sign = min(2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n, 1.0)
    print("  {:22s} {:9s} n={:2d} premium {:+.4f} t={:5.2f} "
          "CI [{:+.4f},{:+.4f}] boot [{:+.4f},{:+.4f}] sign {}/{} p={:.4f}".format(
              label, conv, n, prem.mean(), prem.mean() / se,
              prem.mean() - tcrit * se, prem.mean() + tcrit * se,
              np.percentile(bs, 2.5), np.percentile(bs, 97.5), k, n, p_sign))
    return {"n": n, "mean": float(prem.mean()), "t": float(prem.mean() / se),
            "ci": [float(prem.mean() - tcrit * se), float(prem.mean() + tcrit * se)],
            "boot": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "sign_k": k, "p_sign": float(p_sign)}


def run(prefix, seed, label):
    d = load(prefix, seed)
    acqs = sorted(set(d["scene"]) & set(d["original"]))
    if len(acqs) < 3:
        print("  {:34s} {} complete folds, skipped".format(label, len(acqs)))
        return None
    res = {"acqs": acqs}
    for conv in ("nanmean", "zerofill", "draw", "draw_zerofill"):
        key = "advantage_" + conv
        if key not in d["scene"][acqs[0]]:
            continue
        adv_s = np.array([d["scene"][a][key] for a in acqs])
        adv_o = np.array([d["original"][a][key] for a in acqs])
        r = stats(adv_o - adv_s, label, conv)
        if r:
            r["ratio"] = float(adv_o.mean() / adv_s.mean()) if adv_s.mean() else None
            r["adv_scene"] = adv_s.tolist()
            r["adv_original"] = adv_o.tolist()
            res[conv] = r
    reuse = np.mean([d["scene"][a]["reuse_avoided"] for a in acqs])
    res["mean_reuse_avoided"] = float(reuse)
    print("  {:22s} mean patch-pixel reuse avoided {:.2f}x".format("", reuse))
    return res


def main():
    if not MOS.exists() or not any(MOS.glob("*.json")):
        raise SystemExit("no mosaic results yet in " + str(MOS))
    print("=" * 118)
    print("THE ARTEFACT PREMIUM UNDER MOSAIC SCORING")
    print("  every source pixel scored once, both arms on an identical pixel set,")
    print("  one empty-class convention, reported both ways")
    print("=" * 118)
    out = {}
    for seed in (42, 7, 123):
        r = run("loao", seed, "standard LOAO s{}".format(seed))
        if r:
            out["loao_s{}".format(seed)] = r
    r = run("dj", 42, "tile-disjoint s42")
    if r:
        out["disjoint_s42"] = r
    (W / "runs" / "mosaic_premium.json").write_text(json.dumps(out, indent=1))
    print("\nwrote {}".format(W / "runs" / "mosaic_premium.json"))


if __name__ == "__main__":
    main()
