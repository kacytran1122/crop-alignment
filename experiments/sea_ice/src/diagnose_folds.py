"""
Why do three leave-one-acquisition-out folds collapse?

KNOWN_ISSUES #2: the headline 0.1164 across-acquisition sd is carried by three
folds. Drop them and it falls to 0.0394, which is smaller than the seed sd it was
being contrasted against. RESULTS names the outliers but never says what is wrong
with them, and a reviewer is entitled to ask whether they are a property of the
benchmark or three broken scenes.

This answers that from the confusion matrices, which record exactly what the model
saw and what it predicted, so no re-inference is needed.

For each fold we recover the held-out acquisition's class prior from the row sums
of its confusion matrix, then ask whether mIoU tracks the rarity of the rarest
present class. If it does, the spread is a class-prevalence artefact of macro-IoU
on scenes that barely contain one of the classes, not evidence that model quality
varies across acquisitions. Those are very different claims and the paper
currently makes the stronger one.
"""
import json
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parents[1]
RUNS = W / "runs"
NAMES = ["thick", "thin", "water"]


def fold_rows(labels):
    out = []
    for d in sorted(RUNS.glob(f"loao_{labels}_*_s42")):
        f = d / "test_metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        cm = np.array(m["cm"], dtype=np.float64)
        support = cm.sum(axis=1)
        total = support.sum()
        prior = support / max(total, 1)
        present = prior > 0
        # macro-IoU as the trainer computes it, and again over present classes only
        per = np.array(m["per_iou"], dtype=np.float64)
        out.append({
            "acq": m["holdout_acq"],
            "miou": m["miou"],
            "miou_present": float(per[present].mean()) if present.any() else float("nan"),
            "n_present": int(present.sum()),
            "prior": prior,
            "rarest_present": float(prior[present].min()) if present.any() else float("nan"),
            "per_iou": per,
            "px": float(total),
            "best_epoch": m.get("best_epoch"),
        })
    return out


def report(labels):
    rows = fold_rows(labels)
    if not rows:
        print(f"no folds for {labels}")
        return None
    rows.sort(key=lambda r: r["miou"])
    print(f"\n{'='*100}\n{labels} labels, {len(rows)} folds\n{'='*100}")
    print(f"{'acquisition':24s} {'mIoU':>7s} {'thick%':>7s} {'thin%':>7s} "
          f"{'water%':>7s} {'rarest':>8s} {'IoU_rare':>9s} {'ep':>3s}")
    for r in rows:
        p = r["prior"] * 100
        rare_idx = int(np.argmin(np.where(r["prior"] > 0, r["prior"], np.inf)))
        print(f"{r['acq']:24s} {r['miou']:7.4f} {p[0]:7.2f} {p[1]:7.2f} {p[2]:7.2f} "
              f"{r['rarest_present']*100:7.3f}% {r['per_iou'][rare_idx]:9.4f} "
              f"{str(r['best_epoch']):>3s}")

    miou = np.array([r["miou"] for r in rows])
    rare = np.array([r["rarest_present"] for r in rows])
    print(f"\nsd over all {len(rows)} folds        {miou.std(ddof=1):.4f}")
    print(f"sd dropping the 3 lowest      {miou[3:].std(ddof=1):.4f}")

    # Does mIoU track the rarity of the rarest class?
    lr, lm = np.log10(np.maximum(rare, 1e-12)), miou
    r_pearson = float(np.corrcoef(lr, lm)[0, 1])
    order_r, order_m = np.argsort(np.argsort(rare)), np.argsort(np.argsort(miou))
    r_spearman = float(np.corrcoef(order_r, order_m)[0, 1])
    print(f"\ncorr(mIoU, log10 rarest-class prevalence)  pearson {r_pearson:+.3f}"
          f"   spearman {r_spearman:+.3f}")

    lo = [r for r in rows[:3]]
    hi = [r for r in rows[3:]]
    lo_prevalence = ", ".join(
        f"{row['rarest_present'] * 100:.3f}%" for row in lo
    )
    print(f"\nrarest-class prevalence, 3 lowest folds:  "
          f"{lo_prevalence}")
    print(f"rarest-class prevalence, other {len(hi)} folds: "
          f"median {np.median([r['rarest_present'] for r in hi])*100:.3f}%, "
          f"min {min(r['rarest_present'] for r in hi)*100:.3f}%")

    # The counterfactual: score only classes with a non-trivial share.
    for floor in (0.0, 0.001, 0.005, 0.01):
        vals = []
        for r in rows:
            keep = r["prior"] >= floor
            if keep.any():
                vals.append(r["per_iou"][keep].mean())
        v = np.array(vals)
        print(f"  macro-IoU over classes with prevalence >= {floor*100:5.2f}%   "
              f"mean {v.mean():.4f}   sd {v.std(ddof=1):.4f}")
    return rows


def main():
    out = {}
    for labels in ("scene", "original"):
        rows = report(labels)
        if rows:
            out[labels] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in r.items()} for r in rows]
    (W / "work" / "fold_diagnosis.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {W / 'work' / 'fold_diagnosis.json'}")


if __name__ == "__main__":
    main()
