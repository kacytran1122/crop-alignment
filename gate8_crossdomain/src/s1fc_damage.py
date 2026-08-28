"""Score the constructed flood arms against Sen1Floods11 expert labels.

This is a descriptive expert-label evaluation, not a causal estimate of the cost of
crop-dependence. The dial changes fitting support, mean threshold, class balance,
boundaries and raw label quality together.

Sen1Floods11 carries expert labels for the same chips. This scorer evaluates the
complete rebuilt J1 product: seven arms, eleven held-out events and three seeds (231
models). Those models were trained, validated and tested against their own generator,
so the expert-label question can be asked directly without training anything new:

    how does expert-label performance move across the constructed label dial?

SCORING. One decision per source pixel, by majority vote of the crops covering it,
scored once against the expert label. This is the mosaic convention the paper argues
for elsewhere; using patch scoring here would count each pixel around sixteen times
and would also hand the crop-dependent arms the very reward the paper says patch
scoring hands them, which would manufacture the result rather than measure it.

The alpha = 0 arm is trained on the published per-event algorithmic labels, not on
expert truth. The endpoint difference therefore describes replacing that published
field with predominantly per-crop Otsu labels; it does not hold everything else
fixed.
"""
import collections
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
C = W / "data" / "sen1floods11" / "cache"
CR = C / "crop"
R = W / "runs_s1fc"
IGNORE, NC = 255, 2


def iou_from_cm(cm):
    out = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        den = cm[k].sum() + cm[:, k].sum() - tp
        out.append(np.nan if den == 0 else tp / den)
    return np.array(out, np.float64)


def cell(out, a, e):
    """The value for one arm and event: the mean over the seeds in that cell."""
    return float(np.mean([r["miou_vs_expert"] for r in out[(a, e)]]))


def main():
    import segmentation_models_pytorch as smp

    meta = json.loads((C / "meta.json").read_text())
    cmeta = json.loads((CR / "meta.json").read_text())
    CROP, STARTS = cmeta["crop"], cmeta["starts"]

    ev = np.load(C / "events.npy")
    y_ref = np.load(C / "y_ref.npy")
    th_chip = np.load(CR / "th_chip.npy")
    th_crop = np.load(CR / "th_crop.npy")
    usable = np.isfinite(th_chip) & np.isfinite(th_crop).all(axis=(1, 2))
    imgs = np.load(C / "images.npy", mmap_mode="r")
    mu = np.array(meta["mean"], np.float32)
    sd = np.array(meta["std"], np.float32)

    pat = sys.argv[1] if len(sys.argv) > 1 else "j1_*_s*"
    runs = sorted(glob.glob(str(R / pat / "test_metrics.json")))
    if not runs:
        raise SystemExit("no runs match {!r} under {}".format(pat, R))
    run_meta = [(Path(f), json.loads(Path(f).read_text())) for f in runs]

    # Fail before loading a checkpoint if the selected runs are not the frozen J1
    # design. The previous default selected the superseded 55-run, seed-42 sweep;
    # cached J1 outputs could then make a partial rerun appear complete downstream.
    design_file = next((p for p in (W / "DESIGN.json", W.parent / "DESIGN.json")
                        if p.exists()), None)
    if design_file is None:
        raise SystemExit("DESIGN.json not found; refusing to infer the expected J1 "
                         "cells from the runs that happen to exist")
    design = json.loads(design_file.read_text())["flood"]
    expected = {(a, e, int(s)) for a in design["arms"]
                for e in design["events"] for s in design["seeds"]}
    got_list = [(tm.get("arm"), tm.get("holdout_event"), int(tm.get("seed")))
                for _, tm in run_meta]
    counts = collections.Counter(got_list)
    got = set(counts)
    missing, extra = sorted(expected - got), sorted(got - expected)
    dupes = sorted(k for k, n in counts.items() if n != 1)
    if missing or extra or dupes or len(run_meta) != len(expected):
        for x in missing[:6]:
            print("  MISSING J1 DAMAGE CELL: arm {} event {} seed {}".format(*x))
        for x in extra[:6]:
            print("  UNEXPECTED J1 DAMAGE CELL: arm {} event {} seed {}".format(*x))
        for x in dupes[:6]:
            print("  DUPLICATE J1 DAMAGE CELL: arm {} event {} seed {}".format(*x))
        raise SystemExit(
            "selected damage runs are not the exact {} arms x {} events x {} "
            "seeds = {} J1 cells ({} files selected)".format(
                len(design["arms"]), len(design["events"]), len(design["seeds"]),
                len(expected), len(run_meta)))
    print("scoring exact J1 design: {} runs matching {!r}".format(len(runs), pat))
    dev = torch.device("cuda")
    # Keyed by (arm, event), and a cell holds every seed rather than the last one.
    # It used to hold one record, so with three seeds per arm and event two of every
    # three runs were silently overwritten: the table announced 231 runs scored and
    # printed 77 numbers, each a single arbitrary seed, chosen by where the seed
    # suffix fell in a lexicographic sort. The default pre-J1 pattern is seed 42
    # only, so nothing collided there and the fault stayed invisible until the J1
    # arms, where it put the repository's own results file into disagreement with
    # the paper on the permutation arm, 0.4765 against 0.4399.
    out = collections.defaultdict(list)
    model = None
    for f, tm in run_meta:
        name, event = tm["name"], tm["holdout_event"]
        # named arms carry arm=<name> and alpha=None; dial runs carry alpha
        alpha = tm.get("arm") if tm.get("arm") is not None else tm.get("alpha")
        # A fold-locally normalised model scored with global statistics is wrong
        # in a way that raises nothing. Each J1 run records what it was trained
        # with; older runs have no such field and used the global values.
        if tm.get("norm_mean") is not None and tm.get("norm_std") is not None:
            mu_r = np.array(tm["norm_mean"], np.float32)
            sd_r = np.array(tm["norm_std"], np.float32)
        else:
            mu_r, sd_r = mu, sd
        cached = R / "damage" / (name + ".json")
        norm_source = ("fold-local" if tm.get("norm_mean") is not None and
                       tm.get("norm_std") is not None else "global-legacy")
        cache_provenance = {
            "name": name,
            "event": event,
            "arm": tm.get("arm"),
            "seed": int(tm["seed"]),
            "norm_source": norm_source,
            "test_metrics_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
        if cached.exists():
            old = json.loads(cached.read_text())
            if all(old.get(k) == v for k, v in cache_provenance.items()):
                out[(alpha, event)].append(old)
                continue
            print("  stale or unverifiable cache {}; rescoring".format(cached.name),
                  flush=True)
        if model is None:
            model = smp.Unet(tm["encoder"], encoder_weights=None, in_channels=2,
                             classes=NC).to(dev)
        model.load_state_dict(torch.load(Path(f).parent / "best.pt",
                                         map_location=dev,
                                         weights_only=False)["model"])
        model.eval()

        ti = np.where((ev == event) & usable)[0]
        cm = np.zeros((NC, NC), np.int64)
        with torch.no_grad():
            for ci in ti:
                x = ((np.asarray(imgs[ci], np.float32) - mu_r) / sd_r).transpose(2, 0, 1)
                xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
                pv = np.zeros((NC, 512, 512), np.int32)
                grid = [(r0, c0) for r0 in STARTS for c0 in STARTS]
                for i in range(0, len(grid), 64):
                    g = grid[i:i + 64]
                    xb = torch.stack([xt[:, r0:r0 + CROP, c0:c0 + CROP]
                                      for r0, c0 in g])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pr = model(xb).float().argmax(1).cpu().numpy()
                    for k, (r0, c0) in enumerate(g):
                        for kk in range(NC):
                            pv[kk, r0:r0 + CROP, c0:c0 + CROP] += (pr[k] == kk)
                # one decision per source pixel: majority vote of covering crops
                covered = pv.sum(0) > 0
                pred = pv.argmax(0)
                m = covered & (y_ref[ci] != IGNORE)
                if m.any():
                    t = y_ref[ci][m].astype(np.int64); q = pred[m]
                    cm += np.bincount(t * NC + q, minlength=NC ** 2).reshape(NC, NC)
        rec = dict(cache_provenance)
        rec.update({"alpha": tm.get("alpha"),
               "miou_vs_expert": float(np.nanmean(iou_from_cm(cm))),
               "pix_acc_vs_expert": float(np.trace(cm) / max(cm.sum(), 1)),
               "cm": cm.tolist()})
        (R / "damage").mkdir(parents=True, exist_ok=True)
        (R / "damage" / (name + ".json")).write_text(json.dumps(rec, indent=1))
        out[(alpha, event)].append(rec)
        print("  {:34s} mIoU vs expert {:.4f}".format(name, rec["miou_vs_expert"]),
              flush=True)

    alphas = sorted({k[0] for k in out}, key=lambda x: (isinstance(x, str), x))
    events = sorted({k[1] for k in out})
    full = [e for e in events if all((a, e) in out for a in alphas)]

    # State the seeds per cell, so a cell silently collapsing to one run is visible
    # in the output instead of being discoverable only by comparing two files.
    sizes = {len(v) for k, v in out.items() if k[1] in full}
    scored = sum(len(v) for k, v in out.items() if k[1] in full)
    print("cells {}, seeds per cell {}, runs represented {} of {}".format(
        len(alphas) * len(full),
        sorted(sizes)[0] if len(sizes) == 1 else "UNEVEN " + str(sorted(sizes)),
        scored, len(runs)))

    print("\n" + "=" * 94)
    print("DOES THE ARTEFACT COST ANYTHING AGAINST HUMAN TRUTH?")
    print("  mIoU against expert labels, mosaic scored: one decision per source")
    print("  pixel by majority vote of covering crops, scored once")
    print("=" * 94)
    print("{:12s}".format("event") + "".join(
        "{:>11s}".format(a if isinstance(a, str) else "a=" + format(a, ".2f"))
        for a in alphas))
    for e in full:
        print("{:12s}".format(e[:12]) + "".join(
            "{:>11s}".format("{:.4f}".format(cell(out, a, e)))
            for a in alphas))
    print("-" * 94)
    means = [np.mean([cell(out, a, e) for e in full]) for a in alphas]
    print("{:12s}".format("mean") + "".join(
        "{:>11s}".format("{:.4f}".format(m)) for m in means))

    # The J1 arms are named; alphabetical column order is not a treatment order.
    # The current paper reports only descriptive by-name contrasts, which the exact
    # product summary computes after seed averaging.
    if not all(isinstance(a, str) for a in alphas):
        raise SystemExit("non-J1 numeric dial arm reached the J1 damage scorer")
    print("\n  columns are named arms, not an inferential dial: by-name descriptive")
    print("  contrasts are computed in s1fc_j1_summary.py after seed averaging.")

    (R / "damage_summary.json").write_text(json.dumps(
        {"alphas": alphas, "events": full,
         "miou_vs_expert": {str(a): [cell(out, a, e) for e in full]
                            for a in alphas},
         "cost_mean": None, "cost_t": None},
        indent=1))
    print("\nwrote " + str(R / "damage_summary.json"))


if __name__ == "__main__":
    main()
