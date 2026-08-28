"""Crop alignment in the flood domain, with the artefact set held fixed.

Same estimand as the sea-ice pass. For a source pixel covered by several crops that
disagree about its label, kappa asks how much more often the model predicts the
label belonging to the crop it is currently reading than the label belonging to a
different crop covering the same pixel:

    kappa = P(pred = own crop label | A) - P(pred = other crop label | A)

and it is 0 for any crop-invariant predictor, for a reason that owes nothing to
this dataset: such a predictor returns the same class regardless of which crop it
reads, so which covering crop is called "own" cannot change the answer, and the
two probabilities are the same quantity computed twice.

The artefact set A is defined ONCE, by the predominantly crop-fitted ``a100``
field, and every model is scored on it whatever arm it trained on. That matters. If A
were rebuilt per alpha it would shrink as alpha fell and vanish at alpha = 0,
leaving the control with no pixels to be null on and no number to report. Holding A
fixed is also what the sea-ice control does: the scene-trained model is evaluated
against the original label set, on pixels it was never trained to reproduce.

The ``a000`` model, trained on published per-event crop-invariant labels, is an
empirical negative control. A network can still be crop-sensitive for other reasons,
so this control is expected near zero rather than forced to zero. Only the
``--predictor threshold`` branch is crop-invariant by construction and therefore
forced algebraically to zero.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
C = W / "data" / "sen1floods11" / "cache"
CR = C / "crop"
IGNORE, NC = 255, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grid-step", type=int, default=1,
                    help="take every Nth crop start position. 1 is the grid used "
                         "throughout the paper (stride 32); 2 gives stride 64 and "
                         "4 gives stride 128. Sparser grids are strict subsets, so "
                         "no thresholds are recomputed.")
    ap.add_argument("--predictor", default="model",
                    choices=["model", "threshold"],
                    help="threshold runs the estimator on a per-pixel cut of the "
                         "chip-level smoothed VH, which is crop-invariant by "
                         "construction, so the proposition forces kappa = 0 and "
                         "any nonzero result is a bug in this file.")
    ap.add_argument("--eval-arm", default="a100",
                    help="arm whose labels define the artefact set A; every run is "
                         "scored on the same A so the arms are comparable")
    ap.add_argument("--eval-alpha", type=float, default=1.0,
                    help="alpha whose label set defines the artefact set A and "
                         "the per-crop labels. Fixed at 1.0 so every arm is "
                         "scored on the same pixels.")
    args = ap.parse_args()

    import segmentation_models_pytorch as smp
    dev = torch.device("cuda")
    ARMS = CR / "arms"
    rd = W / "runs_s1fc" / args.run
    tm = json.loads((rd / "test_metrics.json").read_text())
    event = tm["holdout_event"]
    # named arms carry arm=<name> and alpha=None; dial runs carry alpha and no arm
    arm = tm.get("arm")
    alpha = tm.get("alpha")
    label = arm if arm is not None else alpha

    meta = json.loads((C / "meta.json").read_text())
    cmeta = json.loads((CR / "meta.json").read_text())
    CROP, STARTS_ALL = cmeta["crop"], cmeta["starts"]
    step = max(1, int(args.grid_step))
    # keep the index into th_crop aligned with the subsampled start positions
    IDX = list(range(0, len(STARTS_ALL), step))
    STARTS = [STARTS_ALL[i] for i in IDX]
    ns = len(STARTS)

    ev = np.load(C / "events.npy")
    y_ref = np.load(C / "y_ref.npy")
    th_chip = np.load(CR / "th_chip.npy")
    th_crop = np.load(CR / "th_crop.npy")
    usable = np.isfinite(th_chip) & np.isfinite(th_crop).all(axis=(1, 2))
    ti = np.where((ev == event) & usable)[0]
    if not len(ti):
        raise SystemExit("no usable chips for " + event)

    imgs = np.load(C / "images.npy", mmap_mode="r")
    # Normalise the way the model was TRAINED. Training derives fold-local mu/sd
    # from the training chips alone, so the held-out event never enters them, and
    # records them per run. Scoring with the global cache statistics fed every
    # model inputs shifted by about two global standard deviations; s1fc_damage.py
    # already read the per-run values back, this scorer did not.
    if tm.get("norm_mean") is not None and tm.get("norm_std") is not None:
        mu = np.array(tm["norm_mean"], np.float32)
        sd = np.array(tm["norm_std"], np.float32)
        norm_source = "fold-local"
    else:
        mu = np.array(meta["mean"], np.float32)
        sd = np.array(meta["std"], np.float32)
        norm_source = "global"
    vh_s = np.load(CR / "vh_smooth.npy")

    # Labels that DEFINE the artefact set. Fixed at the crop-fitted endpoint for
    # every run, so all arms are scored on identical pixels; rebuilding A per arm
    # would shrink it as crop-dependence fell and empty it at the zero setting,
    # which would make the negative control unmeasurable rather than negative.
    a = args.eval_alpha
    eval_arm = ARMS / ("th_" + args.eval_arm + ".npy")
    if eval_arm.exists():
        TH = np.load(eval_arm).astype(np.float32)
    else:
        TH = ((1.0 - a) * th_chip[:, None, None] + a * th_crop).astype(np.float32)

    model = None
    if args.predictor == "model":
        model = smp.Unet(tm["encoder"], encoder_weights=None, in_channels=2,
                         classes=NC).to(dev)
        model.load_state_dict(torch.load(rd / "best.pt", map_location=dev,
                                         weights_only=False)["model"])
        model.eval()

    print("{}: event={} arm={} evalA={} chips={} "
          "grid={}x{} (stride {})".format(
              args.run, event, label, args.eval_arm, len(ti), ns, ns,
              cmeta["stride"] * step), flush=True)

    own = other = n_inst = 0.0
    pairs_tot = pairs_dis = 0.0
    n_A = n_cov = 0
    # Two weightings of the same quantity, both reported.
    #
    # `own`/`other`/`n_inst` average over (pixel, crop) INSTANCES, so a source pixel
    # covered by m(p) crops carries m(p) times the weight of a pixel covered by two.
    # That is a real property of the estimator and the draft described it wrongly:
    # the 1/(m-1) inside the other-term equalises the crops within a pixel, not the
    # pixels within the average. Coverage here runs from 1 to over 100.
    #
    # kp_* average over SOURCE PIXELS instead: each pixel in A contributes the mean
    # of (own - other) over the crops covering it, once. Proposition 1 survives
    # unchanged, because its proof shows the inner sum over crops vanishes for each
    # p separately, so any per-pixel weight gives exactly zero.
    kp_num = kp_den = 0.0

    for ci in ti:
        valid = y_ref[ci] != IGNORE
        acc = np.zeros((512, 512), np.float64)   # per-chip (own - other) per pixel
        v = vh_s[ci].astype(np.float32)
        # per-crop labels for this chip, on the 512x512 canvas, one plane per crop
        votes = np.zeros((NC, 512, 512), np.uint16)
        labs = np.empty((ns, ns, CROP, CROP), np.uint8)
        for p, r0 in enumerate(STARTS):
            for q, c0 in enumerate(STARTS):
                sub = v[r0:r0 + CROP, c0:c0 + CROP]
                lab = (sub <= TH[ci, IDX[p], IDX[q]]).astype(np.uint8)
                labs[p, q] = lab
                m = valid[r0:r0 + CROP, c0:c0 + CROP]
                for k in range(NC):
                    votes[k, r0:r0 + CROP, c0:c0 + CROP] += ((lab == k) & m)
        cnt = votes.sum(0).astype(np.int32)
        split = (cnt > 0) & (votes.max(0) < cnt) & valid       # artefact set A
        n_A += int(split.sum())
        n_cov += int(((cnt > 0) & valid).sum())

        x = ((np.asarray(imgs[ci], np.float32) - mu) / sd).transpose(2, 0, 1)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev)

        pvotes = np.zeros((NC, 512, 512), np.uint16)
        grid = [(p, q) for p in range(ns) for q in range(ns)]
        with torch.no_grad():
            for i in range(0, len(grid), args.batch):
                g = grid[i:i + args.batch]
                if model is None:
                    # crop-invariant by construction: one pixel in, one class out,
                    # with no dependence on the window it was read through
                    pr = np.stack([
                        (v[STARTS[p]:STARTS[p] + CROP,
                           STARTS[q]:STARTS[q] + CROP] <= th_chip[ci])
                        for p, q in g]).astype(np.uint8)
                else:
                    xb = torch.stack([xt[:, STARTS[p]:STARTS[p] + CROP,
                                         STARTS[q]:STARTS[q] + CROP] for p, q in g])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        pr = model(xb).float().argmax(1).cpu().numpy().astype(np.uint8)
                for k, (p, q) in enumerate(g):
                    r0, c0 = STARTS[p], STARTS[q]
                    yh = pr[k]
                    yc = labs[p, q]
                    m = split[r0:r0 + CROP, c0:c0 + CROP]
                    vm = valid[r0:r0 + CROP, c0:c0 + CROP]
                    for kk in range(NC):
                        pvotes[kk, r0:r0 + CROP, c0:c0 + CROP] += ((yh == kk) & vm)
                    if not m.any():
                        continue
                    yh_m, yc_m = yh[m], yc[m]
                    nm = cnt[r0:r0 + CROP, c0:c0 + CROP][m]
                    # how many covering crops give p the class the model predicted
                    # while reading THIS crop
                    vsel = np.take_along_axis(
                        votes[:, r0:r0 + CROP, c0:c0 + CROP][:, m],
                        yh_m[None, :].astype(np.intp), axis=0)[0].astype(np.float64)
                    hit_own = (yh_m == yc_m).astype(np.float64)
                    denom = np.maximum(nm - 1, 1).astype(np.float64)
                    own += hit_own.sum()
                    other += ((vsel - hit_own) / denom).sum()
                    n_inst += hit_own.size
                    # basic slicing gives a view, so the masked add writes through
                    acc[r0:r0 + CROP, c0:c0 + CROP][m] += (
                        hit_own - (vsel - hit_own) / denom)

        # one contribution per source pixel in A, not one per (pixel, crop)
        if split.any():
            kp_num += float((acc[split] / cnt[split].astype(np.float64)).sum())
            kp_den += int(split.sum())

        pc = pvotes[:, valid & (cnt > 0)].astype(np.float64)
        n_p = pc.sum(0)
        ok = n_p > 1
        agree = (pc[:, ok] * (pc[:, ok] - 1)).sum(0)
        tot = n_p[ok] * (n_p[ok] - 1)
        pairs_tot += tot.sum()
        pairs_dis += (tot - agree).sum()

    kappa = (own - other) / max(n_inst, 1)
    kappa_pixel = kp_num / max(kp_den, 1)
    omega = pairs_dis / max(pairs_tot, 1)
    print("  covered {:,}   artefact set A {:,}  ({:.2%})".format(
        n_cov, n_A, n_A / max(n_cov, 1)))
    print("  P(pred = own crop label   | A) = {:.4f}".format(own / max(n_inst, 1)))
    print("  P(pred = other crop label | A) = {:.4f}".format(other / max(n_inst, 1)))
    print("  kappa = {:+.4f} (crop-read weighted)   kappa_pixel = {:+.4f}   "
          "Omega = {:.4f}".format(kappa, kappa_pixel, omega), flush=True)

    if args.predictor == "threshold":
        print("  predictor is crop-invariant by construction; the proposition")
        print("  forces kappa = 0 under either weighting. Measured |kappa| = {:.3e}, "
              "|kappa_pixel| = {:.3e}, |Omega| = {:.3e}".format(
                  abs(kappa), abs(kappa_pixel), abs(omega)))

    out = {"run": args.run, "event": event, "alpha": alpha, "arm": arm,
           "norm_source": norm_source,
           "predictor": args.predictor, "grid_step": step,
           "grid_stride": cmeta["stride"] * step, "n_crops_per_chip": ns * ns,
           "eval_alpha": a, "seed": tm["seed"], "n_covered": n_cov,
           "n_artefact": n_A, "frac_artefact": n_A / max(n_cov, 1),
           "p_own": own / max(n_inst, 1), "p_other": other / max(n_inst, 1),
           "kappa": kappa, "kappa_pixel": kappa_pixel,
           "n_pixels_A": int(kp_den), "omega": omega,
           "n_instances": int(n_inst)}
    d = W / "runs_s1fc" / "kappa"
    d.mkdir(parents=True, exist_ok=True)
    tag = "" if args.predictor == "model" else "_thr"
    if step != 1:
        tag += "_step{}".format(step)
    (d / (args.run + tag + ".json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
