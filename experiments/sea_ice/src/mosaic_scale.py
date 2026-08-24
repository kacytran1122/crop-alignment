"""Mosaic evaluation at scale, with BOTH arms on an identical pixel set.

This closes the most damaging objection to the paper, which is that the paper does
not follow its own advice. Recommendation 5 says score each source pixel once, by
mosaicking. Every result at scale instead accumulated over patch pixels, at a
measured reuse of 13.97x. An audit that violates its own protocol is the easiest
possible reject.

Two further defects fall out for free, because they share a cause:

  - the U-Net was scored on all 16,384 pixels of every patch while the threshold
    was scored on 120 random pixels per patch, so the two arms never saw the same
    sample;
  - the trainer scored an absent class as IoU 0.0 and the threshold scored it as
    NaN under nanmean, so on folds missing a class the two arms were divided by
    different denominators.

One canvas per scene fixes all three. Softmax is accumulated per source pixel and
argmaxed once; the brightness canvas is written from the same patches and
thresholded with the parameters fitted for that fold; both are scored on exactly
the pixels that are covered, valid, and labelled, under one convention. Whatever
the premium is after this, it is computed under the rules the paper asks of
others.

Empty classes are reported both ways so the choice is visible instead of buried.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

W = Path(__file__).resolve().parents[1]
CACHE = Path("/dev/shm/seaice_expanded")
SCENE = W / "work" / "scene_all"
SUB = W / "work" / "expansion" / "tokens_subsample.parquet"
FEATJSON = W / "work" / "expansion" / "feature_cols.json"
THR = W / "results_threshold_expanded.json"
NC, IGNORE, HALF = 3, 255, 64
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def ious(cm):
    cm = cm.astype(np.float64)
    out = []
    for c in range(NC):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        d = tp + fp + fn
        out.append(tp / d if d > 0 else np.nan)
    return np.array(out)


def summarise(cm):
    v = ious(cm)
    present = ~np.isnan(v)
    return {"miou_nanmean": float(np.nanmean(v)) if present.any() else float("nan"),
            "miou_zerofill": float(np.nan_to_num(v, nan=0.0).mean()),
            "per_iou": [None if np.isnan(x) else float(x) for x in v],
            "n_present": int(present.sum()),
            "pix_acc": float(np.trace(cm) / max(cm.sum(), 1)),
            "n_pixels": int(cm.sum()), "cm": cm.astype(np.int64).tolist()}


def confusion(y_t, y_p):
    return np.bincount(y_t * NC + y_p, minlength=NC ** 2).reshape(NC, NC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the draw convention, not for the model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(W / "src"))
    from train import FusionNet

    rd = W / "runs" / args.run
    tm = json.loads((rd / "test_metrics.json").read_text())
    acq, label_set = tm["holdout_acq"], tm["labels"]
    if acq is None:
        raise SystemExit(args.run + " is not a leave-one-acquisition-out run")

    dev = torch.device("cuda")
    feat_cols = json.loads(FEATJSON.read_text())["feature_cols"]
    tok = pd.read_parquet(SUB).reset_index(drop=True)

    model = FusionNet(len(feat_cols), use_photon=(tm["photon"] != "none")).to(dev)
    ck = torch.load(rd / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()

    # feature normalisation must match training: fitted on the non-holdout rows
    X = np.nan_to_num(tok[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    tr = (tok["acq_id"] != acq).to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-6] = 1.0
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)

    imgs = np.load(CACHE / "images.npy", mmap_mode="r")
    # Ground truth must come from the SAME label set the model was trained on.
    # The scene canvas in work/scene_all only exists for the crop-invariant
    # scheme; the per-crop scheme has no canvas by construction, because a pixel
    # covered by several crops can carry several labels. That disagreement is the
    # phenomenon under study, so resolve it by vote and report how often the
    # crops actually agree.
    lf = {"scene": "labels.npy", "original": "labels_original.npy",
          "kmeans": "labels_kmeans.npy"}[label_set]
    plabs = np.load(CACHE / lf, mmap_mode="r")
    thr = json.loads(THR.read_text())[acq][label_set]
    tw, tt = thr["t_water"], thr["t_thick"]

    test = tok[tok["acq_id"] == acq]
    scenes = sorted(test.groupby(["tile", "s2_datetime"]).groups.keys())
    print("{}: acq={} labels={} patches={:,} scenes={} t_water={} t_thick={}".format(
        args.run, acq, label_set, len(test), len(scenes), tw, tt), flush=True)

    cm_u = np.zeros((NC, NC), np.int64)
    cm_t = np.zeros((NC, NC), np.int64)
    cm_ud = np.zeros((NC, NC), np.int64)
    cm_td = np.zeros((NC, NC), np.int64)
    mean_d, std_d = IM_MEAN.to(dev), IM_STD.to(dev)
    total_patch_px = 0
    vote_tot = vote_top = 0
    rng = np.random.default_rng(args.seed)

    for tile, dt in scenes:
        stem = tile + "_" + pd.Timestamp(dt).strftime("%Y%m%dT%H%M%S")
        vp = SCENE / (stem + "_valid.npy")
        if not vp.exists():
            cand = sorted(SCENE.glob(tile + "_*_valid.npy"))
            if len(cand) != 1:
                print("  skip {}: {} candidate validity masks".format(stem, len(cand)),
                      flush=True)
                continue
            vp = cand[0]
        valid = np.load(vp)
        H, Wd = valid.shape

        sel = test.index[(test["tile"] == tile) & (test["s2_datetime"] == dt)].to_numpy()
        prob = np.zeros((NC, H, Wd), np.float32)
        votes = np.zeros((NC, H, Wd), np.uint16)
        cnt = np.zeros((H, Wd), np.uint16)
        vcanvas = np.zeros((H, Wd), np.int16)

        with torch.no_grad():
            for i in range(0, len(sel), args.batch):
                ids = sel[i:i + args.batch]
                raw = np.ascontiguousarray(imgs[ids])
                im = torch.from_numpy(raw).to(dev)
                im = im.permute(0, 3, 1, 2).float().div_(255).sub_(mean_d).div_(std_d)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = F.softmax(model(im, Xn[ids]).float(), dim=1)
                p = p.float().cpu().numpy()
                v = raw[..., :3].max(axis=3).astype(np.int16)   # V = max over RGB
                pl = np.ascontiguousarray(plabs[ids])
                for k, t in enumerate(ids):
                    y, x = int(tok.at[t, "pix_y"]), int(tok.at[t, "pix_x"])
                    r0, r1 = max(y - HALF, 0), min(y + HALF, H)
                    c0, c1 = max(x - HALF, 0), min(x + HALF, Wd)
                    if r1 <= r0 or c1 <= c0:
                        continue
                    a0, b0 = r0 - (y - HALF), c0 - (x - HALF)
                    prob[:, r0:r1, c0:c1] += p[k][:, a0:a0 + (r1 - r0), b0:b0 + (c1 - c0)]
                    cnt[r0:r1, c0:c1] += 1
                    vcanvas[r0:r1, c0:c1] = v[k][a0:a0 + (r1 - r0), b0:b0 + (c1 - c0)]
                    sub = pl[k][a0:a0 + (r1 - r0), b0:b0 + (c1 - c0)]
                    for c in range(NC):
                        votes[c, r0:r1, c0:c1] += (sub == c)
                total_patch_px += len(ids) * 128 * 128

        vtot = votes.sum(axis=0)
        score = (cnt > 0) & valid & (vtot > 0)
        if not score.any():
            continue
        vote_tot += int(vtot[score].sum())
        vote_top += int(votes.max(0)[score].sum())

        # Two ground-truth conventions, because neither is obviously correct and
        # the difference between them is itself a result.
        #
        #   vote  -- majority label over the crops covering the pixel. Scores each
        #            pixel once, but the vote REPAIRS part of the crop-dependence
        #            before it is measured, which biases the premium DOWN.
        #   draw  -- one covering crop chosen at random per pixel, with
        #            probability equal to its share of the votes. Also scores each
        #            pixel once, but preserves the crop-dependence in expectation,
        #            which is the property under study.
        #
        # For the crop-invariant scheme the two are identical by construction,
        # which is a free check on the sampler.
        v_sorted = votes[:, score].astype(np.float64)
        lab_vote = votes.argmax(0)[score].astype(np.int64)
        cum = np.cumsum(v_sorted, axis=0)
        u = rng.random(cum.shape[1]) * cum[-1]
        lab_draw = (u[None, :] >= cum).sum(axis=0).clip(0, NC - 1).astype(np.int64)

        pred_u = prob.argmax(0)[score].astype(np.int64)
        vv = vcanvas[score]
        pred_t = np.where(vv <= tw, 2, np.where(vv >= tt, 0, 1)).astype(np.int64)
        cm_u += confusion(lab_vote, pred_u)
        cm_t += confusion(lab_vote, pred_t)
        cm_ud += confusion(lab_draw, pred_u)
        cm_td += confusion(lab_draw, pred_t)

    u, t = summarise(cm_u), summarise(cm_t)
    ud, td = summarise(cm_ud), summarise(cm_td)
    reuse = total_patch_px / max(int(cm_u.sum()), 1)
    unanimity = vote_top / max(vote_tot, 1)
    print("  unique source pixels {:,}   patch-pixel reuse avoided {:.2f}x".format(
        int(cm_u.sum()), reuse))
    print("  crop label unanimity {:.4f}  (1.0 means every crop covering a pixel "
          "agreed)".format(unanimity))
    for nm, r in (("U-Net  vote", u), ("thresh vote", t),
                  ("U-Net  draw", ud), ("thresh draw", td)):
        print("  {:12s} mIoU nanmean {:.4f}  zerofill {:.4f}  present {}".format(
            nm, r["miou_nanmean"], r["miou_zerofill"], r["n_present"]))
    print("  advantage  vote {:+.4f}   draw {:+.4f}".format(
        u["miou_nanmean"] - t["miou_nanmean"],
        ud["miou_nanmean"] - td["miou_nanmean"]))

    out = {"run": args.run, "acq": acq, "labels": label_set, "seed": tm["seed"],
           "disjoint_tiles": tm.get("disjoint_tiles", False),
           "t_water": tw, "t_thick": tt, "reuse_avoided": reuse,
           "crop_unanimity": unanimity,
           "unet": u, "threshold": t, "unet_draw": ud, "threshold_draw": td,
           "advantage_nanmean": u["miou_nanmean"] - t["miou_nanmean"],
           "advantage_zerofill": u["miou_zerofill"] - t["miou_zerofill"],
           "advantage_draw": ud["miou_nanmean"] - td["miou_nanmean"],
           "advantage_draw_zerofill": ud["miou_zerofill"] - td["miou_zerofill"]}
    dest = Path(args.out) if args.out else (W / "runs" / "mosaic" / (args.run + ".json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
