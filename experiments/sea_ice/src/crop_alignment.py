"""Does the model reproduce the label of the crop it is currently looking at?

The paper claims a deep model is partly paid to reproduce the label generator
rather than the ice. So far that has been argued through a fourth-order contrast
of four aggregate mIoU numbers, whose noise is dominated by how well a brightness
threshold happens to fit each scene -- a quantity with nothing to do with the
claim. At 17 acquisitions that contrast cannot resolve the effect, and no
reweighting of it can: the required n is in the hundreds.

The claim itself is about a per-pixel object that is directly observable.

Because crops overlap about 33-fold, one source pixel p is seen inside many crops
c. Under the per-crop labelling scheme those crops can disagree about p, since
Otsu and two min-max normalisations are recomputed per crop. Let

    A = { p : the crops covering p do not all agree on its label }

For p in A, ask what the model predicts when it reads p through crop c:

    kappa = P( yhat(p|c) = y_c(p)  | p in A )          <- the crop it is reading
          - P( yhat(p|c) = y_c'(p) | p in A, c' != c ) <- a different crop

The physical state of p does not depend on which crop it is viewed through. So for
ANY crop-invariant predictor kappa is exactly zero. It is a structural null, not
an estimated one, and kappa > 0 admits no explanation except that the model has
learned the crop-conditional generator.

Also reported, because it is nearly free and checks the pipeline:

    Omega = E_p [ P( yhat(p|c) != yhat(p|c') ) ]   over crop pairs covering p

the model's own crop-dependence. Omega is exactly 0 for the two-parameter
threshold, since V is a per-pixel function and the thresholds are per-fold
constants. A U-Net trained on crop-invariant labels still has Omega > 0 from
padding and boundary effects; that is the nuisance floor, measured rather than
assumed, and it is why the scene-label model is the control rather than zero.
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
NC, IGNORE, HALF = 3, 255, 64
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def place(y, x, H, Wd):
    r0, r1 = max(y - HALF, 0), min(y + HALF, H)
    c0, c1 = max(x - HALF, 0), min(x + HALF, Wd)
    return r0, r1, c0, c1, r0 - (y - HALF), c0 - (x - HALF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--eval-labels", default=None,
                    choices=["scene", "original", "kmeans"],
                    help="label set that defines the artefact set A and the "
                         "per-crop labels, if different from the one the model "
                         "was trained on. Running a scene-trained model against "
                         "the original label set is the negative control: it has "
                         "never seen a crop-dependent label, so its kappa should "
                         "sit at the structural null even on pixels where the "
                         "crops disagree.")
    ap.add_argument("--tag", default="")
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
    model.load_state_dict(torch.load(rd / "best.pt", map_location=dev,
                                     weights_only=False)["model"])
    model.eval()

    X = np.nan_to_num(tok[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0,
                      neginf=0.0)
    tr = (tok["acq_id"] != acq).to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-6] = 1.0
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)

    imgs = np.load(CACHE / "images.npy", mmap_mode="r")
    eval_labels = args.eval_labels or label_set
    lf = {"scene": "labels.npy", "original": "labels_original.npy",
          "kmeans": "labels_kmeans.npy"}[eval_labels]
    plabs = np.load(CACHE / lf, mmap_mode="r")

    test = tok[tok["acq_id"] == acq]
    scenes = sorted(test.groupby(["tile", "s2_datetime"]).groups.keys())
    print("{}: acq={} trained_on={} evaluated_on={} scenes={}".format(
        args.run, acq, label_set, eval_labels, len(scenes)), flush=True)

    own = other = n_inst = 0.0
    pairs_tot = pairs_dis = 0.0
    n_A = n_cov = 0
    mean_d, std_d = IM_MEAN.to(dev), IM_STD.to(dev)

    for tile, dt in scenes:
        stem = tile + "_" + pd.Timestamp(dt).strftime("%Y%m%dT%H%M%S")
        vp = SCENE / (stem + "_valid.npy")
        if not vp.exists():
            cand = sorted(SCENE.glob(tile + "_*_valid.npy"))
            if len(cand) != 1:
                continue
            vp = cand[0]
        valid = np.load(vp)
        H, Wd = valid.shape
        sel = test.index[(test["tile"] == tile) & (test["s2_datetime"] == dt)].to_numpy()

        # pass 1: label votes per source pixel, no GPU
        votes = np.zeros((NC, H, Wd), np.uint16)
        for t in sel:
            y, x = int(tok.at[t, "pix_y"]), int(tok.at[t, "pix_x"])
            r0, r1, c0, c1, a0, b0 = place(y, x, H, Wd)
            if r1 <= r0 or c1 <= c0:
                continue
            sub = plabs[t][a0:a0 + (r1 - r0), b0:b0 + (c1 - c0)]
            for k in range(NC):
                votes[k, r0:r1, c0:c1] += (sub == k)
        cnt = votes.sum(0).astype(np.int32)
        split = (cnt > 0) & (votes.max(0) < cnt) & valid      # the artefact set A
        n_A += int(split.sum())
        n_cov += int(((cnt > 0) & valid).sum())

        # pass 2: model predictions per crop, scored against own vs other crops
        pvotes = np.zeros((NC, H, Wd), np.uint16)
        with torch.no_grad():
            for i in range(0, len(sel), args.batch):
                ids = sel[i:i + args.batch]
                im = torch.from_numpy(np.ascontiguousarray(imgs[ids])).to(dev)
                im = im.permute(0, 3, 1, 2).float().div_(255).sub_(mean_d).div_(std_d)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = model(im, Xn[ids]).float().argmax(1)
                pr = pr.cpu().numpy().astype(np.uint8)
                pl = np.ascontiguousarray(plabs[ids])
                for k, t in enumerate(ids):
                    y, x = int(tok.at[t, "pix_y"]), int(tok.at[t, "pix_x"])
                    r0, r1, c0, c1, a0, b0 = place(y, x, H, Wd)
                    if r1 <= r0 or c1 <= c0:
                        continue
                    hh, ww = r1 - r0, c1 - c0
                    yh = pr[k][a0:a0 + hh, b0:b0 + ww]
                    yc = pl[k][a0:a0 + hh, b0:b0 + ww]
                    m = split[r0:r1, c0:c1]
                    if not m.any():
                        for kk in range(NC):
                            pvotes[kk, r0:r1, c0:c1] += (yh == kk)
                        continue
                    yh_m, yc_m = yh[m], yc[m]
                    nm = cnt[r0:r1, c0:c1][m]
                    # how many crops covering p assign p the class the model
                    # predicted while reading THIS crop
                    vsel = np.take_along_axis(
                        votes[:, r0:r1, c0:c1][:, m], yh_m[None, :].astype(np.intp),
                        axis=0)[0].astype(np.float64)
                    hit_own = (yh_m == yc_m).astype(np.float64)
                    denom = np.maximum(nm - 1, 1).astype(np.float64)
                    own += hit_own.sum()
                    other += ((vsel - hit_own) / denom).sum()
                    n_inst += hit_own.size
                    for kk in range(NC):
                        pvotes[kk, r0:r1, c0:c1] += (yh == kk)

        # model's own crop-dependence over the covered pixels
        pc = pvotes[:, valid & (cnt > 0)].astype(np.float64)
        n_p = pc.sum(0)
        ok = n_p > 1
        agree = (pc[:, ok] * (pc[:, ok] - 1)).sum(0)
        tot = n_p[ok] * (n_p[ok] - 1)
        pairs_tot += tot.sum()
        pairs_dis += (tot - agree).sum()

    kappa = (own - other) / max(n_inst, 1)
    print("  covered pixels {:,}   artefact set A {:,}  ({:.2%} of covered)".format(
        n_cov, n_A, n_A / max(n_cov, 1)))
    print("  P(pred = own crop label   | A) = {:.4f}".format(own / max(n_inst, 1)))
    print("  P(pred = other crop label | A) = {:.4f}".format(other / max(n_inst, 1)))
    print("  kappa = {:+.4f}   (structurally 0 for any crop-invariant predictor)".format(
        kappa))
    print("  Omega = {:.4f}   (model crop-dependence; 0 exactly for the threshold)".format(
        pairs_dis / max(pairs_tot, 1)))

    out = {"run": args.run, "acq": acq, "labels": label_set,
           "eval_labels": eval_labels, "seed": tm["seed"],
           "n_covered": n_cov, "n_artefact": n_A,
           "frac_artefact": n_A / max(n_cov, 1),
           "p_own": own / max(n_inst, 1), "p_other": other / max(n_inst, 1),
           "kappa": kappa, "omega": pairs_dis / max(pairs_tot, 1),
           "n_instances": int(n_inst)}
    d = W / "runs" / "kappa"
    d.mkdir(parents=True, exist_ok=True)
    (d / (args.run + args.tag + ".json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
