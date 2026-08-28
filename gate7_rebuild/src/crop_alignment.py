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
an estimated one. A nonzero kappa certifies that the predictor is crop-dependent,
which is all the proposition gives. It does NOT by itself say the model learned
the crop-conditional generator: zero padding, truncated receptive fields and any
normalisation over the crop make a network crop-dependent on their own. That is
what the matched control is for, and even with it the two label sets differ in
boundary structure and class balance, so the contrast supports attribution to the
labeller without isolating it. The paper says this; so should this file.

Also reported, because it is nearly free and checks the pipeline:

    Omega = E_p [ P( yhat(p|c) != yhat(p|c') ) ]   over crop pairs covering p

the model's own crop-dependence. Omega is exactly 0 for the two-parameter
threshold, since V is a per-pixel function and the thresholds are per-fold
constants. A U-Net trained on crop-invariant labels still has Omega > 0 from
padding and boundary effects; that is the nuisance floor, measured rather than
assumed, and it is why the scene-label model is the control rather than zero.
"""
import argparse, hashlib, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_fingerprint import code_digest, scoring_data_digest

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
CACHE = Path("/dev/shm/seaice_expanded")
SCENE = W / "work" / "scene_all"
SUB = W / "work" / "expansion" / "tokens_subsample.parquet"
FEATJSON = W / "work" / "expansion" / "feature_cols.json"
NC, IGNORE, HALF = 3, 255, 64
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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

    rd = W / "runs" / args.run
    metrics_path = rd / "test_metrics.json"
    checkpoint_path = rd / "best.pt"
    metrics_bytes = metrics_path.read_bytes()
    metadata_before = hashlib.sha256(metrics_bytes).hexdigest()
    tm = json.loads(metrics_bytes.decode("utf-8-sig"))
    acq, label_set = tm["holdout_acq"], tm["labels"]
    photon_mode = tm.get("photon")
    if photon_mode not in ("none", "true"):
        raise SystemExit(
            "crop_alignment.py supports optical-only and real-photon runs; "
            "{} records photon={!r}, whose training-time feature transformation "
            "is not replayed by this scorer".format(args.run, photon_mode))
    if acq is None:
        raise SystemExit(args.run + " is not a leave-one-acquisition-out run")

    eval_labels = args.eval_labels or label_set
    lf = {"scene": "labels.npy", "original": "labels_original.npy",
          "kmeans": "labels_kmeans.npy"}[eval_labels]
    # Bind the exact bytes before any schema, table, array, mask or checkpoint is
    # consumed. The checkpoint is loaded from these captured bytes, not reopened.
    scoring_data_before = scoring_data_digest(
        str(CACHE), str(SUB), str(FEATJSON), str(SCENE), eval_labels)
    scoring_code_before = code_digest(
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("train.py")),
        str(Path(__file__).with_name("data_fingerprint.py")))
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_before = hashlib.sha256(checkpoint_bytes).hexdigest()
    if tm.get("checkpoint_sha256") != checkpoint_before:
        raise SystemExit("best.pt is not the checkpoint bound by test_metrics.json")

    from validate_primary_runs import problems as run_problems
    invalid_run = run_problems(args.run)
    if invalid_run:
        raise SystemExit("refusing invalid training run {}: {}".format(
            args.run, "; ".join(invalid_run[:6])))

    sys.path.insert(0, str(W / "src"))
    from train import FusionNet

    dev = torch.device("cuda")
    feat_cols = json.loads(FEATJSON.read_text())["feature_cols"]
    tok = pd.read_parquet(SUB).reset_index(drop=True)
    model = FusionNet(len(feat_cols), use_photon=(tm["photon"] != "none")).to(dev)
    checkpoint_record = torch.load(io.BytesIO(checkpoint_bytes), map_location=dev,
                                   weights_only=False)
    if checkpoint_record.get("training_run_id") != tm.get("training_run_id"):
        raise SystemExit("checkpoint invocation ID does not match training metadata")
    model.load_state_dict(checkpoint_record["model"])
    model.eval()

    X = np.nan_to_num(tok[feat_cols].to_numpy(np.float32), nan=0.0, posinf=0.0,
                      neginf=0.0)
    tr = (tok["acq_id"] != acq).to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-6] = 1.0
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)

    imgs = np.load(CACHE / "images.npy", mmap_mode="r")
    plabs = np.load(CACHE / lf, mmap_mode="r")

    test = tok[tok["acq_id"] == acq]
    scenes = sorted(test.groupby(["tile", "s2_datetime"]).groups.keys())
    print("{}: acq={} trained_on={} evaluated_on={} scenes={}".format(
        args.run, acq, label_set, eval_labels, len(scenes)), flush=True)

    own = other = n_inst = 0.0
    pairs_tot = pairs_dis = 0.0
    n_A = n_cov = 0
    # Two weightings, both reported. own/other/n_inst average over (pixel, crop)
    # INSTANCES, so a pixel covered by m(p) crops carries m(p) times the weight.
    # Here that is not a small effect: sea-ice coverage runs from 1 to over 100
    # crops per source pixel, because the patches follow the satellite track.
    # kp_* average over SOURCE PIXELS instead, one contribution each. Proposition 1
    # is untouched: its proof kills the inner sum over crops for each p separately,
    # so any per-pixel weight still gives exactly zero.
    kp_num = kp_den = 0.0
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
        acc = np.zeros((H, Wd), np.float64)   # per-scene (own - other) per pixel

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
                    # basic slicing gives a view, so this writes through to acc
                    acc[r0:r1, c0:c1][m] += hit_own - (vsel - hit_own) / denom
                    for kk in range(NC):
                        pvotes[kk, r0:r1, c0:c1] += (yh == kk)

        # one contribution per source pixel in A, not one per (pixel, crop)
        if split.any():
            kp_num += float((acc[split] / cnt[split].astype(np.float64)).sum())
            kp_den += int(split.sum())

        # model's own crop-dependence over the covered pixels
        pc = pvotes[:, valid & (cnt > 0)].astype(np.float64)
        n_p = pc.sum(0)
        ok = n_p > 1
        agree = (pc[:, ok] * (pc[:, ok] - 1)).sum(0)
        tot = n_p[ok] * (n_p[ok] - 1)
        pairs_tot += tot.sum()
        pairs_dis += (tot - agree).sum()

    kappa = (own - other) / max(n_inst, 1)
    kappa_pixel = kp_num / max(kp_den, 1)
    print("  covered pixels {:,}   artefact set A {:,}  ({:.2%} of covered)".format(
        n_cov, n_A, n_A / max(n_cov, 1)))
    print("  P(pred = own crop label   | A) = {:.4f}".format(own / max(n_inst, 1)))
    print("  P(pred = other crop label | A) = {:.4f}".format(other / max(n_inst, 1)))
    print("  kappa = {:+.4f} (crop-read weighted)   kappa_pixel = {:+.4f}   "
          "(both structurally 0 for any crop-invariant predictor)".format(
              kappa, kappa_pixel))
    print("  Omega = {:.4f}   (model crop-dependence; 0 exactly for the threshold)".format(
        pairs_dis / max(pairs_tot, 1)))

    input_mode = ("optical-only" if photon_mode == "none" else
                  "optical+ICESat-2 photon")
    scoring_data_digest.cache_clear()
    scoring_data_after = scoring_data_digest(
        str(CACHE), str(SUB), str(FEATJSON), str(SCENE), eval_labels)
    if scoring_data_after != scoring_data_before:
        raise RuntimeError("scoring inputs changed while inference was in progress")
    code_digest.cache_clear()
    scoring_code_after = code_digest(
        str(Path(__file__).resolve()),
        str(Path(__file__).with_name("train.py")),
        str(Path(__file__).with_name("data_fingerprint.py")))
    if scoring_code_after != scoring_code_before:
        raise RuntimeError("scoring/model code changed while inference was in progress")
    if sha256_file(checkpoint_path) != checkpoint_before:
        raise RuntimeError("checkpoint changed while inference was in progress")
    if sha256_file(metrics_path) != metadata_before:
        raise RuntimeError("training metadata changed while inference was in progress")
    out = {"run": args.run, "acq": acq, "labels": label_set,
           "eval_labels": eval_labels, "seed": tm["seed"],
           "photon_mode": photon_mode, "input_mode": input_mode,
           "optical_normalisation": "fixed ImageNet RGB mean/std",
           "photon_normalisation": ("not used" if photon_mode == "none" else
                                     "fold-local non-test-acquisition mean/std"),
           # Bind a reusable score to the exact model and training record.  A run
           # directory can be retrained under the same name; filenames alone are
           # not cache keys.
           "checkpoint_sha256": checkpoint_before,
           "training_metadata_sha256": metadata_before,
           "token_table_sha256": sha256_file(SUB),
           "feature_schema_sha256": sha256_file(FEATJSON),
           "scoring_data_sha256": scoring_data_before,
           "scoring_code_sha256": scoring_code_before,
           "n_covered": n_cov, "n_artefact": n_A,
           "frac_artefact": n_A / max(n_cov, 1),
           "p_own": own / max(n_inst, 1), "p_other": other / max(n_inst, 1),
           "kappa": kappa, "kappa_pixel": kappa_pixel,
           "n_pixels_A": int(kp_den),
           "omega": pairs_dis / max(pairs_tot, 1),
           "n_instances": int(n_inst)}
    d = W / "runs" / "kappa"
    d.mkdir(parents=True, exist_ok=True)
    (d / (args.run + args.tag + ".json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
