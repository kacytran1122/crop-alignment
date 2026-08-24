"""Does the released file reproduce the pipeline the paper actually ran?

crop_alignment.py was written to be readable and dependency-light, separately from
s1f_crop_kappa.py which was written to be fast. Two implementations of the same
estimand are worth nothing if they disagree, and a released tool that does not
reproduce the paper's own numbers is worse than no tool at all.

This runs both over the same event, the same crop grid, the same model weights and
the same labels, and requires agreement to floating-point tolerance.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
C = W / "data" / "sen1floods11" / "cache"
CR = C / "crop"
IGNORE, NC = 255, 2
sys.path.insert(0, str(ROOT))
from crop_alignment import CropAlignment

RUN = sys.argv[1] if len(sys.argv) > 1 else "s1fc_a100_Bolivia_s42"


def main():
    import segmentation_models_pytorch as smp
    dev = torch.device("cuda")
    rd = W / "runs_s1fc" / RUN
    tm = json.loads((rd / "test_metrics.json").read_text())
    event = tm["holdout_event"]

    meta = json.loads((C / "meta.json").read_text())
    cmeta = json.loads((CR / "meta.json").read_text())
    CROP, STARTS = cmeta["crop"], cmeta["starts"]

    ev = np.load(C / "events.npy")
    y_ref = np.load(C / "y_ref.npy")
    th_chip = np.load(CR / "th_chip.npy")
    th_crop = np.load(CR / "th_crop.npy")
    usable = np.isfinite(th_chip) & np.isfinite(th_crop).all(axis=(1, 2))
    ti = np.where((ev == event) & usable)[0]

    imgs = np.load(C / "images.npy", mmap_mode="r")
    mu = np.array(meta["mean"], np.float32)
    sd = np.array(meta["std"], np.float32)
    vh_s = np.load(CR / "vh_smooth.npy")
    TH = th_crop.astype(np.float32)          # eval alpha = 1.0

    model = smp.Unet(tm["encoder"], encoder_weights=None, in_channels=2,
                     classes=NC).to(dev)
    model.load_state_dict(torch.load(rd / "best.pt", map_location=dev,
                                     weights_only=False)["model"])
    model.eval()

    ca = CropAlignment(NC)
    for ci in ti:
        valid = y_ref[ci] != IGNORE
        v = vh_s[ci].astype(np.float32)
        x = ((np.asarray(imgs[ci], np.float32) - mu) / sd).transpose(2, 0, 1)
        xt = torch.from_numpy(np.ascontiguousarray(x)).to(dev)

        boxes, labs = [], []
        for p, r0 in enumerate(STARTS):
            for q, c0 in enumerate(STARTS):
                boxes.append((r0, c0))
                labs.append((v[r0:r0 + CROP, c0:c0 + CROP]
                             <= TH[ci, p, q]).astype(np.int64))

        preds = []
        with torch.no_grad():
            for i in range(0, len(boxes), 64):
                bb = boxes[i:i + 64]
                xb = torch.stack([xt[:, r0:r0 + CROP, c0:c0 + CROP]
                                  for r0, c0 in bb])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = model(xb).float().argmax(1).cpu().numpy()
                preds.extend(list(pr))

        ca.add_scene((512, 512), valid)
        for b, lab in zip(boxes, labs):
            ca.add_labels(b, lab)
        ca.finalize_labels()
        for b, lab, pr in zip(boxes, labs, preds):
            ca.add_predictions(b, lab, pr)

    r = ca.result()
    ref = json.loads((W / "runs_s1fc" / "kappa" / (RUN + ".json")).read_text())

    print("=" * 74)
    print("crop_alignment.py (released)  vs  s1f_crop_kappa.py (pipeline)")
    print("  run {}   {} chips".format(RUN, len(ti)))
    print("=" * 74)
    rows = [("kappa", r["kappa"], ref["kappa"]),
            ("p_own", r["p_own"], ref["p_own"]),
            ("p_other", r["p_other"], ref["p_other"]),
            ("Omega", r["omega"], ref["omega"]),
            ("frac_artefact", r["frac_artefact"], ref["frac_artefact"])]
    worst = 0.0
    for name, a, b in rows:
        d = abs(a - b)
        worst = max(worst, d)
        print("  {:16s} released {:+.10f}   pipeline {:+.10f}   |diff| {:.2e}".format(
            name, a, b, d))
    print("  {:16s} released {:,}   pipeline {:,}".format(
        "n_instances", r["n_instances"], ref["n_instances"]))

    ok = worst < 1e-9 and r["n_instances"] == ref["n_instances"]
    print("\n  worst absolute difference {:.2e}".format(worst))
    print("  " + ("AGREE -- the released file reproduces the paper's numbers"
                  if ok else "DISAGREE -- do not release until this is resolved"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
