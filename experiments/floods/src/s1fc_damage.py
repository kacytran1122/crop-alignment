"""Does the thing kappa detects actually cost anything?

The paper can now show that crop alignment is real, that its null is structural,
that the estimator is correct to 1e-15, and that it rises with a generator's
crop-dependence in a controlled sweep. None of that establishes that anyone should
care. A statistic can be perfectly well behaved and measure something harmless.

Sen1Floods11 carries expert labels for the same chips, and the 55 models from the
alpha sweep have never been scored against them -- they were trained on, validated
on and tested against their own generator. So the question can be asked directly and
without training anything new:

    as the generator's crop-dependence rises, does the model get worse AGAINST THE
    HUMAN?

If it does, then kappa is not merely detectable, it is a leading indicator of
damage, and the dial gives the damage a dose-response of its own. If it does not,
the honest finding is that closed-loop labels are measurable but benign at this
scale, and the paper must say so -- which would be a real result and a considerable
deflation of the recommendations.

SCORING. One decision per source pixel, by majority vote of the crops covering it,
scored once against the expert label. This is the mosaic convention the paper argues
for elsewhere; using patch scoring here would count each pixel around sixteen times
and would also hand the crop-dependent arms the very reward the paper says patch
scoring hands them, which would manufacture the result rather than measure it.

The alpha = 0 arm is the baseline. It is not trained on truth -- it is trained on a
chip-level Otsu labeller, which is still algorithmic and still wrong in its own way.
So the quantity below is the ADDITIONAL cost of making that same labeller
crop-dependent, holding everything else fixed.
"""
import glob
import json
from pathlib import Path

import numpy as np
import torch

W = Path(__file__).resolve().parents[1]
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


def main():
    import segmentation_models_pytorch as smp
    dev = torch.device("cuda")

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

    runs = sorted(glob.glob(str(R / "s1fc_a*_s42" / "test_metrics.json")))
    out = {}
    model = None
    for f in runs:
        tm = json.loads(Path(f).read_text())
        name, event, alpha = tm["name"], tm["holdout_event"], tm["alpha"]
        cached = R / "damage" / (name + ".json")
        if cached.exists():
            out[(alpha, event)] = json.loads(cached.read_text())
            continue
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
                x = ((np.asarray(imgs[ci], np.float32) - mu) / sd).transpose(2, 0, 1)
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
        rec = {"name": name, "event": event, "alpha": alpha,
               "miou_vs_expert": float(np.nanmean(iou_from_cm(cm))),
               "pix_acc_vs_expert": float(np.trace(cm) / max(cm.sum(), 1)),
               "cm": cm.tolist()}
        (R / "damage").mkdir(parents=True, exist_ok=True)
        (R / "damage" / (name + ".json")).write_text(json.dumps(rec, indent=1))
        out[(alpha, event)] = rec
        print("  {:34s} mIoU vs expert {:.4f}".format(name, rec["miou_vs_expert"]),
              flush=True)

    alphas = sorted({k[0] for k in out})
    events = sorted({k[1] for k in out})
    full = [e for e in events if all((a, e) in out for a in alphas)]

    print("\n" + "=" * 94)
    print("DOES THE ARTEFACT COST ANYTHING AGAINST HUMAN TRUTH?")
    print("  mIoU against expert labels, mosaic scored: one decision per source")
    print("  pixel by majority vote of covering crops, scored once")
    print("=" * 94)
    print("{:12s}".format("event") + "".join(
        "{:>11s}".format("a=" + format(a, ".2f")) for a in alphas))
    for e in full:
        print("{:12s}".format(e[:12]) + "".join(
            "{:>11s}".format("{:.4f}".format(out[(a, e)]["miou_vs_expert"]))
            for a in alphas))
    print("-" * 94)
    means = [np.mean([out[(a, e)]["miou_vs_expert"] for e in full]) for a in alphas]
    print("{:12s}".format("mean") + "".join(
        "{:>11s}".format("{:.4f}".format(m)) for m in means))

    base = np.array([out[(alphas[0], e)]["miou_vs_expert"] for e in full])
    top = np.array([out[(alphas[-1], e)]["miou_vs_expert"] for e in full])
    d = top - base
    se = d.std(ddof=1) / np.sqrt(len(d))
    print("\n  cost of full crop-dependence, alpha {:.2f} minus alpha {:.2f}".format(
        alphas[-1], alphas[0]))
    print("    mean {:+.4f}  sd {:.4f}  t {:+.2f}   {}/{} events worse".format(
        d.mean(), d.std(ddof=1), d.mean() / se if se else float("nan"),
        int((d < 0).sum()), len(d)))

    # the chain the paper wants: more closed loop -> higher kappa -> worse vs human
    kf = {}
    for f in glob.glob(str(R / "kappa" / "*.json")):
        if "_thr" in f:
            continue
        k = json.loads(Path(f).read_text())
        kf[(round(k["alpha"], 2), k["event"])] = k["kappa"]
    pairs = [(kf[(a, e)], out[(a, e)]["miou_vs_expert"])
             for a in alphas for e in full if (a, e) in kf]
    if len(pairs) > 4:
        kk = np.array([p[0] for p in pairs]); mm = np.array([p[1] for p in pairs])
        # within-event, so the between-event spread cannot drive it
        kc, mc = [], []
        for e in full:
            ke = np.array([kf[(a, e)] for a in alphas if (a, e) in kf])
            me = np.array([out[(a, e)]["miou_vs_expert"] for a in alphas
                           if (a, e) in kf])
            kc.append(ke - ke.mean()); mc.append(me - me.mean())
        kc = np.concatenate(kc); mc = np.concatenate(mc)
        print("\n  kappa against mIoU-vs-expert, pooled   r = {:+.3f}  (n={})".format(
            float(np.corrcoef(kk, mm)[0, 1]), len(kk)))
        print("  within event, event means removed      r = {:+.3f}".format(
            float(np.corrcoef(kc, mc)[0, 1])))
        A = np.column_stack([np.ones(len(kc)), kc])
        beta, *_ = np.linalg.lstsq(A, mc, rcond=None)
        resid = mc - A @ beta
        s2 = resid @ resid / (len(kc) - 2)
        seb = np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1])
        print("  slope {:+.4f} mIoU per unit kappa   se {:.4f}   t {:+.2f}".format(
            beta[1], seb, beta[1] / seb))

    (R / "damage_summary.json").write_text(json.dumps(
        {"alphas": alphas, "events": full,
         "miou_vs_expert": {str(a): [out[(a, e)]["miou_vs_expert"] for e in full]
                            for a in alphas},
         "cost_mean": float(d.mean()), "cost_t": float(d.mean() / se) if se else None},
        indent=1))
    print("\nwrote " + str(R / "damage_summary.json"))


if __name__ == "__main__":
    main()
