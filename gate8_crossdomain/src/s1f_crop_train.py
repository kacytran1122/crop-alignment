"""Train on explicit crop-label fields built before optimisation.

One model per (held-out event, named saved arm, seed). J1 reads the explicit
threshold field built by ``s1fc_arms.py``; its dial is
``(1-alpha)*tau_event + alpha*tau_crop``. The legacy ``--alpha`` path below instead
interpolates chip to crop and is not used for J1. Given a fixed seed, all non-label
settings are held fixed. Contrasts therefore identify the saved label field as a
whole, not crop-dependence in isolation.

Labels are built on the fly rather than cached, because the whole label set is one
comparison against a threshold that costs less than reading it from disk would:

    th = saved_arm[chip, a, b]
    label(p) = 1 if vh_smooth(p) <= th else 0,  IGNORE where the chip is not scored

vh_smooth is the chip-level 9x9 focal mean recovered from the published thresholds,
identical for every crop, so alpha is the only thing that can make two crops
disagree about a pixel.

Held out by EVENT, as in the whole-chip arm: chips inside an event share flooding,
terrain and acquisition geometry, so a chip-level split leaks.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import os
W = Path(os.environ.get("SEAICE_ROOT",
                        Path(__file__).resolve().parents[1]))
C = W / "data" / "sen1floods11" / "cache"
CR = C / "crop"
IGNORE, NC = 255, 2


def iou_from_cm(cm):
    out = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        den = cm[k].sum() + cm[:, k].sum() - tp
        out.append(np.nan if den == 0 else tp / den)
    return np.array(out, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-event", required=True)
    ap.add_argument("--arm", default=None,
                    help="named threshold field under cache/crop/arms; "
                         "mutually exclusive with --alpha")
    ap.add_argument("--alpha", type=float, default=None,
                    help="legacy exploratory chip-to-crop dial; J1 uses --arm. "
                         "The alpha=1 endpoint is predominantly crop-fitted, "
                         "with recorded fallbacks.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--crops-per-chip", type=int, default=24,
                    help="crops sampled per chip per epoch, out of 169")
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    import segmentation_models_pytorch as smp
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    meta = json.loads((C / "meta.json").read_text())
    cmeta = json.loads((CR / "meta.json").read_text())
    CROP, STARTS = cmeta["crop"], cmeta["starts"]
    ns = len(STARTS)

    ev = np.load(C / "events.npy")
    y_ref = np.load(C / "y_ref.npy")
    th_chip = np.load(CR / "th_chip.npy")
    th_crop = np.load(CR / "th_crop.npy")

    # a chip with no usable chip-level threshold has no label set at any alpha
    if (args.arm is None) == (args.alpha is None):
        raise SystemExit("give exactly one of --arm or --alpha")
    if args.arm is not None:
        af = CR / "arms" / ("th_" + args.arm + ".npy")
        if not af.exists():
            raise SystemExit("no such arm: " + str(af))
        TH_np = np.load(af)
        if TH_np.shape != th_crop.shape:
            raise SystemExit("arm {} has shape {}, expected {}".format(
                args.arm, TH_np.shape, th_crop.shape))
    else:
        TH_np = ((1.0 - args.alpha) * th_chip[:, None, None]
                 + args.alpha * th_crop)
    # a chip is usable only if the arm actually assigns it a threshold everywhere
    usable = np.isfinite(th_chip) & np.isfinite(TH_np).all(axis=(1, 2))
    test_m = (ev == args.holdout_event) & usable
    if not test_m.any():
        raise SystemExit("no usable chips for event " + args.holdout_event)
    # Model selection gets its own EVENT, not a random draw of chips from every
    # training event. A 15% chip draw put the same event on both sides of early
    # stopping, which is the leak the paper's own recommendation forbids.
    others = sorted(set(ev[usable & (ev != args.holdout_event)].tolist()))
    if len(others) < 2:
        raise SystemExit("need at least two non-test events, have " + str(others))
    val_event = others[int(np.random.default_rng(args.seed).integers(len(others)))]
    val_c = np.where((ev == val_event) & usable)[0]
    train_c = np.where(usable & (ev != args.holdout_event) & (ev != val_event))[0]
    if not len(val_c) or not len(train_c):
        raise SystemExit("empty split: val={} train={}".format(len(val_c), len(train_c)))

    imgs = np.load(C / "images.npy", mmap_mode="r")
    # Fold-local statistics. meta["mean"]/["std"] were computed over all 446 chips
    # including the held-out event, so every fold was normalised with numbers that
    # had seen its own test set.
    tr_img = np.asarray(imgs[train_c], np.float32)
    mu = tr_img.reshape(-1, tr_img.shape[-1]).mean(0).astype(np.float32)
    sd = tr_img.reshape(-1, tr_img.shape[-1]).std(0).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    del tr_img
    X = torch.from_numpy(((np.asarray(imgs, np.float32) - mu) / sd)
                         ).permute(0, 3, 1, 2).contiguous().to(dev)
    VH = torch.from_numpy(np.load(CR / "vh_smooth.npy").astype(np.float32)).to(dev)
    VALID = torch.from_numpy(y_ref != IGNORE).to(dev)
    TH = torch.from_numpy(TH_np.astype(np.float32)).to(dev)

    def batch_of(triples):
        """(chip, a, b) -> normalised crops and their alpha-dependent labels."""
        ci = torch.tensor([t[0] for t in triples], device=dev, dtype=torch.long)
        xs, ys = [], []
        for c, a, b in triples:
            r0, c0 = STARTS[a], STARTS[b]
            xs.append(X[c, :, r0:r0 + CROP, c0:c0 + CROP])
            v = VH[c, r0:r0 + CROP, c0:c0 + CROP]
            lab = (v <= TH[c, a, b]).long()
            lab = torch.where(VALID[c, r0:r0 + CROP, c0:c0 + CROP], lab,
                              torch.full_like(lab, IGNORE))
            ys.append(lab)
        return torch.stack(xs), torch.stack(ys), ci

    def sample(chips, k):
        t = [(int(c), int(rng.randint(ns)), int(rng.randint(ns)))
             for c in chips for _ in range(k)]
        rng.shuffle(t)
        return t

    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=2,
                     classes=NC).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)

    name = args.name or "s1fc_a{:03d}_{}_s{}".format(
        int(round(args.alpha * 100)), args.holdout_event, args.seed)
    out = W / "runs_s1fc" / name
    out.mkdir(parents=True, exist_ok=True)
    print("{}: holdout={} alpha={} train_chips={} val={} test={}".format(
        name, args.holdout_event, args.alpha, len(train_c), len(val_c),
        int(test_m.sum())), flush=True)

    val_t = sample(val_c, args.crops_per_chip)

    def evaluate(triples):
        model.eval()
        cm = np.zeros((NC, NC), np.int64)
        with torch.no_grad():
            for i in range(0, len(triples), args.batch):
                xb, yb, _ = batch_of(triples[i:i + args.batch])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(xb).float().argmax(1)
                m = yb != IGNORE
                if m.any():
                    t = yb[m].cpu().numpy(); q = p[m].cpu().numpy()
                    cm += np.bincount(t * NC + q, minlength=NC ** 2).reshape(NC, NC)
        return float(np.nanmean(iou_from_cm(cm)))

    best, best_ep, bad = -1.0, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr = sample(train_c, args.crops_per_chip)
        for i in range(0, len(tr), args.batch):
            xb, yb, _ = batch_of(tr[i:i + args.batch])
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        v = evaluate(val_t)
        if v > best:
            best, best_ep, bad = v, ep, 0
            torch.save({"model": model.state_dict()}, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(torch.load(out / "best.pt", weights_only=False)["model"])

    test_t = sample(np.where(test_m)[0], args.crops_per_chip)
    res = {"name": name, "holdout_event": args.holdout_event,
           "alpha": args.alpha, "arm": args.arm, "val_event": val_event,
           "norm_mean": [float(x) for x in mu], "norm_std": [float(x) for x in sd],
           "seed": args.seed, "encoder": args.encoder,
           "crop": CROP, "stride": cmeta["stride"],
           "best_epoch": best_ep, "best_val_miou": best, "epochs": args.epochs,
           "n_train_chips": len(train_c), "n_val_chips": len(val_c),
           "n_test_chips": int(test_m.sum()),
           "test_miou_vs_own_labels": evaluate(test_t),
           "minutes": (time.time() - t0) / 60.0}
    (out / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print("  test mIoU vs its own label set {:.4f}   [{:.1f} min, best epoch {}]".format(
        res["test_miou_vs_own_labels"], res["minutes"], best_ep), flush=True)


if __name__ == "__main__":
    main()
