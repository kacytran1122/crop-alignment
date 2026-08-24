"""Sen1Floods11: train on one label set, score against both.

The sea-ice audit could measure how much of a deep model's advantage tracks the
label generator, but never how much of it was real, because sea ice has no
reference labels. Sen1Floods11 has both, so the quantity that matters here is not
the premium but the TRANSFER COLLAPSE:

    collapse = mIoU(trained on algorithmic labels, scored against expert)
             - mIoU(trained on expert labels,      scored against expert)

which asks directly how much of the advantage survives contact with a human. It
needs no baseline-family argument and no reviewer can dispute what it means.

Every run therefore reports mIoU against BOTH label sets on identical pixels, and
the context-free baseline is refit on whichever label set the model was trained on,
so neither arm is handicapped.

Model input is Sentinel-1 only. The expert labels were produced by analysts
correcting a Sentinel-2 index classification, so the reference arm is open-loop
only while the model never reads S2. Feeding S2 would contaminate both arms and
destroy the comparison.

Held out by EVENT, not by chip. All chips in an event share one generator
threshold and therefore one artefact, so the event is the independent unit; a
chip-level split would leak the artefact across the boundary and inflate everything.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

W = Path(__file__).resolve().parents[1]
C = W / "data" / "sen1floods11" / "cache"
IGNORE, NC = 255, 2


def iou_from_cm(cm):
    cm = cm.astype(np.float64)
    out = []
    for c in range(NC):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        d = tp + fp + fn
        out.append(tp / d if d > 0 else np.nan)
    return np.array(out)


def score(pred, y):
    m = y != IGNORE
    if not m.any():
        return None
    yt = y[m].astype(np.int64); yp = pred[m].astype(np.int64)
    cm = np.bincount(yt * NC + yp, minlength=NC ** 2).reshape(NC, NC)
    v = iou_from_cm(cm)
    return {"miou": float(np.nanmean(v)),
            "per_iou": [None if np.isnan(x) else float(x) for x in v],
            "pix_acc": float(np.trace(cm) / max(cm.sum(), 1)),
            "n_px": int(cm.sum()), "cm": cm.astype(np.int64).tolist()}


def fit_threshold(vh, y, lo=-30.0, hi=-10.0, step=0.05):
    """Context-free baseline: one scalar on VH, refit on the arm's own labels."""
    m = y != IGNORE
    v, t = vh[m], y[m]
    k = min(len(v), 3_000_000)
    sel = np.random.RandomState(0).choice(len(v), k, replace=False)
    v, t = v[sel], t[sel]
    best, bt = -1.0, None
    for th in np.arange(lo, hi + step, step):
        pred = (v <= th).astype(np.int64)
        cm = np.bincount(t.astype(np.int64) * NC + pred,
                         minlength=NC ** 2).reshape(NC, NC)
        mi = float(np.nanmean(iou_from_cm(cm)))
        if mi > best:
            best, bt = mi, float(th)
    return bt, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-event", required=True)
    ap.add_argument("--labels", required=True, choices=["ref", "alg"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    import segmentation_models_pytorch as smp
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ev = np.load(C / "events.npy")
    meta = json.loads((C / "meta.json").read_text())
    imgs = np.load(C / "images.npy").astype(np.float32)
    y_ref = np.load(C / "y_ref.npy")
    y_alg = np.load(C / "y_alg.npy")
    y_tr = y_alg if args.labels == "alg" else y_ref

    test = ev == args.holdout_event
    if not test.any():
        raise SystemExit("no chips for event " + args.holdout_event)
    rest = np.where(~test)[0]
    rng = np.random.RandomState(args.seed)
    rng.shuffle(rest)
    nval = max(1, int(0.15 * len(rest)))
    val, train = rest[:nval], rest[nval:]

    mu = np.array(meta["mean"], np.float32)
    sd = np.array(meta["std"], np.float32)
    X = torch.from_numpy(((imgs - mu) / sd)).permute(0, 3, 1, 2).contiguous().to(dev)
    Ytr = torch.from_numpy(y_tr.astype(np.int64)).to(dev)

    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=2,
                     classes=NC).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)

    name = args.name or "s1f_{}_{}_s{}".format(args.labels, args.holdout_event,
                                               args.seed)
    out = W / "runs_s1f" / name
    out.mkdir(parents=True, exist_ok=True)
    print("{}: holdout={} labels={} train={} val={} test={}".format(
        name, args.holdout_event, args.labels, len(train), len(val),
        int(test.sum())), flush=True)

    def evaluate(idx):
        model.eval()
        cm = np.zeros((NC, NC), np.int64)
        with torch.no_grad():
            for i in range(0, len(idx), args.batch):
                b = idx[i:i + args.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(X[b]).float().argmax(1)
                yy = Ytr[b]
                m = yy != IGNORE
                if m.any():
                    t = yy[m].cpu().numpy(); q = p[m].cpu().numpy()
                    cm += np.bincount(t * NC + q,
                                      minlength=NC ** 2).reshape(NC, NC)
        return float(np.nanmean(iou_from_cm(cm)))

    best, best_ep, bad = -1.0, 0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        order = train.copy(); rng.shuffle(order)
        for i in range(0, len(order), args.batch):
            b = order[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(X[b]), Ytr[b])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        v = evaluate(val)
        if v > best:
            best, best_ep, bad = v, ep, 0
            torch.save({"model": model.state_dict()}, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(torch.load(out / "best.pt", weights_only=False)["model"])

    # predict once on the held-out event, then score against BOTH label sets
    model.eval()
    ti = np.where(test)[0]
    preds = np.zeros((len(ti), 512, 512), np.uint8)
    with torch.no_grad():
        for i in range(0, len(ti), args.batch):
            b = ti[i:i + args.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds[i:i + len(b)] = model(X[b]).float().argmax(1).cpu().numpy()

    res = {"name": name, "holdout_event": args.holdout_event,
           "labels": args.labels, "seed": args.seed, "encoder": args.encoder,
           "best_epoch": best_ep, "best_val_miou": best, "epochs": args.epochs,
           "n_train": len(train), "n_val": len(val), "n_test": int(test.sum()),
           "minutes": (time.time() - t0) / 60.0,
           "unet_vs_ref": score(preds, y_ref[ti]),
           "unet_vs_alg": score(preds, y_alg[ti])}

    # context-free baseline, refit on the arm's own labels using training events
    vh_tr = imgs[train][..., 1]
    th, fit = fit_threshold(vh_tr, y_tr[train])
    vh_te = imgs[ti][..., 1]
    bpred = (vh_te <= th).astype(np.uint8)
    res.update({"thr_vh": th, "thr_fit_miou": fit,
                "thr_vs_ref": score(bpred, y_ref[ti]),
                "thr_vs_alg": score(bpred, y_alg[ti])})

    (out / "test_metrics.json").write_text(json.dumps(res, indent=1))
    print("  U-Net vs expert {:.4f}   vs algorithmic {:.4f}".format(
        res["unet_vs_ref"]["miou"], res["unet_vs_alg"]["miou"]))
    print("  thresh(VH<={:.2f}) vs expert {:.4f}   vs algorithmic {:.4f}   "
          "[{:.1f} min, best epoch {}]".format(
              th, res["thr_vs_ref"]["miou"], res["thr_vs_alg"]["miou"],
              res["minutes"], best_ep), flush=True)


if __name__ == "__main__":
    main()
