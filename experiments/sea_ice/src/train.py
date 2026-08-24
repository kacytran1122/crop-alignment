"""
Gate 1/3 trainer: fully GPU-resident, no DataLoader.

Speed design. The whole cache is 2.24 GB (1.68 GB images + 0.56 GB labels), so
it is uploaded ONCE to the GPU as uint8 tensors. Batching is then a device-side
index_select and normalisation happens on-GPU. There is no host->device traffic
and no worker processes in the steady state, so the CPU cannot starve the GPU
and utilisation is bounded only by the model itself.

Photon modes double as Gate-3 controls:
  true     real set-encoded photon descriptor
  zero     descriptor replaced by zeros        (is the branch doing anything?)
  shuffle  descriptors permuted within split   (is it the PHOTON or a prior?)
  noise    Gaussian matched to feature stats   (parameter-matched noise)
  dummy    branch present, input constant 1    (capacity control)
  none     optical-only, no fusion branch      (clean unimodal baseline)

Fusion is the paper's original broadcast+SE so the re-baseline is like-for-like.
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

W = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("SEAICE_CACHE", "/dev/shm/seaice_cache"))
TOK = Path(os.environ.get("SEAICE_TOKENS",
           str(W / "work" / "tokens" / "tokens_split.parquet")))
FEATJSON = Path(os.environ.get("SEAICE_FEATS",
                str(W / "work" / "tokens" / "feature_cols.json")))

NUM_CLASSES, IGNORE, PATCH = 3, 255, 128
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- model
class SqueezeExcitation(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        b = max(ch // reduction, 4)
        self.fc1, self.fc2 = nn.Linear(ch, b), nn.Linear(b, ch)

    def forward(self, x):
        z = x.mean(dim=(2, 3))
        z = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))
        return x * z[:, :, None, None]


class FusionNet(nn.Module):
    def __init__(self, n_feats, fusion_ch=16, use_photon=True, mode="deep",
                 encoder="resnet18"):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.use_photon = use_photon
        self.unet = smp.Unet(encoder, encoder_weights="imagenet",
                             in_channels=3, classes=fusion_ch)
        self.mode = mode
        if use_photon:
            self.photon = nn.Sequential(
                nn.Linear(n_feats, 96), nn.ELU(inplace=True), nn.Dropout(0.4),
                nn.Linear(96, 16), nn.ELU(inplace=True), nn.Dropout(0.4),
                nn.Linear(16, fusion_ch))
            self.se = SqueezeExcitation(fusion_ch * 2)
            head_in = fusion_ch * 2 if mode in ("deep", "hybrid") else fusion_ch
            if mode in ("late", "hybrid"):
                # decision-level branches: each modality emits its own logits
                self.img_head = nn.Conv2d(fusion_ch, NUM_CLASSES, 1)
                self.pho_head = nn.Linear(fusion_ch, NUM_CLASSES)
                # learned blend weight, trained by gradient descent (not tuned on
                # val or test). weight decay is excluded for this parameter so it
                # is not regularised toward 0.5.
                self.log_alpha = nn.Parameter(torch.zeros(1))
        else:
            head_in = fusion_ch
        self.head = nn.Sequential(
            nn.Conv2d(head_in, fusion_ch, 3, padding=1),
            nn.BatchNorm2d(fusion_ch), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(fusion_ch, NUM_CLASSES, 1))

    def forward(self, img, feats):
        f = self.unet(img)
        if not self.use_photon:
            return self.head(f)

        pv = self.photon(feats)
        pmap = pv[:, :, None, None].expand(-1, -1, f.shape[2], f.shape[3])

        if self.mode == "deep":
            return self.head(self.se(torch.cat([f, pmap], dim=1)))

        # decision-level pieces, shared by late and hybrid
        img_logits = self.img_head(f)
        pho_logits = self.pho_head(pv)[:, :, None, None].expand_as(img_logits)
        a = torch.sigmoid(self.log_alpha)

        if self.mode == "late":
            # branches stay separate; combine probabilities, return log-probs so
            # the focal loss (which applies log_softmax) is a no-op on them.
            prob = a * F.softmax(img_logits, 1) + (1 - a) * F.softmax(pho_logits, 1)
            return torch.log(prob.clamp_min(1e-8))

        # hybrid: feature-level fusion blended with the unimodal decisions
        deep_logits = self.head(self.se(torch.cat([f, pmap], dim=1)))
        uni = 0.5 * (F.softmax(img_logits, 1) + F.softmax(pho_logits, 1))
        prob = a * F.softmax(deep_logits, 1) + (1 - a) * uni
        return torch.log(prob.clamp_min(1e-8))


class WeightedCE(nn.Module):
    """Inverse-frequency weighted cross-entropy, the loss the manuscript claims."""
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer("w", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        return F.cross_entropy(logits.float(), target.long(), weight=self.w,
                               ignore_index=IGNORE)


class DiceLoss(nn.Module):
    def __init__(self, alpha=None, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        valid = target != IGNORE
        t = torch.where(valid, target, torch.zeros_like(target))
        p = F.softmax(logits.float(), dim=1)
        oh = F.one_hot(t, NUM_CLASSES).permute(0, 3, 1, 2).float()
        m = valid.unsqueeze(1).float()
        p, oh = p * m, oh * m
        inter = (p * oh).sum(dim=(0, 2, 3))
        denom = p.sum(dim=(0, 2, 3)) + oh.sum(dim=(0, 2, 3))
        return 1.0 - ((2 * inter + self.eps) / (denom + self.eps)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, target):
        valid = target != IGNORE
        t = torch.where(valid, target, torch.zeros_like(target))
        logp = F.log_softmax(logits.float(), dim=1)
        lp = logp.gather(1, t.unsqueeze(1)).squeeze(1)
        pt = lp.exp().clamp(1e-8, 1 - 1e-8)
        loss = -self.alpha[t] * (1 - pt).pow(self.gamma) * lp
        loss = loss * valid
        return loss.sum() / valid.sum().clamp_min(1)


# ---------------------------------------------------------------- metrics
def cm_update(cm, pred, tgt):
    valid = tgt != IGNORE
    p, t = pred[valid], tgt[valid]
    idx = t.long() * NUM_CLASSES + p.long()
    cm += torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(cm):
    cm = cm.double()
    iou, f1 = [], []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        iou.append((tp / (tp + fp + fn)).item() if (tp + fp + fn) > 0 else 0.0)
        pr = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        rc = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        f1.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    return {"miou": float(np.mean(iou)), "per_iou": iou, "macro_f1": float(np.mean(f1)),
            "pix_acc": (cm.diag().sum() / cm.sum()).item(), "cm": cm.long().tolist()}


# ---------------------------------------------------------------- data
def load_all(dev, feat_cols, label_set="scene", split_col="split", holdout_acq=None,
             n_train_acq=None, disjoint_tiles=False):
    tok = pd.read_parquet(TOK).reset_index(drop=True)
    imgs = torch.from_numpy(np.load(CACHE / "images.npy", mmap_mode="r").copy())
    lf = {"scene": "labels.npy", "original": "labels_original.npy",
          "kmeans": "labels_kmeans.npy"}[label_set]
    labs = torch.from_numpy(np.load(CACHE / lf, mmap_mode="r").copy())
    imgs = imgs.permute(0, 3, 1, 2).contiguous().to(dev, non_blocking=True)
    labs = labs.to(dev, non_blocking=True)

    X = tok[feat_cols].to_numpy(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if holdout_acq is not None:
        tr = (tok["acq_id"] != holdout_acq).to_numpy()
    else:
        tr = (tok[split_col] == "train").to_numpy()
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-6] = 1.0
    X = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(dev)

    if holdout_acq is not None:
        # leave-one-acquisition-out: test = holdout, val = next two in sorted
        # order, train = the rest. Acquisitions differ in date and orbit, but the
        # tile grid is FIXED, so by default the same ground can appear on both
        # sides of the boundary on different dates. --disjoint-tiles removes that
        # overlap so the two designs can be compared and the leak measured.
        acqs = sorted(tok["acq_id"].unique().tolist())
        assert holdout_acq in acqs, f"{holdout_acq} not among {len(acqs)} acquisitions"
        rest = [a for a in acqs if a != holdout_acq]
        val_acq = set(rest[:2])
        col = np.where(tok["acq_id"].to_numpy() == holdout_acq, "test",
                       np.where(tok["acq_id"].isin(val_acq).to_numpy(), "val", "train"))
        if disjoint_tiles:
            held = set(tok.loc[tok["acq_id"] == holdout_acq, "tile"].unique())
            shared = tok["tile"].isin(held).to_numpy() & (col != "test")
            print(json.dumps({"disjoint_tiles": True, "held_tiles": sorted(held),
                              "dropped_patches": int(shared.sum()),
                              "kept_train_val": int((~shared & (col != "test")).sum())}),
                  flush=True)
            col = np.where(shared, "unused", col)
        idx = {s: torch.from_numpy(np.where(col == s)[0]).to(dev)
               for s in ("train", "val", "test")}
    else:
        col = tok[split_col].to_numpy().astype(object).copy()
        if n_train_acq is not None:
            # learning curve: keep only the first n training acquisitions, in
            # sorted order so the subset is deterministic and nested
            tr_acqs = sorted(tok.loc[tok[split_col] == "train", "acq_id"].unique())
            keep = set(tr_acqs[:n_train_acq])
            drop = (col == "train") & (~tok["acq_id"].isin(keep).to_numpy())
            col[drop] = "unused"
        idx = {s: torch.from_numpy(np.where(col == s)[0]).to(dev)
               for s in ("train", "val", "test")}
    return imgs, labs, X, idx, (mu, sd)


def apply_photon_mode(X, idx_split, mode, gen):
    if mode == "true":
        return X
    if mode == "zero":
        return torch.zeros_like(X)
    if mode == "dummy":
        return torch.ones_like(X)
    if mode == "noise":
        return torch.randn(X.shape, generator=gen, device=X.device)
    if mode == "shuffle":
        Y = X.clone()
        for ids in idx_split.values():
            perm = ids[torch.randperm(len(ids), generator=gen, device=X.device)]
            Y[ids] = X[perm]
        return Y
    raise ValueError(mode)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photon", default="true",
                    choices=["true", "zero", "shuffle", "noise", "dummy", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--name", default=None)
    ap.add_argument("--bench", type=int, default=0, help="benchmark N steps and exit")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--labels", default="scene", choices=["scene", "original", "kmeans"],
                    help="scene = Gate-1 repaired labels; original = per-patch pipeline")
    ap.add_argument("--fusion", default="deep", choices=["deep", "late", "hybrid"],
                    help="where the two modalities are combined")
    ap.add_argument("--encoder", default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50", "efficientnet-b0"])
    ap.add_argument("--loss", default="focal", choices=["focal", "wce", "dice"])
    ap.add_argument("--n-train-acq", type=int, default=None,
                    help="learning curve: use only the first N training acquisitions")
    ap.add_argument("--holdout-acq", default=None,
                    help="leave-one-acquisition-out: this acquisition becomes the test set")
    ap.add_argument("--disjoint-tiles", action="store_true",
                    help="LOAO: also drop train/val patches on tiles present in the holdout")
    ap.add_argument("--split-col", default="split", choices=["split", "split_rev"],
                    help="split = test on 20191126; split_rev = test on 20191104")
    args = ap.parse_args()

    name = args.name or f"{args.photon}_s{args.seed}"
    out = W / "runs" / name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")

    feat_cols = json.loads(FEATJSON.read_text())["feature_cols"]
    imgs, labs, X, idx, _ = load_all(dev, feat_cols, args.labels, args.split_col,
                                     args.holdout_acq, args.n_train_acq,
                                     args.disjoint_tiles)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    X = apply_photon_mode(X, idx, args.photon, gen) if args.photon != "none" else X

    use_photon = args.photon != "none"
    model = FusionNet(len(feat_cols), use_photon=use_photon, mode=args.fusion,
                      encoder=args.encoder).to(dev).to(memory_format=torch.channels_last)
    crit = {"focal": FocalLoss, "wce": WeightedCE,
            "dice": DiceLoss}[args.loss]([0.05, 0.45, 0.60]).to(dev)
    decay = [p for n, p in model.named_parameters() if n != "log_alpha"]
    nodecay = [p for n, p in model.named_parameters() if n == "log_alpha"]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": nodecay, "weight_decay": 0.0}], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    mean = IM_MEAN.to(dev); std = IM_STD.to(dev)

    def batch(ids, train):
        im = imgs[ids].float().div_(255).sub_(mean).div_(std).to(memory_format=torch.channels_last)
        lb = labs[ids].long()
        if train:
            if torch.rand(1, generator=gen, device=dev).item() < 0.5:
                im, lb = im.flip(-1), lb.flip(-1)
            if torch.rand(1, generator=gen, device=dev).item() < 0.5:
                im, lb = im.flip(-2), lb.flip(-2)
        return im, X[ids], lb

    # ---------------- benchmark mode
    if args.bench:
        tr = idx["train"]
        model.train()
        for i in range(args.bench + 10):
            if i == 10:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            ids = tr[torch.randint(0, len(tr), (args.batch,), generator=gen, device=dev)]
            im, xf, lb = batch(ids, True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = crit(model(im, xf), lb)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ips = args.bench * args.batch / dt
        print(json.dumps({"name": name, "batch": args.batch, "amp": args.amp,
                          "steps": args.bench, "sec": round(dt, 2),
                          "img_per_s": round(ips, 1),
                          "sec_per_epoch": round(len(tr) / ips, 2),
                          "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}))
        return

    # ---------------- train
    best, bad, hist = -1.0, 0, []
    t_start = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr = idx["train"][torch.randperm(len(idx["train"]), generator=gen, device=dev)]
        tot, nb = 0.0, 0
        for i in range(0, len(tr) - args.batch + 1, args.batch):
            ids = tr[i:i + args.batch]
            im, xf, lb = batch(ids, True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = crit(model(im, xf), lb)
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        model.eval()
        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=dev)
        with torch.no_grad():
            va = idx["val"]
            for i in range(0, len(va), args.batch):
                ids = va[i:i + args.batch]
                im, xf, lb = batch(ids, False)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                    lo = model(im, xf)
                cm_update(cm, lo.argmax(1), lb)
        vm = metrics(cm)
        hist.append({"epoch": ep, "train_loss": tot / max(nb, 1), "val_miou": vm["miou"]})
        print(f"ep{ep:02d} loss {tot/max(nb,1):.4f} val_mIoU {vm['miou']:.4f}", flush=True)

        if vm["miou"] > best:
            best, bad = vm["miou"], 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val": vm}, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop @ {ep}"); break

    ck = torch.load(out / "best.pt", weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=dev)
    with torch.no_grad():
        te = idx["test"]
        for i in range(0, len(te), args.batch):
            ids = te[i:i + args.batch]
            im, xf, lb = batch(ids, False)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                lo = model(im, xf)
            cm_update(cm, lo.argmax(1), lb)
    tm = metrics(cm)
    tm.update({"photon": args.photon, "seed": args.seed, "labels": args.labels, "split_col": args.split_col, "holdout_acq": args.holdout_acq, "disjoint_tiles": args.disjoint_tiles, "n_train_acq": args.n_train_acq, "fusion": args.fusion, "encoder": args.encoder, "loss": args.loss, "best_val_miou": best,
               "best_epoch": ck["epoch"], "minutes": round((time.perf_counter()-t_start)/60, 2)})
    if hasattr(model, "log_alpha"):
        tm["learned_alpha"] = float(torch.sigmoid(model.log_alpha).item())
    (out / "test_metrics.json").write_text(json.dumps(tm, indent=2))
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    print(f"TEST mIoU {tm['miou']:.4f}  per_iou {[round(x,4) for x in tm['per_iou']]}  "
          f"({tm['minutes']} min)")


if __name__ == "__main__":
    main()
