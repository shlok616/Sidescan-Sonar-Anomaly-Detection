"""
Train the hybrid CNN-Mamba-Transformer detector on the side-scan sonar dataset.

    python train.py --data ../dtst2 --epochs 60

Writes weights/sonar_hybrid.pt (model + anomaly statistics + metrics).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import IMG_SIZE, SonarDetectionDataset
from hybrid_model import CLASS_NAMES, HybridSonarDetector, decode, iou


def focal_loss(logits, gt):
    pred = logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg_w = (1 - gt) ** 4
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred ** 2 * neg_w * (1 - pos)
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n


def gather(feat, ind):  # feat (B,C,H,W), ind (B,M) -> (B,M,C)
    B, C = feat.shape[:2]
    f = feat.view(B, C, -1).permute(0, 2, 1)
    return f.gather(1, ind.unsqueeze(-1).expand(-1, -1, C))


def compute_loss(out, batch):
    l_hm = focal_loss(out["heatmap"], batch["hm"])
    m = batch["mask"].unsqueeze(-1)
    n = m.sum().clamp(min=1)
    l_wh = (F.smooth_l1_loss(gather(out["wh"], batch["ind"]), batch["wh"], reduction="none") * m).sum() / n
    l_off = (F.l1_loss(gather(out["offset"], batch["ind"]), batch["off"], reduction="none") * m).sum() / n
    # image-level class (dominant class) for the prototype's global head
    present = batch["hm"].flatten(2).max(-1).values.eq(1).float()
    l_cls = F.binary_cross_entropy_with_logits(out["cls_logits"], present)
    return l_hm + 0.5 * l_wh + l_off + 0.2 * l_cls, dict(hm=l_hm.item(), wh=l_wh.item(), off=l_off.item(), cls=l_cls.item())


def average_precision(dets, gts, cls, thr=0.5):
    """VOC-style AP for one class. dets: list per image of [x1,y1,x2,y2,s,c]; gts: list per image of [x1,y1,x2,y2,c]."""
    recs = [(s, i, d) for i, ds in enumerate(dets) for d in ds if int(d[5]) == cls for s in [d[4]]]
    recs.sort(key=lambda r: -r[0])
    npos = sum(1 for g in gts for b in g if int(b[4]) == cls)
    if npos == 0:
        return None
    used = [set() for _ in gts]
    tp, fp = [], []
    for s, i, d in recs:
        best, bj = 0, -1
        for j, g in enumerate(gts[i]):
            if int(g[4]) != cls or j in used[i]:
                continue
            o = iou(d, g)
            if o > best:
                best, bj = o, j
        if best >= thr:
            used[i].add(bj); tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / npos if len(tp) else np.array([0.0])
    prec = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.array([0.0])
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


@torch.no_grad()
def evaluate(model, ds, device):
    model.eval()
    dets, gts = [], []
    for b in DataLoader(ds, batch_size=8):
        out = model(b["image"].to(device))
        dets += decode(out, IMG_SIZE, thresh=0.05)
        for g in b["gt"]:
            gts.append([[*r[:4].tolist(), int(r[4])] for r in g if r[5] > 0])
    aps = {CLASS_NAMES[c]: average_precision(dets, gts, c) for c in range(len(CLASS_NAMES))}
    valid = [v for v in aps.values() if v is not None]
    # class-agnostic "is there debris here" AP (what the human reviewer acts on)
    agn = average_precision([[d[:5] + [0] for d in ds_] for ds_ in dets], [[g[:4] + [0] for g in gs] for gs in gts], 0)
    return {"mAP50": float(np.mean(valid)), "AP50_per_class": aps, "AP50_class_agnostic": agn}


@torch.no_grad()
def fit_anomaly_stats(model, ds, device):
    """PaDiM-style: Gaussian of background (seabed) features at stride 8. Distance from it = anomaly."""
    model.eval()
    feats = []
    for b in DataLoader(ds, batch_size=8):
        out = model(b["image"].to(device), return_features=True)
        f = out["features"]                                    # (B,C,H,W)
        bg = (b["hm"].max(1).values < 0.05)                    # far from any labelled object
        feats.append(f.permute(0, 2, 3, 1)[bg.to(device)].cpu())
    X = torch.cat(feats)
    if len(X) > 60000:
        X = X[torch.randperm(len(X))[:60000]]
    mu = X.mean(0)
    cov = torch.cov((X - mu).T) + 0.01 * torch.eye(X.shape[1])
    icov = torch.linalg.inv(cov)
    d = torch.sqrt((((X - mu) @ icov) * (X - mu)).sum(1))
    return {"mu": mu, "icov": icov, "p50": float(d.quantile(0.5)), "p99": float(d.quantile(0.99))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "dtst2"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pretrained", default=os.path.join(os.path.dirname(__file__), "weights", "resnet18_imagenet.pth"),
                    help="ImageNet ResNet-18 weights for the CNN stem (optional)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "weights", "sonar_hybrid.pt"))
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_ds = SonarDetectionDataset(args.data, "train", augment=True)
    val_ds = SonarDetectionDataset(args.data, "valid")
    test_ds = SonarDetectionDataset(args.data, "test")
    dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True)
    model = HybridSonarDetector()
    if args.pretrained and os.path.exists(args.pretrained):
        print("CNN stem initialised from", args.pretrained, model.backbone_stem.load_imagenet(args.pretrained))
    else:
        print("WARNING: no ImageNet weights found - training the CNN stem from scratch (expect much lower mAP). See README.")
    model = model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    stem = list(model.backbone_stem.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone_stem.")]
    opt = torch.optim.AdamW([{"params": stem, "lr": args.lr * 0.25}, {"params": rest, "lr": args.lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, [args.lr * 0.25, args.lr], total_steps=args.epochs * len(dl), pct_start=0.1)
    history, best, best_state = [], -1, None
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), {}
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            loss, parts = compute_loss(model(b["image"]), b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            opt.step(); sched.step()
            for k, v in parts.items():
                tot[k] = tot.get(k, 0) + v / len(dl)
        row = {"epoch": ep, **{k: round(v, 4) for k, v in tot.items()}, "sec": round(time.time() - t0, 1)}
        if ep % 5 == 0 or ep == args.epochs:
            m = evaluate(model, val_ds, device)
            row["val_mAP50"] = round(m["mAP50"], 4)
            row["val_agnostic_AP50"] = round(m["AP50_class_agnostic"], 4)
            if m["mAP50"] + m["AP50_class_agnostic"] > best:
                best = m["mAP50"] + m["AP50_class_agnostic"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(row)
        print(json.dumps(row), flush=True)

    model.load_state_dict(best_state)
    metrics = {"valid": evaluate(model, val_ds, device), "test": evaluate(model, test_ds, device)}
    print("Final:", json.dumps(metrics, indent=1))
    stats = fit_anomaly_stats(model, SonarDetectionDataset(args.data, "train"), device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": best_state, "classes": CLASS_NAMES, "img_size": IMG_SIZE,
                "anomaly": stats, "metrics": metrics, "history": history}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
