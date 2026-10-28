import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from data import build_loaders
from engine import evaluate, train_one_epoch
from models import GoogLeNetMini, LeNet5, PretrainedResnet, ResNet18, n_params

MODELS = {
    "lenet_avg": dict(img=64, lr=1e-3),
    "lenet_max": dict(img=64, lr=1e-3),
    "googlenet": dict(img=64, lr=1e-3),
    "resnet18": dict(img=96, lr=1e-3),
    "resnet50_fe": dict(img=224, lr=1e-3),
    "resnet50_ft": dict(img=224, lr=1e-4),
}


def build_model(name, img, num_classes):
    if name == "lenet_avg":
        return LeNet5(3, img, num_classes, pool="avg")
    if name == "lenet_max":
        return LeNet5(3, img, num_classes, pool="max")
    if name == "googlenet":
        return GoogLeNetMini(num_classes)
    if name == "resnet18":
        return ResNet18(num_classes)
    if name == "resnet50_fe":
        return PretrainedResnet(num_classes, freeze_backbone=True)
    if name == "resnet50_ft":
        return PretrainedResnet(num_classes, freeze_backbone=False)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--data_root", default="VinaFood21")
    ap.add_argument("--cache_dir", default="cache")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no_aug", action="store_true")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = MODELS[a.model]
    img = cfg["img"]
    lr = a.lr if a.lr is not None else cfg["lr"]
    name = a.model + (f"_{a.tag}" if a.tag else "")
    out = Path(a.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_dl, val_dl, test_dl, classes = build_loaders(
        a.data_root, img, a.batch_size, a.cache_dir,
        augment=not a.no_aug, num_workers=a.num_workers, seed=42)
    model = build_model(a.model, img, len(classes)).to(device)
    total_p, train_p = n_params(model), n_params(model, True)
    print(f"{name}: img {img}, params {total_p:,}, trainable {train_p:,}, "
          f"train {len(train_dl.dataset)}, val {len(val_dl.dataset)}, "
          f"test {len(test_dl.dataset)}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    hist, best_f1, best_ep, times = [], -1.0, 0, []
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, yt, yp = evaluate(model, val_dl, criterion, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        _, _, va_f1, _ = precision_recall_fscore_support(
            yt, yp, average="macro", zero_division=0)
        hist.append((ep, tr_loss, tr_acc, va_loss, va_acc, va_f1, dt))
        print(f"epoch {ep:>2}  train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"val {va_loss:.4f}/{va_acc:.4f}  f1 {va_f1:.4f}  {dt:.1f}s",
              flush=True)
        if va_f1 > best_f1:
            best_f1, best_ep = va_f1, ep
            torch.save(model.state_dict(), out / "best.pt")

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                    "val_macro_f1", "seconds"])
        w.writerows(hist)

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    _, _, yt, yp = evaluate(model, test_dl, criterion, device)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro",
                                                   zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(yt, yp, zero_division=0)
    metrics = dict(
        model=name, img=img, lr=lr, epochs=a.epochs, augment=not a.no_aug,
        params=total_p, trainable=train_p, best_epoch=best_ep,
        sec_per_epoch=float(np.mean(times)),
        accuracy=accuracy_score(yt, yp), precision=p, recall=r, f1=f1)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    with open(out / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for i, c in enumerate(classes):
            w.writerow([c, f"{pc[i]:.4f}", f"{rc[i]:.4f}", f"{fc[i]:.4f}", int(sc[i])])
    cm = confusion_matrix(yt, yp, labels=range(len(classes)))
    np.save(out / "confusion.npy", cm)
    json.dump(list(classes), open(out / "classes.json", "w"))

    ep_, trl, va = [h[0] for h in hist], [h[1] for h in hist], [h[3] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.plot(ep_, trl, label="train loss")
    plt.plot(ep_, va, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    plt.yticks(range(len(classes)), classes, fontsize=7)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
