import csv
import json
from pathlib import Path

import numpy as np

R = Path("results")
RUNS = ["lenet_avg", "lenet_max", "googlenet", "resnet18_aug", "resnet18_noaug",
        "resnet50_fe", "resnet50_ft"]


def hist(name):
    with open(R / name / "history.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    classes = json.load(open(R / "lenet_avg" / "classes.json"))
    per = {}
    for n in RUNS:
        with open(R / n / "per_class.csv") as f:
            per[n] = list(csv.DictReader(f))
    sup = [int(r["support"]) for r in per["lenet_avg"]]
    print("support min/max:", min(sup), max(sup), "n_test", sum(sup))
    print("classes:", classes)

    cols = ["lenet_avg", "lenet_max", "googlenet", "resnet18_aug",
            "resnet50_fe", "resnet50_ft"]
    lines = []
    for i, c in enumerate(classes):
        vals = " & ".join(f"{float(per[n][i]['f1']):.3f}" for n in cols)
        lines.append(f"{c} & {sup[i]} & {vals} \\\\")
    Path("per_class_rows.tex").write_text("\n".join(lines) + "\n")

    for n in RUNS:
        h = hist(n)
        m = json.load(open(R / n / "metrics.json"))
        bv = int(np.argmin(h["val_loss"]))
        print(f"\n{n}: best_ep {m['best_epoch']} min val loss {h['val_loss'][bv]:.3f} "
              f"at ep {bv + 1}; final train {h['train_loss'][-1]:.3f} val {h['val_loss'][-1]:.3f}; "
              f"final train acc {h['train_acc'][-1]:.3f} val acc {h['val_acc'][-1]:.3f}; "
              f"sec/ep {h['seconds'].mean():.2f}")
        gap = h["val_loss"] - h["train_loss"]
        first = int(np.argmax(gap > 0.1)) + 1 if (gap > 0.1).any() else None
        print("  first epoch val-train > 0.1:", first)
        cm = np.load(R / n / "confusion.npy")
        off = cm.copy()
        np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:6], off.shape))[0]
        print("  top confusions:", [(classes[a], classes[b], int(off[a, b])) for a, b in top])
        f1 = [float(r["f1"]) for r in per[n]]
        order = np.argsort(f1)
        print("  worst:", [(classes[j], f1[j]) for j in order[:3]],
              "best:", [(classes[j], f1[j]) for j in order[-3:]])
        mm = json.load(open(R / n / "metrics.json"))
        print("  metrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in mm.items()})


if __name__ == "__main__":
    main()
