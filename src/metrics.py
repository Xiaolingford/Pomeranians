from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import csv
import datetime as dt

from dataset import MicroplasticDataset
from model import DualBranchSwinTinyClassifier

DEVICE = "cpu"

def predict_probs(model, ds):
    model.eval()
    probs = []
    gts = []
    paths = []
    with torch.no_grad():
        for i in range(len(ds)):
            x_std, x_cla, y = ds[i]
            x_std = x_std.unsqueeze(0).to(DEVICE)
            x_cla = x_cla.unsqueeze(0).to(DEVICE)
            logits = model(x_std, x_cla)
            p = F.softmax(logits, dim=1)[0, 1].item()
            probs.append(p)
            gts.append(int(y))
            paths.append(str(ds.samples[i][0]))
    return np.array(gts), np.array(probs), paths

def plot_confusion(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(5,4), dpi=160)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def bar_macro(prec, rec, f1, out_path):
    fig, ax = plt.subplots(figsize=(5,4), dpi=160)
    ax.bar(["Precision","Recall","F1"], [prec, rec, f1])
    ax.set_ylim(0,1)
    ax.set_title("Macro metrics")
    for i, v in enumerate([prec,rec,f1]):
        ax.text(i, v+0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data_resized")
    ap.add_argument("--weights", type=str, default="classifier.pth")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold for class=1")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    data_root = ROOT / args.data
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    val_ds = MicroplasticDataset(root_dir=data_root, split="val", return_both=True, verbose=True)
    n_algae = sum(1 for y in val_ds.labels_only if y == 0)
    n_micro = sum(1 for y in val_ds.labels_only if y == 1)
    print(f"Val set: {len(val_ds)} images (algae={n_algae}, microplastics={n_micro})")

    model = DualBranchSwinTinyClassifier(num_classes=2, freeze_backbone=False).to(DEVICE)
    ckpt = ROOT / args.weights
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"Loaded {ckpt.name}")

    y_true, p_micro, paths = predict_probs(model, val_ds)
    thr = float(args.threshold)
    y_pred = (p_micro >= thr).astype(int)

    print("=== Metrics on validation subset ===")
    print(classification_report(y_true, y_pred, target_names=["algae","microplastics"], digits=2))

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plot_confusion(cm, ["algae","microplastics"], results_dir / "confusion_matrix.png")

    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bar_macro(prec, rec, f1, results_dir / "macro_metrics.png")

    if args.csv:
        outfile = results_dir / "predictions.csv"
        rows = zip(paths, y_true.tolist(), y_pred.tolist(), p_micro.tolist())
        try:
            with open(outfile, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path","true","pred","prob_micro","threshold"])
                for r in rows:
                    w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}", thr])
            print(f"Saved CSV: {outfile}")
        except PermissionError:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = results_dir / f"predictions_{ts}.csv"
            with open(alt, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path","true","pred","prob_micro","threshold"])
                for r in rows:
                    w.writerow([r[0], r[1], r[2], f"{r[3]:.6f}", thr])
            print(f"[WARN] predictions.csv in use. Wrote: {alt}")

if __name__ == "__main__":
    main()
