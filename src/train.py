from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import MicroplasticDataset, IM_SIZE
from model import DualBranchSwinTinyClassifier

DEVICE = "cpu"

def class_weights(dataset: MicroplasticDataset) -> torch.Tensor:
    labels = np.array([y for y in dataset.labels_only], dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    # inverse frequency
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def make_loaders(root, batch=16, workers=0):
    train_ds = MicroplasticDataset(root_dir=root, split="train", return_both=True, verbose=True)
    val_ds   = MicroplasticDataset(root_dir=root, split="val",   return_both=True, verbose=True)

    # Weighted sampler for imbalance
    labels = np.array(train_ds.labels_only)
    class_sample_counts = np.bincount(labels, minlength=2)
    weights_per_class = 1.0 / np.maximum(class_sample_counts, 1)
    weights = weights_per_class[labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False,  num_workers=workers, pin_memory=False)
    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x_std, x_cla, y in loader:
        x_std = x_std.to(device)
        x_cla = x_cla.to(device)
        y = y.to(device)
        logits = model(x_std, x_cla)
        loss = ce(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total,1), correct / max(total,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data_resized", help="use data_resized if you preprocessed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--try-pretrained", action="store_true", help="try to fetch pretrained Swin (requires internet)")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    data_root = (ROOT / args.data).resolve()
    print(f"Using data from: {data_root}")

    train_loader, val_loader = make_loaders(data_root, batch=args.batch)

    model = DualBranchSwinTinyClassifier(num_classes=2, freeze_backbone=False, try_pretrained=args.try_pretrained).to(DEVICE)
    # class weights for CE
    w = class_weights(train_loader.dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = -1.0
    ckpt_path = ROOT / "classifier.pth"

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        n = 0
        for x_std, x_cla, y in train_loader:
            x_std = x_std.to(DEVICE)
            x_cla = x_cla.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_std, x_cla)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            n += y.size(0)

        tr_loss = running / max(n,1)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        print(f"[epoch {epoch}] train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ New best. Saved to {ckpt_path}")

    print(f"Done. Best val_acc={best_acc:.3f}")

if __name__ == "__main__":
    main()
