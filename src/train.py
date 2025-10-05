import argparse
import os
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from torch.cuda import amp

# -------------------------
# Imports 
# -------------------------
from model import (
    DualBranchSwinCNNDetector,  # unified detector that supports classification via task="classification"
    DetectionLoss,
)
from dataset import (
    get_dataloaders,
    get_dataloaders_auto,
    get_yolo_dataloaders,
)


# -------------------------
# Utility: classification eval
# -------------------------
def eval_epoch(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            ps.extend(out.argmax(1).cpu().tolist())
            ys.extend(y.cpu().tolist())
    acc = accuracy_score(ys, ps)
    P, R, F1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    return acc, P, R, F1


# -------------------------
# Training Function
# -------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    log_file = "train_log.txt"

    # Create log header if not exists
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("mode,epoch,train_loss,val_loss,acc,P,R,F1\n")

    # ======================================
    # CLASSIFICATION TRAINING
    # ======================================
    if args.task == "classification":
        print(f"[mode] Training CLASSIFICATION model")

        # ---- Dataloaders ----
        if args.auto_split:
            tr, va, te, classes = get_dataloaders_auto(args.data, batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=0.1,
        seed=args.seed,)
        else:
            tr, va, classes = get_dataloaders(
                os.path.join(args.data, "train"),
                os.path.join(args.data, "val"),
                args.batch_size,
            )

        num_classes = len(classes)
        # Use the unified detector class in classification mode
        model = DualBranchSwinCNNDetector(num_classes=num_classes, pretrained=True, task="classification").to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type if device.type in ["cuda", "cpu"] else "cuda")


        # ---- Early stopping & resume ----
        best_val_acc = 0.0
        patience = 5
        counter = 0
        resume_path = "checkpoints/classifier_latest.pth"

        start_epoch = 1
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(resume_path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["opt_state"])
                best_val_acc = checkpoint.get("best_val_acc", 0.0)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"[resume] Loaded checkpoint from {resume_path}")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint: {e}")

        # ---- Training loop ----
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for x, y in tqdm(tr, desc=f"[Class] Epoch {epoch}/{args.epochs}", unit="batch"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu"):
                    out = model(x)
                    loss = criterion(out, y)

                # scale -> backward -> unscale -> clip -> step -> update
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # IMPORTANT: unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * x.size(0)

            train_loss = running_loss / (len(tr.dataset) if len(tr.dataset) > 0 else 1)

            # ---- Validation ----
            val_acc, P, R, F1 = eval_epoch(model, va, device)
            print(f"[epoch {epoch:02d}] loss={train_loss:.4f}, acc={val_acc:.4f}, P={P:.4f}, R={R:.4f}, F1={F1:.4f}")

            # Log metrics (write NaN placeholders for val_loss here)
            with open(log_file, "a") as f:
                f.write(f"class,{epoch},{train_loss:.4f},,,{val_acc:.4f},{P:.4f},{R:.4f},{F1:.4f}\n")

            # ---- Checkpointing ----
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), f"checkpoints/classifier_best_epoch{epoch}.pth")
                print(f" Saved new best model (val_acc={val_acc:.4f})")
            else:
                counter += 1
                print(f" No improvement for {counter}/{patience} epochs")
                if counter >= patience:
                    print(" Early stopping triggered.")
                    break

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, resume_path)

        print(f"\n Training complete! Best validation accuracy: {best_val_acc:.4f}")
        torch.save(model.state_dict(), args.output_model)
        print(f"[save] Final model saved to {args.output_model}")


    # ======================================
    # DETECTION TRAINING
    # ======================================
    elif args.task == "detection":
        print(f"[mode] Training DETECTION model with {args.num_classes} classes")

        tr, va = get_yolo_dataloaders(args.data, batch_size=args.batch_size, img_size=args.img_size)

        # Data sanity check: protect against empty loader
        try:
            sample_imgs, sample_targets = next(iter(tr))
            print(f"[check] Example target[0]: {sample_targets[0]}")
        except Exception as e:
            print(f"[check] Could not fetch a sample from train loader: {e}")

        model = DualBranchSwinCNNDetector(num_classes=args.num_classes, pretrained=True, task="detection").to(device)
        criterion = DetectionLoss(lambda_box=2.0, lambda_cls=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler(device.type if device.type in ["cuda", "cpu"] else "cuda")


        best_loss = float("inf")
        resume_path = "checkpoints/detector_latest.pth"

        start_epoch = 1
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(resume_path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
                optimizer.load_state_dict(checkpoint["opt_state"])
                best_loss = checkpoint.get("best_loss", best_loss)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"[resume] Loaded checkpoint from {resume_path}")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint: {e}")

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0

            for imgs, targets in tqdm(tr, desc=f"[Det] Epoch {epoch}/{args.epochs}", unit="batch"):
                imgs = imgs.to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu"):
                    pred_boxes, pred_logits = model(imgs)
                    loss = criterion(pred_boxes, pred_logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # IMPORTANT: unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * imgs.size(0)

            avg_loss = total_loss / (len(tr.dataset) if len(tr.dataset) > 0 else 1)

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in va:
                    imgs = imgs.to(device)
                    pred_boxes, pred_logits = model(imgs)
                    loss = criterion(pred_boxes, pred_logits, targets)
                    val_loss += loss.item() * imgs.size(0)
            val_loss = val_loss / (len(va.dataset) if len(va.dataset) > 0 else 1)

            print(f"[epoch {epoch:02d}] train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

            # Log detection losses
            with open(log_file, "a") as f:
                f.write(f"det,{epoch},{avg_loss:.4f},{val_loss:.4f},,,,\n")

            # ---- Checkpointing (save resume info first) ----
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "best_loss": best_loss
            }, resume_path)

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "checkpoints/detector_best.pth")
                print(f"Saved new best detection model (val_loss={val_loss:.4f})")

        print(f"\n Detection training complete! Best loss: {best_loss:.4f}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Train microplastic detection/classification model")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--auto_split", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--task", choices=["classification", "detection"], default="classification")
    ap.add_argument("--output_model", default="student_model.pth")
    args = ap.parse_args()

    print(f"\n Starting training task: {args.task.upper()}")
    train(args)


if __name__ == "__main__":
    main()
