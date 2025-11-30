import argparse
import os
from collections import OrderedDict

import torch
import torch._dynamo
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# amp helpers
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# -------------------------
# Imports (project)
# -------------------------
from model import (
    DualBranchSwinCNNDetector,
    DetectionLoss,
)
from dataset import (
    get_dataloaders,
    get_dataloaders_auto,
    get_yolo_dataloaders,
)
# -------------------------
# NMS / threshold helpers
# -------------------------
from torchvision.ops import nms, box_iou
import tempfile
import numpy as np

def apply_score_thresh_and_nms(pb, ps, score_thr=0.5, nms_iou=0.45):
    keep = ps > score_thr
    pb = pb[keep]
    ps = ps[keep]
    if ps.numel() == 0:
        return pb, ps

    # make sure they're float32 for NMS
    boxes_f = pb.to(torch.float32)
    scores_f = ps.to(torch.float32)

    keep_idx = nms(boxes_f, scores_f, nms_iou)
    return pb[keep_idx], ps[keep_idx]

def atomic_save(obj, path):
    dirn = os.path.dirname(path)
    os.makedirs(dirn or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirn, prefix=".tmp_ckpt_", suffix=".pth")
    os.close(fd)
    try:
        torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# -------------------------
# Helpers
# -------------------------
def _safe_load_state_dict(model, state_dict):
    """
    Try loading the given state_dict into model. Handles common issues
    such as 'module.' prefixes from DataParallel/Distributed wrappers
    and attempts a non-strict load as last resort.
    """
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError:
        # try stripping "module." prefix
        try:
            new_sd = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
            model.load_state_dict(new_sd)
            return True
        except Exception:
            pass
        # try non-strict load
        try:
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False

def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def compute_map(pred_boxes_all, pred_scores_all, true_boxes_all, iou_thresh=0.5, num_points=101):
    """Compute mAP@0.5 (VOC-style interpolation)."""
    all_scores = []
    all_tp = []
    all_fp = []
    n_positives = sum(len(tb) for tb in true_boxes_all)

    for pb, ps, tb in zip(pred_boxes_all, pred_scores_all, true_boxes_all):
        if len(pb) == 0:
            continue

        ious = box_iou(pb, tb) if len(tb) > 0 else torch.zeros((len(pb), 0))
        detected = set()
        tp = torch.zeros(len(pb))
        fp = torch.zeros(len(pb))
        for i in range(len(pb)):
            if len(tb) == 0:
                fp[i] = 1
                continue
            max_iou, max_j = ious[i].max(0)
            if max_iou >= iou_thresh and max_j.item() not in detected:
                tp[i] = 1
                detected.add(max_j.item())
            else:
                fp[i] = 1

        all_scores.append(ps)
        all_tp.append(tp)
        all_fp.append(fp)

    if not all_scores:
        return 0.0

    all_scores = torch.cat(all_scores)
    all_tp = torch.cat(all_tp)
    all_fp = torch.cat(all_fp)

    # sort by confidence
    sort_idx = torch.argsort(all_scores, descending=True)
    all_tp = all_tp[sort_idx]
    all_fp = all_fp[sort_idx]

    cum_tp = torch.cumsum(all_tp, dim=0)
    cum_fp = torch.cumsum(all_fp, dim=0)

    recalls = cum_tp / (n_positives + 1e-6)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

    # VOC-style 11-point (or 101-point) interpolation
    recall_points = torch.linspace(0, 1, num_points)
    precision_interp = torch.zeros(num_points)
    for i, r in enumerate(recall_points):
        mask = recalls >= r
        precision_interp[i] = precisions[mask].max() if mask.any() else 0.0

    return precision_interp.mean().item()

# -------------------------
# Utility: detection eval (returns val_loss if criterion provided)
# -------------------------
def evaluate_detection_batch(pred_boxes, true_boxes, iou_thresh=0.5):
    """Compute basic detection metrics for one batch."""
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0
    if len(true_boxes) == 0:
        return 0.0, 0.0, 0.0

    ious = box_iou(pred_boxes, true_boxes)  # [N_pred, N_true]
    # greedy matching: find highest IoU, match, then remove row+col
    ious_cp = ious.clone()
    tp = 0
    matched_ious = []
    while True:
        max_val = ious_cp.max()
        if max_val < iou_thresh:
            break
        # get indices of max
        idx = (ious_cp == max_val).nonzero(as_tuple=False)[0]
        pred_idx, true_idx = int(idx[0].item()), int(idx[1].item())
        tp += 1
        matched_ious.append(max_val.item())
        # zero out matched pred row and true col
        ious_cp[pred_idx, :] = -1
        ious_cp[:, true_idx] = -1

    fp = len(pred_boxes) - tp
    fn = len(true_boxes) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)   # denominator == len(true_boxes)
    mean_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0
    return precision, recall, mean_iou

# -------------------------
# Utility: classification eval (returns val_loss if criterion provided)
# -------------------------
def eval_epoch(model, loader, device, use_amp, criterion: nn.Module = None):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    ctx = autocast(device_type="cuda", enabled=(device.type == "cuda" and use_amp))

    with torch.no_grad():
        with ctx:
            for x, y in loader:
                if device.type == "cuda":
                    x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                else:
                    x = x.to(device)
                    y = y.to(device)

                out = model(x)
                preds = out.argmax(1)
                ps.extend(preds.cpu().tolist())
                ys.extend(y.cpu().tolist())

                if criterion is not None:
                    total_loss += criterion(out, y).item() * x.size(0)

    acc = accuracy_score(ys, ps)
    P, R, F1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    val_loss = None
    if criterion is not None:
        val_loss = total_loss / max(len(getattr(loader, "dataset", loader.dataset)), 1)
    return (acc, P, R, F1, val_loss) if val_loss is not None else (acc, P, R, F1)


# -------------------------
# Warmup + Cosine LR Scheduler
# -------------------------
class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr_scale = self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / self.base_lrs[0])
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale


# -------------------------
# Training Function
# -------------------------
def train(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    # reproducibility & perf hints
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs("checkpoints", exist_ok=True)
    log_file = "train_log.txt"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("mode,epoch,train_loss,val_loss,acc_or_blank,P,R,IoU_or_F1,mAP_or_blank\n")


    # --------------------------------
    # CLASSIFICATION
    # --------------------------------
    if args.task == "classification":
        print("[mode] Training CLASSIFICATION model")

        num_workers = min(4, os.cpu_count() or 4)

        if args.auto_split:
            tr, va, te, classes = get_dataloaders_auto(
                args.data,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                test_ratio=0.1,
                seed=args.seed,
                num_workers=num_workers,
            )
        else:
            tr, va, classes = get_dataloaders(
                train_dir="./data_preprocessed/train",
                val_dir="./data/val",
                batch_size=args.batch_size,
                num_workers=num_workers,
            )

        num_classes = len(classes)
        model = DualBranchSwinCNNDetector(num_classes=num_classes, pretrained=True, task="classification").to(device)
        
        # ✅ Print model info
        print(f"[model] Parameters: {count_parameters(model):.2f}M")

        # criterion & optimizer created BEFORE resume so we can load opt state too
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
        use_amp = (device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # ---- resume BEFORE compiling
        best_val_acc = 0.0
        patience = 5
        counter = 0
        resume_path = "checkpoints/classifier_latest.pth"
        start_epoch = 1
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(resume_path, map_location=device)
                loaded = _safe_load_state_dict(model, checkpoint.get("model_state", checkpoint))
                if not loaded:
                    print("[resume] Warning: could not fully load model state_dict (attempted fallbacks).")
                # attempt to load optimizer if present
                if "opt_state" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["opt_state"])
                    except Exception as e:
                        print(f"[resume] Warning: failed to load optimizer state: {e}")
                best_val_acc = checkpoint.get("best_val_acc", 0.0)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"[resume] Loaded checkpoint from {resume_path} (epoch {start_epoch-1})")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint: {e}")

        # performance tweaks (CUDA-only) and then compile
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
            # compile after loading checkpoint (safer)
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                except Exception:
                    pass

        # ---- Training loop
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for x, y in tqdm(tr, desc=f"[Class] Epoch {epoch}/{args.epochs}", unit="batch"):
                if device.type == "cuda":
                    x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                else:
                    x = x.to(device)
                    y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * x.size(0)

            train_loss = running_loss / max(len(getattr(tr, "dataset", tr.dataset)), 1)

            # ---- Validation (compute real val_loss as well)
            val_acc, P, R, F1, val_loss = eval_epoch(model, va, device, use_amp, criterion=criterion)
            print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={val_acc:.4f}, P={P:.4f}, R={R:.4f}, F1={F1:.4f}")

            # log
            with open(log_file, "a") as f:
                f.write(f"class,{epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f},{P:.4f},{R:.4f},{F1:.4f}\n")

            # scheduler step (after epoch)
            scheduler.step()

            # checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
                torch.save(model.state_dict(), f"checkpoints/classifier_best_epoch{epoch}.pth")
                print(f" ✅ Saved new best model (val_acc={val_acc:.4f})")
            else:
                counter += 1
                print(f" No improvement for {counter}/{patience} epochs")
                if counter >= patience:
                    print(" Early stopping triggered.")
                    break

            # save resume checkpoint (model + optimizer)
            try:
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "best_val_acc": best_val_acc
                }, resume_path)
            except Exception as e:
                print(f"[save] Warning: failed to save resume checkpoint: {e}")

        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
        try:
            torch.save(model.state_dict(), args.output_model)
            print(f"[save] Final model saved to {args.output_model}")
        except Exception as e:
            print(f"[save] Warning: failed to save final model: {e}")

    # --------------------------------
    # DETECTION
    # --------------------------------
    elif args.task == "detection":
        print(f"[mode] Training DETECTION model with {args.num_classes} classes")
        print(f"[config] conf_thresh={args.conf_thresh}, nms_iou={args.nms_iou}")

        num_workers = min(4, os.cpu_count() or 4)
        tr, va = get_yolo_dataloaders(
            train_root="./data_yolo_preprocessed",
            val_root="./data_yolo",
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=num_workers,
        )

        try:
            sample_imgs, sample_targets = next(iter(tr))
            print(f"[check] Example target[0]: {sample_targets[0]}")
        except Exception as e:
            print(f"[check] Could not fetch a sample from train loader: {e}")

        model = DualBranchSwinCNNDetector(num_classes=args.num_classes, pretrained=True, task="detection").to(device)
        
        # ✅ Print model info
        print(f"[model] Parameters: {count_parameters(model):.2f}M")

        # ✅ Updated criterion with correct parameters
        criterion = DetectionLoss(
            lambda_box=2.0, 
            lambda_cls=1.0, 
            lambda_obj=1.0,
            pos_iou_thresh=0.5,  # Match improved model
            neg_iou_thresh=0.3
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        # ✅ Use warmup scheduler for better training
        warmup_epochs = min(5, args.epochs // 10)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, args.epochs)
        print(f"[scheduler] Using warmup ({warmup_epochs} epochs) + cosine annealing")
        
        use_amp = (device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        best_loss = float("inf")
        best_map = 0.0
        resume_path = "checkpoints/detector_latest.pth"
        start_epoch = 1
        if os.path.exists(resume_path):
            try:
                checkpoint = torch.load(resume_path, map_location=device)
                loaded = _safe_load_state_dict(model, checkpoint.get("model_state", checkpoint))
                if not loaded:
                    print("[resume] Warning: could not fully load detector state_dict (attempted fallbacks).")
                if "opt_state" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["opt_state"])
                    except Exception as e:
                        print(f"[resume] Warning: failed to load detector optimizer state: {e}")
                best_loss = checkpoint.get("best_loss", best_loss)
                best_map = checkpoint.get("best_map", best_map)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"[resume] Loaded checkpoint from {resume_path} (epoch {start_epoch-1})")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint: {e}")

        # performance tweaks & compile
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
            if hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)
                except Exception:
                    pass

        # -------------------------
        # EPOCH LOOP (train + val)
        # -------------------------
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0
            
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']

            # ---------- TRAIN ----------
            pbar = tqdm(tr, desc=f"[Det] Epoch {epoch}/{args.epochs} (LR={current_lr:.2e})", unit="batch")
            for imgs, targets in pbar:
                # move images
                if device.type == "cuda":
                    imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
                else:
                    imgs = imgs.to(device)

                # move targets to device (batch)
                for t in targets:
                    if device.type == "cuda":
                        t["boxes"] = t["boxes"].to(device, non_blocking=True)
                        if "labels" in t:
                            t["labels"] = t["labels"].to(device, non_blocking=True)
                    else:
                        t["boxes"] = t["boxes"].to(device)
                        if "labels" in t:
                            t["labels"] = t["labels"].to(device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=use_amp):
                    pred_boxes, pred_logits, pred_obj = model(imgs)
                    loss = criterion(pred_boxes, pred_logits, pred_obj, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * imgs.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            # epoch-level bookkeeping
            avg_loss = total_loss / max(len(getattr(tr, "dataset", tr.dataset)), 1)
            scheduler.step()

            # ---------- VALIDATION (once per epoch) ----------
            model.eval()
            val_loss = 0.0
            val_precision, val_recall, val_iou = 0.0, 0.0, 0.0
            n_images = 0
            pred_boxes_all, pred_scores_all, true_boxes_all = [], [], []

            with torch.no_grad():
                with autocast(device_type="cuda", enabled=use_amp):
                    for imgs, targets in va:
                        # move imgs and targets to device (validation)
                        if device.type == "cuda":
                            imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
                            for t in targets:
                                t["boxes"] = t["boxes"].to(device, non_blocking=True)
                                if "labels" in t:
                                    t["labels"] = t["labels"].to(device, non_blocking=True)
                        else:
                            imgs = imgs.to(device)
                            for t in targets:
                                t["boxes"] = t["boxes"].to(device)
                                if "labels" in t:
                                    t["labels"] = t["labels"].to(device)

                        # forward + loss
                        pred_boxes, pred_logits, pred_obj = model(imgs)
                        loss = criterion(pred_boxes, pred_logits, pred_obj, targets)

                        val_loss += loss.item() * imgs.size(0)

                        # compute per-image metrics
                        # ✅ FIX: Take max across classes, not just class 1
                        class_probs = pred_logits.softmax(dim=-1)  # [B, N, num_classes]
                        max_class_probs, pred_labels = class_probs.max(dim=-1)  # [B, N]
                        pred_scores = (max_class_probs * pred_obj.sigmoid()).detach().cpu()
                        
                        for b in range(len(targets)):
                            tb = targets[b]["boxes"].detach().cpu()
                            pb = pred_boxes[b].detach().cpu()
                            ps = pred_scores[b].detach().cpu()

                            # apply score threshold + NMS
                            pb, ps = apply_score_thresh_and_nms(pb, ps, score_thr=args.conf_thresh, nms_iou=args.nms_iou)

                            # Store for mAP computation
                            pred_boxes_all.append(pb)
                            pred_scores_all.append(ps)
                            true_boxes_all.append(tb)

                            if ps.numel() == 0:
                                continue

                            P, R, IoU = evaluate_detection_batch(pb, tb, iou_thresh=0.5)
                            val_precision += P
                            val_recall += R
                            val_iou += IoU
                            n_images += 1

            # --- Aggregate validation results (after loop)
            val_loss = val_loss / max(len(getattr(va, "dataset", va.dataset)), 1)
            val_precision /= max(1, n_images)
            val_recall /= max(1, n_images)
            val_iou /= max(1, n_images)
            val_map = compute_map(pred_boxes_all, pred_scores_all, true_boxes_all, iou_thresh=0.5)
            
            print(f"[epoch {epoch:02d}] train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"P={val_precision:.4f}, R={val_recall:.4f}, IoU={val_iou:.4f}, mAP@0.5={val_map:.4f}")

            # logging + checkpoint
            with open(log_file, "a") as f:
                f.write(
                    f"det,{epoch},{avg_loss:.4f},{val_loss:.4f},,"
                    f"{val_precision:.4f},{val_recall:.4f},{val_iou:.4f},{val_map:.4f}\n"
                )

            try:
                atomic_save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_map": best_map
                }, resume_path)
            except Exception as e:
                print(f"[save] Warning: failed to save detector resume checkpoint: {e}")

            # ✅ Save best model based on mAP (better metric than loss)
            if val_map > best_map:
                best_map = val_map
                atomic_save(model.state_dict(), f"checkpoints/detector_best_epoch{epoch}.pth")
                print(f"✅ New best detector model (mAP@0.5={val_map:.4f})")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Train microplastic detection/classification model")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--auto_split", action="store_true")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conf_thresh", type=float, default=0.25)  # ✅ Lowered default
    ap.add_argument("--nms_iou", type=float, default=0.45)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)  # ✅ Raised default for detection
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--task", choices=["classification", "detection"], default="classification")
    ap.add_argument("--output_model", default="student_model.pth")
    args = ap.parse_args()

    print(f"\n🚀 Starting training task: {args.task.upper()}")
    print(f"[config] Epochs={args.epochs}, LR={args.lr}, Batch={args.batch_size}")
    train(args)


if __name__ == "__main__":
    main()