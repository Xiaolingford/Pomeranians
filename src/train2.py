"""
train2_fixed.py - LDBST-MP Training with Class Imbalance Fixes

Key changes from original:
1. FocalLoss instead of weighted CrossEntropy
2. WeightedRandomSampler for balanced batches
3. Per-class accuracy logging to diagnose issues
4. Stronger regularization (label smoothing)
5. Lower learning rate with longer warmup
"""

import argparse
import os
from collections import OrderedDict, Counter

import torch
import torch._dynamo
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

# AMP helpers
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from torchvision.ops import nms, box_iou
from torch.utils.data import WeightedRandomSampler
import tempfile
import numpy as np

# -------------------------
# Import LDBST-MP model
# -------------------------
from model2 import (
    LDBST_MP,
    DetectionLoss,
    postprocess_detections,
    compute_box_iou,
)

from dataset import (
    get_dataloaders,
    get_dataloaders_auto,
)


# -------------------------
# YOLO Dataset (flexible directory structure)
# -------------------------
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class YOLODataset(Dataset):
    """
    YOLO-format detection dataset with flexible directory structure.
    
    Supports TWO formats:
    
    Format 1 - Flat (simple):
        root_dir/
        ├── images/
        │   └── *.jpg
        └── labels/
            └── *.txt
    
    Format 2 - Ultralytics/nested (your structure):
        root_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
        
        Use: YOLODataset(root_dir, split='train')
    
    Label format (YOLO): class_id x_center y_center width height (normalized 0-1)
    """
    
    def __init__(self, root_dir, img_size=224, augment=False, split=None):
        """
        Args:
            root_dir: Root directory containing images/ and labels/
            img_size: Target image size
            augment: Whether to apply data augmentation
            split: Optional split name ('train', 'val', 'test') for nested structure
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment
        self.split = split
        
        # Determine directory structure
        if split:
            # Ultralytics format: data_yolo/images/train/, data_yolo/labels/train/
            self.img_dir = os.path.join(root_dir, 'images', split)
            self.label_dir = os.path.join(root_dir, 'labels', split)
        else:
            # Flat format: root_dir/images/, root_dir/labels/
            self.img_dir = os.path.join(root_dir, 'images')
            self.label_dir = os.path.join(root_dir, 'labels')
        
        # Check if directories exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Images directory not found: {self.img_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.label_dir}")
        
        # Collect image files
        self.image_files = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        
        for img_name in sorted(os.listdir(self.img_dir)):
            if img_name.lower().endswith(valid_extensions):
                self.image_files.append(img_name)
        
        split_info = f" (split={split})" if split else ""
        print(f"[YOLODataset] Found {len(self.image_files)} images in {self.img_dir}{split_info}")
        print(f"[YOLODataset] Augmentation: {augment}")
        
        # Transforms
        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.aug_transform = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation before resize
        if self.aug_transform:
            image = self.aug_transform(image)
        
        # Apply base transforms
        image = self.to_tensor(image)
        
        # Load labels
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(self.label_dir, base_name + '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format (center, w, h) to (x1, y1, x2, y2)
                        x1 = max(0, min(1, x_center - width / 2))
                        y1 = max(0, min(1, y_center - height / 2))
                        x2 = max(0, min(1, x_center + width / 2))
                        y2 = max(0, min(1, y_center + height / 2))
                        
                        # Only add valid boxes
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        
        target = {
            "boxes": boxes,
            "labels": labels
        }
        
        return image, target


def collate_fn(batch):
    """Custom collate function for detection - handles variable number of boxes per image."""
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets


def get_yolo_dataloaders_flexible(train_dir=None, val_dir=None, data_root=None,
                                   batch_size=8, img_size=224, num_workers=4):
    """
    Create YOLO dataloaders - supports two modes:
    
    Mode 1 - Separate directories (use train_dir and val_dir):
        train_dir/
        ├── images/
        └── labels/
        val_dir/
        ├── images/
        └── labels/
    
    Mode 2 - Ultralytics format (use data_root):
        data_root/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
    
    Args:
        train_dir: Path to training data (Mode 1)
        val_dir: Path to validation data (Mode 1)
        data_root: Root path for Ultralytics format (Mode 2)
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of dataloader workers
    
    Returns:
        train_loader, val_loader
    """
    if data_root:
        # Mode 2: Ultralytics format with splits
        print(f"[dataloader] Using Ultralytics format from {data_root}")
        train_dataset = YOLODataset(data_root, img_size=img_size, augment=True, split='train')
        val_dataset = YOLODataset(data_root, img_size=img_size, augment=False, split='val')
    else:
        # Mode 1: Separate directories
        print(f"[dataloader] Using separate train/val directories")
        train_dataset = YOLODataset(train_dir, img_size=img_size, augment=True, split=None)
        val_dataset = YOLODataset(val_dir, img_size=img_size, augment=False, split=None)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


# -------------------------
# FocalLoss for Class Imbalance
# -------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss - focuses learning on hard examples by down-weighting easy ones.
    Much better than weighted CE for severe class imbalance.
    
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # class weights tensor
        self.gamma = gamma  # focusing parameter (2.0 is common)
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        n_classes = inputs.size(-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        
        # Compute log probabilities
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Focal weight: (1 - pt)^gamma
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy with smoothing
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# -------------------------
# Utilities
# -------------------------
def apply_score_thresh_and_nms(pb, ps, score_thr=0.5, nms_iou=0.45):
    """Apply confidence threshold and NMS to predictions."""
    keep = ps > score_thr
    pb = pb[keep]
    ps = ps[keep]
    if ps.numel() == 0:
        return pb, ps

    boxes_f = pb.to(torch.float32)
    scores_f = ps.to(torch.float32)
    keep_idx = nms(boxes_f, scores_f, nms_iou)
    return pb[keep_idx], ps[keep_idx]


def atomic_save(obj, path):
    """Atomically save checkpoint to prevent corruption."""
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


def _safe_load_state_dict(model, state_dict):
    """Try loading state_dict with fallbacks for module prefix issues."""
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError:
        try:
            new_sd = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
            model.load_state_dict(new_sd)
            return True
        except Exception:
            pass
        try:
            model.load_state_dict(state_dict, strict=False)
            return True
        except Exception:
            return False


def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def get_class_distribution(loader):
    """Count samples per class in a dataloader."""
    class_counts = Counter()
    for _, labels in loader:
        class_counts.update(labels.tolist())
    return dict(sorted(class_counts.items()))


def compute_sample_weights(dataset):
    """Compute per-sample weights for WeightedRandomSampler."""
    # Get all labels
    labels = []
    for _, label in dataset:
        labels.append(label if isinstance(label, int) else label.item())
    
    labels = np.array(labels)
    class_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)
    
    # Weight each class inversely proportional to its frequency
    class_weights = {cls: n_samples / (n_classes * count) 
                     for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = np.array([class_weights[label] for label in labels])
    
    return torch.from_numpy(sample_weights).float()


def eval_epoch_detailed(model, loader, device, use_amp, criterion=None, class_names=None):
    """Evaluate classification model with per-class metrics."""
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

    # Overall metrics
    acc = accuracy_score(ys, ps)
    P, R, F1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    
    # Per-class metrics
    P_per, R_per, F1_per, support = precision_recall_fscore_support(
        ys, ps, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(ys, ps)
    
    val_loss = None
    if criterion is not None:
        val_loss = total_loss / max(len(getattr(loader, "dataset", loader.dataset)), 1)
    
    return {
        'acc': acc,
        'precision': P,
        'recall': R,
        'f1': F1,
        'val_loss': val_loss,
        'per_class_precision': P_per,
        'per_class_recall': R_per,
        'per_class_f1': F1_per,
        'support': support,
        'confusion_matrix': cm,
        'predictions': ps,
        'ground_truth': ys
    }


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
            lr_scale = self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lr_scale = max(lr_scale, self.min_lr / self.base_lrs[0])
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# -------------------------
# Detection utilities (unchanged)
# -------------------------
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

    sort_idx = torch.argsort(all_scores, descending=True)
    all_tp = all_tp[sort_idx]
    all_fp = all_fp[sort_idx]

    cum_tp = torch.cumsum(all_tp, dim=0)
    cum_fp = torch.cumsum(all_fp, dim=0)

    recalls = cum_tp / (n_positives + 1e-6)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

    recall_points = torch.linspace(0, 1, num_points)
    precision_interp = torch.zeros(num_points)
    for i, r in enumerate(recall_points):
        mask = recalls >= r
        precision_interp[i] = precisions[mask].max() if mask.any() else 0.0

    return precision_interp.mean().item()


def evaluate_detection_batch(pred_boxes, true_boxes, iou_thresh=0.5):
    """Compute basic detection metrics for one batch."""
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0
    if len(true_boxes) == 0:
        return 0.0, 0.0, 0.0

    ious = box_iou(pred_boxes, true_boxes)
    ious_cp = ious.clone()
    tp = 0
    matched_ious = []
    
    while True:
        max_val = ious_cp.max()
        if max_val < iou_thresh:
            break
        idx = (ious_cp == max_val).nonzero(as_tuple=False)[0]
        pred_idx, true_idx = int(idx[0].item()), int(idx[1].item())
        tp += 1
        matched_ious.append(max_val.item())
        ious_cp[pred_idx, :] = -1
        ious_cp[:, true_idx] = -1

    fp = len(pred_boxes) - tp
    fn = len(true_boxes) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    mean_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0
    return precision, recall, mean_iou


# -------------------------
# Training Function
# -------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs("checkpoints", exist_ok=True)
    log_file = "ldbst_train_log.txt"
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("mode,epoch,train_loss,val_loss,acc,P,R,F1,class0_acc,class1_acc\n")

    # --------------------------------
    # CLASSIFICATION MODE
    # --------------------------------
    if args.task == "classification":
        print("=" * 70)
        print("[mode] Training LDBST-MP CLASSIFICATION with IMBALANCE FIXES")
        print("=" * 70)

        num_workers = min(4, os.cpu_count() or 4)

        # Load data (we'll recreate loaders with balanced sampling)
        if args.auto_split:
            tr, va, te, classes = get_dataloaders_auto(
                args.data,
                batch_size=args.batch_size,
                val_ratio=args.val_ratio,
                test_ratio=0.1,
                seed=args.seed,
                num_workers=num_workers,
            )
            train_dataset = tr.dataset
            val_dataset = va.dataset
        else:
            train_path = args.train_dir if args.train_dir else os.path.join(args.data, "train")
            val_path = args.val_dir if args.val_dir else os.path.join(args.data, "val")

            tr, va, classes = get_dataloaders(
                train_dir=train_path,
                val_dir=val_path,
                batch_size=args.batch_size,
                num_workers=num_workers,
                img_size=args.img_size,
                use_augmentation=True,  # Enable online augmentation
            )
            train_dataset = tr.dataset
            val_dataset = va.dataset

        num_classes = len(classes)
        print(f"[data] Found {num_classes} classes: {classes}")
        
        # -------------------------
        # FIX 1: Analyze class distribution
        # -------------------------
        print("\n[analysis] Checking class distribution...")
        train_dist = get_class_distribution(tr)
        val_dist = get_class_distribution(va)
        
        print(f"[train] Class distribution: {train_dist}")
        print(f"[val]   Class distribution: {val_dist}")
        
        total_train = sum(train_dist.values())
        total_val = sum(val_dist.values())
        
        for cls_idx, cls_name in enumerate(classes):
            train_pct = train_dist.get(cls_idx, 0) / total_train * 100
            val_pct = val_dist.get(cls_idx, 0) / total_val * 100
            print(f"        Class {cls_idx} ({cls_name}): train={train_pct:.1f}%, val={val_pct:.1f}%")
        
        # -------------------------
        # FIX 2: Compute class weights based on actual distribution
        # -------------------------
        class_counts = np.array([train_dist.get(i, 1) for i in range(num_classes)])
        # Inverse frequency weighting
        class_weights = total_train / (num_classes * class_counts)
        # Normalize so minimum weight is 1.0
        class_weights = class_weights / class_weights.min()
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        
        print(f"\n[weights] Computed class weights: {class_weights.tolist()}")
        
        # -------------------------
        # FIX 3: Create balanced sampler for training
        # -------------------------
        print("[sampler] Creating WeightedRandomSampler for balanced training...")
        sample_weights = compute_sample_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Recreate train loader with sampler (no shuffle when using sampler)
        from torch.utils.data import DataLoader
        tr = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=True
        )
        print(f"[sampler] Balanced loader created with {len(tr)} batches")
        
        # -------------------------
        # Model
        # -------------------------
        model = LDBST_MP(
            img_size=args.img_size,
            num_classes=num_classes,
            num_heads=4,
            task="classification"
        ).to(device)
        
        print(f"[model] LDBST-MP Parameters: {count_parameters(model):.2f}M")

        # -------------------------
        # FIX 4: Use FocalLoss instead of weighted CE
        # -------------------------
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=2.0,  # Focus on hard examples
            label_smoothing=0.1  # Regularization
        )
        print(f"[loss] Using FocalLoss(gamma=2.0, label_smoothing=0.1)")
        
        # -------------------------
        # FIX 5: Lower LR with warmup
        # -------------------------
        base_lr = args.lr * 0.5  # Start lower
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=base_lr, 
            weight_decay=args.weight_decay
        )
        
        warmup_epochs = max(3, args.epochs // 10)
        scheduler = WarmupCosineScheduler(
            optimizer, 
            warmup_epochs=warmup_epochs, 
            total_epochs=args.epochs,
            min_lr=1e-6
        )
        print(f"[optimizer] AdamW(lr={base_lr:.2e}, wd={args.weight_decay})")
        print(f"[scheduler] Warmup({warmup_epochs} epochs) + CosineAnnealing")
        
        use_amp = (device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # Resume checkpoint
        best_val_f1 = 0.0  # Use F1 instead of accuracy for imbalanced data
        best_val_acc = 0.0
        patience = args.patience
        counter = 0
        resume_path = "checkpoints/ldbst_classifier_latest.pth"
        start_epoch = 1
        
        if os.path.exists(resume_path) and not args.fresh_start:
            try:
                checkpoint = torch.load(resume_path, map_location=device)
                loaded = _safe_load_state_dict(model, checkpoint.get("model_state", checkpoint))
                if not loaded:
                    print("[resume] Warning: could not fully load model state_dict")
                if "opt_state" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["opt_state"])
                    except Exception as e:
                        print(f"[resume] Warning: failed to load optimizer state: {e}")
                best_val_f1 = checkpoint.get("best_val_f1", 0.0)
                best_val_acc = checkpoint.get("best_val_acc", 0.0)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"[resume] Loaded checkpoint from {resume_path} (epoch {start_epoch-1})")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint: {e}")

        # Performance optimizations
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        # -------------------------
        # Training loop
        # -------------------------
        print("\n" + "=" * 70)
        print("Starting Training...")
        print("=" * 70)
        
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            current_lr = scheduler.get_lr()

            pbar = tqdm(tr, desc=f"[Class] Epoch {epoch}/{args.epochs} (LR={current_lr:.2e})", unit="batch")
            for x, y in pbar:
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
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss = running_loss / max(len(train_dataset), 1)

            # Detailed validation
            metrics = eval_epoch_detailed(model, va, device, use_amp, criterion=criterion, class_names=classes)
            
            val_acc = metrics['acc']
            P = metrics['precision']
            R = metrics['recall']
            F1 = metrics['f1']
            val_loss = metrics['val_loss']
            
            # Per-class accuracy
            per_class_acc = []
            for cls_idx in range(num_classes):
                mask = np.array(metrics['ground_truth']) == cls_idx
                if mask.sum() > 0:
                    cls_preds = np.array(metrics['predictions'])[mask]
                    cls_acc = (cls_preds == cls_idx).mean()
                    per_class_acc.append(cls_acc)
                else:
                    per_class_acc.append(0.0)
            
            # Print results
            print(f"\n[epoch {epoch:02d}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            print(f"           Overall: acc={val_acc:.4f}, P={P:.4f}, R={R:.4f}, F1={F1:.4f}")
            for cls_idx, cls_name in enumerate(classes):
                cls_p = metrics['per_class_precision'][cls_idx]
                cls_r = metrics['per_class_recall'][cls_idx]
                cls_f1 = metrics['per_class_f1'][cls_idx]
                cls_support = metrics['support'][cls_idx]
                print(f"           {cls_name}: P={cls_p:.4f}, R={cls_r:.4f}, F1={cls_f1:.4f}, n={cls_support}")
            
            # Confusion matrix
            cm = metrics['confusion_matrix']
            print(f"           Confusion Matrix:")
            print(f"                    Predicted")
            print(f"                    {' '.join([f'{c[:8]:>8}' for c in classes])}")
            for i, row in enumerate(cm):
                print(f"           {classes[i][:8]:>8} {' '.join([f'{v:>8}' for v in row])}")

            # Logging
            with open(log_file, "a") as f:
                cls_accs = ','.join([f'{a:.4f}' for a in per_class_acc])
                f.write(f"class,{epoch},{train_loss:.4f},{val_loss:.4f},{val_acc:.4f},"
                       f"{P:.4f},{R:.4f},{F1:.4f},{cls_accs}\n")

            scheduler.step()

            # -------------------------
            # FIX 6: Use F1 for best model selection (better for imbalanced)
            # -------------------------
            improved = False
            if F1 > best_val_f1:
                best_val_f1 = F1
                best_val_acc = val_acc
                improved = True
                counter = 0
                atomic_save(model.state_dict(), f"checkpoints/ldbst_classifier_best_epoch{epoch}.pth")
                print(f"✅ New best model (F1={F1:.4f}, acc={val_acc:.4f})")
            else:
                counter += 1
                print(f"⏳ No improvement for {counter}/{patience} epochs (best F1={best_val_f1:.4f})")
                
                if counter >= patience:
                    print("🛑 Early stopping triggered.")
                    break

            # Save resume checkpoint
            try:
                atomic_save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "best_val_f1": best_val_f1,
                    "best_val_acc": best_val_acc
                }, resume_path)
            except Exception as e:
                print(f"[save] Warning: failed to save checkpoint: {e}")

        print(f"\n{'=' * 70}")
        print(f"✅ Classification training complete!")
        print(f"   Best F1: {best_val_f1:.4f}")
        print(f"   Best Accuracy: {best_val_acc:.4f}")
        print(f"{'=' * 70}")
        
        try:
            torch.save(model.state_dict(), args.output_model)
            print(f"[save] Final model saved to {args.output_model}")
        except Exception as e:
            print(f"[save] Warning: failed to save final model: {e}")

    # --------------------------------
    # DETECTION MODE
    # --------------------------------
    elif args.task == "detection":
        print(f"[mode] Training LDBST-MP DETECTION model with {args.num_classes} classes")
        print(f"[config] conf_thresh={args.conf_thresh}, nms_iou={args.nms_iou}")

        num_workers = min(4, os.cpu_count() or 4)
        
        # Use --train_dir and --val_dir
        # train_dir: folder with images/ and labels/ directly (flat)
        # val_dir: folder with images/val/ and labels/val/ (Ultralytics format)
        train_dir = args.train_dir if args.train_dir else "./data_yolo_preprocessed"
        val_dir = args.val_dir if args.val_dir else "./data_yolo"
        
        print(f"[data] Train directory: {train_dir}")
        print(f"[data]   Expected: {train_dir}/images/ and {train_dir}/labels/")
        print(f"[data] Val directory: {val_dir}")
        print(f"[data]   Expected: {val_dir}/images/val/ and {val_dir}/labels/val/")
        
        # Use dataset.py with proper online augmentation
        from dataset import get_yolo_dataloaders
        
        tr, va = get_yolo_dataloaders(
            train_root=train_dir,
            val_root=val_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=num_workers,
        )

        # Check data
        try:
            sample_imgs, sample_targets = next(iter(tr))
            print(f"[check] Example target[0]: {sample_targets[0]}")
        except Exception as e:
            print(f"[check] Could not fetch a sample from train loader: {e}")

        # Model initialization
        model = LDBST_MP(
            img_size=args.img_size,
            num_classes=args.num_classes,
            num_heads=4,
            task="detection"
        ).to(device)
        
        print(f"[model] LDBST-MP (Detection) Parameters: {count_parameters(model):.2f}M")

        # Load pretrained classification weights if provided
        if args.pretrained and os.path.exists(args.pretrained):
            try:
                print(f"[pretrained] Loading weights from {args.pretrained}")
                pretrained_dict = torch.load(args.pretrained, map_location=device)
                model_dict = model.state_dict()
                
                # Filter out detection head weights, keep backbone
                pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                                 if k in model_dict and "detection_head" not in k and "upsample" not in k}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print(f"[pretrained] Loaded {len(pretrained_dict)} layers from classifier")
            except Exception as e:
                print(f"[pretrained] Failed to load pretrained weights: {e}")

        # Loss and optimizer
        criterion = DetectionLoss(
            lambda_box=5.0,
            lambda_cls=1.0,
            lambda_obj=0.25,
            pos_iou_thresh=0.4,
            neg_iou_thresh=0.2
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Scheduler with warmup
        warmup_epochs = min(5, args.epochs // 10)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, args.epochs)
        print(f"[scheduler] Using warmup ({warmup_epochs} epochs) + cosine annealing")
        
        use_amp = (device.type == "cuda")
        scaler = GradScaler(enabled=use_amp)

        # Resume from checkpoint
        best_loss = float("inf")
        best_map = 0.0
        resume_path = "checkpoints/ldbst_detector_latest.pth"
        start_epoch = 1
        
        if os.path.exists(resume_path) and not args.fresh_start:
            try:
                checkpoint = torch.load(resume_path, map_location=device)
                loaded = _safe_load_state_dict(model, checkpoint.get("model_state", checkpoint))
                if not loaded:
                    print("[resume] Warning: could not fully load state_dict")
                if "opt_state" in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint["opt_state"])
                    except Exception as e:
                        print(f"[resume] Warning: failed to load optimizer state: {e}")
                best_loss = checkpoint.get("best_loss", best_loss)
                best_map = checkpoint.get("best_map", best_map)
                start_epoch = checkpoint.get("epoch", 0) + 1
                print(f"[resume] Loaded checkpoint from {resume_path} (epoch {start_epoch-1})")
            except Exception as e:
                print(f"[resume] Failed to load checkpoint: {e}")

        # Performance optimizations
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        # Training Loop
        print("\n" + "=" * 70)
        print("Starting Detection Training...")
        print("=" * 70)
        
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            total_loss = 0.0
            current_lr = optimizer.param_groups[0]['lr']

            # Training
            pbar = tqdm(tr, desc=f"[Det] Epoch {epoch}/{args.epochs} (LR={current_lr:.2e})", unit="batch")
            for imgs, targets in pbar:
                # Move to device
                if device.type == "cuda":
                    imgs = imgs.to(device, memory_format=torch.channels_last, non_blocking=True)
                else:
                    imgs = imgs.to(device)

                # Move targets to device
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

            avg_loss = total_loss / max(len(getattr(tr, "dataset", tr.dataset)), 1)
            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            val_precision, val_recall, val_iou = 0.0, 0.0, 0.0
            n_images = 0
            pred_boxes_all, pred_scores_all, true_boxes_all = [], [], []
            n_val_batches = 0
            total_gt_boxes = 0
            total_pred_above_thresh = 0

            with torch.no_grad():
                with autocast(device_type="cuda", enabled=use_amp):
                    for imgs, targets in va:
                        n_val_batches += 1
                        
                        # Move to device
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

                        # Count GT boxes
                        for t in targets:
                            total_gt_boxes += len(t["boxes"])

                        # Forward pass
                        pred_boxes, pred_logits, pred_obj = model(imgs)
                        loss = criterion(pred_boxes, pred_logits, pred_obj, targets)
                        val_loss += loss.item() * imgs.size(0)

                        # Compute predictions
                        class_probs = pred_logits.softmax(dim=-1)
                        max_class_probs, pred_labels = class_probs.max(dim=-1)
                        obj_scores = pred_obj.sigmoid()
                        pred_scores = (max_class_probs * obj_scores).detach().cpu()

                        # Debug: check score distribution (first batch only)
                        if n_val_batches == 1:
                            print(f"\n[DEBUG] First val batch:")
                            print(f"  Objectness range: [{obj_scores.min().item():.4f}, {obj_scores.max().item():.4f}]")
                            print(f"  Class prob range: [{max_class_probs.min().item():.4f}, {max_class_probs.max().item():.4f}]")
                            print(f"  Final score range: [{pred_scores.min().item():.4f}, {pred_scores.max().item():.4f}]")
                            print(f"  Scores > {args.conf_thresh}: {(pred_scores > args.conf_thresh).sum().item()}")
                            print(f"  GT boxes in batch: {sum(len(t['boxes']) for t in targets)}")
                            print(f"  Loss value: {loss.item():.4f}")

                        # Per-image evaluation
                        for b in range(len(targets)):
                            tb = targets[b]["boxes"].detach().cpu()
                            pb = pred_boxes[b].detach().cpu()
                            ps = pred_scores[b].detach().cpu()

                            # Count predictions above threshold
                            n_above = (ps > args.conf_thresh).sum().item()
                            total_pred_above_thresh += n_above

                            # Apply threshold and NMS
                            pb, ps = apply_score_thresh_and_nms(
                                pb, ps,
                                score_thr=args.conf_thresh,
                                nms_iou=args.nms_iou
                            )

                            # Store for mAP
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

            # Print debug summary
            print(f"\n[DEBUG] Validation summary:")
            print(f"  Total val batches: {n_val_batches}")
            print(f"  Total GT boxes: {total_gt_boxes}")
            print(f"  Total predictions > {args.conf_thresh}: {total_pred_above_thresh}")
            print(f"  Images with valid detections: {n_images}")

            # Aggregate metrics
            val_loss = val_loss / max(len(getattr(va, "dataset", va.dataset)), 1)
            val_precision /= max(1, n_images)
            val_recall /= max(1, n_images)
            val_iou /= max(1, n_images)
            val_map = compute_map(pred_boxes_all, pred_scores_all, true_boxes_all, iou_thresh=0.5)

            print(f"\n[epoch {epoch:02d}] train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            print(f"           P={val_precision:.4f}, R={val_recall:.4f}, IoU={val_iou:.4f}, mAP@0.5={val_map:.4f}")

            # Logging
            log_file = "ldbst_train_log.txt"
            with open(log_file, "a") as f:
                f.write(f"det,{epoch},{avg_loss:.4f},{val_loss:.4f},,"
                       f"{val_precision:.4f},{val_recall:.4f},{val_iou:.4f},{val_map:.4f}\n")

            # Save checkpoint
            try:
                atomic_save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "best_map": best_map
                }, resume_path)
            except Exception as e:
                print(f"[save] Warning: failed to save checkpoint: {e}")

            # Save best model based on mAP
            if val_map > best_map:
                best_map = val_map
                atomic_save(model.state_dict(), f"checkpoints/ldbst_detector_best_epoch{epoch}.pth")
                print(f"✅ New best model (mAP@0.5={val_map:.4f})")

        print(f"\n{'=' * 70}")
        print(f"✅ Detection training complete!")
        print(f"   Best mAP@0.5: {best_map:.4f}")
        print(f"{'=' * 70}")
        
        try:
            torch.save(model.state_dict(), args.output_model)
            print(f"[save] Final model saved to {args.output_model}")
        except Exception as e:
            print(f"[save] Warning: failed to save final model: {e}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Train LDBST-MP with class imbalance fixes")
    
    # Task and data
    ap.add_argument("--task", choices=["classification", "detection"], default="classification",
                    help="Training task")
    ap.add_argument("--data", default="./data", help="Data directory")
    ap.add_argument("--train_dir", default=None, help="Override training directory") 
    ap.add_argument("--val_dir", default=None, help="Override validation directory")    
    ap.add_argument("--auto_split", action="store_true", help="Auto-split data")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    
    # Training hyperparameters
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5, help="Learning rate (lower default)")
    ap.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay (higher default)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    
    # Detection-specific
    ap.add_argument("--conf_thresh", type=float, default=0.01, help="Confidence threshold for detection (low for early training)")
    ap.add_argument("--nms_iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--num_classes", type=int, default=5)
    
    # Transfer learning
    ap.add_argument("--pretrained", default="", help="Path to pretrained weights")
    ap.add_argument("--fresh_start", action="store_true", help="Ignore existing checkpoints")
    
    # Output
    ap.add_argument("--output_model", default="ldbst_classifier.pth", help="Output model path")
    
    args = ap.parse_args()

    print(f"\n{'=' * 70}")
    print(f"🚀 LDBST-MP Training: {args.task.upper()} (with imbalance fixes)")
    print(f"{'=' * 70}")
    print(f"[config] Epochs={args.epochs}, LR={args.lr}, Batch={args.batch_size}")
    print(f"[config] Weight Decay={args.weight_decay}, Patience={args.patience}")
    print(f"[config] Image size={args.img_size}x{args.img_size}")
    
    train(args)


if __name__ == "__main__":
    main()