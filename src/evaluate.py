"""
evaluate.py - Evaluate trained model on test set

Usage:
    python evaluate.py --task classification --model classifier_v2.pth --test_dir ./data_test_ready --num_classes 2
    python evaluate.py --task detection --model detector_v2.pth --test_dir ./data_yolo/images/test --labels_dir ./data_yolo/labels/test --num_classes 2
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# ============================================================
# METRICS
# ============================================================

def compute_classification_metrics(y_true, y_pred, class_names=None):
    """Compute precision, recall, F1 for each class and overall."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_classes = len(np.unique(y_true))
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]
    
    results = {}
    
    # Per-class metrics
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        tn = np.sum((y_true != c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[class_names[c]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(y_true == c)),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    # Overall metrics
    accuracy = np.mean(y_true == y_pred)
    
    # Macro average
    macro_p = np.mean([results[c]['precision'] for c in class_names])
    macro_r = np.mean([results[c]['recall'] for c in class_names])
    macro_f1 = np.mean([results[c]['f1'] for c in class_names])
    
    # Weighted average
    total = len(y_true)
    weighted_p = sum(results[c]['precision'] * results[c]['support'] for c in class_names) / total
    weighted_r = sum(results[c]['recall'] * results[c]['support'] for c in class_names) / total
    weighted_f1 = sum(results[c]['f1'] * results[c]['support'] for c in class_names) / total
    
    results['_overall'] = {
        'accuracy': accuracy,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_p,
        'weighted_recall': weighted_r,
        'weighted_f1': weighted_f1,
        'total': total
    }
    
    return results


def print_confusion_matrix(y_true, y_pred, class_names):
    """Print confusion matrix."""
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    # Print header
    max_name_len = max(len(name) for name in class_names)
    header = " " * (max_name_len + 2) + "Predicted"
    print(header)
    
    # Column headers
    col_header = " " * (max_name_len + 2)
    for name in class_names:
        col_header += f"{name[:8]:>10}"
    print(col_header)
    
    # Rows
    for i, name in enumerate(class_names):
        row = f"{name:>{max_name_len}}  "
        for j in range(n_classes):
            row += f"{cm[i, j]:>10}"
        print(row)
    
    return cm


# ============================================================
# CLASSIFICATION EVALUATION
# ============================================================

def evaluate_classification(args):
    """Evaluate classification model on test set."""
    from model2 import LDBST_MP
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] Using: {device}")
    
    # Load model
    print(f"[model] Loading from {args.model}")
    model = LDBST_MP(img_size=224, num_classes=args.num_classes, task="classification")
    
    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"[model] Parameters: {model.count_parameters():,}")
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    class_names = test_dataset.classes
    print(f"[data] Test set: {len(test_dataset)} images")
    print(f"[data] Classes: {class_names}")
    
    # Count per class
    class_counts = defaultdict(int)
    for _, label in test_dataset:
        class_counts[class_names[label]] += 1
    print(f"[data] Distribution: {dict(class_counts)}")
    
    # Inference
    print("\n[eval] Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Compute metrics
    results = compute_classification_metrics(all_labels, all_preds, class_names)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Accuracy':<20} {results['_overall']['accuracy']:>10.4f}")
    print(f"{'Macro F1':<20} {results['_overall']['macro_f1']:>10.4f}")
    print(f"{'Weighted F1':<20} {results['_overall']['weighted_f1']:>10.4f}")
    
    print(f"\n{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 57)
    for name in class_names:
        r = results[name]
        print(f"{name:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['support']:>10}")
    
    print("\nConfusion Matrix:")
    cm = print_confusion_matrix(all_labels, all_preds, class_names)
    
    # Detailed error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    errors = np.array(all_preds) != np.array(all_labels)
    n_errors = np.sum(errors)
    print(f"Total errors: {n_errors}/{len(all_labels)} ({n_errors/len(all_labels)*100:.2f}%)")
    
    for i, name in enumerate(class_names):
        mask = np.array(all_labels) == i
        class_errors = errors & mask
        n_class_errors = np.sum(class_errors)
        n_class_total = np.sum(mask)
        print(f"  {name}: {n_class_errors}/{n_class_total} misclassified ({n_class_errors/n_class_total*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("SUMMARY FOR YOUR PAPER")
    print("=" * 60)
    print(f"""
Model: LDBST-MP (Classification)
Test Set Size: {len(test_dataset)} images
Classes: {', '.join(class_names)}

Results:
  Accuracy:  {results['_overall']['accuracy']*100:.2f}%
  Macro F1:  {results['_overall']['macro_f1']:.4f}
  
Per-Class Performance:
""")
    for name in class_names:
        r = results[name]
        print(f"  {name}: P={r['precision']:.4f}, R={r['recall']:.4f}, F1={r['f1']:.4f}")
    
    return results


# ============================================================
# DETECTION EVALUATION 
# ============================================================

def evaluate_detection(args):
    """Evaluate detection model on test set."""
    from model2 import LDBST_MP, postprocess_detections
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] Using: {device}")
    
    # Load model
    print(f"[model] Loading from {args.model}")
    model = LDBST_MP(img_size=224, num_classes=args.num_classes, task="detection")
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"[model] Parameters: {model.count_parameters():,}")
    print(f"[config] Confidence threshold: {args.conf_thresh}")
    
    # Load test images and labels
    import cv2
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    
    transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    
    images_dir = Path(args.test_dir) / "images"
    labels_dir = Path(args.test_dir) / "labels"
    
    if not images_dir.exists():
        images_dir = Path(args.test_dir)
        labels_dir = Path(args.labels_dir) if args.labels_dir else Path(args.test_dir).parent / "labels"
    
    image_files = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    print(f"[data] Test images: {len(image_files)}")
    print(f"[data] Labels dir: {labels_dir}")
    
    # Metrics accumulators
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_iou = []
    
    # For mAP calculation
    all_detections = []  # (confidence, is_tp, iou)
    total_gt = 0
    
    print(f"\n[eval] Running inference at conf_thresh={args.conf_thresh}...")
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Testing"):
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Transform
            transformed = transform(image=img)
            img_t = transformed["image"].unsqueeze(0).to(device)
            
            # Inference
            pred_boxes, pred_logits, pred_obj = model(img_t)
            
            # Postprocess with specified threshold
            dets = postprocess_detections(
                pred_boxes[0], pred_logits[0], pred_obj[0],
                conf_thresh=args.conf_thresh,
                iou_thresh=0.45,
                img_size=224
            )
            
            # Load ground truth
            label_path = labels_dir / (img_path.stem + ".txt")
            gt_boxes = []
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls, xc, yc, w, h = int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            # Convert YOLO to xyxy (pixel coords)
                            x1 = (xc - w/2) * 224
                            y1 = (yc - h/2) * 224
                            x2 = (xc + w/2) * 224
                            y2 = (yc + h/2) * 224
                            gt_boxes.append([x1, y1, x2, y2, cls])
            
            gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 5))
            total_gt += len(gt_boxes)
            
            # Match predictions to GT
            matched_gt = set()
            
            for det in dets:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    # Calculate IoU
                    gx1, gy1, gx2, gy2 = gt[:4]
                    
                    inter_x1 = max(x1, gx1)
                    inter_y1 = max(y1, gy1)
                    inter_x2 = min(x2, gx2)
                    inter_y2 = min(y2, gy2)
                    
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (gx2 - gx1) * (gy2 - gy1)
                    union = area1 + area2 - inter_area
                    
                    iou = inter_area / (union + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # TP if IoU > 0.5
                if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                    all_tp += 1
                    matched_gt.add(best_gt_idx)
                    all_iou.append(best_iou)
                    all_detections.append((conf, True, best_iou))
                else:
                    all_fp += 1
                    all_detections.append((conf, False, 0))
            
            # Count FN (unmatched GT)
            all_fn += len(gt_boxes) - len(matched_gt)
    
    # Calculate metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(all_iou) if all_iou else 0
    
    # Calculate AP (Area under PR curve)
    all_detections.sort(key=lambda x: -x[0])  # Sort by confidence descending
    
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    
    for conf, is_tp, iou in all_detections:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        
        p = tp_cumsum / (tp_cumsum + fp_cumsum)
        r = tp_cumsum / total_gt if total_gt > 0 else 0
        precisions.append(p)
        recalls.append(r)
    
    # AP using 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_recall:
            ap += max(prec_at_recall) / 11
    
    # Print results
    print("\n" + "=" * 60)
    print(f"TEST SET RESULTS (conf_thresh={args.conf_thresh})")
    print("=" * 60)
    
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'Precision':<20} {precision:>10.4f}")
    print(f"{'Recall':<20} {recall:>10.4f}")
    print(f"{'F1 Score':<20} {f1:>10.4f}")
    print(f"{'mAP@0.5':<20} {ap:>10.4f}")
    print(f"{'Avg IoU':<20} {avg_iou:>10.4f}")
    
    print(f"\n{'Detection Counts':<20}")
    print("-" * 32)
    print(f"{'True Positives':<20} {all_tp:>10}")
    print(f"{'False Positives':<20} {all_fp:>10}")
    print(f"{'False Negatives':<20} {all_fn:>10}")
    print(f"{'Total GT':<20} {total_gt:>10}")
    
    print("\n" + "=" * 60)
    print("SUMMARY FOR YOUR PAPER")
    print("=" * 60)
    print(f"""
Model: LDBST-MP (Detection)
Test Set Size: {len(image_files)} images
Confidence Threshold: {args.conf_thresh}

Results:
  Precision:  {precision*100:.2f}%
  Recall:     {recall*100:.2f}%
  F1 Score:   {f1:.4f}
  mAP@0.5:    {ap*100:.2f}%
  Avg IoU:    {avg_iou:.4f}

Objective Check:
  mAP@0.5 > 90%:  {"✅ PASS" if ap > 0.9 else f"❌ {ap*100:.2f}%"}
  F1 > 0.90:      {"✅ PASS" if f1 > 0.9 else f"❌ {f1:.4f}"}
""")




def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test set")
    
    parser.add_argument("--task", required=True, choices=["classification", "detection"])
    parser.add_argument("--model", required=True, help="Path to trained model (.pth)")
    parser.add_argument("--test_dir", required=True, help="Path to test data")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Detection-specific
    parser.add_argument("--labels_dir", help="Path to test labels (detection only)")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if args.task == "classification":
        evaluate_classification(args)
    else:
        evaluate_detection(args)


if __name__ == "__main__":
    main()