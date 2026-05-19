"""
visualize.py - Generate visualizations for detection and classification results

Usage:
    python visualize.py --task detection --model detector_v3.pth --test_dir ./data_yolo_test_ready --output ./visuals_detection --conf_thresh 0.5
    python visualize.py --task classification --model classifier_v3.pth --test_dir ./data_test_ready --output ./visuals_classification
"""

import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ============================================================
# DETECTION VISUALIZATION
# ============================================================

def visualize_detection(args):
    """Generate detection visualizations with TP, FP, FN boxes AND ground truth overlay."""
    from model2 import LDBST_MP, postprocess_detections
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
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
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "all").mkdir(exist_ok=True)
    (output_dir / "errors").mkdir(exist_ok=True)
    
    # Transforms
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    transform = A.Compose([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
    
    # Find images and labels
    images_dir = Path(args.test_dir) / "images"
    labels_dir = Path(args.test_dir) / "labels"
    
    if not images_dir.exists():
        images_dir = Path(args.test_dir)
        labels_dir = Path(args.test_dir).parent / "labels"
    
    image_files = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    
    print(f"[data] Found {len(image_files)} test images")
    print(f"[config] Confidence threshold: {args.conf_thresh}")
    print(f"[output] Saving to {output_dir}")
    
    # Class names
    class_names = ["algae", "microplastics"] if args.num_classes == 2 else [f"class_{i}" for i in range(args.num_classes)]
    
    # Colors (BGR for OpenCV)
    COLOR_TP = (0, 255, 0)      # Green - True Positive (prediction matches GT)
    COLOR_FP = (0, 0, 255)      # Red - False Positive (prediction without matching GT)
    COLOR_FN = (0, 255, 255)    # Yellow - False Negative (missed GT)
    COLOR_GT = (255, 165, 0)    # Orange - Ground Truth (drawn alongside TP predictions)
    
    # Stats
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    all_ious = []
    all_detections = []
    error_images = []
    
    print("\n[visualize] Processing images...")
    
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Visualizing"):
            # Load original image for visualization
            img_orig = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            
            # Transform for model
            transformed = transform(image=img_rgb)
            img_t = transformed["image"].unsqueeze(0).to(device)
            
            # Inference
            pred_boxes, pred_logits, pred_obj = model(img_t)
            
            # Postprocess
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
                            cls = int(float(parts[0]))
                            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            x1 = int((xc - w/2) * 224)
                            y1 = int((yc - h/2) * 224)
                            x2 = int((xc + w/2) * 224)
                            y2 = int((yc + h/2) * 224)
                            gt_boxes.append([x1, y1, x2, y2, cls])
            
            # Create visualization image
            vis_img = img_orig.copy()
            
            # Match predictions to GT
            matched_gt = set()
            matched_pred = set()
            
            img_tp = 0
            img_fp = 0
            img_fn = 0
            
            # ----------------------------------------------------------
            # FIRST PASS: Draw ALL ground truth boxes as orange outline
            # This gives the reviewer a clear picture of what SHOULD be detected
            # regardless of whether the model found it. TP boxes will later
            # be drawn in green on top, creating a clear GT vs prediction
            # side-by-side comparison.
            # ----------------------------------------------------------
            for gt in gt_boxes:
                gx1, gy1, gx2, gy2, gcls = gt
                # Thin dashed-look outline using 1px line
                cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), COLOR_GT, 1)
                gt_label = f"GT: {class_names[gcls]}"
                (tw, th), _ = cv2.getTextSize(gt_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                # Small GT tag in top-left corner of the GT box
                cv2.rectangle(vis_img, (gx1, gy1 - th - 4), (gx1 + tw + 4, gy1 - 1), COLOR_GT, -1)
                cv2.putText(vis_img, gt_label, (gx1 + 2, gy1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # ----------------------------------------------------------
            # SECOND PASS: Match predictions against GT and draw TP / FP
            # ----------------------------------------------------------
            for pred_idx, det in enumerate(dets):
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(cls)
                
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    gx1, gy1, gx2, gy2, gcls = gt
                    
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
                
                if best_iou >= 0.5 and best_gt_idx not in matched_gt:
                    # True Positive
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                    color = COLOR_TP
                    label = f"TP: {class_names[cls]} ({conf:.2f}) IoU={best_iou:.2f}"
                    img_tp += 1
                    all_ious.append(best_iou)
                    all_detections.append((conf, True, best_iou))
                else:
                    # False Positive
                    color = COLOR_FP
                    label = f"FP: {class_names[cls]} ({conf:.2f})"
                    img_fp += 1
                    all_detections.append((conf, False, 0))
                
                # Draw prediction box (thicker, on top of GT outline)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                
                # Label below the prediction box to avoid clashing with GT label above
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis_img, (x1, y2), (x1 + tw + 4, y2 + th + 8), color, -1)
                cv2.putText(vis_img, label, (x1 + 2, y2 + th + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # ----------------------------------------------------------
            # THIRD PASS: Mark unmatched GT boxes as False Negatives
            # (drawn in yellow, replacing the orange GT outline for clarity)
            # ----------------------------------------------------------
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx not in matched_gt:
                    gx1, gy1, gx2, gy2, gcls = gt
                    # Redraw as yellow FN (thicker, on top of orange)
                    cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), COLOR_FN, 2)
                    label = f"FN: {class_names[gcls]} (missed)"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(vis_img, (gx1, gy2), (gx1 + tw + 4, gy2 + th + 8), COLOR_FN, -1)
                    cv2.putText(vis_img, label, (gx1 + 2, gy2 + th + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    img_fn += 1
            
            total_tp += img_tp
            total_fp += img_fp
            total_fn += img_fn
            total_gt += len(gt_boxes)
            
            # Extended legend with GT entry
            legend_y = 20
            cv2.putText(vis_img, "Legend:", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(vis_img, (80, legend_y - 12), (95, legend_y + 3), COLOR_GT, -1)
            cv2.putText(vis_img, "GT", (100, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(vis_img, (130, legend_y - 12), (145, legend_y + 3), COLOR_TP, -1)
            cv2.putText(vis_img, "TP", (150, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(vis_img, (180, legend_y - 12), (195, legend_y + 3), COLOR_FP, -1)
            cv2.putText(vis_img, "FP", (200, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(vis_img, (230, legend_y - 12), (245, legend_y + 3), COLOR_FN, -1)
            cv2.putText(vis_img, "FN", (250, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save outputs
            cv2.imwrite(str(output_dir / "all" / f"{img_path.stem}_vis.jpg"), vis_img)
            if img_fp > 0 or img_fn > 0:
                cv2.imwrite(str(output_dir / "errors" / f"{img_path.stem}_vis.jpg"), vis_img)
                error_images.append(img_path.stem)
    
    # Summary chart + metrics (unchanged)
    print("\n[visualize] Creating summary...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = ['True Positive\n(Correct)', 'False Positive\n(Wrong Detection)', 'False Negative\n(Missed)']
    values = [total_tp, total_fp, total_fn]
    colors = ['#2ecc71', '#e74c3c', '#f1c40f']
    
    axes[0].bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Detection Results Summary', fontsize=14, fontweight='bold')
    for i, v in enumerate(values):
        axes[0].text(i, v + 1, str(v), ha='center', fontsize=12, fontweight='bold')
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = np.mean(all_ious) if all_ious else 0
    
    all_detections.sort(key=lambda x: -x[0])
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
    
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if prec_at_recall:
            ap += max(prec_at_recall) / 11
    
    metrics_text = f"""
    Detection Metrics @ conf={args.conf_thresh}
    
    True Positives:  {total_tp}
    False Positives: {total_fp}
    False Negatives: {total_fn}
    Total GT:        {total_gt}
    
    Precision: {precision:.4f} ({precision*100:.2f}%)
    Recall:    {recall:.4f} ({recall*100:.2f}%)
    F1 Score:  {f1:.4f}
    mAP@0.5:   {ap:.4f} ({ap*100:.2f}%)
    Avg IoU:   {avg_iou:.4f}
    
    Total Images: {len(image_files)}
    Images with Errors: {len(error_images)}
    """
    
    axes[1].text(0.05, 0.5, metrics_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1].axis('off')
    axes[1].set_title('Metrics Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[done] Visualizations saved to {output_dir}")
    print(f"       - all/: {len(image_files)} images with GT + predictions")
    print(f"       - errors/: {len(error_images)} images with FP or FN")
    print(f"       - summary.png: metrics summary")
    print(f"\n[metrics] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"[metrics] mAP@0.5: {ap:.4f} ({ap*100:.2f}%), Avg IoU: {avg_iou:.4f}")


# ============================================================
# CLASSIFICATION VISUALIZATION (unchanged)
# ============================================================

def visualize_classification(args):
    """Generate classification visualizations with misclassified samples and confusion matrix."""
    from model2 import LDBST_MP
    from torchvision import transforms
    from PIL import Image
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] Using: {device}")
    
    print(f"[model] Loading from {args.model}")
    model = LDBST_MP(img_size=224, num_classes=args.num_classes, task="classification")
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "correct").mkdir(exist_ok=True)
    (output_dir / "misclassified").mkdir(exist_ok=True)
    
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    test_dir = Path(args.test_dir)
    class_names = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
    print(f"[data] Found {len(class_names)} classes: {class_names}")
    
    samples = []
    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = test_dir / cls_name
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                samples.append((img_path, cls_idx, cls_name))
    
    print(f"[data] Found {len(samples)} test images")
    print(f"[output] Saving to {output_dir}")
    
    y_true, y_pred, y_probs = [], [], []
    misclassified, correct = [], []
    
    with torch.no_grad():
        for img_path, true_idx, true_name in tqdm(samples, desc="Classifying"):
            img = Image.open(img_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            
            logits = model(img_t)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            pred_prob = probs[0, pred_idx].item()
            
            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_probs.append(pred_prob)
            
            pred_name = class_names[pred_idx]
            
            if pred_idx != true_idx:
                misclassified.append({'path': img_path, 'true': true_name,
                                      'pred': pred_name, 'prob': pred_prob})
            else:
                correct.append({'path': img_path, 'true': true_name, 'prob': pred_prob})
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    accuracy = np.mean(y_true == y_pred)
    per_class = {}
    for cls_idx, cls_name in enumerate(class_names):
        tp = np.sum((y_true == cls_idx) & (y_pred == cls_idx))
        fp = np.sum((y_true != cls_idx) & (y_pred == cls_idx))
        fn = np.sum((y_true == cls_idx) & (y_pred != cls_idx))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class[cls_name] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    for item in misclassified:
        img = cv2.imread(str(item['path']))
        text1 = f"TRUE: {item['true']}"
        text2 = f"PRED: {item['pred']} ({item['prob']*100:.1f}%)"
        cv2.rectangle(img, (0, 0), (224, 50), (0, 0, 200), -1)
        cv2.putText(img, text1, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, text2, (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(str(output_dir / "misclassified" / f"{item['path'].stem}_mis.jpg"), img)
    
    correct_by_class = defaultdict(list)
    for item in correct:
        correct_by_class[item['true']].append(item)
    for cls_name, items in correct_by_class.items():
        for item in items[:5]:
            img = cv2.imread(str(item['path']))
            text = f"{item['true']} ({item['prob']*100:.1f}%)"
            cv2.rectangle(img, (0, 0), (224, 25), (0, 150, 0), -1)
            cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(str(output_dir / "correct" / f"{item['path'].stem}_correct.jpg"), img)
    
    if len(misclassified) > 0:
        n_show = min(20, len(misclassified))
        cols = 5
        rows = (n_show + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten() if n_show > 1 else [axes]
        for i, item in enumerate(misclassified[:n_show]):
            img = Image.open(item['path']).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(f"True: {item['true']}\nPred: {item['pred']}\n({item['prob']*100:.1f}%)",
                              fontsize=9, color='red')
            axes[i].axis('off')
        for i in range(n_show, len(axes)):
            axes[i].axis('off')
        plt.suptitle(f"Misclassified Samples ({len(misclassified)} total)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "misclassified_grid.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(class_names))
    width = 0.25
    precisions = [per_class[c]['precision'] for c in class_names]
    recalls = [per_class[c]['recall'] for c in class_names]
    f1s = [per_class[c]['f1'] for c in class_names]
    axes[0].bar(x - width, precisions, width, label='Precision', color='#3498db')
    axes[0].bar(x, recalls, width, label='Recall', color='#2ecc71')
    axes[0].bar(x + width, f1s, width, label='F1', color='#9b59b6')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names)
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    
    macro_f1 = np.mean(f1s)
    summary_text = f"""
    Classification Results
    
    Total Images:      {len(samples)}
    Correct:           {len(correct)} ({len(correct)/len(samples)*100:.1f}%)
    Misclassified:     {len(misclassified)} ({len(misclassified)/len(samples)*100:.1f}%)
    
    Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)
    Macro F1:          {macro_f1:.4f}
    
    Per-Class Performance:
    """
    for cls_name in class_names:
        m = per_class[cls_name]
        summary_text += f"\n      {cls_name}: P={m['precision']:.3f}, R={m['recall']:.3f}, F1={m['f1']:.3f}"
    
    axes[1].text(0.1, 0.5, summary_text, transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[1].axis('off')
    axes[1].set_title('Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[done] Visualizations saved to {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize detection/classification results")
    parser.add_argument("--task", choices=["classification", "detection"], required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test data")
    parser.add_argument("--output", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold (detection only)")
    
    args = parser.parse_args()
    
    if args.task == "detection":
        visualize_detection(args)
    else:
        visualize_classification(args)


if __name__ == "__main__":
    main()