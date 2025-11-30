from ultralytics import YOLO
import matplotlib.pyplot as plt
import json
import cv2
import os
import numpy as np
import torch
import time
from thop import profile

SAVE_DIR = "runs/test_visualized"
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------------------------------------
#  Plot AP vs IoU
# ----------------------------------------------------------
def visualize_metrics(metrics):
    ap_per_iou = metrics.get("AP_per_iou", {})
    if not ap_per_iou:
        print("⚠️ No AP per IoU data found.")
        return

    ious = list(map(float, ap_per_iou.keys()))
    aps = list(ap_per_iou.values())

    plt.figure(figsize=(6, 4))
    plt.plot(ious, aps, marker='o')
    plt.title("AP per IoU")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Average Precision (AP)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ap_vs_iou.png"))
    plt.show()


# ----------------------------------------------------------
#  Draw TP / FP / FN
# ----------------------------------------------------------
def draw_tp_fp_fn(image_path, pred_boxes, pred_cls, gt_labels, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return
    
    h, w = img.shape[:2]

    # Convert GT xywh (normalized) → xyxy (pixel coords)
    gt_boxes = []
    for t in gt_labels:
        cls, x, y, ww, hh = t
        x1 = int((x - ww/2) * w)
        y1 = int((y - hh/2) * h)
        x2 = int((x + ww/2) * w)
        y2 = int((y + hh/2) * h)
        gt_boxes.append([int(cls), x1, y1, x2, y2])

    matched_pred = set()
    matched_gt = set()

    # IoU matching
    for gi, gt in enumerate(gt_boxes):
        gt_cls, gx1, gy1, gx2, gy2 = gt
        best_iou = 0
        best_pi = -1

        for pi, pb in enumerate(pred_boxes):
            px1, py1, px2, py2 = map(int, pb)

            inter_x1 = max(gx1, px1)
            inter_y1 = max(gy1, py1)
            inter_x2 = min(gx2, px2)
            inter_y2 = min(gy2, py2)

            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            gt_area = (gx2 - gx1) * (gy2 - gy1)
            pred_area = (px2 - px1) * (py2 - py1)

            iou = inter_area / (gt_area + pred_area - inter_area + 1e-7)

            if iou > best_iou:
                best_iou = iou
                best_pi = pi

        if best_iou >= 0.5:
            matched_gt.add(gi)
            matched_pred.add(best_pi)

            # TP (green)
            cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(img, "TP", (gx1, gy1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # FP (red)
    for pi, pb in enumerate(pred_boxes):
        if pi not in matched_pred:
            x1, y1, x2, y2 = map(int, pb)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "FP", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    # FN (purple/pink)
    FN_COLOR = (255, 0, 180)
    for gi, gt in enumerate(gt_boxes):
        if gi not in matched_gt:
            _, x1, y1, x2, y2 = gt
            cv2.rectangle(img, (x1, y1), (x2, y2), FN_COLOR, 2)
            cv2.putText(img, "FN", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, FN_COLOR, 2)

    cv2.imwrite(save_path, img)


# ----------------------------------------------------------
#  Helper function to convert numpy/torch types to native Python
# ----------------------------------------------------------
def convert_to_serializable(obj):
    """Convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


# ----------------------------------------------------------
#  Load ground truth labels
# ----------------------------------------------------------
def load_label_file(label_path):
    """Load ground truth labels from YOLO format txt file."""
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])
    return labels


# ----------------------------------------------------------
#  MAIN
# ----------------------------------------------------------
def main():
    model_path = r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\runs\detect\yolov8n_detection_microplastics_algae3\weights\best.pt"
    model = YOLO(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device).eval()

    # --- Efficiency evaluation ---
    img_size = 640
    sample_input = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        # Warm-up
        for _ in range(5):
            _ = model.model(sample_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(20):
            _ = model.model(sample_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start_time) / 20 * 1000  # ms per image

    # FLOPs using thop
    try:
        flops, _ = profile(model.model, inputs=(sample_input,), verbose=False)
        flops = flops / 1e9  # GigaFLOPs
    except Exception:
        flops = float("nan")

    # --- Validation ---
    results = model.val(
        data=r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataD\data.yaml",
        split='test'
    )

    # JSON metrics - convert all values to native Python types
    metrics = {
        "mAP@0.5": float(results.box.map50),
        "mAP@[0.5:0.95]": float(results.box.map),
        "mean_precision": float(results.box.mp),
        "mean_recall": float(results.box.mr),
        "AP_per_iou": {
            f"{iou:.2f}": float(ap)
            for iou, ap in zip([0.50 + 0.05 * i for i in range(10)], results.box.all_ap.mean(axis=0).tolist())
        },
        "per_class_AP@0.5": {
            name: float(results.box.maps[idx])
            for idx, name in enumerate(results.names.values())
        },
        "n_images": int(sum(results.nt_per_image)),
        "efficiency": {
            "params_million": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
            "flops_giga": round(float(flops), 2),
            "inference_time_ms_per_image": round(elapsed, 2)
        }
    }

    # Ensure all values are JSON serializable
    metrics = convert_to_serializable(metrics)

    print(json.dumps(metrics, indent=2))
    visualize_metrics(metrics)

    # TP / FP / FN visualization
    print("\nGenerating TP/FP/FN images...")
    val_images_folder = r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\data_yolo\images\test"
    val_labels_folder = r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\data_yolo\labels\test"
    val_image_files = sorted([f for f in os.listdir(val_images_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])

    # Run predictions to get individual results
    print("Running predictions for visualization...")
    for i, img_file in enumerate(val_image_files):
        img_path = os.path.join(val_images_folder, img_file)
        
        # Get prediction for this image
        pred_results = model.predict(img_path, verbose=False)
        
        if len(pred_results) == 0 or pred_results[0].boxes is None:
            print(f"⚠️ No predictions for {img_file}")
            continue
        
        pred = pred_results[0]
        pred_boxes = pred.boxes.xyxy.cpu().numpy()
        pred_cls = pred.boxes.cls.cpu().numpy()
        
        # Load ground truth labels
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(val_labels_folder, label_file)
        gt_labels = load_label_file(label_path)
        
        # Draw and save
        save_path = os.path.join(SAVE_DIR, f"val_{i:04d}.jpg")
        draw_tp_fp_fn(img_path, pred_boxes, pred_cls, gt_labels, save_path)
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(val_image_files)} images")

    print(f"\n✅ Saved visualized images to: {SAVE_DIR}")


if __name__ == "__main__":
    main()