
"""
metrics_detection.py

Evaluate a DualBranchSwinCNNDetector checkpoint on a folder of images + YOLO-style labels.
Computes mAP@0.5 and COCO-style mAP@[0.5:0.95], per-class AP, precision/recall/IoU aggregates,
and saves qualitative visualizations showing TP (green), FP (red) and FN (magenta).


"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import box_iou, nms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm
import math
import os
from thop import profile
import time


# -------------------------
# Helper functions (matching style used in train)
# -------------------------
def load_checkpoint_state(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        # common keys
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        # Some saves contain "model_state_dict" or similar:
        for k in ["model_state_dict", "model", "model_state"]:
            if k in ckpt:
                return ckpt[k]
        # If keys look like a state_dict, return as-is
        return ckpt
    else:
        return ckpt


def yolo_txt_to_boxes(txt_path, img_size):
    """
    Read YOLO-format txt (class cx cy w h normalized) and convert to:
    - boxes: tensor [N,4] in pixel coords (x1,y1,x2,y2) scaled to img_size
    - labels: tensor [N] (int)
    """
    boxes = []
    labels = []
    if not txt_path.exists():
        return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.long)
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 5:
                continue
            cls = int(toks[0])
            cx = float(toks[1]); cy = float(toks[2]); w = float(toks[3]); h = float(toks[4])
            x1 = (cx - 0.5 * w) * img_size
            y1 = (cy - 0.5 * h) * img_size
            x2 = (cx + 0.5 * w) * img_size
            y2 = (cy + 0.5 * h) * img_size
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    else:
        return torch.zeros((0,4)), torch.zeros((0,), dtype=torch.long)


def boxes_xyxy_to_xywh(boxes):
    # [x1,y1,x2,y2] -> [cx,cy,w,h]
    x1y1 = boxes[:, :2]
    x2y2 = boxes[:, 2:4]
    cxcy = (x1y1 + x2y2) / 2.0
    wh = (x2y2 - x1y1)
    return torch.cat([cxcy, wh], dim=1)


# compute AP (VOC-style interpolation) for single IoU threshold
def compute_map(pred_boxes_all, pred_scores_all, true_boxes_all, iou_thresh=0.5, num_points=101):
    """
    pred_boxes_all: list of tensors [K_i,4] (pixel coords)
    pred_scores_all: list of tensors [K_i] confidence
    true_boxes_all: list of tensors [M_i,4]
    returns scalar AP (mean precision over recall points)
    """
    all_scores = []
    all_tp = []
    all_fp = []
    n_positives = sum([tb.shape[0] for tb in true_boxes_all])

    for pb, ps, tb in zip(pred_boxes_all, pred_scores_all, true_boxes_all):
        if pb.numel() == 0:
            continue
        # compute IoUs [Npred, Ntrue]
        ious = box_iou(pb, tb) if tb.shape[0] > 0 else torch.zeros((pb.shape[0], 0))
        detected = set()
        tp = torch.zeros(pb.shape[0])
        fp = torch.zeros(pb.shape[0])
        for i in range(pb.shape[0]):
            if tb.shape[0] == 0:
                fp[i] = 1
                continue
            max_iou, max_j = ious[i].max(0)
            if max_iou >= iou_thresh and int(max_j.item()) not in detected:
                tp[i] = 1
                detected.add(int(max_j.item()))
            else:
                fp[i] = 1
        all_scores.append(ps)
        all_tp.append(tp)
        all_fp.append(fp)

    if len(all_scores) == 0:
        return 0.0

    all_scores = torch.cat(all_scores)
    all_tp = torch.cat(all_tp)
    all_fp = torch.cat(all_fp)

    # sort by score desc
    sort_idx = torch.argsort(all_scores, descending=True)
    all_tp = all_tp[sort_idx]
    all_fp = all_fp[sort_idx]

    cum_tp = torch.cumsum(all_tp, dim=0)
    cum_fp = torch.cumsum(all_fp, dim=0)

    recalls = cum_tp / (n_positives + 1e-8)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

    recall_points = torch.linspace(0, 1, num_points)
    precision_interp = torch.zeros(num_points)
    for i, r in enumerate(recall_points):
        mask = recalls >= r
        precision_interp[i] = precisions[mask].max() if mask.any() else 0.0

    return float(precision_interp.mean().item())


def compute_map_coco(pred_boxes_all, pred_scores_all, true_boxes_all):
    iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    aps = []
    for thr in iou_thresholds:
        aps.append(compute_map(pred_boxes_all, pred_scores_all, true_boxes_all, iou_thresh=thr))
    return float(np.mean(aps)), aps, iou_thresholds


# matching helper returning matches info per image
def match_detections_to_gt(pb, ps, pl, tb, tl, iou_thresh=0.5):
    """
    pb: [K,4] predicted boxes (pixel coords)
    ps: [K] scores
    pl: [K] predicted labels (int)
    tb: [M,4] gt boxes
    tl: [M] gt labels
    returns:
      tp_indices (list of pred indices matched as TP),
      fp_indices (list of pred indices considered FP),
      fn_indices (list of gt indices not matched)
    This matching is label-aware: predictions match only GTs of same label.
    """
    K = pb.shape[0]
    M = tb.shape[0]
    if K == 0:
        return [], [], list(range(M))
    if M == 0:
        return [], list(range(K)), []

    # compute pairwise IoU
    ious = box_iou(pb, tb)  # [K, M]
    matched_gt = set()
    tp_idx = []
    fp_idx = []

    # Greedy matching by descending score
    order = torch.argsort(ps, descending=True)
    for idx in order.tolist():
        # find best gt for this pred
        iou_row = ious[idx]
        best_iou, best_j = iou_row.max(0)
        best_j = int(best_j.item())
        if best_iou >= iou_thresh and int(tl[best_j].item()) == int(pl[idx].item()) and best_j not in matched_gt:
            tp_idx.append(idx)
            matched_gt.add(best_j)
        else:
            fp_idx.append(idx)

    fn_idx = [j for j in range(M) if j not in matched_gt]
    return tp_idx, fp_idx, fn_idx


# -------------------------
# Visualization utilities
# -------------------------
def draw_boxes_on_image(img_pil, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels,
                        tp_idx, fp_idx, fn_idx, class_names=None, out_path=None):
    """
    img_pil: PIL Image (resized to img_size)
    boxes coords assumed to be in pixel coords (0..img_size).
    Colors:
      GT unmatched (FN) -> magenta
      TP predictions -> green
      FP predictions -> red
    """
    draw = ImageDraw.Draw(img_pil)
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # draw FN (missed GT) - magenta
    for j in fn_idx:
        x1, y1, x2, y2 = map(float, gt_boxes[j].tolist())
        draw.rectangle([x1, y1, x2, y2], outline="magenta", width=2)
        lbl = f"GT:{class_names[gt_labels[j]] if class_names else int(gt_labels[j])}"
        if font:
            draw.text((x1+2, y1+2), lbl, fill="magenta", font=font)

    # draw TP predicted boxes - green
    for i in tp_idx:
        x1, y1, x2, y2 = map(float, pred_boxes[i].tolist())
        s = float(pred_scores[i].item())
        lbl = f"{class_names[pred_labels[i]] if class_names else int(pred_labels[i])}:{s:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        if font:
            draw.text((x1+2, y1+2), lbl, fill="lime", font=font)

    # draw FP predicted boxes - red
    for i in fp_idx:
        x1, y1, x2, y2 = map(float, pred_boxes[i].tolist())
        s = float(pred_scores[i].item())
        lbl = f"{class_names[pred_labels[i]] if class_names else int(pred_labels[i])}:{s:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        if font:
            draw.text((x1+2, y1+2), lbl, fill="red", font=font)

    if out_path:
        img_pil.save(out_path)


# -------------------------
# Main evaluation loop
# -------------------------
def evaluate(weights, images_dir, labels_dir, out_dir="results_detection",
             img_size=224, conf_thresh=0.05, nms_iou=0.45,
             num_classes=2, class_names=None, visualize_n=20, device=None):

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # load model (import from same module or from your model.py)
    from model import DualBranchSwinCNNDetector, postprocess_detections  # expects these names available
    model = DualBranchSwinCNNDetector(num_classes=num_classes, pretrained=False, task="detection")
    state = load_checkpoint_state(weights)

    # strip prefixes if necessary
    stripped = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk.replace("module.", "", 1)
        if nk.startswith("_orig_mod."):
            nk = nk.replace("_orig_mod.", "", 1)
        stripped[nk] = v
    try:
        missing, unexpected = model.load_state_dict(stripped, strict=False)
    except Exception as e:
        # fallback: try loading direct dict
        model.load_state_dict(state, strict=False)

    model.to(device).eval()

    # --- Efficiency evaluation ---
    sample_input = torch.randn(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        # Warm-up
        for _ in range(5):
            _ = model(sample_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        for _ in range(20):
            _ = model(sample_input)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.time() - start_time) / 20 * 1000  # ms/image average


    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6  # million

    # Estimate FLOPs (if thop installed)
    try:
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
        flops = flops / 1e9  # to GigaFLOPs
    except Exception:
        flops = float("nan")

    print(f"[Efficiency] Params: {params:.2f}M | FLOPs: {flops:.2f}G | Inference: {elapsed:.2f} ms/image")

    # image transform: resize to img_size and normalize (same as training)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # If you used standard ImageNet normalization during training:
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    image_paths = sorted([p for p in Path(images_dir).glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    pred_boxes_all = []
    pred_scores_all = []
    true_boxes_all = []

    per_image_records = []

    # random subset for visualization
    vis_paths = []
    if visualize_n > 0:
        vis_paths = list(np.random.choice([p for p in image_paths], min(visualize_n, len(image_paths)), replace=False))

    for img_path in tqdm(image_paths, desc="Evaluating images"):
        # corresponding label path
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        # load and transform image
        img = Image.open(img_path).convert("RGB")
        img_trans = tf(img)
        # batch dim
        x = img_trans.unsqueeze(0).to(device)

        # Run model: try using predict() if available (which does postprocess)
        try:
            results = model.predict(x, conf_thresh=conf_thresh, iou_thresh=nms_iou)
            # results is list length 1 containing [K,6]: x1,y1,x2,y2,score,label
            dets = results[0].cpu()
            if dets.numel() == 0:
                pred_boxes = torch.zeros((0,4))
                pred_scores = torch.zeros((0,))
                pred_labels = torch.zeros((0,), dtype=torch.long)
            else:
                pred_boxes = dets[:, :4].detach().cpu()
                pred_scores = dets[:, 4].detach().cpu()
                pred_labels = dets[:, 5].long().detach().cpu()
        except Exception:
            # fallback: run forward + postprocess_detections
            with torch.no_grad():
                pb, pl, po = model(x)
                dets = postprocess_detections(pb[0], pl[0], po[0],
                                              conf_thresh=conf_thresh, iou_thresh=nms_iou, img_size=img_size)
                dets = dets.cpu()
                if dets.numel() == 0:
                    pred_boxes = torch.zeros((0,4))
                    pred_scores = torch.zeros((0,))
                    pred_labels = torch.zeros((0,), dtype=torch.long)
                else:
                    pred_boxes = dets[:, :4].detach().cpu()
                    pred_scores = dets[:, 4].detach().cpu()
                    pred_labels = dets[:, 5].long().detach().cpu()

        # load GT boxes from YOLO label (normalized) -> convert to pixel coords on img_size
        gt_boxes, gt_labels = yolo_txt_to_boxes(Path(label_path), img_size=img_size)

        # store lists for mAP
        pred_boxes_all.append(pred_boxes)
        pred_scores_all.append(pred_scores)
        true_boxes_all.append(gt_boxes)

        # compute per-image P/R/IoU with greedy matching (for logging)
        if pred_boxes.shape[0] == 0 and gt_boxes.shape[0] == 0:
            P = 1.0
            R = 1.0
            mean_iou = 0.0
        elif pred_boxes.shape[0] == 0:
            P = 0.0
            R = 0.0
            mean_iou = 0.0
        elif gt_boxes.shape[0] == 0:
            P = 0.0
            R = 0.0
            mean_iou = 0.0
        else:
            # greedy matching similar to training code
            ious = box_iou(pred_boxes, gt_boxes)
            ious_cp = ious.clone()
            tp = 0
            matched_ious = []
            while True:
                if ious_cp.numel() == 0:
                    break
                max_val = ious_cp.max()
                if max_val < 0.5:
                    break
                idx = (ious_cp == max_val).nonzero(as_tuple=False)[0]
                pred_idx, gt_idx = int(idx[0].item()), int(idx[1].item())
                tp += 1
                matched_ious.append(max_val.item())
                ious_cp[pred_idx, :] = -1
                ious_cp[:, gt_idx] = -1
            fp = pred_boxes.shape[0] - tp
            fn = gt_boxes.shape[0] - tp
            P = tp / (tp + fp + 1e-8)
            R = tp / (tp + fn + 1e-8)
            mean_iou = (sum(matched_ious) / len(matched_ious)) if matched_ious else 0.0

        per_image_records.append({
            "image": str(img_path.name),
            "n_gt": int(gt_boxes.shape[0]),
            "n_pred": int(pred_boxes.shape[0]),
            "precision": float(P),
            "recall": float(R),
            "mean_iou": float(mean_iou),
        })

        # qualitative visualization for a subset (vis_paths)
        if img_path in vis_paths:
            # compute label-aware matching to categorize TP/FP/FN
            tp_idx, fp_idx, fn_idx = match_detections_to_gt(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5)
            # convert image to size img_size to draw
            img_resized = img.resize((img_size, img_size))
            draw_out = outp / "visuals"
            draw_out.mkdir(parents=True, exist_ok=True)
            out_file = draw_out / f"{img_path.stem}_vis.png"
            draw_boxes_on_image(img_resized, gt_boxes, gt_labels, pred_boxes, pred_scores, pred_labels, tp_idx, fp_idx, fn_idx, class_names=class_names, out_path=out_file)

    # --- After loop: compute dataset-level metrics
    mAP50 = compute_map(pred_boxes_all, pred_scores_all, true_boxes_all, iou_thresh=0.5)
    mAPcoco, ap_list, iou_thresholds = compute_map_coco(pred_boxes_all, pred_scores_all, true_boxes_all)

    # per-class AP (compute AP per class at IoU=0.5)
    per_class_aps = []

    # To compute per-class AP we need predicted labels; re-run lighter loop to collect pred_labels_all
    pred_labels_all = []
    for img_path in tqdm(image_paths, desc="Collecting labels for per-class AP"):
        # load + preprocess
        img = Image.open(img_path).convert("RGB")
        img_trans = tf(img)
        x = img_trans.unsqueeze(0).to(device)
        with torch.no_grad():
            try:
                results = model.predict(x, conf_thresh=conf_thresh, iou_thresh=nms_iou)
                dets = results[0].cpu()
                if dets.numel() == 0:
                    pred_boxes = torch.zeros((0,4))
                    pred_scores = torch.zeros((0,))
                    pred_labels = torch.zeros((0,), dtype=torch.long)
                else:
                    pred_boxes = dets[:, :4].detach().cpu()
                    pred_scores = dets[:, 4].detach().cpu()
                    pred_labels = dets[:, 5].long().detach().cpu()
            except Exception:
                with torch.no_grad():
                    pb, pl, po = model(x)
                    dets = postprocess_detections(pb[0], pl[0], po[0], conf_thresh=conf_thresh, iou_thresh=nms_iou, img_size=img_size)
                    dets = dets.cpu()
                    if dets.numel() == 0:
                        pred_boxes = torch.zeros((0,4)); pred_scores = torch.zeros((0,)); pred_labels = torch.zeros((0,), dtype=torch.long)
                    else:
                        pred_boxes = dets[:, :4].detach().cpu()
                        pred_scores = dets[:, 4].detach().cpu()
                        pred_labels = dets[:, 5].long().detach().cpu()
        pred_labels_all.append(pred_labels)

    # Now compute per-class AP at IoU=0.5
    per_class_results = {}
    for cls in range(num_classes):
        cls_pred_boxes, cls_pred_scores, cls_true_boxes = [], [], []
        for pb, ps, pl, tb in zip(pred_boxes_all, pred_scores_all, pred_labels_all, true_boxes_all):
            # filter preds by class
            if pl.numel() == 0:
                cls_pred_boxes.append(torch.zeros((0,4)))
                cls_pred_scores.append(torch.zeros((0,)))
            else:
                keep = (pl == cls)
                if keep.any():
                    cls_pred_boxes.append(pb[keep])
                    cls_pred_scores.append(ps[keep])
                else:
                    cls_pred_boxes.append(torch.zeros((0,4)))
                    cls_pred_scores.append(torch.zeros((0,)))


    # Re-load GT per image to ensure we have GT labels
    gt_boxes_all = []
    gt_labels_all = []
    for img_path in image_paths:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        boxes, labels = yolo_txt_to_boxes(Path(label_path), img_size=img_size)
        gt_boxes_all.append(boxes)
        gt_labels_all.append(labels)

    # compute per-class AP properly now
    per_class_ap_values = []
    for cls in range(num_classes):
        cls_pred_boxes = []
        cls_pred_scores = []
        cls_true_boxes = []
        for pb, ps, pl, tb, tl in zip(pred_boxes_all, pred_scores_all, pred_labels_all, gt_boxes_all, gt_labels_all):
            # preds filtered by label
            if pl.numel() == 0:
                cls_pred_boxes.append(torch.zeros((0,4))); cls_pred_scores.append(torch.zeros((0,)))
            else:
                keep = (pl == cls)
                if keep.any():
                    cls_pred_boxes.append(pb[keep])
                    cls_pred_scores.append(ps[keep])
                else:
                    cls_pred_boxes.append(torch.zeros((0,4))); cls_pred_scores.append(torch.zeros((0,)))
            # gt filtered by label
            if tl.numel() == 0:
                cls_true_boxes.append(torch.zeros((0,4)))
            else:
                keepg = (tl == cls)
                if keepg.any():
                    cls_true_boxes.append(tb[keepg])
                else:
                    cls_true_boxes.append(torch.zeros((0,4)))
        ap = compute_map(cls_pred_boxes, cls_pred_scores, cls_true_boxes, iou_thresh=0.5)
        per_class_ap_values.append(ap)
        per_class_results[class_names[cls] if class_names else str(cls)] = float(ap)

    # ---- Aggregate dataset-level stats ----
    mean_P = np.mean([r["precision"] for r in per_image_records])
    mean_R = np.mean([r["recall"] for r in per_image_records])
    mean_IoU = np.mean([r["mean_iou"] for r in per_image_records])

    summary = {
        "mAP@0.5": mAP50,
        "mAP@[0.5:0.95]": mAPcoco,
        "mean_precision": round(mean_P, 4),
        "mean_recall": round(mean_R, 4),
        "mean_iou": round(mean_IoU, 4),
        "AP_per_iou": {str(i): float(ap) for i, ap in zip(iou_thresholds, ap_list)},
        "per_class_AP@0.5": per_class_results,
        "n_images": len(image_paths),
        "efficiency": {
            "params_million": round(params, 2),
            "flops_giga": round(flops, 2),
            "inference_time_ms_per_image": round(elapsed, 2)
        }
    }

    (outp / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    (outp / "per_image_records.json").write_text(json.dumps(per_image_records, indent=2))

    print("=== DETECTION METRICS SUMMARY ===")
    print(f"Images evaluated: {len(image_paths)}")
    print(f"mAP@0.5 = {mAP50:.4f}")
    print(f"mAP@[0.5:0.95] = {mAPcoco:.4f}")
    print(f"Mean Precision = {mean_P:.4f}, Mean Recall = {mean_R:.4f}, Mean IoU = {mean_IoU:.4f}")
    print("Per-class AP@0.5:")
    for k, v in per_class_results.items():
        print(f"  {k}: {v:.4f}")

    return summary



# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to detector checkpoint (.pth)")
    ap.add_argument("--images", required=True, help="Folder with images")
    ap.add_argument("--labels", required=True, help="Folder with YOLO-style txt labels (same stem as images)")
    ap.add_argument("--out", default="results_detection", help="Output directory")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--conf_thresh", type=float, default=0.05)
    ap.add_argument("--nms_iou", type=float, default=0.45)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--class_names", nargs="*", default=None)
    ap.add_argument("--visualize_n", type=int, default=20, help="How many random images to save visualizations for")
    args = ap.parse_args()

    evaluate(args.weights, args.images, args.labels, out_dir=args.out,
             img_size=args.img_size, conf_thresh=args.conf_thresh, nms_iou=args.nms_iou,
             num_classes=args.num_classes, class_names=args.class_names, visualize_n=args.visualize_n)
