import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
import matplotlib.pyplot as plt
import pandas as pd  # for Excel export
from dataset import _tfm_classification
import random, torch, numpy as np

# ---- Force deterministic behavior ----
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------- CAM utilities (used by detect.py too) ----------
def generate_cam(model, image_tensor, class_idx=None):
    """
    Grad-CAM on the CNN branch's last conv layer (ResNet18 layer4).
    image_tensor: 3D (C,H,W) normalized as in training.
    Returns a [0,1] heatmap (h, w).
    """
    model.eval()
    feature_maps, gradients = {}, {}

    def forward_hook(module, inp, out):
        feature_maps['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # NOTE: register_full_backward_hook would silence a deprecation warning,
    # but layer4 supports this standard hook fine for our use.
    hook_f = model.cnn.layer4.register_forward_hook(forward_hook)
    hook_b = model.cnn.layer4.register_backward_hook(backward_hook)

    outputs = model(image_tensor.unsqueeze(0))
    if class_idx is None:
        class_idx = int(outputs.argmax(dim=1))
    model.zero_grad()
    target = outputs[0, class_idx]
    target.backward()

    fmap = feature_maps['value'][0]   # [C, h, w]
    grads = gradients['value'][0]     # [C, h, w]
    hook_f.remove(); hook_b.remove()

    weights = grads.mean(dim=(1, 2))  # [C]
    cam_map = torch.zeros_like(fmap[0])
    for i, w in enumerate(weights):
        cam_map += w * fmap[i]
    cam_map = torch.relu(cam_map)
    cam_map = cam_map - cam_map.min()
    cam_map = cam_map / (cam_map.max() + 1e-8)
    return cam_map.cpu().numpy()


def overlay_cam_on_image(img: Image.Image, cam_map: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm
    H, W = img.size[1], img.size[0]
    heatmap = (cm.jet(cam_map) * 255).astype(np.uint8)[:, :, :3]
    heatmap = Image.fromarray(heatmap).resize((W, H), resample=Image.BILINEAR)
    heatmap = np.array(heatmap)
    img_np  = np.array(img.convert('RGB'))
    overlay = (0.5 * img_np + 0.5 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


# ---------- Full evaluation over a folder ----------
@torch.no_grad()
def evaluate_folder(weights, data_dir, out_dir="results", class_names=None, batch_size=32, num_classes=2, num_visuals=50):
    from model import DualBranchSwinCNNDetector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchSwinCNNDetector(num_classes=num_classes, pretrained=False).to(device)

    # --- Load weights (single call) ---
    state = torch.load(weights, map_location=device)

    # If this is a dict with model weights inside
    if isinstance(state, dict):
        if "model_state" in state:
            state = state["model_state"]
        elif "state_dict" in state:
            state = state["state_dict"]

    # Strip common prefixes (DataParallel / compiled models)
    def _strip_prefixes(sd):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            for prefix in ["module.", "_orig_mod."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            new_sd[nk] = v
        return new_sd

    if isinstance(state, dict):
        state = _strip_prefixes(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights (strict=False). Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print("  Missing keys:", missing[:10])
    if unexpected:
        print("  Unexpected keys:", unexpected[:10])

    model.eval()

            # --- Compute model efficiency metrics ---
    print("\n[Analyzing model efficiency...]")
    sample_input = torch.randn(1, 3, 224, 224).to(device)  # for classification
    params, flops, time_ms = model_efficiency(model, sample_input, device)
    print(f"[Efficiency] Params: {params:.2f}M | FLOPs: {flops:.2f}G | Inference: {time_ms:.2f} ms/image\n")
    
    # --- Dataset + DataLoader ---
    tf = _tfm_classification()
    ds = datasets.ImageFolder(data_dir, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Try to inspect classifier head (unchanged) ---
    def _find_head_linear(mod):
        if hasattr(mod, "head") and isinstance(mod.head, torch.nn.Sequential):
            for i in reversed(range(len(mod.head))):
                layer = mod.head[i]
                if isinstance(layer, torch.nn.Linear):
                    return (mod.head, i, layer)
        return (None, None, None)
    parent, idx, linear = _find_head_linear(model)

    # Track predictions + file paths
    y_true, y_pred = [], []
    paths_all = [p for p, _ in ds.samples]
    path_cursor = 0
    per_image_rows = []

    # ---------- MAIN LOOP: collect preds and per-image rows ----------
    for x, y in dl:
        bs = x.size(0)
        batch_paths = paths_all[path_cursor:path_cursor+bs]
        path_cursor += bs

        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy().tolist())

        # per-image log (y is still on CPU)
        for pth, t, pr in zip(batch_paths, y.numpy().tolist(), preds):
            per_image_rows.append({
                "filename": Path(pth).name,
                "true_class": class_names[t] if class_names else str(t),
                "predicted_class": class_names[pr] if class_names else str(pr),
                "correct": (t == pr)
            })

    # ---- Prepare output path(s) ----
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---- Visualize predictions (optionally with CAM) ----
    vis_dir = out_path / "visuals"
    vis_dir.mkdir(exist_ok=True)

    print("Generating visual prediction gallery...")

    # avoid trying to sample when no images
    if len(per_image_rows) > 0:
        sample_rows = random.sample(per_image_rows, min(len(per_image_rows), num_visuals))
    else:
        sample_rows = []

    # reuse the transform instance for CAM preprocessing if needed
    cam_tfm = _tfm_classification()

    for row in sample_rows:
        img_path = Path(data_dir) / row["true_class"] / row["filename"]
        if not img_path.exists():
            # try absolute path fallback (some ImageFolder samples may store relative vs absolute)
            print(f"Missing image (skipping): {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        label_true = row["true_class"]
        label_pred = row["predicted_class"]
        correct = row["correct"]

        # Attempt Grad-CAM overlay safely
        img_overlay = img
        try:
            # Determine class_idx safely
            if class_names:
                try:
                    class_idx = int(class_names.index(label_pred))
                except ValueError:
                    # fallback if predicted label not in class_names
                    class_idx = None
            else:
                class_idx = None

            x_for_cam = cam_tfm(img).to(device)
            cam_map = generate_cam(model, x_for_cam, class_idx=class_idx)
            if cam_map is not None:
                img_overlay = overlay_cam_on_image(img, cam_map)
        except Exception as e:
            # CAM failed; fall back to original image (don't crash)
            # Uncomment next line to debug CAM problems:
            # print(f"Grad-CAM failed for {img_path}: {e}")
            img_overlay = img

        # save visualization
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_overlay)
        ax.axis("off")
        color = "green" if correct else "red"
        ax.set_title(f"T: {label_true} | P: {label_pred}", color=color, fontsize=10, weight="bold")
        save_name = f"{img_path.stem}_pred.png"
        fig.savefig(vis_dir / save_name, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

    # ---------- Metrics ----------
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(num_classes), zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=range(num_classes),
        target_names=class_names if class_names else None,
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Save JSON
    with open(out_path / "metrics_report.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "precision_per_class": prec.tolist(),
            "recall_per_class": rec.tolist(),
            "f1_per_class": f1.tolist(),
            "support_per_class": support.tolist(),
            "classification_report": report,
            "efficiency": {
                "params_million": round(params, 2),
                "flops_giga": round(flops, 2),
                "inference_time_ms_per_image": round(time_ms, 2)
            }
        }, f, indent=2)


    # Confusion matrix PNG
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    ticks = class_names if class_names else list(range(num_classes))
    plt.xticks(range(num_classes), ticks, rotation=45)
    plt.yticks(range(num_classes), ticks)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path / "confusion_matrix.png")
    plt.close(fig)

    # Excel/CSV export (unchanged)
    per_image_df = pd.DataFrame(per_image_rows)
    per_class_df = pd.DataFrame({
        "class": class_names if class_names else list(range(num_classes)),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "support": support
    })
    summary_df = pd.DataFrame([{
        "accuracy": acc,
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "samples": int(report["weighted avg"]["support"])
    }])

    xlsx_path = out_path / "metrics_report.xlsx"
    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            per_image_df.to_excel(writer, index=False, sheet_name="per_image")
            per_class_df.to_excel(writer, index=False, sheet_name="per_class")
            summary_df.to_excel(writer, index=False, sheet_name="summary")
        print(f"Saved Excel metrics → {xlsx_path}")
    except Exception as e:
        per_image_df.to_csv(out_path / "per_image.csv", index=False)
        per_class_df.to_csv(out_path / "per_class.csv", index=False)
        summary_df.to_csv(out_path / "summary.csv", index=False)
        print(f"Excel write failed ({e}). Wrote CSVs instead to {out_path}")

    # Console summary
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved metrics to {out_path}/metrics_report.json, metrics_report.xlsx (or CSVs), and confusion_matrix.png")
    return acc, cm, report


import time
import torch
from thop import profile
from fvcore.nn import FlopCountAnalysis

def model_efficiency(model, sample_input, device="cuda"):
    """
    Computes parameters, FLOPs, and inference time for a given model.
    Works for classification and detection models.
    """
    model = model.to(device)
    model.eval()

    # --- Parameters ---
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # in millions

    # --- FLOPs ---
    try:
        flops = FlopCountAnalysis(model, sample_input).total() / 1e9  # in GFLOPs
    except Exception:
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
        flops /= 1e9

    # --- Inference time ---
    sample_input = sample_input.to(device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model(sample_input)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000  # ms
    time_per_image = elapsed / sample_input.size(0)

    return params, flops, time_per_image

def main():
    ap = argparse.ArgumentParser("Compute accuracy/F1/recall/precision + confusion matrix for a folder")
    ap.add_argument("--data", required=True, help="Folder with class subfolders (e.g., algae/ microplastics/)")
    ap.add_argument("--weights", required=True, help="Path to classifier .pth")
    ap.add_argument("--out", default="results", help="Where to save metrics outputs")
    ap.add_argument("--class_names", nargs="*", default=["algae","microplastics"], help="Class names in order")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--num_visuals", type=int, default=50, help="Number of images to visualize")

    args = ap.parse_args()

    evaluate_folder(args.weights, args.data, out_dir=args.out,
            class_names=args.class_names,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            num_visuals=args.num_visuals)

if __name__ == "__main__":
    main()
