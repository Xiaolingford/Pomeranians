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
import pandas as pd
import random, torch, numpy as np
from ultralytics import YOLO
import time
from thop import profile
from fvcore.nn import FlopCountAnalysis


# ---- Force deterministic behavior ----
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------- Grad-CAM ----------
def generate_cam(model, image_tensor, class_idx=None):
    model.eval()
    feature_maps, gradients = {}, {}

    def forward_hook(module, inp, out):
        
        feature_maps["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    outputs = model(image_tensor.unsqueeze(0))
    if class_idx is None:
        class_idx = int(outputs.argmax(dim=1))
    model.zero_grad()
    target = outputs[0, class_idx]
    target.backward()

    fmap = feature_maps["value"][0]
    grads = gradients["value"][0]
    hook_f.remove(); hook_b.remove()

    weights = grads.mean(dim=(1, 2))
    cam_map = torch.zeros_like(fmap[0])
    for i, w in enumerate(weights):
        cam_map += w * fmap[i]
    cam_map = torch.relu(cam_map)
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() + 1e-8)
    return cam_map.cpu().numpy()


def overlay_cam_on_image(img: Image.Image, cam_map: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm
    W, H = img.size
    heatmap = (cm.jet(cam_map) * 255).astype(np.uint8)[:, :, :3]
    heatmap = Image.fromarray(heatmap).resize((W, H), resample=Image.BILINEAR)
    overlay = (0.5 * np.array(img.convert("RGB")) + 0.5 * np.array(heatmap)).astype(np.uint8)
    return Image.fromarray(overlay)


# ---------- Model efficiency ----------
def model_efficiency(model, sample_input, device="cuda"):
    model = model.to(device)
    model.eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    try:
        flops = FlopCountAnalysis(model, sample_input).total() / 1e9
    except Exception:
        flops, _ = profile(model, inputs=(sample_input,), verbose=False)
        flops /= 1e9

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model(sample_input)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    return params, flops, elapsed


# ---------- Main evaluation ----------
@torch.no_grad()
def evaluate_folder(weights, data_dir, out_dir="results", batch_size=32, num_visuals=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading YOLOv8 classification model from {weights} ...")
    yolo_wrapper = YOLO(weights)
    model = yolo_wrapper.model.to(device).eval()
    class_names = list(yolo_wrapper.names.values())
    num_classes = len(class_names)
    print(f"Loaded model with {num_classes} classes: {class_names}")

    # --- Compute model efficiency ---
    print("\n[Analyzing model efficiency...]")
    sample_input = torch.randn(1, 3, 224, 224).to(device)
    params, flops, time_ms = model_efficiency(model, sample_input, device)
    print(f"[Efficiency] Params: {params:.2f}M | FLOPs: {flops:.2f}G | Inference: {time_ms:.2f} ms/image\n")

    # --- YOLOv8-like transform ---
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # scale to [0,1]
    ])
    ds = datasets.ImageFolder(data_dir, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Evaluation loop ---
    y_true, y_pred, per_image_rows = [], [], []
    paths_all = [p for p, _ in ds.samples]
    path_cursor = 0

    for x, y in dl:
        bs = x.size(0)
        batch_paths = paths_all[path_cursor:path_cursor+bs]
        path_cursor += bs

        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy().tolist()
        y_pred.extend(preds)
        y_true.extend(y.cpu().numpy().tolist())

        for pth, t, pr in zip(batch_paths, y.numpy().tolist(), preds):
            per_image_rows.append({
                "filename": Path(pth).name,
                "true_class": class_names[t],
                "predicted_class": class_names[pr],
                "correct": (t == pr)
            })

    # --- Metrics ---
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # --- Save JSON ---
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

    # --- Confusion Matrix ---
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path / "confusion_matrix.png")
    plt.close(fig)

    # --- Excel/CSV export ---
    per_image_df = pd.DataFrame(per_image_rows)
    per_class_df = pd.DataFrame({
        "class": class_names,
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
        "samples": int(report["weighted avg"]["support"])
    }])

    xlsx_path = out_path / "metrics_report.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        per_image_df.to_excel(writer, index=False, sheet_name="per_image")
        per_class_df.to_excel(writer, index=False, sheet_name="per_class")
        summary_df.to_excel(writer, index=False, sheet_name="summary")

    print(f"\n✅ Accuracy: {acc:.4f}")
    print(f"Saved metrics to: {out_path}/metrics_report.json, metrics_report.xlsx, confusion_matrix.png")

    return acc, cm, report


def main():
    ap = argparse.ArgumentParser("Compute YOLOv8-matched metrics")
    ap.add_argument("--data", required=True, help="Folder with class subfolders (e.g., algae/ microplastics/)")
    ap.add_argument("--weights", required=True, help="Path to YOLOv8 .pt weights")
    ap.add_argument("--out", default="results", help="Output directory for reports")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_visuals", type=int, default=50)
    args = ap.parse_args()

    evaluate_folder(
        weights=args.weights,
        data_dir=args.data,
        out_dir=args.out,
        batch_size=args.batch_size,
        num_visuals=args.num_visuals
    )


if __name__ == "__main__":
    main()
