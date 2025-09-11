from pathlib import Path
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from dataset import IM_SIZE, letterbox_pad, clahe_rgb
from model import DualBranchSwinTinyClassifier

DEVICE = "cpu"

def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_rgb).permute(2,0,1).float()/255.0  # CHW

@torch.no_grad()
def predict_prob(model, x_std: torch.Tensor, x_cla: torch.Tensor) -> float:
    logits = model(x_std, x_cla)
    p = F.softmax(logits, dim=1)[0,1].item()
    return p

def saliency_heatmap(model, x_std: torch.Tensor, x_cla: torch.Tensor, class_idx: int = 1) -> np.ndarray:
    """
    Input-gradient saliency wrt x_std to get a spatial map (no fragile hooks).
    Returns (H,W) float in [0,1].
    """
    model.zero_grad(set_to_none=True)
    x_std.requires_grad_(True)
    x_cla.requires_grad_(False)
    logits = model(x_std, x_cla)  # (1,2)
    score = logits[0, class_idx]
    score.backward()
    g = x_std.grad.detach()[0]  # (3,224,224)
    g = g.abs().mean(0).cpu().numpy()  # (224,224)
    g -= g.min()
    if g.max() > 1e-6:
        g /= g.max()
    return g

def boxes_from_heatmap(hm: np.ndarray, thresh: float = 0.25, min_area: int = 20, orig_shape=None):
    """
    hm: (224,224) float [0,1]; threshold + contour → boxes in *original image* coordinates if orig_shape provided.
    """
    m = (hm * 255).astype(np.uint8)
    _, bw = cv2.threshold(m, int(255 * thresh), 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w*h >= min_area:
            boxes.append([x, y, x+w, y+h])

    # if original shape is given, map 224x224 coords back
    if orig_shape is not None:
        H, W = orig_shape[:2]
        sx, sy = W/IM_SIZE, H/IM_SIZE
        boxes = [[int(a*sx), int(b*sy), int(c*sx), int(d*sy)] for (a,b,c,d) in boxes]
    return boxes

def annotate_one(model, in_path: Path, out_path: Path,
                 cls_thresh: float, heat_thresh: float, min_area: int):
    # read image (RGB)
    img_bgr = cv2.imread(str(in_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read: {in_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # preprocess both branches
    letter = letterbox_pad(img_rgb, IM_SIZE)
    cla = clahe_rgb(letter)

    x_std = to_tensor(letter).unsqueeze(0).to(DEVICE)
    x_cla = to_tensor(cla).unsqueeze(0).to(DEVICE)

    p_micro = predict_prob(model, x_std, x_cla)

    # saliency only if predicted (or close to) microplastics
    hm = saliency_heatmap(model, x_std.clone().requires_grad_(True), x_cla, class_idx=1)

    # default: looser heatmap threshold to draw something
    boxes = boxes_from_heatmap(hm, heat_thresh, min_area, orig_shape=img_rgb.shape)

    # draw
    vis = img_bgr.copy()
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(vis, f"p(microplastics)={p_micro:.2f}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if p_micro>=cls_thresh else (255,255,0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    return len(boxes), p_micro, out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, help="single image path")
    ap.add_argument("--input", type=str, help="directory of images")
    ap.add_argument("--outdir", type=str, default="./results")
    ap.add_argument("--thresh", type=float, default=0.55, help="classification threshold for microplastics (default 0.55)")
    ap.add_argument("--heat", type=float, default=0.25, help="heatmap threshold for box extraction (default 0.25)")
    ap.add_argument("--min-area", type=int, default=20, help="min contour area in heatmap (default 20)")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    ckpt = ROOT / "classifier.pth"
    out_dir = (ROOT / args.outdir).resolve()

    model = DualBranchSwinTinyClassifier(num_classes=2, freeze_backbone=False, try_pretrained=False).to(DEVICE)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"Loaded weights from {ckpt}")
    else:
        print("WARNING: classifier.pth not found; using random weights.")

    model.eval()

    if args.image:
        p = Path(args.image)
        n, pm, outp = annotate_one(model, p, out_dir / (p.stem + "_det.png"), args.thresh, args.heat, args.min_area)
        print(f"{p.name} → {outp.name} ({n} box(es), p_micro={pm:.2f})")
        return

    if args.input:
        inp = Path(args.input)
        files = sorted([*inp.glob("*.jpg"), *inp.glob("*.jpeg"), *inp.glob("*.png")])
        for i, p in enumerate(files, 1):
            try:
                n, pm, outp = annotate_one(model, p, out_dir / (p.stem + "_det.png"), args.thresh, args.heat, args.min_area)
                print(f"[{i}/{len(files)}] {p.name} → {outp.name} ({n} box(es), p_micro={pm:.2f})")
            except Exception as e:
                print(f"[{i}/{len(files)}] {p.name} → ERROR: {e}")
        return

    print("Nothing to do. Provide --image or --input.")

if __name__ == "__main__":
    main()
