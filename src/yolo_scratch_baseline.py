"""
YOLO From-Scratch Baseline — Train + Evaluate + Benchmark
==========================================================
Trains YOLOv8n, YOLOv8s, or YOLOv5n from RANDOM INITIALIZATION
(no COCO pretrained weights) for fair comparison with LDBST-MP.

C5 Revision: All YOLO baselines are trained from scratch to match
the training conditions of LDBST-MP (same dataset, same epochs,
same optimizer, no external pretraining).

Usage:
    # Train + benchmark a single model
    python yolo_scratch_baseline.py --model yolov8n --task both --data data.yaml --epochs 100 --batch 16 --img_size 224
    python yolo_scratch_baseline.py --model yolov8s --task both --data data.yaml --epochs 100 --batch 16 --img_size 224
    python yolo_scratch_baseline.py --model yolov5n --task both --data data.yaml --epochs 100 --batch 16 --img_size 224

    # Run all three models back-to-back (recommended for Table 2 revision)
    for model in yolov8n yolov8s yolov5n; do
        python yolo_scratch_baseline.py --model $model --task both --data data.yaml --epochs 100 --batch 16 --img_size 224
    done

    # Benchmark only (requires --weights)
    python yolo_scratch_baseline.py --model yolov8s --task benchmark --data data.yaml --weights runs/detect/yolov8s_scratch/weights/best.pt
"""

import argparse
import os
import time
import random
import json
import yaml
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Maps CLI model name → (yaml config for scratch init, display name)
# Using .yaml loads architecture only — no pretrained weights
MODEL_CONFIGS = {
    "yolov8n": ("yolov8n.yaml", "YOLOv8n"),
    "yolov8s": ("yolov8s.yaml", "YOLOv8s"),
    "yolov5n": ("yolov5n.yaml", "YOLOv5n"),
}


# -----------------------------------------------------------------------
# Dataset YAML
# -----------------------------------------------------------------------
def create_dataset_yaml(data_dir, output_path="dataset.yaml"):
    """Create YAML config file for YOLO training."""
    data_dir = Path(data_dir).resolve()
    config = {
        'path': str(data_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'algae',
            1: 'microplastics'
        }
    }
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[setup] Created dataset config: {output_path}")
    return output_path


# -----------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------
def benchmark_model(weights_path: str, model_display_name: str,
                    img_size: int = 224, n_warmup: int = 50, n_runs: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[benchmark] Device: {device}")
    print(f"[benchmark] Loading weights from: {weights_path}")

    yolo = YOLO(weights_path)
    model = yolo.model.to(device).eval()

    params = sum(p.numel() for p in model.parameters())
    params_m = params / 1e6
    model_size_mb = os.path.getsize(weights_path) / (1024 ** 2)

    print(f"[benchmark] Parameters: {params:,} ({params_m:.2f}M)")
    print(f"[benchmark] Model size: {model_size_mb:.2f} MB")

    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Clear cached memory and reset tracking BEFORE warmup — matches old
    # yolo_baseline.py methodology and LDBST-MP benchmark so VRAM numbers
    # are directly comparable across all models.
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    print(f"[benchmark] Running {n_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
        # Reset peak stats AFTER warmup so only inference-time VRAM is counted
        # (warmup allocations are excluded, matching LDBST-MP measurement).
        torch.cuda.reset_peak_memory_stats(device)

    print(f"[benchmark] Running {n_runs} timed iterations...")
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times = np.array(times)
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "model": model_display_name,
        "initialization": "scratch",   # C5: explicitly recorded for paper
        "weights": weights_path,
        "params_total": params,
        "params_million": round(params_m, 2),
        "model_size_mb": round(model_size_mb, 2),
        "inference_ms_mean": round(mean_ms, 2),
        "inference_ms_std": round(std_ms, 2),
        "fps": round(fps, 1),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "img_size": img_size,
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "device": str(device),
    }


# -----------------------------------------------------------------------
# Detection evaluation
# -----------------------------------------------------------------------
def evaluate_detection(weights_path: str, data_yaml: str, img_size: int = 224,
                       conf: float = 0.5, iou: float = 0.45):
    print(f"\n[evaluate] Running detection evaluation...")
    print(f"[evaluate] Weights: {weights_path}")
    print(f"[evaluate] Data:    {data_yaml}")
    print(f"[evaluate] conf={conf}, iou={iou}")

    yolo = YOLO(weights_path)
    metrics = yolo.val(
        data=data_yaml,
        split="test",
        imgsz=img_size,
        conf=conf,
        iou=iou,
        batch=1,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=True,
        save=False,
        plots=False,
    )

    precision = float(metrics.box.p.mean()) if hasattr(metrics.box.p, 'mean') else float(metrics.box.p)
    recall    = float(metrics.box.r.mean()) if hasattr(metrics.box.r, 'mean') else float(metrics.box.r)
    map50     = float(metrics.box.map50)
    map50_95  = float(metrics.box.map)
    f1        = 2 * (precision * recall) / (precision + recall + 1e-6)

    return {
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "map50":       round(map50, 4),
        "map50_95":    round(map50_95, 4),
        "conf_thresh": conf,
        "iou_thresh":  iou,
    }


# -----------------------------------------------------------------------
# Training (from scratch)
# -----------------------------------------------------------------------
def train_model(model_key: str, data_yaml: str, epochs: int, batch: int,
                img_size: int, device: str, name: str):
    yaml_cfg, display_name = MODEL_CONFIGS[model_key]

    print("\n" + "=" * 70)
    print(f"Training {display_name} FROM SCRATCH (C5 fair comparison)")
    print("=" * 70)
    print(f"[config] Architecture: {yaml_cfg}  (no pretrained weights)")
    print(f"[config] Data:         {data_yaml}")
    print(f"[config] Epochs:       {epochs}")
    print(f"[config] Batch:        {batch}")
    print(f"[config] Img size:     {img_size}")
    print(f"[config] Device:       {device}")

    # Load from .yaml → random weight initialization, no COCO weights
    model = YOLO(yaml_cfg)
    print(f"\n[model] Loaded architecture from {yaml_cfg} — weights randomly initialized")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        device=device,
        name=name,
        patience=20,
        save=True,
        workers=4,
        pretrained=False,       # C5: explicitly disabled
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        conf=0.001,
        iou=0.5,
    )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\n[train] Done. Best weights: {best_weights}")
    return str(best_weights), display_name


# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------
def save_results(benchmark: dict, detection: dict, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_name = benchmark["model"]
    slug = model_name.lower().replace(" ", "_")

    combined = {**benchmark, **detection}

    json_path = Path(out_dir) / f"{slug}_scratch_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)

    txt_path = Path(out_dir) / f"{slug}_scratch_benchmark.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{model_name} SCRATCH BENCHMARK RESULTS\n")
        f.write("Initialization: random (no COCO pretraining) — C5 revision\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model info\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Parameters:   {benchmark['params_total']:,} ({benchmark['params_million']}M)\n")
        f.write(f"  Model size:   {benchmark['model_size_mb']} MB\n")
        f.write(f"  Device:       {benchmark['device']}\n\n")

        f.write(f"Detection metrics (conf={detection['conf_thresh']})\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Precision:    {detection['precision']:.4f}\n")
        f.write(f"  Recall:       {detection['recall']:.4f}\n")
        f.write(f"  F1-Score:     {detection['f1']:.4f}\n")
        f.write(f"  mAP@0.5:      {detection['map50']:.4f} ({detection['map50']*100:.2f}%)\n")
        f.write(f"  mAP@0.5:0.95: {detection['map50_95']:.4f}\n\n")

        f.write("Inference efficiency\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Inference:    {benchmark['inference_ms_mean']} ± {benchmark['inference_ms_std']} ms\n")
        f.write(f"  FPS:          {benchmark['fps']}\n")
        f.write(f"  Peak VRAM:    {benchmark['peak_vram_mb']} MB\n\n")

        f.write("=" * 70 + "\n")
        f.write("Table 2 row (copy into paper):\n")
        f.write(f"  {model_name} (scratch) | {benchmark['params_million']}M | "
                f"{detection['map50']*100:.2f}% | {detection['f1']:.4f} | "
                f"{benchmark['fps']} FPS | {benchmark['peak_vram_mb']} MB\n")
        f.write("=" * 70 + "\n")

    print(f"\n[save] Results → {txt_path}")
    print(f"[save]          → {json_path}")

    print("\n" + "=" * 70)
    print("TABLE 2 ROW (copy into paper):")
    print("=" * 70)
    print(f"  {model_name} (scratch) | {benchmark['params_million']}M | "
          f"{detection['map50']*100:.2f}% | {detection['f1']:.4f} | "
          f"{benchmark['fps']} FPS | {benchmark['peak_vram_mb']} MB")
    print("=" * 70)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train YOLO baselines from scratch for fair C5 comparison"
    )
    ap.add_argument("--model",    choices=list(MODEL_CONFIGS.keys()), required=True,
                    help="Which YOLO variant to run")
    ap.add_argument("--task",     choices=["train", "benchmark", "both"], default="both")
    ap.add_argument("--data",     default=None,          help="Path to data.yaml")
    ap.add_argument("--data_dir", default="./data_yolo", help="Auto-generate data.yaml from this folder")
    ap.add_argument("--epochs",   type=int, default=100)
    ap.add_argument("--batch",    type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device",   default="0")
    ap.add_argument("--weights",  default=None,          help="Existing weights for benchmark-only mode")
    ap.add_argument("--n_warmup", type=int, default=50)
    ap.add_argument("--n_runs",   type=int, default=500)
    ap.add_argument("--conf",     type=float, default=0.5)
    ap.add_argument("--iou",      type=float, default=0.45)
    ap.add_argument("--out",      default="benchmark_results_scratch")
    args = ap.parse_args()

    if args.data is None:
        args.data = create_dataset_yaml(args.data_dir)

    _, display_name = MODEL_CONFIGS[args.model]
    name = f"{args.model}_scratch"

    print("\n" + "=" * 70)
    print(f"{display_name} — From-Scratch Baseline (C5 Revision)")
    print("=" * 70)

    best_weights = args.weights

    if args.task in ("train", "both"):
        best_weights, display_name = train_model(
            model_key=args.model, data_yaml=args.data,
            epochs=args.epochs, batch=args.batch,
            img_size=args.img_size, device=args.device, name=name,
        )

    if args.task in ("benchmark", "both"):
        if best_weights is None:
            raise ValueError("--weights required when task=benchmark")

        benchmark = benchmark_model(
            weights_path=best_weights, model_display_name=display_name,
            img_size=args.img_size, n_warmup=args.n_warmup, n_runs=args.n_runs,
        )
        detection = evaluate_detection(
            weights_path=best_weights, data_yaml=args.data,
            img_size=args.img_size, conf=args.conf, iou=args.iou,
        )
        save_results(benchmark, detection, out_dir=args.out)

        print("\n" + "=" * 70)
        print("FULL BENCHMARK SUMMARY")
        print("=" * 70)
        for k, v in {**benchmark, **detection}.items():
            print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()