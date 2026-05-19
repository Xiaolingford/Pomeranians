"""
YOLOv8n Detection — Training + Evaluation + Benchmark
======================================================
Nano-scale YOLO baseline for microplastic detection.
Uses Ultralytics native pipeline with settings matched as closely
as possible to the LDBST-MP training recipe.

Note: Ultralytics uses its own internal training loop (SGD/AdamW,
mosaic augmentation, warmup, etc.). We match what we can:
  - AdamW optimizer
  - weight_decay=1e-2
  - warmup_epochs=5
  - comparable augmentation settings
  - same image size, batch size, epochs
  - same conf/iou thresholds for evaluation

Usage:
    # Train + benchmark
    python src/yolov8n_det.py --task both --data data.yaml --epochs 100 --batch 16 --img_size 224

    # Train only
    python src/yolov8n_det.py --task train --data data.yaml --epochs 100

    # Benchmark only (requires --weights)
    python src/yolov8n_det.py --task benchmark --data data.yaml --weights runs/detect/yolov8n/weights/best.pt
"""

import argparse
import os
import time
import random
import json
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

MODEL_NAME = "YOLOv8n"


# -----------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------
def benchmark_model(weights_path: str, img_size: int = 224, n_warmup: int = 50, n_runs: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[benchmark] Device: {device}")
    print(f"[benchmark] Loading weights from: {weights_path}")

    yolo = YOLO(weights_path)
    model = yolo.model.to(device).eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = params / 1e6
    model_size_mb = os.path.getsize(weights_path) / (1024 ** 2)

    print(f"[benchmark] Parameters: {params:,} ({params_m:.2f}M)")
    print(f"[benchmark] Model size: {model_size_mb:.2f} MB")

    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Warmup
    print(f"[benchmark] Running {n_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
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

    results = {
        "model": MODEL_NAME,
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
    return results


# -----------------------------------------------------------------------
# Detection evaluation
# -----------------------------------------------------------------------
def evaluate_detection(weights_path: str, data_yaml: str, img_size: int = 224,
                       conf: float = 0.5, iou: float = 0.45):
    print(f"\n[evaluate] Running detection evaluation...")
    print(f"[evaluate] Weights: {weights_path}")
    print(f"[evaluate] Data: {data_yaml}")
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
    recall = float(metrics.box.r.mean()) if hasattr(metrics.box.r, 'mean') else float(metrics.box.r)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

    detection_results = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "conf_thresh": conf,
        "iou_thresh": iou,
    }
    return detection_results


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------
def train_model(data_yaml: str, epochs: int, batch: int, img_size: int,
                device: str, project: str, name: str, pretrained: bool):
    print("\n" + "=" * 70)
    print(f"Training {MODEL_NAME} for Microplastic Detection")
    print("=" * 70)
    print(f"[config] Data:       {data_yaml}")
    print(f"[config] Epochs:     {epochs}")
    print(f"[config] Batch:      {batch}")
    print(f"[config] Img size:   {img_size}")
    print(f"[config] Device:     {device}")
    print(f"[config] Pretrained: {pretrained}")

    weights = "yolov8n.pt" if pretrained else "yolov8n.yaml"
    model = YOLO(weights)

    print(f"\n[model] Loaded: {weights}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        seed=SEED,
        deterministic=True,
        verbose=True,
        # Optimizer — match LDBST-MP where possible
        optimizer="AdamW",
        lr0=1e-4,
        weight_decay=1e-2,
        warmup_epochs=5,
        # Augmentation — comparable to LDBST-MP pipeline
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.3,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        blur=0.01,
        close_mosaic=10,
        save=True,
        save_period=-1,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    print(f"\n[train] Training complete. Best weights: {best_weights}")
    return str(best_weights)


# -----------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------
def save_results(benchmark: dict, detection: dict, out_dir: str = "benchmark_results"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    combined = {**benchmark, **detection}
    json_path = Path(out_dir) / f"{MODEL_NAME.lower()}_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)

    txt_path = Path(out_dir) / f"{MODEL_NAME.lower()}_benchmark.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{MODEL_NAME} BENCHMARK RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("Model Info\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Parameters:   {benchmark['params_total']:,} ({benchmark['params_million']}M)\n")
        f.write(f"  Model Size:   {benchmark['model_size_mb']} MB\n")
        f.write(f"  Device:       {benchmark['device']}\n\n")

        f.write(f"Detection Metrics (conf={detection['conf_thresh']})\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Precision:    {detection['precision']:.4f}\n")
        f.write(f"  Recall:       {detection['recall']:.4f}\n")
        f.write(f"  F1-Score:     {detection['f1']:.4f}\n")
        f.write(f"  mAP@0.5:      {detection['map50']:.4f} ({detection['map50']*100:.2f}%)\n")
        f.write(f"  mAP@0.5:0.95: {detection['map50_95']:.4f}\n\n")

        f.write("Inference Efficiency\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Inference:    {benchmark['inference_ms_mean']} ± {benchmark['inference_ms_std']} ms\n")
        f.write(f"  FPS:          {benchmark['fps']}\n")
        f.write(f"  Peak VRAM:    {benchmark['peak_vram_mb']} MB\n\n")

        f.write("=" * 70 + "\n")
        f.write("For Table in your paper:\n")
        f.write(f"  {MODEL_NAME} | {benchmark['params_million']}M | {detection['map50']*100:.2f}% | "
                f"{detection['f1']:.4f} | {benchmark['fps']} FPS | {benchmark['peak_vram_mb']} MB\n")
        f.write("=" * 70 + "\n")

    print(f"\n[save] Results saved to:")
    print(f"       {txt_path}")
    print(f"       {json_path}")

    print("\n" + "=" * 70)
    print("TABLE ROW (copy into your paper):")
    print("=" * 70)
    print(f"{MODEL_NAME} | {benchmark['params_million']}M | {detection['map50']*100:.2f}% | "
          f"{detection['f1']:.4f} | {benchmark['fps']} FPS | {benchmark['peak_vram_mb']} MB")
    print("=" * 70)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=f"Train and benchmark {MODEL_NAME} for microplastic detection")

    ap.add_argument("--task", choices=["train", "benchmark", "both"], default="both")
    ap.add_argument("--data", required=True, help="Path to data.yaml")

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="yolov8n_microplastic")
    ap.add_argument("--pretrained", action="store_true")

    # Benchmark
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n_warmup", type=int, default=50)
    ap.add_argument("--n_runs", type=int, default=500)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default="benchmark_results")

    args = ap.parse_args()

    print("\n" + "=" * 70)
    print(f"{MODEL_NAME} — Microplastic Detection Baseline")
    print("=" * 70)

    best_weights = args.weights

    if args.task in ("train", "both"):
        best_weights = train_model(
            data_yaml=args.data, epochs=args.epochs, batch=args.batch,
            img_size=args.img_size, device=args.device,
            project=args.project, name=args.name, pretrained=args.pretrained,
        )

    if args.task in ("benchmark", "both"):
        if best_weights is None:
            raise ValueError("--weights required when task=benchmark")

        print("\n" + "=" * 70)
        print(f"Benchmarking {MODEL_NAME}")
        print("=" * 70)

        benchmark = benchmark_model(
            weights_path=best_weights, img_size=args.img_size,
            n_warmup=args.n_warmup, n_runs=args.n_runs,
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