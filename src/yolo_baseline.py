"""
yolo_baseline.py - Train and evaluate YOLOv8/YOLOv5 for detection comparison

Benchmark uses raw model(tensor) forward pass (no predict pipeline)
to match LDBST-MP benchmark methodology.

Usage:
    # Train YOLOv8-nano
    python yolo_baseline.py --model yolov8n --data_dir ./data_yolo --epochs 100 --imgsz 224
    
    # Train YOLOv5-nano
    python yolo_baseline.py --model yolov5n --data_dir ./data_yolo --epochs 100 --imgsz 224
    
    # Evaluate only (if already trained)
    python yolo_baseline.py --model yolov8n --data_dir ./data_yolo --eval_only --weights runs/detect/yolov8n_microplastic/weights/best.pt
    
    # Benchmark only
    python yolo_baseline.py --model yolov8n --benchmark_only --weights runs/detect/yolov8n_microplastic/weights/best.pt
"""

import argparse
import torch
import numpy as np
import time
import os
import yaml
from pathlib import Path
from tqdm import tqdm

# ============================================================
# SETUP
# ============================================================

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


# ============================================================
# TRAINING
# ============================================================

def train_yolo(args):
    """Train YOLO model."""
    from ultralytics import YOLO
    
    # Create dataset config
    yaml_path = create_dataset_yaml(args.data_dir)
    
    # Load model
    if args.model == "yolov8n":
        model = YOLO("yolov8n.pt")
        project_name = "yolov8n_microplastic"
    elif args.model == "yolov5n":
        model = YOLO("yolov5nu.pt")  # YOLOv5-nano ultralytics version
        project_name = "yolov5n_microplastic"
    elif args.model == "yolov8s":
        model = YOLO("yolov8s.pt")
        project_name = "yolov8s_microplastic"
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"\n[training] Starting {args.model} training...")
    print(f"[training] Epochs: {args.epochs}, Image size: {args.imgsz}, Batch: {args.batch_size}")
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        name=project_name,
        patience=20,  # Early stopping patience
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=4,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        conf=0.001,  # Low conf for training
        iou=0.5,
    )
    
    # Get actual save directory (handles auto-increment)
    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\n[saved] Best weights: {best_weights}")
    
    return best_weights


# ============================================================
# EVALUATION
# ============================================================

def evaluate_yolo(args, weights_path):
    """Evaluate YOLO model on test set."""
    from ultralytics import YOLO
    
    print(f"\n[eval] Loading model from {weights_path}")
    model = YOLO(weights_path)
    
    # Create dataset config if not exists
    yaml_path = "dataset.yaml"
    if not os.path.exists(yaml_path):
        yaml_path = create_dataset_yaml(args.data_dir)
    
    # Get model info (count ALL params, not just requires_grad)
    num_params = sum(p.numel() for p in model.model.parameters())
    print(f"[model] Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Evaluate on test set
    print(f"\n[eval] Evaluating on test set (conf={args.conf_thresh})...")
    
    metrics = model.val(
        data=yaml_path,
        split='test',
        imgsz=args.imgsz,
        batch=args.batch_size,
        conf=args.conf_thresh,
        iou=0.5,
        device=0 if torch.cuda.is_available() else 'cpu',
        verbose=False,
    )
    
    # Extract metrics
    results = {
        'precision': float(metrics.box.mp),  # Mean precision
        'recall': float(metrics.box.mr),     # Mean recall
        'map50': float(metrics.box.map50),   # mAP@0.5
        'map50_95': float(metrics.box.map),  # mAP@0.5:0.95
        'f1': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr + 1e-6),
    }
    
    return results, num_params


def benchmark_yolo(args, weights_path):
    """
    Benchmark YOLO model for FPS and VRAM.
    
    Uses raw model(tensor) forward pass — NOT model.predict() —
    to match LDBST-MP benchmark methodology (pure inference, no
    preprocessing or postprocessing overhead).
    """
    from ultralytics import YOLO
    
    print(f"\n[benchmark] Loading model from {weights_path}")
    print(f"[benchmark] Method: raw model(tensor) — matches LDBST-MP benchmark")
    
    yolo = YOLO(weights_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = yolo.model.to(device).eval()
    
    # Count ALL params (not just requires_grad — stripped after training)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[benchmark] Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Create dummy input (batch size 1, matching LDBST-MP)
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    
    # Reset VRAM tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Warmup (not timed)
    n_warmup = 50
    print(f"[benchmark] Warmup ({n_warmup} runs)...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Timed runs
    n_runs = 500
    print(f"[benchmark] Timing ({n_runs} runs)...")
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(dummy_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
    
    # Calculate stats
    times = np.array(times)
    avg_time_ms = np.mean(times)
    std_time_ms = np.std(times)
    fps = 1000 / avg_time_ms
    
    # Get VRAM
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0
    
    return {
        'avg_time_ms': avg_time_ms,
        'std_time_ms': std_time_ms,
        'fps': fps,
        'peak_vram_mb': peak_vram_mb,
        'num_params': num_params,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO baselines")
    parser.add_argument("--model", type=str, default="yolov8n",
                        choices=["yolov8n", "yolov5n", "yolov8s"])
    parser.add_argument("--data_dir", type=str, default="./data_yolo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--benchmark_only", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--num_runs", type=int, default=100)
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"YOLO BASELINE: {args.model.upper()}")
    print("=" * 70)
    
    # Determine weights path
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = Path(f"runs/detect/{args.model}_microplastic/weights/best.pt")
    
    # Training
    if not args.eval_only and not args.benchmark_only:
        weights_path = train_yolo(args)
    
    # Check weights exist
    if not weights_path.exists():
        print(f"[error] Weights not found: {weights_path}")
        print(f"[error] Please train first or provide --weights path")
        return
    
    # Evaluation
    if not args.benchmark_only:
        results, num_params = evaluate_yolo(args, weights_path)
        
        print("\n" + "=" * 70)
        print(f"TEST SET RESULTS ({args.model.upper()}) @ conf={args.conf_thresh}")
        print("=" * 70)
        print(f"\n{'Metric':<20} {'Value':>12}")
        print("-" * 34)
        print(f"{'Precision':<20} {results['precision']:>12.4f}")
        print(f"{'Recall':<20} {results['recall']:>12.4f}")
        print(f"{'F1 Score':<20} {results['f1']:>12.4f}")
        print(f"{'mAP@0.5':<20} {results['map50']:>12.4f}")
        print(f"{'mAP@0.5:0.95':<20} {results['map50_95']:>12.4f}")
    else:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        num_params = sum(p.numel() for p in model.model.parameters())
        results = {}
    
    # Benchmark
    benchmark = benchmark_yolo(args, weights_path)
    
    # Use param count from benchmark (more reliable)
    num_params = benchmark.get('num_params', num_params)
    
    print(f"\n{'Hardware Performance'}")
    print("-" * 34)
    print(f"{'Parameters':<20} {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"{'Inference Time':<20} {benchmark['avg_time_ms']:>10.2f} ms")
    print(f"{'FPS':<20} {benchmark['fps']:>10.1f}")
    print(f"{'Peak VRAM':<20} {benchmark['peak_vram_mb']:>10.2f} MB")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR COMPARISON TABLE")
    print("=" * 70)
    
    model_size_mb = os.path.getsize(weights_path) / (1024 * 1024) if weights_path.exists() else 0
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  Model: {args.model.upper():<59} │
├─────────────────────────────────────────────────────────────────────┤
│  Parameters:    {num_params:,} ({num_params/1e6:.2f}M){' '*(40-len(f'{num_params:,} ({num_params/1e6:.2f}M)'))}│
│  Model Size:    {model_size_mb:.2f} MB{' '*(51-len(f'{model_size_mb:.2f} MB'))}│""")
    
    if results:
        print(f"""│  Precision:     {results['precision']*100:.2f}%{' '*(52-len(f"{results['precision']*100:.2f}%"))}│
│  Recall:        {results['recall']*100:.2f}%{' '*(52-len(f"{results['recall']*100:.2f}%"))}│
│  F1 Score:      {results['f1']:.4f}{' '*(52-len(f"{results['f1']:.4f}"))}│
│  mAP@0.5:       {results['map50']*100:.2f}%{' '*(52-len(f"{results['map50']*100:.2f}%"))}│""")
    
    print(f"""│  FPS:           {benchmark['fps']:.1f}{' '*(53-len(f"{benchmark['fps']:.1f}"))}│
│  VRAM:          {benchmark['peak_vram_mb']:.2f} MB{' '*(49-len(f"{benchmark['peak_vram_mb']:.2f} MB"))}│
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Save results
    results_file = f"{args.model}_detection_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")
        if results:
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1 Score: {results['f1']:.4f}\n")
            f.write(f"mAP@0.5: {results['map50']:.4f}\n")
            f.write(f"mAP@0.5:0.95: {results['map50_95']:.4f}\n")
        f.write(f"FPS: {benchmark['fps']:.1f}\n")
        f.write(f"VRAM: {benchmark['peak_vram_mb']:.2f} MB\n")
    
    print(f"[saved] Results saved to {results_file}")


if __name__ == "__main__":
    main()