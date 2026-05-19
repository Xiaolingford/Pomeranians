"""
benchmark.py - Measure inference speed (FPS) and VRAM usage

Usage:
    python benchmark.py --model detector_v4.pth --task detection --num_runs 100
    python benchmark.py --model classifier_v3.pth --task classification --num_runs 100
    python benchmark.py --model detector_v4.pth --task detection --num_runs 100 --cpu  # Force CPU
"""

import argparse
import torch
import numpy as np
import time
import os
from pathlib import Path

def get_model_size_mb(model_path):
    """Get model file size in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters())

def benchmark_model(args):
    """Run benchmark for FPS and VRAM measurement."""
    from model2 import LDBST_MP
    
    # Determine device
    if args.cpu:
        device = torch.device("cpu")
        device_name = "CPU"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU (CUDA not available)"
    
    print("\n" + "=" * 70)
    print(f"LDBST-MP BENCHMARK: {args.task.upper()}")
    print("=" * 70)
    print(f"\n[config] Model: {args.model}")
    print(f"[config] Task: {args.task}")
    print(f"[config] Device: {device_name}")
    print(f"[config] Warmup runs: {args.warmup}")
    print(f"[config] Benchmark runs: {args.num_runs}")
    print(f"[config] Image size: 224x224")
    
    # Load model
    print("\n[model] Loading...")
    model = LDBST_MP(img_size=224, num_classes=args.num_classes, task=args.task)
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Model info
    num_params = count_parameters(model)
    model_size_mb = get_model_size_mb(args.model)
    
    print(f"[model] Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"[model] File size: {model_size_mb:.2f} MB")
    
    # Create dummy input (batch size 1)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # =========================================
    # GPU BENCHMARK (if available)
    # =========================================
    if device.type == "cuda":
        print(f"\n[benchmark] Running GPU benchmark on {device_name}...")
        
        # Reset VRAM tracking
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Get baseline VRAM (before model loads to GPU)
        baseline_vram = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Warmup runs (not timed)
        print(f"[benchmark] Warmup ({args.warmup} runs)...")
        with torch.no_grad():
            for _ in range(args.warmup):
                if args.task == "detection":
                    _ = model(dummy_input)
                else:
                    _ = model(dummy_input)
        
        # Synchronize GPU
        torch.cuda.synchronize()
        
        # Reset stats after warmup
        torch.cuda.reset_peak_memory_stats()
        
        # Timed runs
        print(f"[benchmark] Timing ({args.num_runs} runs)...")
        times = []
        
        with torch.no_grad():
            for i in range(args.num_runs):
                # Synchronize before timing
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                if args.task == "detection":
                    pred_boxes, pred_logits, pred_obj = model(dummy_input)
                else:
                    logits = model(dummy_input)
                
                # Synchronize after inference
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append((end - start) * 1000)  # Convert to ms
        
        # Get peak VRAM
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        current_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Calculate stats
        times = np.array(times)
        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)
        min_time_ms = np.min(times)
        max_time_ms = np.max(times)
        fps = 1000 / avg_time_ms
        
        gpu_results = {
            'device': device_name,
            'avg_time_ms': avg_time_ms,
            'std_time_ms': std_time_ms,
            'min_time_ms': min_time_ms,
            'max_time_ms': max_time_ms,
            'fps': fps,
            'peak_vram_mb': peak_vram_mb,
            'current_vram_mb': current_vram_mb,
        }
    else:
        gpu_results = None
    
    # =========================================
    # CPU BENCHMARK
    # =========================================
    if args.include_cpu or device.type == "cpu":
        print(f"\n[benchmark] Running CPU benchmark...")
        
        # Move model to CPU if needed
        if device.type != "cpu":
            model_cpu = model.cpu()
            dummy_input_cpu = dummy_input.cpu()
        else:
            model_cpu = model
            dummy_input_cpu = dummy_input
        
        # Warmup
        print(f"[benchmark] Warmup ({args.warmup} runs)...")
        with torch.no_grad():
            for _ in range(args.warmup):
                if args.task == "detection":
                    _ = model_cpu(dummy_input_cpu)
                else:
                    _ = model_cpu(dummy_input_cpu)
        
        # Timed runs
        print(f"[benchmark] Timing ({args.num_runs} runs)...")
        times_cpu = []
        
        with torch.no_grad():
            for i in range(args.num_runs):
                start = time.perf_counter()
                
                if args.task == "detection":
                    pred_boxes, pred_logits, pred_obj = model_cpu(dummy_input_cpu)
                else:
                    logits = model_cpu(dummy_input_cpu)
                
                end = time.perf_counter()
                times_cpu.append((end - start) * 1000)
        
        # Calculate stats
        times_cpu = np.array(times_cpu)
        avg_time_cpu_ms = np.mean(times_cpu)
        std_time_cpu_ms = np.std(times_cpu)
        min_time_cpu_ms = np.min(times_cpu)
        max_time_cpu_ms = np.max(times_cpu)
        fps_cpu = 1000 / avg_time_cpu_ms
        
        cpu_results = {
            'avg_time_ms': avg_time_cpu_ms,
            'std_time_ms': std_time_cpu_ms,
            'min_time_ms': min_time_cpu_ms,
            'max_time_ms': max_time_cpu_ms,
            'fps': fps_cpu,
        }
    else:
        cpu_results = None
    
    # =========================================
    # PRINT RESULTS
    # =========================================
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model Information'}")
    print("-" * 40)
    print(f"  Task:                 {args.task}")
    print(f"  Parameters:           {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Model File Size:      {model_size_mb:.2f} MB")
    print(f"  Input Size:           224 x 224 x 3")
    
    if gpu_results:
        print(f"\n{'GPU Performance'} ({gpu_results['device']})")
        print("-" * 40)
        print(f"  Average Time:         {gpu_results['avg_time_ms']:.2f} ms/image")
        print(f"  Std Dev:              ±{gpu_results['std_time_ms']:.2f} ms")
        print(f"  Min/Max:              {gpu_results['min_time_ms']:.2f} / {gpu_results['max_time_ms']:.2f} ms")
        print(f"  FPS:                  {gpu_results['fps']:.1f} frames/second")
        print(f"  Peak VRAM Usage:      {gpu_results['peak_vram_mb']:.2f} MB")
        print(f"  Current VRAM Usage:   {gpu_results['current_vram_mb']:.2f} MB")
    
    if cpu_results:
        print(f"\n{'CPU Performance'}")
        print("-" * 40)
        print(f"  Average Time:         {cpu_results['avg_time_ms']:.2f} ms/image")
        print(f"  Std Dev:              ±{cpu_results['std_time_ms']:.2f} ms")
        print(f"  Min/Max:              {cpu_results['min_time_ms']:.2f} / {cpu_results['max_time_ms']:.2f} ms")
        print(f"  FPS:                  {cpu_results['fps']:.1f} frames/second")
    
    # =========================================
    # PAPER-READY SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY FOR YOUR PAPER")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  LDBST-MP {args.task.capitalize()} - Hardware Feasibility                        │
├─────────────────────────────────────────────────────────────────────┤
│  Model Specifications:                                              │
│    • Parameters:        {num_params:,} ({num_params/1e6:.2f}M)                       │
│    • Model Size:        {model_size_mb:.2f} MB                                      │
│                                                                     │""")
    
    if gpu_results:
        print(f"""│  GPU Performance ({gpu_results['device'][:30]}):              │
│    • Inference Time:    {gpu_results['avg_time_ms']:.2f} ± {gpu_results['std_time_ms']:.2f} ms                              │
│    • FPS:               {gpu_results['fps']:.1f} frames/second                          │
│    • VRAM Usage:        {gpu_results['peak_vram_mb']:.2f} MB                                     │
│                                                                     │""")
    
    if cpu_results:
        print(f"""│  CPU Performance:                                                   │
│    • Inference Time:    {cpu_results['avg_time_ms']:.2f} ± {cpu_results['std_time_ms']:.2f} ms                              │
│    • FPS:               {cpu_results['fps']:.1f} frames/second                           │
│                                                                     │""")
    
    print(f"""│  Consumer-Grade Hardware Feasibility:                              │""")
    
    if gpu_results:
        fps_check = "✅ PASS" if gpu_results['fps'] >= 30 else "❌ FAIL"
        vram_check = "✅ PASS" if gpu_results['peak_vram_mb'] < 2000 else "❌ FAIL"
        print(f"""│    • Real-time (>30 FPS):     {fps_check} ({gpu_results['fps']:.1f} FPS)                   │
│    • Low VRAM (<2GB):         {vram_check} ({gpu_results['peak_vram_mb']:.2f} MB)                  │""")
    
    print("""└─────────────────────────────────────────────────────────────────────┘""")
    
    # Save results to file
    results_file = Path(args.model).stem + "_benchmark.txt"
    with open(results_file, 'w') as f:
        f.write(f"LDBST-MP Benchmark Results\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n\n")
        
        if gpu_results:
            f.write(f"GPU: {gpu_results['device']}\n")
            f.write(f"  Inference Time: {gpu_results['avg_time_ms']:.2f} ± {gpu_results['std_time_ms']:.2f} ms\n")
            f.write(f"  FPS: {gpu_results['fps']:.1f}\n")
            f.write(f"  Peak VRAM: {gpu_results['peak_vram_mb']:.2f} MB\n\n")
        
        if cpu_results:
            f.write(f"CPU:\n")
            f.write(f"  Inference Time: {cpu_results['avg_time_ms']:.2f} ± {cpu_results['std_time_ms']:.2f} ms\n")
            f.write(f"  FPS: {cpu_results['fps']:.1f}\n")
    
    print(f"\n[saved] Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LDBST-MP model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--task", choices=["classification", "detection"], required=True)
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup runs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU benchmark only")
    parser.add_argument("--include_cpu", action="store_true", help="Include CPU benchmark alongside GPU")
    
    args = parser.parse_args()
    benchmark_model(args)


if __name__ == "__main__":
    main()