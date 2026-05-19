"""
run_sensitivity.py — C6 Hyperparameter Sensitivity Analysis
=============================================================
Tests how LDBST-MP performance changes when key hyperparameters vary
from their chosen values. Re-trains the model with identical setup
except for the hyperparameter being tested — this isolates the effect
of each individual choice.

Hyperparameters tested (one at a time, others fixed at paper defaults):

    1. Number of attention heads ∈ {2, 4, 8}     (paper uses 4)
    2. Window size                ∈ {4, 7, 14}    (paper uses 7)
    3. Box loss weight λ_box     ∈ {1.0, 5.0, 10.0}  (paper uses 5.0)

Total runs: 9 (3 per hyperparameter). Default values are run once
each — so variant "heads=4" is the same as paper's main result.

Usage:
    python src/run_sensitivity.py \
        --train_dir ./data_yolo_train_ready \
        --val_dir   ./data_yolo \
        --num_classes 2 \
        --epochs 150 \
        --batch_size 8 \
        --lr 1e-4

    # Skip training, just benchmark existing checkpoints:
    python src/run_sensitivity.py --skip_train

    # Run only one hyperparameter group:
    python src/run_sensitivity.py --groups heads
    python src/run_sensitivity.py --groups window_size
    python src/run_sensitivity.py --groups loss_weights
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model2 import LDBST_MP, DetectionLoss
from dataset import get_yolo_dataloaders

from train2 import (
    WarmupCosineScheduler,
    compute_map,
    evaluate_detection_batch,
    apply_score_thresh_and_nms,
    atomic_save,
    count_parameters,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------
# Experiment grid — one hyperparameter varied at a time
# -----------------------------------------------------------------------
EXPERIMENTS = {
    # Format: run_name → {param_name: value, ...}
    # Paper defaults: heads=4, window=7, lambda_box=5.0

    # Group: attention heads
    "heads_2":       {"group": "heads",         "num_heads": 2,  "window_size": 7, "lambda_box": 5.0},
    "heads_4":       {"group": "heads",         "num_heads": 4,  "window_size": 7, "lambda_box": 5.0},   # default
    "heads_8":       {"group": "heads",         "num_heads": 8,  "window_size": 7, "lambda_box": 5.0},

    # Group: window size
    # Note: window_size must divide both 56 (Stage 1 feature map) and 14 (Stage 2).
    # Valid values: 1, 2, 7, 14. Using 2 instead of 4 because 14/4 is not an integer.
    "window_2":      {"group": "window_size",   "num_heads": 4,  "window_size": 2,  "lambda_box": 5.0},
    "window_7":      {"group": "window_size",   "num_heads": 4,  "window_size": 7,  "lambda_box": 5.0},  # default
    "window_14":     {"group": "window_size",   "num_heads": 4,  "window_size": 14, "lambda_box": 5.0},

    # Group: box loss weight (lambda_box)
    "lambdabox_1":   {"group": "loss_weights",  "num_heads": 4,  "window_size": 7, "lambda_box": 1.0},
    "lambdabox_5":   {"group": "loss_weights",  "num_heads": 4,  "window_size": 7, "lambda_box": 5.0},   # default
    "lambdabox_10":  {"group": "loss_weights",  "num_heads": 4,  "window_size": 7, "lambda_box": 10.0},
}


# -----------------------------------------------------------------------
# Note on num_heads divisibility
# -----------------------------------------------------------------------
# LDBST-MP branch dims: 48 (stage 1) and 96 (stage 2). num_heads must
# divide both. 2, 4, 8 all satisfy this (48/8=6, 96/8=12). If you change
# branch dims, update these values accordingly.


# -----------------------------------------------------------------------
# Benchmark (FPS + params)
# -----------------------------------------------------------------------
def benchmark_model(model, img_size=224, n_warmup=50, n_runs=300):
    model = model.to(DEVICE).eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=DEVICE)

    # Clear cached memory and reset tracking BEFORE warmup — matches
    # LDBST-MP benchmark.py so all VRAM numbers come from identical methodology.
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
        # Reset peak stats AFTER warmup so only inference-time VRAM is counted
        torch.cuda.reset_peak_memory_stats(DEVICE)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_vram = torch.cuda.max_memory_allocated(DEVICE) / 1e6 if DEVICE.type == "cuda" else 0.0

    return {
        "params_M":  round(params / 1e6, 2),
        "fps":       round(1000.0 / float(np.mean(times)), 1),
        "ms_mean":   round(float(np.mean(times)), 2),
        "peak_vram": round(peak_vram, 1),
    }


# -----------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------
def _evaluate(model, loader, args, use_amp):
    model.eval()
    pred_boxes_all, pred_scores_all, true_boxes_all = [], [], []
    total_precision = total_recall = 0.0
    n_images = 0

    with torch.no_grad():
        with autocast(device_type="cuda", enabled=use_amp):
            for imgs, targets in loader:
                if DEVICE.type == "cuda":
                    imgs = imgs.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
                    for t in targets:
                        t["boxes"]  = t["boxes"].to(DEVICE, non_blocking=True)
                        t["labels"] = t["labels"].to(DEVICE, non_blocking=True)
                else:
                    imgs = imgs.to(DEVICE)
                    for t in targets:
                        t["boxes"]  = t["boxes"].to(DEVICE)
                        t["labels"] = t["labels"].to(DEVICE)

                pred_boxes, pred_logits, pred_obj = model(imgs)

                class_probs = pred_logits.softmax(dim=-1)
                max_class_probs, _ = class_probs.max(dim=-1)
                pred_scores = (max_class_probs * pred_obj.sigmoid()).detach().cpu()

                for b in range(len(targets)):
                    tb = targets[b]["boxes"].detach().cpu()
                    pb = pred_boxes[b].detach().cpu()
                    ps = pred_scores[b].detach().cpu()

                    pb, ps = apply_score_thresh_and_nms(
                        pb, ps, score_thr=args.conf_thresh, nms_iou=args.nms_iou
                    )
                    pred_boxes_all.append(pb)
                    pred_scores_all.append(ps)
                    true_boxes_all.append(tb)

                    if ps.numel() == 0:
                        continue
                    P, R, _ = evaluate_detection_batch(pb, tb, iou_thresh=0.5)
                    total_precision += P
                    total_recall    += R
                    n_images        += 1

    precision = total_precision / max(1, n_images)
    recall    = total_recall    / max(1, n_images)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    map50     = compute_map(pred_boxes_all, pred_scores_all, true_boxes_all, iou_thresh=0.5)

    return {
        "map50":     round(map50, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# -----------------------------------------------------------------------
# Build LDBST-MP with custom hyperparameters
# -----------------------------------------------------------------------
def build_model_with_hparams(cfg, img_size, num_classes):
    """
    Build the LDBST_MP model, then patch its inner Swin blocks to use
    custom window_size. num_heads is passed through at construction time.
    """
    model = LDBST_MP(
        img_size=img_size,
        num_classes=num_classes,
        num_heads=cfg["num_heads"],
        task="detection",
    )

    # The LDBST_MP __init__ hard-codes window_size=7 for both stages.
    # Rebuild the Swin blocks in each DualBranchStage with the new window_size.
    if cfg["window_size"] != 7:
        from model2 import SwinTransformerBlock

        for stage_name in ["stage1", "stage2"]:
            stage = getattr(model, stage_name)
            dim = stage.swin_block1.dim
            heads = stage.swin_block1.num_heads
            ws = cfg["window_size"]
            stage.swin_block1 = SwinTransformerBlock(dim, heads, ws, shift_size=0)
            stage.swin_block2 = SwinTransformerBlock(dim, heads, ws, shift_size=ws // 2)

    return model


# -----------------------------------------------------------------------
# Single-experiment training (mirrors train2.py exactly)
# -----------------------------------------------------------------------
def train_experiment(run_name, cfg, args):
    ckpt_path   = f"checkpoints/sensitivity_{run_name}_best.pth"
    latest_path = f"checkpoints/sensitivity_{run_name}_latest.pth"

    print(f"\n{'=' * 70}")
    print(f"Sensitivity run: {run_name}")
    print(f"  group={cfg['group']} | heads={cfg['num_heads']} | "
          f"window={cfg['window_size']} | lambda_box={cfg['lambda_box']}")
    print(f"{'=' * 70}")

    # Data
    tr, va = get_yolo_dataloaders(
        train_root=args.train_dir,
        val_root=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=min(4, os.cpu_count() or 4),
    )

    # Model with variant-specific hyperparameters
    model = build_model_with_hparams(cfg, args.img_size, args.num_classes).to(DEVICE)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    print(f"[model] Parameters: {count_parameters(model):.2f}M")

    # Loss with variant-specific lambda_box
    criterion = DetectionLoss(
        lambda_box=cfg["lambda_box"],
        lambda_cls=1.0,
        lambda_obj=0.25,
        pos_iou_thresh=0.4,
        neg_iou_thresh=0.2,
    )

    # Optimizer + scheduler — identical to train2.py
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = min(5, args.epochs // 10)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, args.epochs)

    use_amp = (DEVICE.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    # Resume
    best_map       = 0.0
    start_epoch    = 1
    early_stop_ctr = 0

    if Path(latest_path).exists() and not args.fresh_start:
        try:
            ckpt = torch.load(latest_path, map_location=DEVICE)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            optimizer.load_state_dict(ckpt["opt_state"])
            best_map       = ckpt.get("best_map", 0.0)
            early_stop_ctr = ckpt.get("early_stop_ctr", 0)
            start_epoch    = ckpt.get("epoch", 0) + 1
            print(f"[resume] Resumed from epoch {start_epoch - 1}")
        except Exception as e:
            print(f"[resume] Could not load: {e}")

    # Training loop with early stopping (patience=15, matches train2.py)
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(tr, desc=f"[{run_name}] Epoch {epoch}/{args.epochs} LR={current_lr:.2e}", unit="batch")
        for imgs, targets in pbar:
            if DEVICE.type == "cuda":
                imgs = imgs.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
                for t in targets:
                    t["boxes"]  = t["boxes"].to(DEVICE, non_blocking=True)
                    t["labels"] = t["labels"].to(DEVICE, non_blocking=True)
            else:
                imgs = imgs.to(DEVICE)
                for t in targets:
                    t["boxes"]  = t["boxes"].to(DEVICE)
                    t["labels"] = t["labels"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                pred_boxes, pred_logits, pred_obj = model(imgs)
                loss = criterion(pred_boxes, pred_logits, pred_obj, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / max(len(tr.dataset), 1)
        scheduler.step()

        # Validation
        val_metrics = _evaluate(model, va, args, use_amp)
        val_map = val_metrics["map50"]

        print(f"[epoch {epoch:02d}] loss={avg_loss:.4f} | "
              f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
              f"F1={val_metrics['f1']:.4f} mAP@0.5={val_map:.4f}")

        # Early stopping + save best
        if val_map > best_map:
            best_map       = val_map
            early_stop_ctr = 0
            atomic_save(model.state_dict(), ckpt_path)
            print(f"  ✅ New best (mAP@0.5={val_map:.4f})")
        else:
            early_stop_ctr += 1
            print(f"  ⏳ No improvement for {early_stop_ctr}/{args.patience} epochs")
            if early_stop_ctr >= args.patience:
                print(f"  🛑 Early stopping triggered at epoch {epoch}")
                stopped_early = True

        atomic_save({
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "opt_state":      optimizer.state_dict(),
            "best_map":       best_map,
            "early_stop_ctr": early_stop_ctr,
        }, latest_path)

        if stopped_early:
            break

    print(f"[done] {run_name}: best mAP@0.5={best_map:.4f}")

    # Final eval + benchmark on best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    final_metrics = _evaluate(model, va, args, use_amp)
    bench         = benchmark_model(model, img_size=args.img_size)

    return {**cfg, **bench, **final_metrics, "stopped_early": stopped_early}


# -----------------------------------------------------------------------
# Print + save tables (grouped by hyperparameter)
# -----------------------------------------------------------------------
def print_tables(results):
    groups = {}
    for name, r in results.items():
        groups.setdefault(r["group"], []).append((name, r))

    for group, items in groups.items():
        print(f"\n{'=' * 70}")
        print(f"SENSITIVITY — {group.upper().replace('_', ' ')}")
        print("=" * 70)
        header = f"{'Run':<16} {'H':>3} {'W':>3} {'λbox':>5} | {'Params':>7} {'mAP@0.5':>9} {'F1':>7} {'FPS':>7}"
        print(header)
        print("-" * len(header))
        for name, r in items:
            map50 = f"{r['map50']*100:.2f}%" if r.get("map50") is not None else "N/A"
            f1    = f"{r['f1']:.4f}"          if r.get("f1")    is not None else "N/A"
            print(f"{name:<16} {r['num_heads']:>3} {r['window_size']:>3} "
                  f"{r['lambda_box']:>5.1f} | {str(r['params_M'])+'M':>7} "
                  f"{map50:>9} {f1:>7} {r['fps']:>7}")
    print("=" * 70)


def save_results(results):
    out = Path("sensitivity_results")
    out.mkdir(exist_ok=True)

    with open(out / "sensitivity_table.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(out / "sensitivity_table.txt", "w") as f:
        f.write("LDBST-MP Hyperparameter Sensitivity Analysis — C6 Revision\n")
        f.write("Identical training setup to main model (patience=15, mAP-based early stopping)\n\n")
        for name, r in results.items():
            f.write(f"{name}: {r}\n")

    print(f"\n[save] sensitivity_results/sensitivity_table.json")
    print(f"[save] sensitivity_results/sensitivity_table.txt")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LDBST-MP sensitivity analysis — C6 revision")

    ap.add_argument("--train_dir",    default="./data_yolo_train_ready")
    ap.add_argument("--val_dir",      default="./data_yolo")
    ap.add_argument("--num_classes",  type=int,   default=2)
    ap.add_argument("--epochs",       type=int,   default=150)
    ap.add_argument("--batch_size",   type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--img_size",     type=int,   default=224)
    ap.add_argument("--conf_thresh",  type=float, default=0.01)
    ap.add_argument("--nms_iou",      type=float, default=0.45)
    ap.add_argument("--patience",     type=int,   default=15)
    ap.add_argument("--fresh_start",  action="store_true")
    ap.add_argument("--skip_train",   action="store_true")
    ap.add_argument("--groups",       nargs="+",
                    default=["heads", "window_size", "loss_weights"],
                    choices=["heads", "window_size", "loss_weights"],
                    help="Hyperparameter groups to run")
    ap.add_argument("--runs",         nargs="+",
                    default=None,
                    help="Specific run names to execute, e.g. --runs heads_4 window_7")

    args = ap.parse_args()
    Path("checkpoints").mkdir(exist_ok=True)

    # Filter experiments
    if args.runs:
        selected = {k: v for k, v in EXPERIMENTS.items() if k in args.runs}
    else:
        selected = {k: v for k, v in EXPERIMENTS.items() if v["group"] in args.groups}

    print(f"[plan] Running {len(selected)} experiments: {list(selected.keys())}")

    results = {}
    for run_name, cfg in selected.items():
        if args.skip_train:
            ckpt_path = f"checkpoints/sensitivity_{run_name}_best.pth"
            model = build_model_with_hparams(cfg, args.img_size, args.num_classes)
            if Path(ckpt_path).exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                print(f"[{run_name}] Loaded {ckpt_path}")
            else:
                print(f"[{run_name}] No checkpoint — untrained weights")
            bench = benchmark_model(model, img_size=args.img_size)
            results[run_name] = {**cfg, **bench, "map50": None, "f1": None,
                                 "precision": None, "recall": None}
        else:
            results[run_name] = train_experiment(run_name, cfg, args)

    print_tables(results)
    save_results(results)


if __name__ == "__main__":
    main()