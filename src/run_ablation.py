"""
run_ablation.py — C2 Ablation Study Runner
============================================
Trains all four LDBST-MP ablation variants using the EXACT same training
setup as your main model (train2.py), only the model is swapped per variant.

Matches your training command:
    python src/train2.py --task detection
        --train_dir ./data_yolo_train_ready
        --val_dir   ./data_yolo
        --num_classes 2
        --epochs 150
        --batch_size 8
        --lr 1e-4
        --fresh_start

Usage:
    python src/run_ablation.py \
        --train_dir ./data_yolo_train_ready \
        --val_dir   ./data_yolo \
        --num_classes 2 \
        --epochs 150 \
        --batch_size 8 \
        --lr 1e-4 \
        --fresh_start \
        --variants swin_only cnn_only no_fusion

    # Benchmark only (if checkpoints already exist)
    python src/run_ablation.py --skip_train --variants swin_only cnn_only no_fusion
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

from ablation_model import build_ablation_model
from model2 import DetectionLoss
from dataset import get_yolo_dataloaders

# Reuse helpers already in train2.py — no need to rewrite them
from train2 import (
    WarmupCosineScheduler,
    compute_map,
    evaluate_detection_batch,
    apply_score_thresh_and_nms,
    atomic_save,
    count_parameters,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIANTS = ["swin_only", "cnn_only", "no_fusion", "full"]
VARIANT_LABELS = {
    "swin_only": "A: Swin only",
    "cnn_only":  "B: CNN only",
    "no_fusion": "C: No fusion",
    "full":      "D: Full LDBST-MP",
}


# -----------------------------------------------------------------------
# Benchmark (FPS + params)
# -----------------------------------------------------------------------
def benchmark_model(model, img_size=224, n_warmup=50, n_runs=300):
    model = model.to(DEVICE).eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=DEVICE)

    # Clear cached memory and reset tracking BEFORE warmup — matches
    # LDBST-MP benchmark.py and yolo_scratch_baseline.py so all VRAM
    # numbers across Tables 2 and 3 come from identical methodology.
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
        "ms_std":    round(float(np.std(times)), 2),
        "peak_vram": round(peak_vram, 1),
    }


# -----------------------------------------------------------------------
# Evaluation helper
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
# Single-variant training (mirrors train2.py detection loop exactly)
# -----------------------------------------------------------------------
def train_variant(variant, args):
    ckpt_path   = f"checkpoints/ablation_{variant}_best.pth"
    latest_path = f"checkpoints/ablation_{variant}_latest.pth"

    print(f"\n{'=' * 70}")
    print(f"Ablation variant: {VARIANT_LABELS[variant]}")
    print(f"[early stopping] patience={args.patience} epochs based on mAP@0.5")
    print(f"{'=' * 70}")

    # Data — identical dirs and loader to your main training command
    tr, va = get_yolo_dataloaders(
        train_root=args.train_dir,
        val_root=args.val_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=min(4, os.cpu_count() or 4),
    )

    # Model
    model = build_ablation_model(
        variant,
        img_size=args.img_size,
        num_classes=args.num_classes,
        num_heads=4,
        task="detection",
    ).to(DEVICE)

    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    print(f"[model] Parameters: {count_parameters(model):.2f}M")

    # Loss — identical to train2.py
    criterion = DetectionLoss(
        lambda_box=5.0, lambda_cls=1.0, lambda_obj=0.25,
        pos_iou_thresh=0.4, neg_iou_thresh=0.2,
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
    early_stop_ctr = 0          # epochs since last mAP improvement

    if Path(latest_path).exists() and not args.fresh_start:
        try:
            ckpt = torch.load(latest_path, map_location=DEVICE)
            model.load_state_dict(ckpt.get("model_state", ckpt))
            optimizer.load_state_dict(ckpt["opt_state"])
            best_map       = ckpt.get("best_map", 0.0)
            early_stop_ctr = ckpt.get("early_stop_ctr", 0)
            start_epoch    = ckpt.get("epoch", 0) + 1
            print(f"[resume] Resumed from epoch {start_epoch - 1} "
                  f"(early stop counter={early_stop_ctr}/{args.patience})")
        except Exception as e:
            print(f"[resume] Could not load checkpoint: {e}")

    # Training loop
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        current_lr = optimizer.param_groups[0]["lr"]

        pbar = tqdm(
            tr, desc=f"[{variant}] Epoch {epoch}/{args.epochs} LR={current_lr:.2e}", unit="batch"
        )
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

        # Early stopping logic — identical to train2.py (patience=15, based on mAP)
        if val_map > best_map:
            best_map       = val_map
            early_stop_ctr = 0
            atomic_save(model.state_dict(), ckpt_path)
            print(f"  ✅ New best (mAP@0.5={val_map:.4f}) → {ckpt_path}")
        else:
            early_stop_ctr += 1
            print(f"  ⏳ No improvement for {early_stop_ctr}/{args.patience} epochs "
                  f"(best={best_map:.4f})")
            if early_stop_ctr >= args.patience:
                print(f"  🛑 Early stopping triggered at epoch {epoch}.")
                stopped_early = True

        # Save latest (includes early stop counter for safe resume)
        atomic_save({
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "opt_state":      optimizer.state_dict(),
            "best_map":       best_map,
            "early_stop_ctr": early_stop_ctr,
        }, latest_path)

        if stopped_early:
            break

    print(f"[done] {variant}: best mAP@0.5={best_map:.4f} "
          f"({'early stop' if stopped_early else 'full training'})")

    # Final eval + benchmark on best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    final_metrics = _evaluate(model, va, args, use_amp)
    bench         = benchmark_model(model, img_size=args.img_size)

    return {**bench, **final_metrics, "stopped_early": stopped_early}


# -----------------------------------------------------------------------
# Print + save table
# -----------------------------------------------------------------------
def print_table(results):
    header = (f"{'Variant':<24} {'Params':>7} {'mAP@0.5':>9} "
              f"{'F1':>7} {'Prec':>7} {'Rec':>7} {'FPS':>7} {'VRAM':>8}")
    print("\n" + "=" * 70)
    print("ABLATION STUDY — TABLE 3 (copy into paper)")
    print("=" * 70)
    print(header)
    print("-" * len(header))
    for v, r in results.items():
        map50 = f"{r['map50']*100:.2f}%" if r.get("map50") is not None else "N/A"
        f1    = f"{r['f1']:.4f}"          if r.get("f1")    is not None else "N/A"
        prec  = f"{r['precision']:.4f}"   if r.get("precision") is not None else "N/A"
        rec   = f"{r['recall']:.4f}"      if r.get("recall")    is not None else "N/A"
        print(f"{VARIANT_LABELS[v]:<24} {str(r['params_M'])+'M':>7} {map50:>9} "
              f"{f1:>7} {prec:>7} {rec:>7} {r['fps']:>7} {str(r['peak_vram'])+'MB':>8}")
    print("=" * 70)


def save_results(results):
    out = Path("ablation_results")
    out.mkdir(exist_ok=True)
    json_path = out / "ablation_table.json"
    txt_path  = out / "ablation_table.txt"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w") as f:
        f.write("LDBST-MP Ablation Study — C2 Revision\n")
        f.write("Identical training setup to main model (patience=15, mAP-based early stopping)\n\n")
        for v, r in results.items():
            f.write(f"{VARIANT_LABELS[v]}: {r}\n")

    print(f"\n[save] {json_path}")
    print(f"[save] {txt_path}")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LDBST-MP ablation study — C2 revision")

    # All defaults mirror your main training command
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
    ap.add_argument("--patience",     type=int,   default=15,
                    help="Early stopping patience — matches train2.py default")
    ap.add_argument("--fresh_start",  action="store_true")
    ap.add_argument("--skip_train",   action="store_true",
                    help="Skip training, benchmark existing checkpoints only")
    ap.add_argument("--variants",     nargs="+", default=VARIANTS,
                    choices=VARIANTS,
                    help="Subset of variants to run, e.g. --variants swin_only cnn_only no_fusion")

    args = ap.parse_args()
    Path("checkpoints").mkdir(exist_ok=True)

    results = {}

    for variant in args.variants:
        if args.skip_train:
            ckpt_path = f"checkpoints/ablation_{variant}_best.pth"
            model = build_ablation_model(
                variant, img_size=args.img_size,
                num_classes=args.num_classes, num_heads=4, task="detection",
            )
            if Path(ckpt_path).exists():
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                print(f"[{variant}] Loaded {ckpt_path}")
            else:
                print(f"[{variant}] No checkpoint — benchmarking untrained weights")
            bench = benchmark_model(model, img_size=args.img_size)
            results[variant] = {**bench, "map50": None, "f1": None,
                                 "precision": None, "recall": None}
        else:
            results[variant] = train_variant(variant, args)

    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()