"""
baseline_models.py - Train and evaluate baseline models for comparison

Usage:
    # Classification baselines
    python baseline_models.py --model resnet18 --task classification --train_dir ./data_train_ready --val_dir ./data_val_ready --test_dir ./data_test_ready --epochs 50
    python baseline_models.py --model mobilenetv2 --task classification --train_dir ./data_train_ready --val_dir ./data_val_ready --test_dir ./data_test_ready --epochs 50
    python baseline_models.py --model efficientnet_b0 --task classification --train_dir ./data_train_ready --val_dir ./data_val_ready --test_dir ./data_test_ready --epochs 50
    
    # Evaluate only (if already trained)
    python baseline_models.py --model resnet18 --task classification --test_dir ./data_test_ready --eval_only --checkpoint resnet18_classification.pth
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import os

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_model(model_name, num_classes, pretrained=True):
    """Get a baseline model with modified classifier head."""
    
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# DATA LOADING
# ============================================================

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=4):
    """Create dataloaders with same preprocessing as LDBST-MP."""
    
    # Same normalization as LDBST-MP: [0.5, 0.5, 0.5]
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.classes


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, val_loader, device, args):
    """Train the model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    print(f"\n[training] Starting {args.model} training...")
    print(f"[training] Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"[epoch {epoch+1:02d}] train_loss={train_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✅ New best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set with detailed metrics."""
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Timing for FPS
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            
            # Time inference
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            outputs = model(images)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            inference_times.append((end - start) / images.size(0))
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class metrics
    n_classes = len(class_names)
    per_class = {}
    
    for cls_idx, cls_name in enumerate(class_names):
        tp = np.sum((all_labels == cls_idx) & (all_preds == cls_idx))
        fp = np.sum((all_labels != cls_idx) & (all_preds == cls_idx))
        fn = np.sum((all_labels == cls_idx) & (all_preds != cls_idx))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[cls_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(np.sum(all_labels == cls_idx))
        }
    
    # Macro F1
    macro_f1 = np.mean([m['f1'] for m in per_class.values()])
    
    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    
    # Timing
    avg_time_ms = np.mean(inference_times) * 1000
    fps = 1000 / avg_time_ms
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'per_class': per_class,
        'confusion_matrix': cm,
        'avg_time_ms': avg_time_ms,
        'fps': fps,
    }


def benchmark_model(model, device, num_runs=100):
    """Benchmark model for FPS and VRAM."""
    
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    
    # Reset VRAM tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = model(dummy_input)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
    
    avg_time_ms = np.mean(times)
    fps = 1000 / avg_time_ms
    
    # VRAM
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_vram_mb = 0
    
    return {
        'avg_time_ms': avg_time_ms,
        'fps': fps,
        'peak_vram_mb': peak_vram_mb,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate baseline models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet18", "resnet50", "mobilenetv2", "efficientnet_b0", "vgg16"])
    parser.add_argument("--task", type=str, default="classification", choices=["classification"])
    parser.add_argument("--train_dir", type=str, default="./data_train_ready")
    parser.add_argument("--val_dir", type=str, default="./data_val_ready")
    parser.add_argument("--test_dir", type=str, default="./data_test_ready")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.no_pretrained:
        args.pretrained = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[device] Using: {device}")
    
    # Get model
    print(f"\n[model] Loading {args.model}...")
    model = get_model(args.model, args.num_classes, pretrained=args.pretrained)
    num_params = count_parameters(model)
    print(f"[model] Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    model = model.to(device)
    
    # Get class names
    if os.path.exists(args.test_dir):
        class_names = sorted([d.name for d in Path(args.test_dir).iterdir() if d.is_dir()])
    else:
        class_names = [f"class_{i}" for i in range(args.num_classes)]
    print(f"[data] Classes: {class_names}")
    
    # Load checkpoint if eval only
    if args.eval_only:
        if args.checkpoint is None:
            args.checkpoint = f"{args.model}_classification.pth"
        print(f"[model] Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        # Training
        train_loader, val_loader, test_loader, _ = get_dataloaders(
            args.train_dir, args.val_dir, args.test_dir,
            batch_size=args.batch_size, num_workers=4
        )
        
        print(f"[data] Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        model, best_val_acc = train_model(model, train_loader, val_loader, device, args)
        
        # Save model
        save_path = f"{args.model}_classification.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\n[saved] Model saved to {save_path}")
    
    # Evaluation
    print("\n" + "=" * 70)
    print(f"EVALUATING {args.model.upper()}")
    print("=" * 70)
    
    # Create test loader if not already created
    if args.eval_only:
        _, _, test_loader, _ = get_dataloaders(
            args.train_dir, args.val_dir, args.test_dir,
            batch_size=args.batch_size, num_workers=4
        )
    
    results = evaluate_model(model, test_loader, device, class_names)
    benchmark = benchmark_model(model, device, num_runs=100)
    
    # Print results
    print(f"\n{'Metric':<20} {'Value':>12}")
    print("-" * 34)
    print(f"{'Accuracy':<20} {results['accuracy']:>12.4f}")
    print(f"{'Macro F1':<20} {results['macro_f1']:>12.4f}")
    
    print(f"\n{'Per-Class Performance'}")
    print("-" * 50)
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    for cls_name, metrics in results['per_class'].items():
        print(f"{cls_name:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['support']:>8}")
    
    print(f"\n{'Confusion Matrix'}")
    print("-" * 34)
    print(f"{'':>15}", end="")
    for cls in class_names:
        print(f"{cls[:8]:>10}", end="")
    print()
    for i, cls in enumerate(class_names):
        print(f"{cls:<15}", end="")
        for j in range(len(class_names)):
            print(f"{results['confusion_matrix'][i,j]:>10}", end="")
        print()
    
    print(f"\n{'Hardware Performance'}")
    print("-" * 34)
    print(f"{'Inference Time':<20} {benchmark['avg_time_ms']:>10.2f} ms")
    print(f"{'FPS':<20} {benchmark['fps']:>10.1f}")
    if benchmark['peak_vram_mb'] > 0:
        print(f"{'Peak VRAM':<20} {benchmark['peak_vram_mb']:>10.2f} MB")
    
    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR COMPARISON TABLE")
    print("=" * 70)
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  Model: {args.model.upper():<59} │
├─────────────────────────────────────────────────────────────────────┤
│  Parameters:    {num_params:,} ({num_params/1e6:.2f}M){' '*(40-len(f'{num_params:,} ({num_params/1e6:.2f}M)'))}│
│  Accuracy:      {results['accuracy']*100:.2f}%{' '*49}│
│  Macro F1:      {results['macro_f1']:.4f}{' '*50}│
│  FPS:           {benchmark['fps']:.1f}{' '*(53-len(f'{benchmark['fps']:.1f}'))}│
│  VRAM:          {benchmark['peak_vram_mb']:.2f} MB{' '*(49-len(f'{benchmark['peak_vram_mb']:.2f} MB'))}│
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Save results to file
    results_file = f"{args.model}_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"FPS: {benchmark['fps']:.1f}\n")
        f.write(f"VRAM: {benchmark['peak_vram_mb']:.2f} MB\n")
        f.write(f"\nPer-class:\n")
        for cls_name, metrics in results['per_class'].items():
            f.write(f"  {cls_name}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}\n")
    
    print(f"[saved] Results saved to {results_file}")


if __name__ == "__main__":
    main()