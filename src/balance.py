"""
balance.py - Balance datasets for classification OR detection

Handles class imbalance by duplicating underrepresented samples.

Usage:
    # Classification - balance a specific class
    python balance.py --task classification --input ./data --output ./data_balanced --target 4000 --class algae

    # Detection - balance based on object class frequency
    python balance.py --task detection --input ./data_yolo --output ./data_yolo_balanced --target 4000
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# CLASSIFICATION BALANCING
# ============================================================

def analyze_classification(input_dir: Path, class_names: list = None):
    """Analyze classification dataset distribution."""
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION DATASET ANALYSIS")
    print(f"{'='*60}")
    
    # Find class folders
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(f"ERROR: No class folders found in {input_dir}")
        return None
    
    # Count images per class
    class_counts = {}
    for class_dir in class_dirs:
        images = [p for p in class_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
        class_counts[class_dir.name] = len(images)
    
    total = sum(class_counts.values())
    
    print(f"\nTotal images: {total}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"\nClass Distribution:")
    print(f"{'Class':<20} {'Count':<10} {'%':<10}")
    print("-" * 40)
    
    for cls_name, count in sorted(class_counts.items()):
        pct = count / total * 100 if total > 0 else 0
        print(f"{cls_name:<20} {count:<10} {pct:<10.1f}")
    
    # Imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nImbalance ratio: {imbalance:.2f}x")
    
    return class_counts


def balance_classification(
    input_dir: Path,
    output_dir: Path,
    target_class: str,
    target_count: int,
    max_duplicates: int = 20,
    seed: int = 42
):
    """
    Balance classification dataset by duplicating images from a specific class.
    """
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print("BALANCING CLASSIFICATION DATASET")
    print(f"{'='*60}")
    
    # Find class folders
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    # Step 1: Copy all classes to output
    print("\n[Step 1] Copying all classes...")
    for class_dir in class_dirs:
        out_class_dir = output_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        
        images = [p for p in class_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
        for img in images:
            shutil.copy2(img, out_class_dir / img.name)
        
        print(f"   {class_dir.name}: {len(images)} images copied")
    
    # Step 2: Augment target class
    target_dir = output_dir / target_class
    if not target_dir.exists():
        print(f"ERROR: Target class '{target_class}' not found!")
        return
    
    source_dir = input_dir / target_class
    sources = [p for p in source_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    current = len(sources)
    
    if current >= target_count:
        print(f"\n[Step 2] {target_class} already has {current} >= {target_count}. Done!")
        return
    
    needed = target_count - current
    print(f"\n[Step 2] Duplicating {target_class}: {current} → {target_count} ({needed} needed)")
    
    duplicate_counts = defaultdict(int)
    duplicated = 0
    idx = 0
    
    while duplicated < needed:
        src = sources[idx % len(sources)]
        
        if duplicate_counts[src.stem] >= max_duplicates:
            idx += 1
            continue
        
        dup_idx = duplicate_counts[src.stem] + 1
        new_name = f"{src.stem}_dup{dup_idx:03d}{src.suffix}"
        
        shutil.copy2(src, target_dir / new_name)
        
        duplicate_counts[src.stem] += 1
        duplicated += 1
        idx += 1
        
        if duplicated % 100 == 0:
            print(f"   Duplicated {duplicated}/{needed}")
    
    print(f"   Duplicated {duplicated} images")
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ DONE!")
    analyze_classification(output_dir)


# ============================================================
# DETECTION BALANCING
# ============================================================

def count_detection_classes(labels_dir: Path) -> tuple[Counter, dict]:
    """Count objects per class and track which images contain each class."""
    
    class_counts = Counter()
    class_to_images = defaultdict(list)
    
    for lbl_path in labels_dir.glob("*.txt"):
        if lbl_path.stat().st_size == 0:
            continue
        
        with open(lbl_path) as f:
            classes_in_image = set()
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    class_counts[cls_id] += 1
                    classes_in_image.add(cls_id)
            
            for cls_id in classes_in_image:
                class_to_images[cls_id].append(lbl_path.stem)
    
    return class_counts, dict(class_to_images)


def analyze_detection(images_dir: Path, labels_dir: Path, class_names: list = None):
    """Analyze detection dataset distribution."""
    
    images = [p for p in images_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    labels = list(labels_dir.glob("*.txt"))
    
    print(f"\n{'='*60}")
    print("DETECTION DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    
    class_counts, class_to_images = count_detection_classes(labels_dir)
    
    if not class_counts:
        print("WARNING: No labels found!")
        return None, None
    
    total_objects = sum(class_counts.values())
    
    print(f"\nTotal objects: {total_objects}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"\nClass Distribution:")
    print(f"{'Class':<15} {'Objects':<10} {'%':<10} {'Images':<10}")
    print("-" * 45)
    
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        pct = count / total_objects * 100
        n_images = len(class_to_images[cls_id])
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"Class {cls_id}"
        print(f"{name:<15} {count:<10} {pct:<10.1f} {n_images:<10}")
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nImbalance ratio: {imbalance:.2f}x")
    
    return class_counts, class_to_images


def balance_detection(
    input_images: Path,
    input_labels: Path,
    output_images: Path,
    output_labels: Path,
    target_images: int,
    max_duplicates: int = 10,
    seed: int = 42
):
    """Balance detection dataset by duplicating images with minority classes."""
    
    random.seed(seed)
    
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    class_counts, class_to_images = count_detection_classes(input_labels)
    
    if not class_counts:
        print("ERROR: No labels found!")
        return
    
    all_images = [p for p in input_images.glob("*") if p.suffix.lower() in IMG_EXTS]
    
    print(f"\n{'='*60}")
    print("BALANCING DETECTION DATASET")
    print(f"{'='*60}")
    print(f"Original images: {len(all_images)}")
    print(f"Target images: {target_images}")
    
    # Step 1: Copy all originals
    print("\n[Step 1] Copying original images...")
    copied = 0
    for img_path in all_images:
        shutil.copy2(img_path, output_images / img_path.name)
        
        lbl_path = input_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, output_labels / lbl_path.name)
        
        copied += 1
    
    print(f"   Copied {copied} images")
    
    if copied >= target_images:
        print(f"\n✅ Already have {copied} >= {target_images}. Done!")
        return
    
    # Step 2: Calculate weights (prioritize minority classes)
    needed = target_images - copied
    print(f"\n[Step 2] Need {needed} more images")
    
    total_objects = sum(class_counts.values())
    class_weights = {cls: total_objects / count for cls, count in class_counts.items()}
    
    # Weight each image by its minority class content
    image_weights = {}
    for img_path in all_images:
        lbl_path = input_labels / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        
        weight = 0
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    weight += class_weights.get(cls_id, 1.0)
        
        image_weights[img_path] = weight
    
    total_weight = sum(image_weights.values())
    image_probs = {img: w / total_weight for img, w in image_weights.items()}
    
    # Step 3: Duplicate images
    print(f"\n[Step 3] Duplicating (prioritizing minority classes)...")
    
    images_list = list(image_probs.keys())
    probs_list = [image_probs[img] for img in images_list]
    
    duplicate_counts = defaultdict(int)
    duplicated = 0
    attempts = 0
    max_attempts = needed * 3
    
    while duplicated < needed and attempts < max_attempts:
        attempts += 1
        
        img_path = random.choices(images_list, weights=probs_list, k=1)[0]
        
        if duplicate_counts[img_path.stem] >= max_duplicates:
            continue
        
        dup_idx = duplicate_counts[img_path.stem] + 1
        new_name = f"{img_path.stem}_dup{dup_idx:03d}{img_path.suffix}"
        
        shutil.copy2(img_path, output_images / new_name)
        
        lbl_path = input_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            new_lbl_name = f"{img_path.stem}_dup{dup_idx:03d}.txt"
            shutil.copy2(lbl_path, output_labels / new_lbl_name)
        
        duplicate_counts[img_path.stem] += 1
        duplicated += 1
        
        if duplicated % 500 == 0:
            print(f"   Duplicated {duplicated}/{needed}")
    
    print(f"   Duplicated {duplicated} images")
    
    final_count = len(list(output_images.glob("*")))
    print(f"\n{'='*60}")
    print(f"✅ DONE! Final: {final_count} images")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Balance classification or detection datasets")
    
    parser.add_argument("--task", required=True, choices=["classification", "detection"],
                        help="Task type")
    parser.add_argument("--input", required=True, help="Input data folder")
    parser.add_argument("--output", help="Output data folder (required unless --analyze-only)")
    parser.add_argument("--target", type=int, help="Target number of images")
    parser.add_argument("--max-dup", type=int, default=10, help="Max duplicates per source")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't balance")
    
    # Classification-specific
    parser.add_argument("--class", dest="cls", help="Class to augment (classification only)")
    
    # Detection-specific
    parser.add_argument("--split", default="train", help="Split to balance (detection only)")
    parser.add_argument("--class-names", nargs="+", help="Class names for display")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.task == "classification":
        # Classification
        if args.analyze_only:
            analyze_classification(input_path)
        else:
            if not args.output or not args.target or not args.cls:
                parser.error("--output, --target, and --class required for classification balancing")
            balance_classification(
                input_dir=input_path,
                output_dir=Path(args.output),
                target_class=args.cls,
                target_count=args.target,
                max_duplicates=args.max_dup,
                seed=args.seed
            )
    
    else:
        # Detection
        # Detect structure
        if (input_path / "images" / args.split).exists():
            images_dir = input_path / "images" / args.split
            labels_dir = input_path / "labels" / args.split
        elif (input_path / "images").exists():
            images_dir = input_path / "images"
            labels_dir = input_path / "labels"
        else:
            print(f"ERROR: Cannot find images in {input_path}")
            return
        
        if args.analyze_only:
            analyze_detection(images_dir, labels_dir, args.class_names)
        else:
            if not args.output or not args.target:
                parser.error("--output and --target required for detection balancing")
            
            output_path = Path(args.output)
            balance_detection(
                input_images=images_dir,
                input_labels=labels_dir,
                output_images=output_path / "images",
                output_labels=output_path / "labels",
                target_images=args.target,
                max_duplicates=args.max_dup,
                seed=args.seed
            )


if __name__ == "__main__":
    main()