"""
balance_detection.py - Balance detection dataset by class frequency

This script analyzes your YOLO labels and duplicates images containing
underrepresented classes to achieve better class balance.

Usage:
    python balance_detection.py --input ./data_yolo --output ./data_yolo_balanced --target 4000
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_class_distribution(labels_dir: Path) -> tuple[Counter, dict]:
    """
    Count objects per class and track which images contain each class.
    
    Returns:
        class_counts: Counter of total objects per class
        class_to_images: dict mapping class_id -> list of image stems
    """
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
            
            # Track which images contain each class
            for cls_id in classes_in_image:
                class_to_images[cls_id].append(lbl_path.stem)
    
    return class_counts, dict(class_to_images)


def analyze_dataset(images_dir: Path, labels_dir: Path, class_names: list = None):
    """Analyze and print dataset statistics."""
    
    # Count images
    images = [p for p in images_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    labels = list(labels_dir.glob("*.txt"))
    
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS")
    print(f"{'='*60}")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    
    # Count class distribution
    class_counts, class_to_images = count_class_distribution(labels_dir)
    
    if not class_counts:
        print("WARNING: No labels found!")
        return None, None
    
    total_objects = sum(class_counts.values())
    num_classes = len(class_counts)
    
    print(f"\nTotal objects: {total_objects}")
    print(f"Number of classes: {num_classes}")
    print(f"\nClass Distribution:")
    print(f"{'Class':<10} {'Count':<10} {'%':<10} {'Images':<10}")
    print("-" * 40)
    
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        pct = count / total_objects * 100
        n_images = len(class_to_images[cls_id])
        name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"Class {cls_id}"
        print(f"{name:<10} {count:<10} {pct:<10.1f} {n_images:<10}")
    
    # Imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nImbalance ratio: {imbalance:.2f}x")
    
    return class_counts, class_to_images


def balance_by_duplication(
    input_images: Path,
    input_labels: Path,
    output_images: Path,
    output_labels: Path,
    target_images: int,
    max_duplicates_per_image: int = 10,
    seed: int = 42
):
    """
    Balance dataset by duplicating images containing minority classes.
    
    Strategy:
    1. Copy all original images
    2. Identify which classes are underrepresented
    3. Duplicate images containing those classes until target is reached
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output dirs
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Get class distribution
    class_counts, class_to_images = count_class_distribution(input_labels)
    
    if not class_counts:
        print("ERROR: No labels found!")
        return
    
    # Get all images
    all_images = [p for p in input_images.glob("*") if p.suffix.lower() in IMG_EXTS]
    
    print(f"\n{'='*60}")
    print("BALANCING DATASET")
    print(f"{'='*60}")
    print(f"Original images: {len(all_images)}")
    print(f"Target images: {target_images}")
    
    # Step 1: Copy all originals
    print("\n[Step 1] Copying original images...")
    copied = 0
    for img_path in all_images:
        # Copy image
        shutil.copy2(img_path, output_images / img_path.name)
        
        # Copy label if exists
        lbl_path = input_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, output_labels / lbl_path.name)
        
        copied += 1
    
    print(f"   Copied {copied} images")
    
    if copied >= target_images:
        print(f"\n✅ Already have {copied} >= {target_images} target. Done!")
        return
    
    # Step 2: Calculate how many more images needed
    needed = target_images - copied
    print(f"\n[Step 2] Need {needed} more images")
    
    # Step 3: Prioritize images with minority classes
    # Weight each image by inverse frequency of its classes
    total_objects = sum(class_counts.values())
    class_weights = {cls: total_objects / count for cls, count in class_counts.items()}
    
    # Calculate weight for each image
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
    
    # Normalize weights to probabilities
    total_weight = sum(image_weights.values())
    image_probs = {img: w / total_weight for img, w in image_weights.items()}
    
    # Step 4: Sample images weighted by minority class content
    print(f"\n[Step 3] Duplicating images (prioritizing minority classes)...")
    
    images_list = list(image_probs.keys())
    probs_list = [image_probs[img] for img in images_list]
    
    duplicate_counts = defaultdict(int)
    duplicated = 0
    attempts = 0
    max_attempts = needed * 3
    
    while duplicated < needed and attempts < max_attempts:
        attempts += 1
        
        # Sample image weighted by minority class content
        img_path = random.choices(images_list, weights=probs_list, k=1)[0]
        
        # Check if we've duplicated this image too many times
        if duplicate_counts[img_path.stem] >= max_duplicates_per_image:
            continue
        
        # Create duplicate
        dup_idx = duplicate_counts[img_path.stem] + 1
        new_name = f"{img_path.stem}_dup{dup_idx:03d}{img_path.suffix}"
        
        # Copy image
        shutil.copy2(img_path, output_images / new_name)
        
        # Copy label
        lbl_path = input_labels / (img_path.stem + ".txt")
        if lbl_path.exists():
            new_lbl_name = f"{img_path.stem}_dup{dup_idx:03d}.txt"
            shutil.copy2(lbl_path, output_labels / new_lbl_name)
        
        duplicate_counts[img_path.stem] += 1
        duplicated += 1
        
        if duplicated % 500 == 0:
            print(f"   Duplicated {duplicated}/{needed}")
    
    print(f"   Duplicated {duplicated} images")
    
    # Final count
    final_count = len(list(output_images.glob("*")))
    print(f"\n{'='*60}")
    print(f"✅ DONE! Final dataset: {final_count} images")
    print(f"{'='*60}")
    
    # Show new distribution
    print("\nNew class distribution:")
    analyze_dataset(output_images, output_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Balance YOLO detection dataset by duplicating minority class images"
    )
    parser.add_argument("--input", required=True, help="Input data root (with images/ and labels/)")
    parser.add_argument("--output", required=True, help="Output data root")
    parser.add_argument("--target", type=int, required=True, help="Target number of images")
    parser.add_argument("--max-dup", type=int, default=10, help="Max duplicates per source image")
    parser.add_argument("--split", default="train", help="Split to balance (train/val/test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, don't balance")
    parser.add_argument("--class-names", nargs="+", help="Class names for display")
    
    args = parser.parse_args()
    
    input_root = Path(args.input)
    
    # Detect structure
    if (input_root / "images" / args.split).exists():
        # Ultralytics structure: data/images/train/
        input_images = input_root / "images" / args.split
        input_labels = input_root / "labels" / args.split
    elif (input_root / "images").exists():
        # Flat structure: data/images/
        input_images = input_root / "images"
        input_labels = input_root / "labels"
    else:
        print(f"ERROR: Cannot find images in {input_root}")
        print("Expected: {input_root}/images/ or {input_root}/images/{split}/")
        return
    
    print(f"Input images: {input_images}")
    print(f"Input labels: {input_labels}")
    
    # Analyze
    analyze_dataset(input_images, input_labels, args.class_names)
    
    if args.analyze_only:
        return
    
    # Balance
    output_root = Path(args.output)
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    
    balance_by_duplication(
        input_images=input_images,
        input_labels=input_labels,
        output_images=output_images,
        output_labels=output_labels,
        target_images=args.target,
        max_duplicates_per_image=args.max_dup,
        seed=args.seed
    )


if __name__ == "__main__":
    main()