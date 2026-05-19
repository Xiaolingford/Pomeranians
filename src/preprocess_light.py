"""
preprocess_light.py - MINIMAL preprocessing for detection

Philosophy: Keep images as natural as possible. Let online augmentation
handle variation. Heavy preprocessing REDUCES dataset diversity.

What this does:
    - Resize to target size (required for consistent batching)
    - Optionally: very mild denoising (off by default)
    - Optionally: copy without any processing

What this does NOT do:
    - White balance (removes natural color variation)
    - CLAHE (can create artifacts)
    - Heavy denoising (removes fine details)
    - Contrast stretching (normalizes away useful variation)

Usage:
    # Just resize (recommended)
    python preprocess_light.py --input ./data_yolo_balanced --output ./data_yolo_ready --size 224

    # Copy without processing (if you want online resize)
    python preprocess_light.py --input ./data_yolo_balanced --output ./data_yolo_ready --no-resize
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def resize_image(img, size, keep_aspect=False):
    """Resize image to target size."""
    if keep_aspect:
        h, w = img.shape[:2]
        scale = size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad to square
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - new_h) // 2
        x_offset = (size - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas
    else:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def mild_denoise(img, strength=3):
    """Very mild denoising - preserves details."""
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def process_image(img_path: Path, output_path: Path, args):
    """Process a single image with minimal operations."""
    
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    
    # Resize (if enabled)
    if not args.no_resize:
        img = resize_image(img, args.size, keep_aspect=args.keep_aspect)
    
    # Mild denoise (if enabled, off by default)
    if args.denoise:
        img = mild_denoise(img, strength=args.denoise_strength)
    
    # Save
    cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def process_detection_dataset(args):
    """Process detection dataset (images + labels)."""
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    # Detect structure
    if (input_root / "images").exists():
        input_images = input_root / "images"
        input_labels = input_root / "labels"
    else:
        print(f"ERROR: Cannot find images/ folder in {input_root}")
        return
    
    output_images = output_root / "images"
    output_labels = output_root / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    images = [p for p in input_images.glob("*") if p.suffix.lower() in IMG_EXTS]
    
    print(f"\n{'='*60}")
    print("LIGHT PREPROCESSING")
    print(f"{'='*60}")
    print(f"Input: {input_images}")
    print(f"Output: {output_images}")
    print(f"Images found: {len(images)}")
    print(f"\nSettings:")
    print(f"  Resize: {'No' if args.no_resize else f'{args.size}x{args.size}'}")
    print(f"  Keep aspect: {args.keep_aspect}")
    print(f"  Denoise: {'Yes (strength={})'.format(args.denoise_strength) if args.denoise else 'No'}")
    print(f"{'='*60}\n")
    
    success = 0
    failed = 0
    
    for img_path in tqdm(images, desc="Processing"):
        out_img_path = output_images / img_path.name
        
        if process_image(img_path, out_img_path, args):
            success += 1
            
            # Copy label file (unchanged)
            lbl_path = input_labels / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, output_labels / lbl_path.name)
        else:
            failed += 1
            print(f"[WARN] Failed to process: {img_path.name}")
    
    print(f"\n{'='*60}")
    print(f"✅ DONE!")
    print(f"   Success: {success}")
    print(f"   Failed: {failed}")
    print(f"   Output: {output_root}")
    print(f"{'='*60}")


def process_classification_dataset(args):
    """Process classification dataset (preserves class folders)."""
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    # Find class folders
    class_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"ERROR: No class folders found in {input_root}")
        return
    
    print(f"\n{'='*60}")
    print("LIGHT PREPROCESSING (Classification)")
    print(f"{'='*60}")
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Classes: {[d.name for d in class_dirs]}")
    print(f"\nSettings:")
    print(f"  Resize: {'No' if args.no_resize else f'{args.size}x{args.size}'}")
    print(f"  Denoise: {'Yes' if args.denoise else 'No'}")
    print(f"{'='*60}\n")
    
    total_success = 0
    total_failed = 0
    
    for class_dir in class_dirs:
        out_class_dir = output_root / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)
        
        images = [p for p in class_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
        
        for img_path in tqdm(images, desc=f"Processing {class_dir.name}"):
            out_path = out_class_dir / img_path.name
            
            if process_image(img_path, out_path, args):
                total_success += 1
            else:
                total_failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ DONE!")
    print(f"   Success: {total_success}")
    print(f"   Failed: {total_failed}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Light preprocessing - resize only, preserve natural variation"
    )
    parser.add_argument("--input", required=True, help="Input data folder")
    parser.add_argument("--output", required=True, help="Output data folder")
    parser.add_argument("--task", choices=["detection", "classification"], default="detection")
    parser.add_argument("--size", type=int, default=224, help="Target image size")
    parser.add_argument("--no-resize", action="store_true", help="Skip resize (just copy)")
    parser.add_argument("--keep-aspect", action="store_true", help="Keep aspect ratio (pad to square)")
    parser.add_argument("--denoise", action="store_true", help="Apply mild denoising (off by default)")
    parser.add_argument("--denoise-strength", type=int, default=3, help="Denoise strength (1-10)")
    
    args = parser.parse_args()
    
    if args.task == "detection":
        process_detection_dataset(args)
    else:
        process_classification_dataset(args)


if __name__ == "__main__":
    main()