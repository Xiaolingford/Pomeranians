# src/augment_balance.py
import argparse, random
from pathlib import Path
import cv2
import numpy as np
from augmentations_util import augment_image, augment_image_and_bboxes  # ✅ correct import

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --- Helpers ---
def imread(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def count_images(d: Path) -> int:
    return sum(1 for p in d.glob("*") if p.suffix.lower() in IMG_EXTS)


# --- Classification Workflow ---
def run_classification(args):
    in_root = Path(args.input)
    out_root = Path(args.outdir)
    ensure_dir(out_root)

    # Mirror the dataset first (copy all originals)
    for cls_name in ["algae", "microplastics"]:
        src = in_root / cls_name
        dst = out_root / cls_name
        ensure_dir(dst)
        imgs = [p for p in src.glob("*") if p.suffix.lower() in IMG_EXTS]
        for p in imgs:
            img = cv2.imread(str(p))
            if img is None:
                print(f"[WARN] skip unreadable: {p}")
                continue
            cv2.imwrite(str(dst / p.name), img)

    # Now augment ONLY the requested class
    cls_dir = out_root / args.cls
    cur = count_images(cls_dir)
    if cur >= args.target:
        print(f"[INFO] {args.cls} already has {cur} >= target {args.target}. Nothing to do.")
        return

    source_dir = in_root / args.cls
    sources = [p for p in source_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    if not sources:
        print(f"[ERROR] No images found in {source_dir}")
        return

    print(f"[START] Classification augment '{args.cls}' from {cur} → {args.target}")
    per_src_counts = {p: 0 for p in sources}
    idx = 0
    while cur < args.target:
        p = sources[idx % len(sources)]
        if per_src_counts[p] >= args.max_per_src:
            idx += 1
            continue
        try:
            img = imread(p)
        except Exception as e:
            print(f"[WARN] {e}; skipping.")
            idx += 1
            continue

        aug_img = augment_image(img, use_clahe=args.use_clahe)

        stem, ext = p.stem, p.suffix
        out_name = f"{stem}_aug{per_src_counts[p]+1:03d}{ext}"
        cv2.imwrite(str(cls_dir / out_name), aug_img)

        per_src_counts[p] += 1
        cur += 1
        idx += 1
        if cur % 50 == 0:
            print(f"  → {cur}/{args.target}")

    print(f"[DONE] {args.cls}: {cur} images in {cls_dir}")


# --- Detection Workflow ---
def run_detection(args):
 # Point directly to the train subfolders
    images_root = Path(args.images) / "images" / "train"
    labels_root = Path(args.labels) / "labels" / "train"
    out_images_root = Path(args.outdir) / "images" / "train"
    out_labels_root = Path(args.outdir) / "labels" / "train"

    ensure_dir(out_images_root)
    ensure_dir(out_labels_root)

    imgs = [p for p in images_root.glob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print(f"[ERROR] No images found in {images_root}")
        return

    # Mirror originals first
    for p in imgs:
        dst_img = out_images_root / p.name
        ensure_dir(out_images_root)
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] skip unreadable: {p}")
            continue
        cv2.imwrite(str(dst_img), img)

        # Copy label
        lbl_in = labels_root / p.with_suffix(".txt").name
        if lbl_in.exists():
            lbl_out = out_labels_root / lbl_in.name
            lbl_out.write_text(lbl_in.read_text())

    cur = len(imgs)
    if cur >= args.target:
        print(f"[INFO] Already has {cur} >= target {args.target}. Nothing to do.")
        return

    print(f"[START] Detection augment from {cur} → {args.target}")
    per_src_counts = {p: 0 for p in imgs}
    idx = 0
    while cur < args.target:
        p = imgs[idx % len(imgs)]
        if per_src_counts[p] >= args.max_per_src:
            idx += 1
            continue

        try:
            img = imread(p)
        except Exception as e:
            print(f"[WARN] {e}; skipping.")
            idx += 1
            continue

        label_file = labels_root / p.with_suffix(".txt").name
        if not label_file.exists():
            print(f"[WARN] missing label for {p}, skipping")
            idx += 1
            continue

        # parse YOLO labels into floats
        with open(label_file, "r") as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, bw, bh = parts
                bboxes.append((int(cls), float(xc), float(yc), float(bw), float(bh)))

        # augment image + bboxes
        aug_img, aug_bboxes = augment_image_and_bboxes(img, bboxes, use_clahe=args.use_clahe)
        if aug_img is None or not aug_bboxes:
            idx += 1
            continue

        # convert back to YOLO lines
        aug_lines = [f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                     for cls, xc, yc, bw, bh in aug_bboxes]

        stem, ext = p.stem, p.suffix
        out_name = f"{stem}_aug{per_src_counts[p]+1:03d}{ext}"
        cv2.imwrite(str(out_images_root / out_name), aug_img)
        with open(out_labels_root / f"{stem}_aug{per_src_counts[p]+1:03d}.txt", "w") as f:
            f.writelines(aug_lines)

        per_src_counts[p] += 1
        cur += 1
        idx += 1
        if cur % 50 == 0:
            print(f"  → {cur}/{args.target}")

    print(f"[DONE] Detection: {cur} images in {out_images_root}")


# --- Main Entrypoint ---
def main():
    ap = argparse.ArgumentParser(description="Unified tool to balance datasets with augmentation.")
    ap.add_argument("--task", required=True, choices=["classification", "detection"],
                    help="Which type of dataset to balance.")
    ap.add_argument("--input", help="Input dataset root (required for classification).")
    ap.add_argument("--outdir", required=True, help="Output root (augmented copy).")
    ap.add_argument("--class", dest="cls", help="For classification: class to augment (algae/microplastics).")
    ap.add_argument("--target", type=int, required=True, help="Desired total images.")
    ap.add_argument("--max_per_src", type=int, default=20, help="Max augmented copies per source image.")
    ap.add_argument("--use_clahe", action="store_true", help="Apply CLAHE randomly in augmentation")
    ap.add_argument("--images", help="Root folder of images (for detection).")
    ap.add_argument("--labels", help="Root folder of labels (for detection).")


    args = ap.parse_args()

    if args.task == "classification":
        if not args.cls or not args.input:
            ap.error("--class and --input are required for classification task")
        run_classification(args)
    elif args.task == "detection":
        if not args.images or not args.labels:
            ap.error("--images and --labels are required for detection task")
        run_detection(args)



if __name__ == "__main__":
    main()
