# src/visual_check.py
import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from augmentations_util import augment_image_and_bboxes

# Allowed image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def draw_yolo_boxes(img, bboxes, color=(0,0,255)):
    """
    Draw YOLO-format boxes on an image.
    bboxes = [(cls, xc, yc, w, h), ...]
    """
    h, w = img.shape[:2]
    vis = img.copy()
    for cls, xc, yc, bw, bh in bboxes:
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
    return vis

def load_yolo_labels(label_file):
    """
    Read YOLO-format label file and return list of boxes.
    """
    bboxes = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, xc, yc, bw, bh = parts
                bboxes.append((int(cls), float(xc), float(yc), float(bw), float(bh)))
    return bboxes

def main(img_dir, num_samples=5):
    img_dir = Path(img_dir).resolve()
    print("Looking for images in:", img_dir)

    # If user points to dataset root, go into "images"
    if (img_dir / "images").exists():
        img_dir = img_dir / "images"

    imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        print("[ERROR] No images found in folder.")
        return


    # Pick random images to visualize
    # Pick random images to visualize
    chosen = random.sample(imgs, min(num_samples, len(imgs)))
    
    for p in chosen:
        # Replace "images" with "labels" and change extension to .txt
        rel_path = p.relative_to(img_dir)  # path like img001.jpg
        lbl_file = img_dir.parent / "labels" / rel_path.with_suffix(".txt")

        if not lbl_file.exists():
            print(f"[WARN] No label for {p.name}, skipping.")
            continue


        img = cv2.imread(str(p))
        bboxes = load_yolo_labels(lbl_file)

        # Apply augmentation (just like your dataset script)
        aug_img, aug_bboxes = augment_image_and_bboxes(img, bboxes, use_clahe=True)

        if aug_img is None:
            print(f"[SKIP] Augmentation removed all boxes for {p.name}")
            continue

        # Draw boxes: original in RED, augmented in GREEN
        orig_vis = draw_yolo_boxes(img, bboxes, (0,0,255))
        aug_vis  = draw_yolo_boxes(aug_img, aug_bboxes, (0,255,0))

        # Convert BGR → RGB for matplotlib
        orig_vis = cv2.cvtColor(orig_vis, cv2.COLOR_BGR2RGB)
        aug_vis  = cv2.cvtColor(aug_vis, cv2.COLOR_BGR2RGB)

        # Show side by side
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1); plt.imshow(orig_vis); plt.title(f"Original ({p.name})")
        plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(aug_vis); plt.title("Augmented")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to dataset folder with images + labels.")
    ap.add_argument("--num", type=int, default=5, help="Number of samples to visualize.")
    args = ap.parse_args()
    main(args.input, args.num)
