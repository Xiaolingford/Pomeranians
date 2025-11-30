import os
import shutil
import random
from pathlib import Path

def remap_and_split_dataset(root, mapping, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)
    root = Path(root)

    # Get all image files
    images = sorted([p for p in root.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print(f"Found {len(images)} images in {root}")

    # Shuffle
    random.shuffle(images)

    n_total = len(images)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, files in splits.items():
        img_out = root / "images" / split
        lbl_out = root / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # Copy image
            shutil.copy(img_path, img_out / img_path.name)

            # Copy + remap matching label
            lbl_path = root / (img_path.stem + ".txt")
            if lbl_path.exists():
                new_lines = []
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls = int(parts[0])
                        if cls in mapping:
                            parts[0] = str(mapping[cls])
                        else:
                            raise ValueError(f"Class {cls} not in mapping for {lbl_path}")
                        new_lines.append(" ".join(parts))

                with open(lbl_out / lbl_path.name, "w") as f:
                    f.write("\n".join(new_lines))

        print(f"{split}: {len(files)} images")

if __name__ == "__main__":
    dataset_root = "./data_yolo"  # adjust if needed
    class_mapping = {1: 0, 4: 1}  # old → new
    remap_and_split_dataset(dataset_root, class_mapping, val_ratio=0.2, test_ratio=0.1)
