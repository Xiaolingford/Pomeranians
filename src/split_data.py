
import os
import shutil
import random
from pathlib import Path

def split_classification_dataset(root, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)

    root = Path(root)
    classes = [d for d in root.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes: {[c.name for c in classes]}")

    for cls in classes:
        images = sorted([p for p in cls.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
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
            out_dir = root / split / cls.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in files:
                shutil.copy(img_path, out_dir / img_path.name)

        print(f"{cls.name} → train: {n_train}, val: {n_val}, test: {n_test}")

if __name__ == "__main__":
    dataset_root = "./data" 
    split_classification_dataset(dataset_root, val_ratio=0.2, test_ratio=0.1)

