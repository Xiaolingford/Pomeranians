import os
from pathlib import Path

val_dir = "./data/val"
for class_name in os.listdir(val_dir):
    class_path = Path(val_dir) / class_name
    if class_path.is_dir():
        count = len(list(class_path.glob("*")))
        pct = (count / 228) * 100
        print(f"{class_name}: {count} images ({pct:.2f}%)")