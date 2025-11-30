from pathlib import Path

train_dir = Path(r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC\train\algae")

files = sorted([f.name for f in train_dir.glob("*.jpg")])[:20]
print("First 20 files in dataC/train/algae:")
for f in files:
    print(f"  {f}")