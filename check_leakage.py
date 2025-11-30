import hashlib
from pathlib import Path

def get_image_hash(img_path):
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

data_path = Path(r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC")

print("=== Checking for duplicates between train and val ===\n")

# Get all train image hashes
train_hashes = {}
train_path = data_path / 'train'
for class_folder in train_path.iterdir():
    if class_folder.is_dir():
        for img in list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png')):
            train_hashes[get_image_hash(img)] = img

print(f"Train set: {len(train_hashes)} unique images")

# Check for duplicates in val
val_path = data_path / 'val'
duplicates = []
val_count = 0
for class_folder in val_path.iterdir():
    if class_folder.is_dir():
        for img in list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png')):
            val_count += 1
            img_hash = get_image_hash(img)
            if img_hash in train_hashes:
                duplicates.append((img, train_hashes[img_hash]))

print(f"Val set: {val_count} images")
print(f"\n🚨 DUPLICATES FOUND: {len(duplicates)}")

if duplicates:
    print(f"Leakage: {len(duplicates)/val_count*100:.1f}% of validation set")
    print("\nFirst 5 examples:")
    for val_img, train_img in duplicates[:5]:
        print(f"  Val:   {val_img.relative_to(data_path)}")
        print(f"  Train: {train_img.relative_to(data_path)}\n")
else:
    print("✅ No duplicates - your split is clean!")