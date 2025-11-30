from pathlib import Path

data_path = Path(r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC")

print("=== Dataset Structure ===")
for split in ['train', 'val', 'test']:
    split_path = data_path / split
    if split_path.exists():
        print(f"\n{split}/:")
        for class_folder in split_path.iterdir():
            if class_folder.is_dir():
                num_images = len(list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png')))
                print(f"  {class_folder.name}: {num_images} images")
    else:
        print(f"\n{split}/ - DOES NOT EXIST")

print("\n=== Looking for loose images in dataC root ===")
loose_images = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png'))
if loose_images:
    print(f"WARNING: Found {len(loose_images)} images directly in dataC/ (should be in train/val folders)")
else:
    print("Good - no loose images in root")