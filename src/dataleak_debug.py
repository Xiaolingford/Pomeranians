import os
from pathlib import Path

data_path = Path(r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC")

# Check structure
print("Dataset structure:")
for split in ['train', 'val', 'test']:
    split_path = data_path / split
    if split_path.exists():
        print(f"\n{split}/:")
        for class_folder in split_path.iterdir():
            if class_folder.is_dir():
                num_images = len(list(class_folder.glob('*')))
                print(f"  {class_folder.name}: {num_images} images")
    else:
        print(f"\n{split}/ - MISSING!")
        
        import hashlib

def get_image_hash(img_path):
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

train_hashes = {}
val_hashes = {}

# Get all train image hashes
train_path = data_path / 'train'
for img_path in train_path.rglob('*.[jp][pn]g'):
    train_hashes[get_image_hash(img_path)] = img_path

# Check for duplicates in val
val_path = data_path / 'val'
duplicates = []
for img_path in val_path.rglob('*.[jp][pn]g'):
    img_hash = get_image_hash(img_path)
    if img_hash in train_hashes:
        duplicates.append((img_path, train_hashes[img_hash]))

print(f"\nFound {len(duplicates)} duplicate images between train and val!")
if duplicates:
    print("Examples:")
    for val_img, train_img in duplicates[:5]:
        print(f"  Val: {val_img}")
        print(f"  Train: {train_img}")
        
        from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data=r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC",
        epochs=10,
        imgsz=224,
        batch=8,
        name="yolov8n_classification_microplastics_algae",
        pretrained=True,
        verbose=True  # More detailed output
    )

    # Validate on BOTH train and val to compare
    print("\n=== Training Set Performance ===")
    train_metrics = model.val(split='train')
    print(f"Train Accuracy: {train_metrics.top1}")
    
    print("\n=== Validation Set Performance ===")
    val_metrics = model.val(split='val')
    print(f"Val Accuracy: {val_metrics.top1}")
    
    # Check confusion matrix
    print("\nCheck the confusion matrix at:")
    print(f"runs/classify/yolov8n_classification_microplastics_algae/confusion_matrix.png")

if __name__ == "__main__":
    main()