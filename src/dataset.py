import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


#--------------
#Shared Constraints
#--------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


#--------------
#Classification
#--------------


def _tfm_classification():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_dataloaders(train_dir, val_dir, batch_size=32):
    tfm = _tfm_classification()
    tr = datasets.ImageFolder(train_dir, transform=tfm)
    va = datasets.ImageFolder(val_dir,   transform=tfm)
    print(f"[data] train={len(tr)} | val={len(va)} | classes={tr.classes}")
    return (DataLoader(tr, batch_size=batch_size, shuffle=True),
            DataLoader(va, batch_size=batch_size, shuffle=False),
            tr.classes)

def _split_dataset_auto(full_dataset, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Splits a dataset into train/val/test subsets safely (avoids rounding errors).
    """
    n_total = len(full_dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test  # ensure exact total match

    # Prevent empty splits
    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    n_test = max(n_test, 1)

    # Adjust in case rounding caused overflow
    diff = n_total - (n_train + n_val + n_test)
    if diff != 0:
        n_train += diff  # fix mismatch by adjusting training set

    g = torch.Generator().manual_seed(seed)
    tr, va, te = random_split(full_dataset, [n_train, n_val, n_test], generator=g)
    return tr, va, te



def get_dataloaders_auto(root_dir, batch_size=32, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Automatically split a single folder into train, val, and test sets.
    Returns:
        tr_loader, va_loader, te_loader, classes
    """
    tfm = _tfm_classification()
    full = datasets.ImageFolder(root_dir, transform=tfm)

    tr, va, te = _split_dataset_auto(full, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[data:auto] root={root_dir} | train={len(tr)} | val={len(va)} | test={len(te)} | classes={full.classes}")
    return tr_loader, va_loader, te_loader, full.classes


#-------------------
#Detection YOLOOOO
#-------------------

def _tfm_detection(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.2, rotate_limit=20, p=0.4),
        A.Blur(blur_limit=3, p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    
class YoloDetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, img_size=224):
        self.images = sorted([p for p in Path(images_dir).glob("*") 
                              if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        self.labels_dir = Path(labels_dir)
        self.transform = transform if transform is not None else _tfm_detection(img_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels_dir / (img_path.stem + ".txt")

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load YOLO-format labels
        bboxes, class_labels = [], []
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            with open(lbl_path) as f:
                for line in f:
                    cls, xc, yc, w, h = map(float, line.split())
                    bboxes.append([xc, yc, w, h])
                    class_labels.append(int(cls))

        # Apply transforms
        transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        img = transformed["image"]
        bboxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        labels = torch.tensor(transformed["class_labels"], dtype=torch.long)

        # Convert YOLO xywh → xyxy
        if len(bboxes) > 0:
            x, y, w, h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
            x1, y1 = (x - w / 2) * 224, (y - h / 2) * 224
            x2, y2 = (x + w / 2) * 224, (y + h / 2) * 224
            bboxes = torch.stack([x1, y1, x2, y2], dim=1)


        target = {"boxes": bboxes, "labels": labels}
        return img, target


# -------------------
# Detection Collate Fn to be used outside dataset.py
# -------------------
def detection_collate(batch):
    """Custom collate function for detection DataLoader."""
    imgs, targets = zip(*batch)   # unzip list of tuples
    imgs = torch.stack(imgs, 0)   # stack images into tensor [B,3,H,W]
    return imgs, list(targets)    # keep targets as a list of dicts

# -------------------
# YOLO Detection Dataloaders
# -------------------
def get_yolo_dataloaders(data_root, batch_size=4, img_size=224):
    """
    Creates train/val dataloaders for YOLO-format detection dataset.
    Expected structure:
        data_root/
            images/train/*.jpg
            images/val/*.jpg
            labels/train/*.txt
            labels/val/*.txt
    """
    train_img = Path(data_root) / "images" / "train"
    val_img   = Path(data_root) / "images" / "val"
    train_lbl = Path(data_root) / "labels" / "train"
    val_lbl   = Path(data_root) / "labels" / "val"

    train_ds = YoloDetectionDataset(train_img, train_lbl, transform=_tfm_detection(img_size))
    val_ds   = YoloDetectionDataset(val_img,   val_lbl,   transform=_tfm_detection(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=detection_collate)

    print(f"[data:det] train={len(train_ds)} | val={len(val_ds)} | img_size={img_size}")
    return train_loader, val_loader
