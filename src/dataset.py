import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random
import numpy as np

# ============================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ============================================================
# CONSTANTS
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# UTILS
# ============================================================
def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)


# ============================================================
# CLASSIFICATION DATASETS
# ============================================================
def _tfm_classification(img_size=224):
    """Standard ImageNet normalization and resizing."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _make_loader_args(batch_size, num_workers, pin_memory=True):
    """Build DataLoader kwargs safely."""
    args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )
    if num_workers > 0:
        args.update(dict(prefetch_factor=2, persistent_workers=True))
    return args


def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=None):
    """Standard classification dataloaders with performance hints."""
    tfm = _tfm_classification()
    tr = datasets.ImageFolder(train_dir, transform=tfm)
    va = datasets.ImageFolder(val_dir, transform=tfm)

    num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 4)
    pin_memory = torch.cuda.is_available()

    print(f"[data] train={len(tr)} | val={len(va)} | workers={num_workers} | pin_memory={pin_memory}")

    loader_args = _make_loader_args(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if loader_args.get("worker_init_fn") is None:
        loader_args.pop("worker_init_fn", None)

    train_loader = DataLoader(tr, shuffle=True, **loader_args)
    val_loader = DataLoader(va, shuffle=False, **loader_args)

    return train_loader, val_loader, tr.classes


# ============================================================
# AUTO SPLIT CLASSIFICATION DATA
# ============================================================
def _split_dataset_auto(full_dataset, val_ratio=0.2, test_ratio=0.1, seed=42):
    """Split dataset automatically into train, val, test."""
    n_total = len(full_dataset)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    diff = n_total - (n_train + n_val + n_test)
    if diff != 0:
        n_train += diff

    g = torch.Generator().manual_seed(seed)
    return random_split(full_dataset, [n_train, n_val, n_test], generator=g)


def get_dataloaders_auto(data_dir, batch_size=32, val_ratio=0.2, test_ratio=0.1, seed=42, **loader_kwargs):
    """Automatically split one folder into train, val, and test loaders."""
    tfm = _tfm_classification()
    full = datasets.ImageFolder(data_dir, transform=tfm)
    tr, va, te = _split_dataset_auto(full, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    num_workers = loader_kwargs.pop("num_workers", min(4, os.cpu_count() or 4))
    pin_memory = loader_kwargs.pop("pin_memory", torch.cuda.is_available())

    base_args = _make_loader_args(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if base_args.get("worker_init_fn") is None:
        base_args.pop("worker_init_fn", None)
    base_args.update(loader_kwargs)

    tr_loader = DataLoader(tr, shuffle=True, **base_args)
    va_loader = DataLoader(va, shuffle=False, **base_args)
    te_loader = DataLoader(te, shuffle=False, **base_args)

    print(f"[data:auto] dir={data_dir} | train={len(tr)} | val={len(va)} | test={len(te)} | classes={full.classes}")
    return tr_loader, va_loader, te_loader, full.classes


# ============================================================
# DETECTION DATASETS (YOLO-style)
# ============================================================
def _tfm_detection(img_size=224):
    """Albumentations pipeline for YOLO-style detection."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3, clip=True))


class YoloDetectionDataset(Dataset):
    """YOLO-format dataset with Albumentations transforms.

    Expects:
      - images_dir: folder of images
      - labels_dir: folder of .txt YOLO-format label files
    """
    def __init__(self, images_dir, labels_dir, transform=None, img_size=224):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        self.images = sorted([
            p for p in self.images_dir.glob("*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        self.transform = transform if transform is not None else _tfm_detection(img_size)
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels_dir / (img_path.stem + ".txt")

        # Read image (BGR->RGB)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load YOLO labels (xc,yc,w,h normalized [0..1])
        bboxes = []
        class_labels = []
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            with open(lbl_path) as f:
                for line in f:
                    vals = line.strip().split()
                    if len(vals) != 5:
                        continue
                    try:
                        cls = int(float(vals[0]))
                        xc, yc, w, h = map(float, vals[1:5])
                        
                        # ✅ Validate YOLO format (must be in [0,1])
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            print(f"[warn] Invalid box in {lbl_path.name}: {xc},{yc},{w},{h}")
                            continue
                        
                        bboxes.append([xc, yc, w, h])
                        class_labels.append(int(cls))
                    except Exception as e:
                        print(f"[warn] Failed to parse line in {lbl_path.name}: {line.strip()} | {e}")
                        continue

        # Apply Albumentations transform
        transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
        img_t = transformed["image"]  # ToTensorV2 -> (C,H,W)

        tbboxes = transformed.get("bboxes", [])
        tlabels = transformed.get("class_labels", [])

        # Convert to numpy arrays
        if len(tbboxes) == 0:
            bboxes_np = np.zeros((0, 4), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int64)
        else:
            bboxes_np = np.array(tbboxes, dtype=np.float32)
            labels_np = np.array(tlabels, dtype=np.int64)

        # Convert YOLO (xc,yc,w,h) normalized -> xyxy normalized [0,1]
        if bboxes_np.shape[0] > 0:
            x = bboxes_np[:, 0]
            y = bboxes_np[:, 1]
            w = bboxes_np[:, 2]
            h = bboxes_np[:, 3]
            
            x1 = x - w / 2.0
            y1 = y - h / 2.0
            x2 = x + w / 2.0
            y2 = y + h / 2.0
            
            # ✅ Clip to [0, 1] range (safety measure)
            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)
            
            # ✅ Ensure x2 > x1 and y2 > y1 (valid boxes)
            x2 = np.maximum(x2, x1 + 1e-4)
            y2 = np.maximum(y2, y1 + 1e-4)
            
            xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)

        boxes_t = torch.from_numpy(xyxy)
        labels_t = torch.from_numpy(labels_np).long()

        target = {"boxes": boxes_t, "labels": labels_t}
        return img_t, target


# ============================================================
# COLLATE + DATALOADERS
# ============================================================
def detection_collate(batch):
    """Custom collate for detection batches (handles variable target sizes)."""
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets)


def get_yolo_dataloaders(train_root, val_root=None, batch_size=4, img_size=224, num_workers=None):
    """
    Return YOLO-style dataloaders for train and val.
    
    Expected structure:
      train_root/
        ├── images/train/
        └── labels/train/
      val_root/
        ├── images/val/
        └── labels/val/
    """
    if val_root is None:
        val_root = train_root

    # ✅ Fixed paths - explicitly include train/val subdirectories
    train_img = Path(train_root) / "images" 
    train_lbl = Path(train_root) / "labels" 

    val_img = Path(val_root) / "images" / "val"
    val_lbl = Path(val_root) / "labels" / "val"

    # ✅ Validate directories exist
    for name, path in [("train_img", train_img), ("train_lbl", train_lbl), 
                        ("val_img", val_img), ("val_lbl", val_lbl)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} directory not found: {path}")

    train_ds = YoloDetectionDataset(train_img, train_lbl, transform=_tfm_detection(img_size), img_size=img_size)
    val_ds = YoloDetectionDataset(val_img, val_lbl, transform=_tfm_detection(img_size), img_size=img_size)

    num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 4)
    pin_memory = torch.cuda.is_available()

    loader_args = _make_loader_args(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if loader_args.get("worker_init_fn") is None:
        loader_args.pop("worker_init_fn", None)
    loader_args["collate_fn"] = detection_collate

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print(f"[data:det] train={len(train_ds)} | val={len(val_ds)} | img_size={img_size} | workers={num_workers}")
    return train_loader, val_loader

