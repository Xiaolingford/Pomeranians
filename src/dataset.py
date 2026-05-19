"""
dataset.py - Complete dataset loaders for classification and detection

Classification: Standard ImageNet transforms
Detection: Strong ONLINE augmentation (different every epoch!)

This replaces the original dataset.py with proper augmentation.
"""

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
# Using [0.5, 0.5, 0.5] normalization as per methodology document
# This scales pixel values from [0,1] to [-1,1]
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# UTILS
# ============================================================
def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)


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


# ============================================================
# CLASSIFICATION AUGMENTATION PIPELINES (NEW - with strong augmentation)
# ============================================================

def get_classification_train_augmentation(img_size=224):
    """
    Strong augmentation for TRAINING classification.
    Applied ONLINE - different every epoch!
    
    Same augmentations as detection, but without bbox handling.
    """
    return A.Compose([
        # === RESIZE ===
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        
        # === GEOMETRIC ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        
        A.ShiftScaleRotate(
            shift_limit=0.1,       # ±10% translation
            scale_limit=0.15,      # ±15% scale
            rotate_limit=15,       # ±15 degrees rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        
        A.OneOf([
            A.Perspective(scale=(0.02, 0.05), p=1.0),
            A.Affine(shear=(-10, 10), p=1.0),
        ], p=0.2),
        
        # === COLOR ===
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.6),
        
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.4
        ),
        
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.Equalize(p=1.0),
        ], p=0.2),
        
        # === NOISE/BLUR ===
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        
        A.OneOf([
            A.GaussNoise(std=(0.1, 0.2), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
        ], p=0.2),
        
        # === QUALITY ===
        A.OneOf([
            A.ImageCompression(quality_range=(75, 95), p=1.0),
            A.Downscale(scale_range=(0.75, 0.95), p=1.0),
        ], p=0.15),
        
        # === OCCLUSION ===
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(img_size // 20, img_size // 10),
            hole_width_range=(img_size // 20, img_size // 10),
            fill=0,
            p=0.2
        ),
        
        # === NORMALIZE ===
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


def get_classification_val_augmentation(img_size=224):
    """
    Validation transforms - NO augmentation, just resize + normalize.
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])


# ============================================================
# CLASSIFICATION DATASET WITH AUGMENTATION
# ============================================================

class ClassificationDataset(Dataset):
    """
    Classification dataset with online augmentation support.
    
    Expects ImageFolder structure:
      root/
        ├── class1/
        │   └── *.jpg
        └── class2/
            └── *.jpg
    """
    
    def __init__(self, root_dir, transform=None, img_size=224):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_size = img_size
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")
        
        # Find class folders
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all images
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((img_path, self.class_to_idx[cls]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {self.root_dir}")
        
        print(f"[ClassificationDataset] Loaded {len(self.samples)} images, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Read image (BGR -> RGB)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply transform (augmentation for train, just resize for val)
        if self.transform:
            transformed = self.transform(image=img)
            img_t = transformed["image"]
        else:
            # No transform - just resize and convert to tensor
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        return img_t, label


# ============================================================
# CLASSIFICATION DATALOADERS (UPDATED with augmentation)
# ============================================================

def _tfm_classification(img_size=224):
    """Standard ImageNet normalization and resizing (for backward compatibility)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_dataloaders(train_dir, val_dir, batch_size=32, num_workers=None, img_size=224, use_augmentation=True):
    """
    Classification dataloaders with ONLINE AUGMENTATION.
    
    Args:
        train_dir: Path to training data (ImageFolder structure)
        val_dir: Path to validation data (ImageFolder structure)
        batch_size: Batch size
        num_workers: Number of workers
        img_size: Image size (default 224)
        use_augmentation: Whether to use online augmentation (default True)
    
    Returns:
        train_loader, val_loader, class_names
    """
    num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 4)
    pin_memory = torch.cuda.is_available()
    
    if use_augmentation:
        # NEW: Use Albumentations with strong augmentation
        train_ds = ClassificationDataset(
            train_dir,
            transform=get_classification_train_augmentation(img_size),
            img_size=img_size
        )
        val_ds = ClassificationDataset(
            val_dir,
            transform=get_classification_val_augmentation(img_size),
            img_size=img_size
        )
        classes = train_ds.classes
        
        print(f"[data:cls] train={len(train_ds)} | val={len(val_ds)} | img_size={img_size}")
        print(f"[data:cls] Train augmentation: ENABLED (strong online)")
        print(f"[data:cls] Val augmentation: DISABLED")
    else:
        # Original: Use torchvision transforms (no augmentation)
        tfm = _tfm_classification(img_size)
        train_ds = datasets.ImageFolder(train_dir, transform=tfm)
        val_ds = datasets.ImageFolder(val_dir, transform=tfm)
        classes = train_ds.classes
        
        print(f"[data:cls] train={len(train_ds)} | val={len(val_ds)} | img_size={img_size}")
        print(f"[data:cls] Train augmentation: DISABLED")
        print(f"[data:cls] Val augmentation: DISABLED")

    loader_args = _make_loader_args(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if loader_args.get("worker_init_fn") is None:
        loader_args.pop("worker_init_fn", None)

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    return train_loader, val_loader, classes


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
# DETECTION AUGMENTATION PIPELINES (NEW - with strong augmentation)
# ============================================================

def get_train_augmentation(img_size=224):
    """
    Strong augmentation for TRAINING detection.
    Applied ONLINE - different every epoch!
    
    Includes:
        - Geometric: flip, rotate, scale, shift, perspective
        - Color: brightness, contrast, hue, saturation
        - Noise: gaussian blur, noise, compression
        - Occlusion: coarse dropout
    """
    return A.Compose([
        # === RESIZE ===
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        
        # === GEOMETRIC ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        
        A.ShiftScaleRotate(
            shift_limit=0.1,       # ±10% translation
            scale_limit=0.15,      # ±15% scale
            rotate_limit=15,       # ±15 degrees rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        
        A.OneOf([
            A.Perspective(scale=(0.02, 0.05), p=1.0),
            A.Affine(shear=(-10, 10), p=1.0),
        ], p=0.2),
        
        # === COLOR ===
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.6),
        
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=15,
            p=0.4
        ),
        
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1.0),
            A.Equalize(p=1.0),
        ], p=0.2),
        
        # === NOISE/BLUR ===
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        
        A.OneOf([
            A.GaussNoise(std=(0.1, 0.2), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
        ], p=0.2),
        
        # === QUALITY ===
        A.OneOf([
            A.ImageCompression(quality_range=(75, 95), p=1.0),
            A.Downscale(scale_range=(0.75, 0.95), p=1.0),
        ], p=0.15),
        
        # === OCCLUSION ===
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(img_size // 20, img_size // 10),
            hole_width_range=(img_size // 20, img_size // 10),
            fill=0,
            p=0.2
        ),
        
        # === NORMALIZE ===
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.2,
        min_area=50,
    ))


def get_val_augmentation(img_size=224):
    """
    Validation transforms - NO augmentation, just resize + normalize.
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.2,
    ))


# ============================================================
# DETECTION DATASET (NEW - with online augmentation support)
# ============================================================

class YoloDetectionDataset(Dataset):
    """
    YOLO-format dataset with online augmentation support.
    
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
            if p.suffix.lower() in IMG_EXTS
        ])
        
        if len(self.images) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.labels_dir / (img_path.stem + ".txt")

        # Read image (BGR -> RGB)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Load YOLO labels (xc, yc, w, h normalized [0..1])
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
                        
                        # Validate YOLO format (must be in [0,1])
                        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            continue
                        
                        bboxes.append([xc, yc, w, h])
                        class_labels.append(cls)
                    except Exception:
                        continue

        # Apply transform (augmentation for train, just resize for val)
        if self.transform:
            try:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
                img_t = transformed["image"]
                bboxes = transformed.get("bboxes", [])
                class_labels = transformed.get("class_labels", [])
            except Exception:
                # Fallback: just resize and normalize
                fallback = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ToTensorV2()
                ])(image=img)
                img_t = fallback["image"]
                bboxes, class_labels = [], []
        else:
            # No transform - just convert to tensor
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Convert YOLO (xc, yc, w, h) to xyxy normalized [0,1]
        if len(bboxes) > 0:
            bboxes_np = np.array(bboxes, dtype=np.float32)
            labels_np = np.array(class_labels, dtype=np.int64)
            
            x = bboxes_np[:, 0]
            y = bboxes_np[:, 1]
            w = bboxes_np[:, 2]
            h = bboxes_np[:, 3]
            
            x1 = np.clip(x - w / 2, 0, 1)
            y1 = np.clip(y - h / 2, 0, 1)
            x2 = np.clip(x + w / 2, 0, 1)
            y2 = np.clip(y + h / 2, 0, 1)
            
            # Ensure valid boxes
            valid = (x2 > x1) & (y2 > y1)
            xyxy = np.stack([x1, y1, x2, y2], axis=1)[valid].astype(np.float32)
            labels_np = labels_np[valid]
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int64)

        boxes_t = torch.from_numpy(xyxy)
        labels_t = torch.from_numpy(labels_np).long()

        return img_t, {"boxes": boxes_t, "labels": labels_t}


# ============================================================
# DETECTION COLLATE + DATALOADERS
# ============================================================

def detection_collate(batch):
    """Custom collate for detection batches (handles variable target sizes)."""
    imgs, targets = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(targets)


def get_yolo_dataloaders(train_root, val_root=None, batch_size=4, img_size=224, num_workers=None):
    """
    Return YOLO-style dataloaders for train and val.
    
    NEW: Training uses STRONG ONLINE AUGMENTATION!
    
    Expected structure:
      train_root/
        ├── images/
        └── labels/
      val_root/
        ├── images/val/
        └── labels/val/
    """
    if val_root is None:
        val_root = train_root

    train_root = Path(train_root)
    val_root = Path(val_root)

    # Detect train structure (flat or nested)
    if (train_root / "images" / "train").exists():
        train_img = train_root / "images" / "train"
        train_lbl = train_root / "labels" / "train"
    else:
        train_img = train_root / "images"
        train_lbl = train_root / "labels"

    # Detect val structure
    if (val_root / "images" / "val").exists():
        val_img = val_root / "images" / "val"
        val_lbl = val_root / "labels" / "val"
    else:
        val_img = val_root / "images"
        val_lbl = val_root / "labels"

    # Validate directories exist
    for name, path in [("train_img", train_img), ("train_lbl", train_lbl),
                       ("val_img", val_img), ("val_lbl", val_lbl)]:
        if not path.exists():
            raise FileNotFoundError(f"{name} directory not found: {path}")

    # Create datasets with appropriate transforms
    # KEY CHANGE: Training uses strong augmentation, validation uses none
    train_ds = YoloDetectionDataset(
        train_img, train_lbl,
        transform=get_train_augmentation(img_size),  # STRONG augmentation
        img_size=img_size
    )
    val_ds = YoloDetectionDataset(
        val_img, val_lbl,
        transform=get_val_augmentation(img_size),    # NO augmentation
        img_size=img_size
    )

    num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 4)
    pin_memory = torch.cuda.is_available()

    loader_args = _make_loader_args(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if loader_args.get("worker_init_fn") is None:
        loader_args.pop("worker_init_fn", None)
    loader_args["collate_fn"] = detection_collate

    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print(f"[data:det] train={len(train_ds)} | val={len(val_ds)} | img_size={img_size}")
    print(f"[data:det] Train augmentation: ENABLED (strong online)")
    print(f"[data:det] Val augmentation: DISABLED")
    
    return train_loader, val_loader


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Augmentation Pipelines")
    print("=" * 60)
    
    # Create dummy data
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bboxes = [[0.5, 0.5, 0.2, 0.3], [0.3, 0.7, 0.15, 0.2]]
    dummy_labels = [0, 1]
    
    # ==========================================
    # Test CLASSIFICATION augmentation
    # ==========================================
    print("\n" + "-" * 60)
    print("CLASSIFICATION Augmentation")
    print("-" * 60)
    
    cls_train_aug = get_classification_train_augmentation(224)
    cls_val_aug = get_classification_val_augmentation(224)
    
    print("\nTrain augmentation (5 runs on same image):")
    for i in range(5):
        result = cls_train_aug(image=dummy_img)
        img = result["image"]
        print(f"  Run {i+1}: shape={tuple(img.shape)}, range=[{img.min():.2f}, {img.max():.2f}]")
    
    print("\nVal augmentation (should be identical):")
    for i in range(3):
        result = cls_val_aug(image=dummy_img)
        img = result["image"]
        print(f"  Run {i+1}: shape={tuple(img.shape)}, range=[{img.min():.2f}, {img.max():.2f}]")
    
    # ==========================================
    # Test DETECTION augmentation
    # ==========================================
    print("\n" + "-" * 60)
    print("DETECTION Augmentation")
    print("-" * 60)
    
    det_train_aug = get_train_augmentation(224)
    det_val_aug = get_val_augmentation(224)
    
    print("\nTrain augmentation (5 runs on same image):")
    for i in range(5):
        result = det_train_aug(image=dummy_img, bboxes=dummy_bboxes, class_labels=dummy_labels)
        img = result["image"]
        boxes = result["bboxes"]
        print(f"  Run {i+1}: shape={tuple(img.shape)}, boxes={len(boxes)}")
    
    print("\nVal augmentation (should be identical):")
    for i in range(3):
        result = det_val_aug(image=dummy_img, bboxes=dummy_bboxes, class_labels=dummy_labels)
        img = result["image"]
        boxes = result["bboxes"]
        print(f"  Run {i+1}: shape={tuple(img.shape)}, boxes={len(boxes)}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset module ready!")
    print("=" * 60)
    print("\nAugmentations applied during TRAINING (both tasks):")
    print("  • HorizontalFlip, VerticalFlip, RandomRotate90")
    print("  • ShiftScaleRotate (±10% shift, ±15% scale, ±15° rotation)")
    print("  • Perspective, Affine")
    print("  • BrightnessContrast, ColorJitter, HSV")
    print("  • GaussianBlur, MotionBlur, GaussNoise")
    print("  • ImageCompression, CoarseDropout")