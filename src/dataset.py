from pathlib import Path
import os
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageFile
import cv2
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

IM_SIZE = 224

def _read_rgb(path: Path) -> np.ndarray:
    # PIL handles Unicode paths better than cv2 on Windows
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im)  # H,W,3 uint8

def letterbox_pad(img: np.ndarray, size: int = IM_SIZE, color=(114,114,114)) -> np.ndarray:
    """Resize with unchanged aspect and pad to square (size x size)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def clahe_rgb(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE per channel in LAB space for contrast enhancement."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

class MicroplasticDataset(Dataset):
    """
    Expects directory structure:
      root/
        algae/*.jpg|png|jpeg
        microplastics/*.jpg|png|jpeg
    """
    def __init__(self,
                 root_dir: Optional[str] = None,
                 split: str = "train",
                 val_ratio: float = 0.2,
                 return_both: bool = False,
                 verbose: bool = True):
        ROOT = Path(__file__).resolve().parents[1]
        self.root = Path(root_dir) if root_dir else (ROOT / "data")
        if not self.root.exists():
            raise FileNotFoundError(f"Data root not found: {self.root}")

        self.return_both = return_both
        cls_map = {"algae": 0, "microplastics": 1}
        samples: List[Tuple[Path, int]] = []
        for cname, label in cls_map.items():
            cdir = self.root / cname
            if not cdir.exists():
                raise FileNotFoundError(f"Missing folder: {cdir}")
            for fn in sorted(os.listdir(cdir)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    samples.append((cdir / fn, label))

        # Stable split by filename sort
        rng = np.random.RandomState(1234)
        idx = np.arange(len(samples))
        # Shuffle deterministically, but keep stability across runs
        rng.shuffle(idx)
        cut = int(round(len(samples) * (1.0 - val_ratio)))
        train_idx = idx[:cut]
        val_idx   = idx[cut:]

        if split == "train":
            sel = train_idx
        elif split == "val":
            sel = val_idx
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.samples = [samples[i] for i in sel]
        self.labels_only = [y for _, y in self.samples]

        if verbose:
            n0 = sum(1 for _,y in self.samples if y==0)
            n1 = sum(1 for _,y in self.samples if y==1)
            print(f"{split.capitalize()} set: {len(self.samples)} images (algae={n0}, microplastics={n1})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = _read_rgb(path)
        img_std = letterbox_pad(img, IM_SIZE)
        img_cla = clahe_rgb(img_std)

        # to CHW float32 in [0,1]
        x_std = torch.from_numpy(img_std).permute(2,0,1).float()/255.0
        x_cla = torch.from_numpy(img_cla).permute(2,0,1).float()/255.0

        if self.return_both:
            return x_std, x_cla, torch.tensor(y, dtype=torch.long)
        else:
            # simple mode (not used by this project)
            return x_std, torch.tensor(y, dtype=torch.long)
