# src/augmentations_util.py
import cv2, random
import numpy as np

def augment_image(img, use_clahe=True):
    """Augmentation for classification (image only)."""
    h, w = img.shape[:2]
    out = img.copy()

    # --- flips ---
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
    if random.random() < 0.2:
        out = cv2.flip(out, 0)

    # --- small rotation + scale ---
    ang = random.uniform(-12, 12)
    scale = random.uniform(0.95, 1.05)
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, scale)
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # --- brightness / contrast ---
    alpha = random.uniform(0.9, 1.1)
    beta  = random.uniform(-12, 12)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # --- HSV jitter ---
    if random.random() < 0.6:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-10, 10), 0, 255)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-4, 4)) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- blur / noise ---
    if random.random() < 0.4:
        if random.random() < 0.5:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        else:
            noise = np.random.normal(0, 6, out.shape).astype(np.int16)
            out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- CLAHE ---
    if use_clahe and random.random() < 0.6:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return out


def augment_image_and_bboxes(img, bboxes, use_clahe=True, prob_keep_empty=False):
    """
    Augmentation for detection (image + YOLO bboxes).
    bboxes = [(cls, xc, yc, w, h), ...] in normalized YOLO format.
    Returns: aug_img, new_bboxes
    """
    h, w = img.shape[:2]
    out = img.copy()

    # ------------------------------
    # Helpers
    # ------------------------------
    def yolo_to_xyxy(box):
        cls, xc, yc, bw, bh = box
        return cls, int((xc - bw/2) * w), int((yc - bh/2) * h), int((xc + bw/2) * w), int((yc + bh/2) * h)

    def xyxy_to_yolo(cls, x1, y1, x2, y2):
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        xc = (x1 + x2) / (2*w)
        yc = (y1 + y2) / (2*h)
        # NEW: clamp to [0,1]
        return (cls,
                np.clip(xc, 0, 1),
                np.clip(yc, 0, 1),
                np.clip(bw, 0, 1),
                np.clip(bh, 0, 1))

    def apply_affine(x, y, M):
        v = np.dot(M, np.array([x, y, 1]))
        return int(v[0]), int(v[1])

    # Convert all boxes first
    boxes_xyxy = [yolo_to_xyxy(b) for b in bboxes]

    # ------------------------------
    # Augmentations
    # ------------------------------

    # --- flips ---
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
        boxes_xyxy = [(cls, w-x2, y1, w-x1, y2) for cls, x1, y1, x2, y2 in boxes_xyxy]

    if random.random() < 0.2:
        out = cv2.flip(out, 0)
        boxes_xyxy = [(cls, x1, h-y2, x2, h-y1) for cls, x1, y1, x2, y2 in boxes_xyxy]

    # --- small rotation + scale ---
    ang = random.uniform(-0, 0)
    scale = random.uniform(1.0, 1.0)
    M = cv2.getRotationMatrix2D((w/2, h/2), ang, scale)
    out = cv2.warpAffine(out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    new_boxes = []
    for cls, x1, y1, x2, y2 in boxes_xyxy:
        pts = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
        pts = [apply_affine(x,y,M) for (x,y) in pts]
        xs, ys = zip(*pts)
        x1n, y1n, x2n, y2n = max(0,min(xs)), max(0,min(ys)), min(w,max(xs)), min(h,max(ys))
        if x2n > x1n and y2n > y1n:
            new_boxes.append((cls, x1n,y1n,x2n,y2n))

    # --- brightness / contrast ---
    alpha = random.uniform(0.9, 1.1)
    beta  = random.uniform(-12, 12)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # --- HSV jitter ---
    if random.random() < 0.6:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-10, 10), 0, 255)
        hsv[..., 0] = (hsv[..., 0] + random.randint(-4, 4)) % 180
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- blur / noise ---
    if random.random() < 0.4:
        if random.random() < 0.5:
            k = random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        else:
            noise = np.random.normal(0, 6, out.shape).astype(np.int16)
            out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # --- CLAHE ---
    if use_clahe and random.random() < 0.6:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ------------------------------
    # Convert back to YOLO format
    # ------------------------------
    new_bboxes = [xyxy_to_yolo(cls,x1,y1,x2,y2) for cls,x1,y1,x2,y2 in new_boxes]

    # NEW: skip images if no boxes survive (optional)
    if not new_bboxes and not prob_keep_empty:
        return None, []

    return out, new_bboxes
