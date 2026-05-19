"""
LDBST-MP Ablation Study — C2 Revision
======================================
Four model variants to quantify the contribution of each key component.
Each variant is a drop-in replacement for the full LDBST_MP model and
uses identical hyperparameters, loss, and training setup.

Variants:
    A: Swin branch only      — removes CNN branch
    B: CNN branch only       — removes Swin branch
    C: No fusion             — both branches run but outputs are summed
                               (no channel-split interaction / concatenation)
    D: Full LDBST-MP         — original model (import from model.py)

Usage (train each variant):
    python train2.py --task detection --model_variant swin_only   --epochs 100 ...
    python train2.py --task detection --model_variant cnn_only    --epochs 100 ...
    python train2.py --task detection --model_variant no_fusion   --epochs 100 ...
    python train2.py --task detection --model_variant full        --epochs 100 ...

    Or use the run_ablation.py helper to run all four sequentially.

Expected ablation table (fill in after training):
    ┌───────────────────────┬────────┬──────────┬──────┬──────────┐
    │ Variant               │ Params │ mAP@0.5  │  F1  │   FPS    │
    ├───────────────────────┼────────┼──────────┼──────┼──────────┤
    │ A: Swin branch only   │  ?M    │   ?%     │  ?   │   ?      │
    │ B: CNN branch only    │  ?M    │   ?%     │  ?   │   ?      │
    │ C: No fusion          │  ?M    │   ?%     │  ?   │   ?      │
    │ D: Full LDBST-MP      │  ?M    │   ?%     │  ?   │   ?      │
    └───────────────────────┴────────┴──────────┴──────┴──────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# Re-use shared components from your original model file
# Adjust the import path to match your project structure
from model2 import (
    PatchEmbedding,
    SwinTransformerBlock,
    PatchMerging,
    postprocess_detections,
    DetectionLoss,
    LDBST_MP,          # Variant D — full model
)


# -----------------------------------------------------------------------
# Variant A: Swin branch only
# -----------------------------------------------------------------------
class DualBranchStage_SwinOnly(nn.Module):
    """
    Ablation A — Swin branch only.
    Removes the CNN branch entirely. Full channel width flows through
    the Swin Transformer path to keep parameter count comparable.
    """
    def __init__(self, dim, num_heads=4, window_size=7):
        super().__init__()
        self.dim = dim

        self.swin_block1 = SwinTransformerBlock(dim, num_heads, window_size, shift_size=0)
        self.swin_block2 = SwinTransformerBlock(dim, num_heads, window_size, shift_size=window_size // 2)
        self.swin_downsample = PatchMerging(dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)              # (B, H, W, C)
        x = self.swin_block1(x)
        x = self.swin_block2(x)
        x = self.swin_downsample(x)             # (B, H/2, W/2, C)
        x = x.permute(0, 3, 1, 2)              # (B, C, H/2, W/2)
        return x


class LDBST_MP_SwinOnly(nn.Module):
    """Variant A: Swin branch only — no CNN component."""
    VARIANT = "swin_only"

    def __init__(self, img_size=224, num_classes=2, num_heads=4, task="detection"):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.task = task

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=4, in_chans=3, embed_dim=96)
        self.stage1 = DualBranchStage_SwinOnly(dim=96, num_heads=num_heads, window_size=7)
        self.patch_merge_s1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
        )
        self.stage2 = DualBranchStage_SwinOnly(dim=192, num_heads=num_heads, window_size=7)
        self._build_head(task, num_classes)

    def _build_head(self, task, num_classes):
        if task == "classification":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(192, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.5), nn.Linear(256, num_classes),
            )
        elif task == "detection":
            self.upsample = nn.Sequential(
                nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
                nn.Conv2d(192, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            )
            self.detection_head = nn.Conv2d(96, 4 + 1 + num_classes, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.patch_merge_s1(x)
        x = self.stage2(x)
        return _detection_forward(self, x) if self.task == "detection" else _cls_forward(self, x)

    def predict(self, x, conf_thresh=0.45, iou_thresh=0.45):
        return _predict(self, x, conf_thresh, iou_thresh)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Variant B: CNN branch only
# -----------------------------------------------------------------------
class DualBranchStage_CNNOnly(nn.Module):
    """
    Ablation B — CNN branch only.
    Removes the Swin Transformer branch entirely. Full channel width
    flows through the CNN path.
    """
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim

        self.cnn_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.cnn_bn   = nn.BatchNorm2d(dim)
        self.cnn_relu = nn.ReLU(inplace=True)
        self.cnn_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cnn_conv(x)
        x = self.cnn_bn(x)
        x = self.cnn_relu(x)
        x = self.cnn_pool(x)    # (B, C, H/2, W/2)
        return x


class LDBST_MP_CNNOnly(nn.Module):
    """Variant B: CNN branch only — no Swin Transformer component."""
    VARIANT = "cnn_only"

    def __init__(self, img_size=224, num_classes=2, num_heads=4, task="detection"):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.task = task

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=4, in_chans=3, embed_dim=96)
        self.stage1 = DualBranchStage_CNNOnly(dim=96)
        self.patch_merge_s1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
        )
        self.stage2 = DualBranchStage_CNNOnly(dim=192)
        self._build_head(task, num_classes)

    def _build_head(self, task, num_classes):
        if task == "classification":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(192, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.5), nn.Linear(256, num_classes),
            )
        elif task == "detection":
            self.upsample = nn.Sequential(
                nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
                nn.Conv2d(192, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            )
            self.detection_head = nn.Conv2d(96, 4 + 1 + num_classes, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.patch_merge_s1(x)
        x = self.stage2(x)
        return _detection_forward(self, x) if self.task == "detection" else _cls_forward(self, x)

    def predict(self, x, conf_thresh=0.45, iou_thresh=0.45):
        return _predict(self, x, conf_thresh, iou_thresh)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Variant C: No fusion (both branches, summation instead of concatenation)
# -----------------------------------------------------------------------
class DualBranchStage_NoFusion(nn.Module):
    """
    Ablation C — Both branches, no feature fusion.
    Both the Swin and CNN branches process half the channels each,
    but their outputs are SUMMED rather than concatenated.
    This keeps the channel dimensions identical to the full model
    while removing the cross-branch interaction that concatenation enables.
    """
    def __init__(self, dim, num_heads=4, window_size=7):
        super().__init__()
        self.dim = dim
        branch_dim = dim // 2

        self.split_conv = nn.Conv2d(dim, dim, kernel_size=1)

        # Swin branch
        self.swin_block1 = SwinTransformerBlock(branch_dim, num_heads, window_size, shift_size=0)
        self.swin_block2 = SwinTransformerBlock(branch_dim, num_heads, window_size, shift_size=window_size // 2)
        self.swin_downsample = PatchMerging(branch_dim)

        # CNN branch
        self.cnn_conv = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1)
        self.cnn_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn_bn   = nn.BatchNorm2d(branch_dim)
        self.cnn_relu = nn.ReLU(inplace=True)

        # 1x1 projection to restore full channel width after summation
        # (branch_dim -> dim, since we sum instead of cat)
        self.proj = nn.Conv2d(branch_dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.split_conv(x)
        swin_input = x[:, :C // 2, :, :]
        cnn_input  = x[:, C // 2:, :, :]

        # Swin branch
        swin_x = swin_input.permute(0, 2, 3, 1)
        swin_x = self.swin_block1(swin_x)
        swin_x = self.swin_block2(swin_x)
        swin_x = self.swin_downsample(swin_x)   # (B, H/2, W/2, C/2)
        swin_x = swin_x.permute(0, 3, 1, 2)    # (B, C/2, H/2, W/2)

        # CNN branch
        cnn_x = self.cnn_conv(cnn_input)
        cnn_x = self.cnn_bn(cnn_x)
        cnn_x = self.cnn_relu(cnn_x)
        cnn_x = self.cnn_pool(cnn_x)            # (B, C/2, H/2, W/2)

        # SUM instead of concatenate — no fusion interaction
        fused = self.proj(swin_x + cnn_x)       # (B, C, H/2, W/2)
        return fused


class LDBST_MP_NoFusion(nn.Module):
    """Variant C: Both branches present, feature fusion replaced with summation."""
    VARIANT = "no_fusion"

    def __init__(self, img_size=224, num_classes=2, num_heads=4, task="detection"):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.task = task

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=4, in_chans=3, embed_dim=96)
        self.stage1 = DualBranchStage_NoFusion(dim=96, num_heads=num_heads, window_size=7)
        self.patch_merge_s1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
        )
        self.stage2 = DualBranchStage_NoFusion(dim=192, num_heads=num_heads, window_size=7)
        self._build_head(task, num_classes)

    def _build_head(self, task, num_classes):
        if task == "classification":
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(192, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.5), nn.Linear(256, num_classes),
            )
        elif task == "detection":
            self.upsample = nn.Sequential(
                nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
                nn.Conv2d(192, 96, kernel_size=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            )
            self.detection_head = nn.Conv2d(96, 4 + 1 + num_classes, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stage1(x)
        x = self.patch_merge_s1(x)
        x = self.stage2(x)
        return _detection_forward(self, x) if self.task == "detection" else _cls_forward(self, x)

    def predict(self, x, conf_thresh=0.45, iou_thresh=0.45):
        return _predict(self, x, conf_thresh, iou_thresh)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------
# Shared forward helpers (avoid code duplication across variants)
# -----------------------------------------------------------------------
def _cls_forward(model, x):
    x = model.pool(x).flatten(1)
    return model.classifier(x)


def _detection_forward(model, x):
    x = model.upsample(x)
    detections = model.detection_head(x)
    B, C, H, W = detections.shape
    detections = detections.permute(0, 2, 3, 1).reshape(B, H * W, C)
    pred_boxes_raw = torch.sigmoid(detections[..., :4])
    pred_obj    = detections[..., 4]
    pred_logits = detections[..., 5:]
    cx, cy = pred_boxes_raw[..., 0], pred_boxes_raw[..., 1]
    w,  h  = pred_boxes_raw[..., 2], pred_boxes_raw[..., 3]
    x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
    y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
    x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
    y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
    pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return pred_boxes, pred_logits, pred_obj


@torch.no_grad()
def _predict(model, x, conf_thresh, iou_thresh):
    model.eval()
    if model.task == "classification":
        return torch.softmax(model.forward(x), dim=-1)
    pred_boxes, pred_logits, pred_obj = model.forward(x)
    return [
        postprocess_detections(
            pred_boxes[b], pred_logits[b], pred_obj[b],
            conf_thresh=conf_thresh, iou_thresh=iou_thresh, img_size=model.img_size,
        ).cpu()
        for b in range(x.size(0))
    ]


# -----------------------------------------------------------------------
# Registry — use this in your training script
# -----------------------------------------------------------------------
ABLATION_VARIANTS = {
    "swin_only":  LDBST_MP_SwinOnly,
    "cnn_only":   LDBST_MP_CNNOnly,
    "no_fusion":  LDBST_MP_NoFusion,
    "full":       LDBST_MP,             # Variant D — original model
}


def build_ablation_model(variant: str, **kwargs) -> nn.Module:
    """
    Factory function. Use in your training script:

        from ablation_model import build_ablation_model
        model = build_ablation_model("swin_only", img_size=224, num_classes=2, task="detection")
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(ABLATION_VARIANTS)}")
    return ABLATION_VARIANTS[variant](**kwargs)


# -----------------------------------------------------------------------
# Quick sanity check
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Ablation model sanity check")
    print("=" * 70)

    x = torch.randn(2, 3, 224, 224)

    for name, cls in ABLATION_VARIANTS.items():
        model = cls(img_size=224, num_classes=2, num_heads=4, task="detection")
        with torch.no_grad():
            pred_boxes, pred_logits, pred_obj = model(x)
        params = model.count_parameters() if hasattr(model, "count_parameters") \
            else sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [{name:<12}]  params={params:,}  boxes={pred_boxes.shape}  logits={pred_logits.shape}")

    print("\nAll variants OK.")