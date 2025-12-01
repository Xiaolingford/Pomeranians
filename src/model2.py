import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou
import math


CONF_THRESH = 0.5
NMS_IOU = 0.45


# ========================================
# UTILITY FUNCTIONS (IoU, GIoU, etc.)
# ========================================
def box_iou_custom(boxes1, boxes2):
    """Compute IoU between two sets of boxes [N,4] and [M,4]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def giou_loss(pred_boxes, target_boxes, reduction='mean'):
    """Generalized IoU loss."""
    x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    area_c = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area_p = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    area_t = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)

    lt = torch.max(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.min(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area_p + area_t - inter
    iou = inter / (union + 1e-6)
    giou = iou - (area_c - union) / (area_c + 1e-6)
    loss = 1 - giou
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


# ========================================
# 1. PATCH EMBEDDING MODULE
# ========================================
class PatchEmbedding(nn.Module):
    """
    Converts 224×224×3 image into 56×56×96 feature map.
    Uses 4×4 patch size with stride 4.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size  # 224 // 4 = 56
        self.num_patches = self.grid_size ** 2
        
        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Conv projection: (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = self.proj(x)
        return x


# ========================================
# 2. WINDOW-BASED MULTI-HEAD SELF-ATTENTION
# ========================================
def window_partition(x, window_size):
    """
    Partition feature map into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H: Height of feature map
        W: Width of feature map
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-head Self-Attention (W-MSA).
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows*B, N, C) where N = window_size * window_size
            mask: (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


# ========================================
# 3. CUSTOM SWIN TRANSFORMER BLOCK
# ========================================
class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with W-MSA or SW-MSA.
    NO Conv-MLP as specified.
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Convert to (B, H, W, C) for window attention
        x = x.permute(0, 2, 3, 1).contiguous()
        
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # Create attention mask for SW-MSA
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            shifted_x = x
            attn_mask = None
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        # Convert back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
    def create_mask(self, H, W):
        """Create attention mask for SW-MSA."""
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


# ========================================
# 4. DUAL-BRANCH MODULE (Stage)
# ========================================
class DualBranchStage(nn.Module):
    """
    One stage of dual-branch processing:
    - Splits channels in half via 1x1 conv
    - Swin branch: 2 Swin blocks (W-MSA + SW-MSA)
    - CNN branch: 3x3 conv + 2x2 maxpool
    - Concatenation
    """
    def __init__(self, in_dim, window_size=7, num_heads=4):
        super().__init__()
        self.in_dim = in_dim
        self.half_dim = in_dim // 2
        
        # Channel splitting
        self.split_conv1 = nn.Conv2d(in_dim, self.half_dim, kernel_size=1)
        self.split_conv2 = nn.Conv2d(in_dim, self.half_dim, kernel_size=1)
        
        # Swin Transformer branch: 2 blocks (W-MSA, then SW-MSA)
        self.swin_block1 = SwinTransformerBlock(
            dim=self.half_dim, 
            num_heads=num_heads, 
            window_size=window_size,
            shift_size=0  # W-MSA
        )
        self.swin_block2 = SwinTransformerBlock(
            dim=self.half_dim, 
            num_heads=num_heads, 
            window_size=window_size,
            shift_size=window_size // 2  # SW-MSA
        )
        
        # CNN branch: simple conv + maxpool
        self.cnn_conv = nn.Conv2d(self.half_dim, self.half_dim, kernel_size=3, padding=1)
        self.cnn_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H//2, W//2) - concatenated and downsampled
        """
        # Split into two branches
        x1 = self.split_conv1(x)  # Swin branch
        x2 = self.split_conv2(x)  # CNN branch
        
        # Swin branch: apply 2 Swin blocks
        x1 = self.swin_block1(x1)  # W-MSA
        x1 = self.swin_block2(x1)  # SW-MSA
        # x1 shape: (B, half_dim, H, W)
        
        # CNN branch: conv + maxpool
        x2 = self.cnn_conv(x2)
        x2 = self.cnn_pool(x2)
        # x2 shape: (B, half_dim, H//2, W//2)
        
        # Downsample Swin output to match CNN resolution
        x1_down = F.adaptive_avg_pool2d(x1, output_size=x2.shape[-2:])
        
        # Concatenate along channel dimension
        out = torch.cat([x1_down, x2], dim=1)  # (B, in_dim, H//2, W//2)
        
        return out


# ========================================
# 5. PATCH MERGING (Downsampling between stages)
# ========================================
class PatchMerging(nn.Module):
    """
    Downsample feature map by 2x and increase channels by 2x.
    Similar to Swin Transformer's patch merging.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.reduction = nn.Linear(4 * in_dim, 2 * in_dim, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, 2C, H//2, W//2)
        """
        B, C, H, W = x.shape
        
        # Ensure H and W are even
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
            H, W = x.shape[2], x.shape[3]
        
        # Reshape and concatenate 2x2 patches
        x0 = x[:, :, 0::2, 0::2]  # (B, C, H//2, W//2)
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # (B, 4C, H//2, W//2)
        
        # Permute for linear layer
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H//2, W//2, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H//2, W//2, 2C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 2C, H//2, W//2)
        
        return x


# ========================================
# 6. FEATURE FUSION MODULE
# ========================================
class FeatureFusion(nn.Module):
    """
    Upsample final feature map and reduce channels.
    7x7x192 -> 56x56x192 -> 56x56x96
    """
    def __init__(self, in_dim=192, out_dim=96, target_size=56):
        super().__init__()
        self.target_size = target_size
        self.conv_reduce = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: (B, 192, 7, 7)
        Returns:
            x: (B, 96, 56, 56)
        """
        # Upsample using bilinear interpolation
        x = F.interpolate(x, size=(self.target_size, self.target_size), 
                         mode='bilinear', align_corners=False)
        # Reduce channels
        x = self.conv_reduce(x)
        return x


# ========================================
# 7. YOLO DETECTION HEAD
# ========================================
class YOLODetectionHead(nn.Module):
    """
    YOLO-style detection head.
    Converts 56x56x96 -> 56x56x21 (for 2 classes: num_classes * 3 + 15)
    Output format per anchor: [x, y, w, h, objectness, class1_prob, class2_prob]
    With 3 anchors: 3 * (4 + 1 + 2) = 21 channels
    """
    def __init__(self, in_dim=96, num_classes=2, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        out_channels = num_anchors * (5 + num_classes)  # 3 * (4+1+2) = 21
        
        self.conv = nn.Conv2d(in_dim, out_channels, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: (B, 96, 56, 56)
        Returns:
            pred_boxes: (B, N, 4) - normalized coordinates
            pred_logits: (B, N, num_classes)
            pred_obj: (B, N) - objectness scores
        """
        B = x.size(0)
        out = self.conv(x)  # (B, 21, 56, 56)
        
        # Reshape: (B, num_anchors, 5+num_classes, H, W) -> (B, num_anchors, H, W, 5+num_classes)
        out = out.view(B, self.num_anchors, 5 + self.num_classes, 56, 56)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # (B, 3, 56, 56, 7)
        
        # Flatten spatial dimensions: (B, 3*56*56, 7)
        out = out.view(B, -1, 5 + self.num_classes)
        
        # Split into boxes, objectness, and class logits
        pred_boxes_raw = out[..., :4]  # (B, N, 4)
        pred_obj = out[..., 4]  # (B, N)
        pred_logits = out[..., 5:]  # (B, N, num_classes)
        
        # Normalize box coordinates to [0, 1]
        pred_boxes_raw = torch.sigmoid(pred_boxes_raw)
        
        cx = pred_boxes_raw[..., 0]
        cy = pred_boxes_raw[..., 1]
        w = pred_boxes_raw[..., 2]
        h = pred_boxes_raw[..., 3]
        
        x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * h).clamp(0.0, 1.0)
        
        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        
        return pred_boxes, pred_logits, pred_obj


# ========================================
# 8. COMPLETE LDBST ARCHITECTURE
# ========================================
class LightweightDualBranchSwinTransformer(nn.Module):
    """
    Complete Lightweight Dual-Branch Swin Transformer for Microplastic Detection.
    
    Architecture:
    Input (224x224x3)
      -> Patch Embedding (56x56x96)
      -> Stage 1: Dual-Branch (28x28x96)
      -> Patch Merging (14x14x192)
      -> Stage 2: Dual-Branch (7x7x192)
      -> Feature Fusion (56x56x96)
      -> YOLO Detection Head
    """
    def __init__(self, num_classes=2, img_size=224):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Patch Embedding: 224x224x3 -> 56x56x96
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=4, embed_dim=96)
        
        # 2. Stage 1: Dual-Branch Processing (56x56x96 -> 28x28x96)
        self.stage1 = DualBranchStage(in_dim=96, window_size=7, num_heads=4)
        
        # 3. Patch Merging: 28x28x96 -> 14x14x192
        self.patch_merge = PatchMerging(in_dim=96)
        
        # 4. Stage 2: Dual-Branch Processing (14x14x192 -> 7x7x192)
        self.stage2 = DualBranchStage(in_dim=192, window_size=7, num_heads=4)
        
        # 5. Feature Fusion: 7x7x192 -> 56x56x96
        self.fusion = FeatureFusion(in_dim=192, out_dim=96, target_size=56)
        
        # 6. YOLO Detection Head
        self.detection_head = YOLODetectionHead(in_dim=96, num_classes=num_classes)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            pred_boxes: (B, N, 4)
            pred_logits: (B, N, num_classes)
            pred_obj: (B, N)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, 96, 56, 56)
        
        # Stage 1
        x = self.stage1(x)  # (B, 96, 28, 28)
        
        # Patch merging
        x = self.patch_merge(x)  # (B, 192, 14, 14)
        
        # Stage 2
        x = self.stage2(x)  # (B, 192, 7, 7)
        
        # Feature fusion
        x = self.fusion(x)  # (B, 96, 56, 56)
        
        # Detection head
        pred_boxes, pred_logits, pred_obj = self.detection_head(x)
        
        return pred_boxes, pred_logits, pred_obj
    
    @torch.no_grad()
    def predict(self, x, conf_thresh=0.45, iou_thresh=0.45):
        """Inference with NMS post-processing."""
        self.eval()
        pred_boxes, pred_logits, pred_obj = self.forward(x)
        
        results = []
        for b in range(x.size(0)):
            dets = postprocess_detections(
                pred_boxes[b], pred_logits[b], pred_obj[b],
                conf_thresh=conf_thresh, iou_thresh=iou_thresh
            )
            results.append(dets.cpu())
        return results


# ========================================
# 9. DETECTION LOSS
# ========================================
class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=5.0, lambda_cls=1.0, lambda_obj=0.25,
                 pos_iou_thresh=0.4, neg_iou_thresh=0.2):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred_boxes, pred_logits, pred_obj, targets):
        device = pred_boxes.device
        total_obj_loss = 0.0
        total_cls_loss = 0.0
        total_box_loss = 0.0
        n_images = len(targets)

        for i in range(n_images):
            pb, pl, po = pred_boxes[i], pred_logits[i], pred_obj[i]
            tgt = targets[i]
            gt_boxes = tgt.get("boxes", torch.empty((0,4), device=device))
            gt_labels = tgt.get("labels", torch.empty((0,), dtype=torch.long, device=device))

            N = pb.size(0)
            obj_target = torch.zeros((N,), device=device)

            if gt_boxes.numel() == 0:
                total_obj_loss += self.bce(po, obj_target)
                continue

            ious = box_iou_custom(pb, gt_boxes)
            best_iou, best_gt = ious.max(dim=1)
            pos_mask = best_iou > self.pos_iou_thresh

            obj_target[pos_mask] = 1.0

            # Objectness loss
            pos_weight = torch.tensor(3.0, device=device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            total_obj_loss += bce(po, obj_target)

            if pos_mask.any():
                pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                matched_gt_boxes = gt_boxes[best_gt[pos_idx]]
                matched_gt_labels = gt_labels[best_gt[pos_idx]]

                # Box loss
                total_box_loss += giou_loss(pb[pos_idx], matched_gt_boxes, reduction='mean')

                # Classification loss
                total_cls_loss += self.ce(pl[pos_idx], matched_gt_labels)

        total_loss =
        
        