import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


# -----------------------------
# IoU + GIoU Loss Functions
# -----------------------------
def compute_box_iou(boxes1, boxes2):
    """Compute IoU between two sets of boxes [N,4] and [M,4]."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((len(boxes1), len(boxes2)), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
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


# -----------------------------
# Detection Loss
# -----------------------------
class DetectionLoss(nn.Module):
    """Detection loss with box regression, objectness, and classification."""
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
        """
        Args:
            pred_boxes: (B, N, 4) - predicted bounding boxes in [0,1]
            pred_logits: (B, N, num_classes) - class logits
            pred_obj: (B, N) - objectness logits
            targets: list of dicts with 'boxes' and 'labels'
        """
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
            cls_loss = 0.0
            box_loss = 0.0

            if gt_boxes.numel() == 0:
                total_obj_loss += self.bce(po, obj_target)
                continue

            ious = compute_box_iou(pb, gt_boxes)
            best_iou, best_gt = ious.max(dim=1)
            pos_mask = best_iou > self.pos_iou_thresh
            neg_mask = best_iou < self.neg_iou_thresh

            obj_target[pos_mask] = 1.0

            # Objectness loss with positive weight
            pos_weight = torch.tensor(3.0, device=device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            total_obj_loss += bce(po, obj_target)

            if pos_mask.any():
                pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                matched_gt_boxes = gt_boxes[best_gt[pos_idx]]
                matched_gt_labels = gt_labels[best_gt[pos_idx]]

                # Box loss (GIoU)
                total_box_loss += giou_loss(pb[pos_idx], matched_gt_boxes, reduction='mean')

                # Classification loss
                total_cls_loss += self.ce(pl[pos_idx], matched_gt_labels)

        total_loss = (
            self.lambda_box * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_cls * total_cls_loss
        ) / max(1, n_images)

        return torch.nan_to_num(total_loss)


# -----------------------------
# NMS and Postprocessing
# -----------------------------
@torch.no_grad()
def postprocess_detections(pred_boxes, pred_logits, pred_obj,
                           conf_thresh=0.45, iou_thresh=0.45, img_size=224):
    """
    Apply NMS and postprocess detections.
    
    Args:
        pred_boxes: (N, 4) - predicted boxes in [0,1]
        pred_logits: (N, num_classes) - class logits
        pred_obj: (N,) - objectness logits
        conf_thresh: confidence threshold
        iou_thresh: NMS IoU threshold
        img_size: image size for scaling boxes
        
    Returns:
        Tensor [K, 6] with columns: x1, y1, x2, y2, conf, label
    """
    # Objectness probability
    obj_probs = torch.sigmoid(pred_obj)
    
    # Class probabilities
    class_probs = F.softmax(pred_logits, dim=-1)
    scores, labels = class_probs.max(dim=-1)
    
    # Final confidence = objectness * class probability
    final_scores = scores * obj_probs
    
    # Filter by confidence
    keep_mask = final_scores > conf_thresh
    if keep_mask.sum() == 0:
        return torch.empty((0, 6), device=pred_boxes.device)
    
    boxes = pred_boxes[keep_mask].clamp(0, 1) * img_size
    scores = final_scores[keep_mask]
    labels = labels[keep_mask]
    
    # Class-wise NMS
    kept_indices = []
    for c in labels.unique():
        idxs = torch.nonzero(labels == c).squeeze(1)
        if len(idxs) == 0:
            continue
        c_boxes = boxes[idxs]
        c_scores = scores[idxs]
        keep = torchvision.ops.nms(c_boxes, c_scores, iou_thresh)
        kept_indices.append(idxs[keep])
    
    if len(kept_indices) == 0:
        return torch.empty((0, 6), device=pred_boxes.device)
    
    kept_indices = torch.cat(kept_indices, dim=0)
    boxes, scores, labels = boxes[kept_indices], scores[kept_indices], labels[kept_indices]
    
    return torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)


class PatchEmbedding(nn.Module):
    """Patch embedding layer: 224x224x3 -> 56x56x96"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, 3, 224, 224) -> (B, 96, 56, 56)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self Attention"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with W-MSA or SW-MSA"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging(nn.Module):
    """Patch Merging layer: downsamples and increases channels"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        # Downsample by 2x2
        x = F.avg_pool2d(x.permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class DualBranchStage(nn.Module):
    """Dual-branch stage with Swin Transformer and CNN branches"""
    def __init__(self, dim, num_heads=4, window_size=7):
        super().__init__()
        self.dim = dim
        branch_dim = dim // 2
        
        # Channel splitting
        self.split_conv = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Swin Transformer branch
        self.swin_block1 = SwinTransformerBlock(branch_dim, num_heads, window_size, shift_size=0)
        self.swin_block2 = SwinTransformerBlock(branch_dim, num_heads, window_size, shift_size=window_size//2)
        self.swin_downsample = PatchMerging(branch_dim)
        
        # CNN branch
        self.cnn_conv = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1)
        self.cnn_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn_bn = nn.BatchNorm2d(branch_dim)
        self.cnn_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Split channels
        x = self.split_conv(x)
        swin_input = x[:, :C//2, :, :]
        cnn_input = x[:, C//2:, :, :]
        
        # Swin branch: (B, C/2, H, W) -> (B, H, W, C/2)
        swin_x = swin_input.permute(0, 2, 3, 1)
        swin_x = self.swin_block1(swin_x)
        swin_x = self.swin_block2(swin_x)
        swin_x = self.swin_downsample(swin_x)  # (B, H/2, W/2, C/2)
        swin_x = swin_x.permute(0, 3, 1, 2)  # (B, C/2, H/2, W/2)
        
        # CNN branch: (B, C/2, H, W) -> (B, C/2, H/2, W/2)
        cnn_x = self.cnn_conv(cnn_input)
        cnn_x = self.cnn_bn(cnn_x)
        cnn_x = self.cnn_relu(cnn_x)
        cnn_x = self.cnn_pool(cnn_x)
        
        # Concatenate branches
        fused = torch.cat([swin_x, cnn_x], dim=1)  # (B, C, H/2, W/2)
        
        return fused


class LDBST_MP(nn.Module):
    """Lightweight Dual-Branch Swin Transformer for Microplastic Detection and Classification"""
    def __init__(self, img_size=224, num_classes=5, num_heads=4, task="detection"):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.task = task
        
        # Patch Embedding: 224x224x3 -> 56x56x96
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=4, in_chans=3, embed_dim=96)
        
        # Stage 1: 56x56x96 -> 28x28x96
        self.stage1 = DualBranchStage(dim=96, num_heads=num_heads, window_size=7)
        
        # Patch merging between stages: 28x28x96 -> 14x14x192
        self.patch_merge_s1 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192)
        )
        
        # Stage 2: 14x14x192 -> 7x7x192
        self.stage2 = DualBranchStage(dim=192, num_heads=num_heads, window_size=7)
        
        if task == "classification":
            # Classification head: global average pooling + MLP
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(192, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),  # ← Stronger dropout
                nn.Linear(256, num_classes)
            )
        elif task == "detection":
            # Feature fusion and upsampling: 7x7x192 -> 56x56x96
            self.upsample = nn.Sequential(
                nn.Upsample(size=(56, 56), mode='bilinear', align_corners=False),
                nn.Conv2d(192, 96, kernel_size=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True)
            )
            
            # YOLO-style detection head: 56x56x96 -> 56x56x(4+1+num_classes)
            out_channels = 4 + 1 + num_classes  # bbox + objectness + classes
            self.detection_head = nn.Conv2d(96, out_channels, kernel_size=1)
        else:
            raise ValueError(f"task must be 'classification' or 'detection', got {task}")
        
    def forward(self, x):
        """
        Classification mode returns: (B, num_classes) logits
        Detection mode returns: pred_boxes (B, N, 4), pred_logits (B, N, num_classes), pred_obj (B, N)
        """
        # x: (B, 3, 224, 224)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 96, 56, 56)
        
        # Stage 1
        x = self.stage1(x)  # (B, 96, 28, 28)
        
        # Merge for stage 2
        x = self.patch_merge_s1(x)  # (B, 192, 14, 14)
        
        # Stage 2
        x = self.stage2(x)  # (B, 192, 7, 7)
        
        if self.task == "classification":
            # Global pooling and classification
            x = self.pool(x).flatten(1)  # (B, 192)
            logits = self.classifier(x)  # (B, num_classes)
            return logits
            
        elif self.task == "detection":
            # Upsample and reduce channels
            x = self.upsample(x)  # (B, 96, 56, 56)
            
            # Detection head
            detections = self.detection_head(x)  # (B, out_channels, 56, 56)
            
            # Parse output into components
            B, C, H, W = detections.shape
            detections = detections.permute(0, 2, 3, 1).reshape(B, H*W, C)  # (B, N, C)
            
            # Split into bbox, objectness, and class logits
            pred_boxes_raw = detections[..., :4]  # (B, N, 4)
            pred_obj = detections[..., 4]  # (B, N)
            pred_logits = detections[..., 5:]  # (B, N, num_classes)
            
            # Convert box predictions to normalized coordinates [0,1]
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
    
    @torch.no_grad()
    def predict(self, x, conf_thresh=0.45, iou_thresh=0.45):
        """
        Run inference.
        
        Classification mode: Returns class probabilities (B, num_classes)
        Detection mode: Returns list of detection tensors [K, 6] for each image
        """
        self.eval()
        
        if self.task == "classification":
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)
            
        elif self.task == "detection":
            pred_boxes, pred_logits, pred_obj = self.forward(x)
            
            results = []
            for b in range(x.size(0)):
                dets = postprocess_detections(
                    pred_boxes[b], 
                    pred_logits[b], 
                    pred_obj[b],
                    conf_thresh=conf_thresh, 
                    iou_thresh=iou_thresh,
                    img_size=self.img_size
                )
                results.append(dets.cpu())
            return results
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("LDBST-MP MODEL EXAMPLES")
    print("=" * 70)
    
    # -------------------------
    # CLASSIFICATION MODE
    # -------------------------
    print("\n[1] CLASSIFICATION MODE")
    print("-" * 70)
    
    model_cls = LDBST_MP(img_size=224, num_classes=2, num_heads=4, task="classification")
    print(f"Total parameters: {model_cls.count_parameters():,}")
    
    x = torch.randn(2, 3, 224, 224)
    logits = model_cls(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}  # (B, num_classes)")
    
    # Inference
    probs = model_cls.predict(x)
    print(f"Class probabilities: {probs.shape}")
    print(f"Example: Image 1 -> Class 0: {probs[0, 0]:.3f}, Class 1: {probs[0, 1]:.3f}")
    
    # -------------------------
    # DETECTION MODE
    # -------------------------
    print("\n[2] DETECTION MODE")
    print("-" * 70)
    
    model_det = LDBST_MP(img_size=224, num_classes=5, num_heads=4, task="detection")
    print(f"Total parameters: {model_det.count_parameters():,}")
    
    x = torch.randn(2, 3, 224, 224)
    pred_boxes, pred_logits, pred_obj = model_det(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predicted boxes shape: {pred_boxes.shape}  # (B, N, 4)")
    print(f"Predicted logits shape: {pred_logits.shape}  # (B, N, num_classes)")
    print(f"Predicted objectness shape: {pred_obj.shape}  # (B, N)")
    
    # Test loss computation
    print("\n[3] LOSS COMPUTATION (Detection)")
    print("-" * 70)
    
    targets = [
        {
            "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]),
            "labels": torch.tensor([0, 1])
        },
        {
            "boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]]),
            "labels": torch.tensor([2])
        }
    ]
    
    criterion = DetectionLoss(
        lambda_box=5.0,
        lambda_cls=1.0,
        lambda_obj=0.25,
        pos_iou_thresh=0.4,
        neg_iou_thresh=0.2
    )
    
    loss = criterion(pred_boxes, pred_logits, pred_obj, targets)
    print(f"Detection loss: {loss.item():.4f}")
    
    # Inference with NMS
    print("\n[4] INFERENCE WITH NMS (Detection)")
    print("-" * 70)
    
    detections = model_det.predict(x, conf_thresh=0.25, iou_thresh=0.45)
    
    for i, dets in enumerate(detections):
        print(f"Image {i+1}: {len(dets)} detections")
        if len(dets) > 0:
            print(f"  Format: [x1, y1, x2, y2, confidence, class_label]")
            for j, det in enumerate(dets[:3]):
                x1, y1, x2, y2, conf, label = det
                print(f"  Detection {j+1}: bbox=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}], "
                      f"conf={conf:.3f}, class={int(label)}")
    
    # Training example
    print("\n[5] TRAINING WORKFLOW")
    print("-" * 70)
    print("""
# STEP 1: Train classification model (if you have image-level labels)
python train2.py --task classification \\
    --data ./data_no_bbox \\
    --num_classes 2 \\
    --epochs 30 \\
    --batch_size 32 \\
    --auto_split

# STEP 2: Train detection model (with bounding box annotations)
python train2.py --task detection \\
    --num_classes 5 \\
    --epochs 50 \\
    --batch_size 16 \\
    --pretrained checkpoints/ldbst_classifier_best_epoch30.pth

# Or train detection only (if you only have bbox data)
python train2.py --task detection \\
    --num_classes 5 \\
    --epochs 50 \\
    --batch_size 32
    """)