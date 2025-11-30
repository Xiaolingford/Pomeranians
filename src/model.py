import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from timm import create_model
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou, generalized_box_iou


CONF_THRESH = 0.45
NMS_IOU = 0.45

# -----------------------------
# IoU + GIoU
# -----------------------------
def box_iou(boxes1, boxes2):
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
# Detection Loss (ORIGINAL 74% SETTINGS)
# -----------------------------
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
            cls_loss = 0.0
            box_loss = 0.0

            if gt_boxes.numel() == 0:
                total_obj_loss += self.bce(po, obj_target)
                continue

            ious = box_iou(pb, gt_boxes)
            best_iou, best_gt = ious.max(dim=1)
            pos_mask = best_iou > self.pos_iou_thresh
            neg_mask = best_iou < self.neg_iou_thresh

            obj_target[pos_mask] = 1.0

            # Objectness loss (weighted)
            pos_weight = torch.tensor(3.0, device=device)
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            total_obj_loss += bce(po, obj_target)

            if pos_mask.any():
                pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                matched_gt_boxes = gt_boxes[best_gt[pos_idx]]
                matched_gt_labels = gt_labels[best_gt[pos_idx]]

                # Box loss
                total_box_loss += giou_loss(pb[pos_idx], matched_gt_boxes, reduction='mean')

                # Classification
                total_cls_loss += self.ce(pl[pos_idx], matched_gt_labels)

        total_loss = (
            self.lambda_box * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_cls * total_cls_loss
        ) / max(1, n_images)

        return torch.nan_to_num(total_loss)


# -----------------------------
# NMS + Postprocessing
# -----------------------------
@torch.no_grad()
def postprocess_detections(pred_boxes, pred_logits, pred_obj,
                           conf_thresh=CONF_THRESH, iou_thresh=0.45, img_size=224):
    """
    pred_boxes: (N,4), pred_logits: (N,num_classes), pred_obj: (N,)
    returns: Tensor [K, 6] -> x1,y1,x2,y2,conf,label
    """
    # objectness -> prob
    obj_probs = torch.sigmoid(pred_obj)  # [N]

    # class probs (per-class) but multiply by objectness to get final score
    class_probs = F.softmax(pred_logits, dim=-1)  # [N, C]
    scores, labels = class_probs.max(dim=-1)      # per-pred best class score
    final_scores = scores * obj_probs

    keep_mask = final_scores > conf_thresh
    if keep_mask.sum() == 0:
        return torch.empty((0, 6), device=pred_boxes.device)

    boxes = pred_boxes[keep_mask].clamp(0, 1) * img_size
    scores = final_scores[keep_mask]
    labels = labels[keep_mask]

    # class-wise NMS
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


# -----------------------------
# Detection Head (ORIGINAL SIMPLE VERSION)
# -----------------------------
class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_preds=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_preds = num_preds
        self.neck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Simple 1-layer heads (ORIGINAL)
        self.box_pred = nn.Linear(in_channels, 4)
        self.cls_pred = nn.Linear(in_channels, num_classes)
        self.obj_pred = nn.Linear(in_channels, 1)

    def forward(self, feat):
        feat = self.neck(feat)
        B, C, H, W = feat.shape
        x = feat.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        # Limit to num_preds for stability
        if x.size(1) > self.num_preds:
            x = x[:, :self.num_preds, :]

        # Separate prediction branches
        pred_boxes_raw = self.box_pred(x)          # (B, N, 4)
        pred_logits = self.cls_pred(x)             # (B, N, num_classes)
        pred_obj = self.obj_pred(x).squeeze(-1)    # (B, N)

        # Normalize box coordinates (0..1)
        pred_boxes_raw = torch.sigmoid(pred_boxes_raw)

        cx = pred_boxes_raw[..., 0]
        cy = pred_boxes_raw[..., 1]
        w  = pred_boxes_raw[..., 2]
        h  = pred_boxes_raw[..., 3]

        x1 = (cx - 0.5 * w).clamp(0.0, 1.0)
        y1 = (cy - 0.5 * h).clamp(0.0, 1.0)
        x2 = (cx + 0.5 * w).clamp(0.0, 1.0)
        y2 = (cy + 0.5 * h).clamp(0.0, 1.0)

        pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        return pred_boxes, pred_logits, pred_obj


# -----------------------------
# Dual-Branch Detector
# -----------------------------
class DualBranchSwinCNNDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, task="classification"):
        super().__init__()
        self.task = task

        # CNN branch
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.cnn.fc = nn.Identity()

        # Swin branch
        self.swin = create_model("swin_tiny_patch4_window7_224", pretrained=pretrained)
        self.swin.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Determine feature dims
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            c = self.cnn(dummy)
            s = self._swin_feats(dummy)
        fused_dim = c.shape[1] + s.shape[1]

        if task == "classification":
            self.head = nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        elif task == "detection":
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(s.shape[1] + c.shape[1], s.shape[1], 1),
                nn.BatchNorm2d(s.shape[1]),
                nn.ReLU(inplace=True)
            )
            self.det_head = DetectionHead(in_channels=s.shape[1], num_classes=num_classes)
        else:
            raise ValueError("task must be 'classification' or 'detection'")

    def _swin_feats(self, x):
        f = self.swin.forward_features(x)
        if f.ndim == 3:
            side = int(f.size(1) ** 0.5)
            f = f.transpose(1, 2).reshape(f.size(0), f.size(2), side, side)
        return f

    def forward(self, x):
        if self.task == "classification":
            c = self.cnn(x)
            s = self.pool(self._swin_feats(x)).flatten(1)
            return self.head(torch.cat([c, s], dim=1))
        elif self.task == "detection":
            # CNN branch (get spatial features)
            c = self.cnn.conv1(x)
            c = self.cnn.bn1(c)
            c = self.cnn.relu(c)
            c = self.cnn.layer1(c)
            c = self.cnn.layer2(c)
            c = self.cnn.layer3(c)
            c = self.cnn.layer4(c)

            # Swin branch (patch-based features)
            s = self._swin_feats(x)

            # Match feature dims
            if c.shape[-2:] != s.shape[-2:]:
                s = F.adaptive_avg_pool2d(s, output_size=c.shape[-2:])

            # Fuse both features
            fused = torch.cat([c, s], dim=1)
            fused = self.fusion_conv(fused)

            # Detection head
            pred_boxes, pred_logits, pred_obj = self.det_head(fused)
            return pred_boxes, pred_logits, pred_obj

    @torch.no_grad()
    def predict(self, x, conf_thresh=0.25, iou_thresh=0.45):
        self.eval()
        pred_boxes, pred_logits, pred_obj = self.forward(x)

        results = []
        for b in range(x.size(0)):
            dets = postprocess_detections(pred_boxes[b], pred_logits[b], pred_obj[b],
                              conf_thresh=conf_thresh, iou_thresh=iou_thresh)
            results.append(dets.cpu())
        return results


# -----------------------------
# Test block
# -----------------------------
if __name__ == "__main__":
    det_model = DualBranchSwinCNNDetector(num_classes=2, task="detection")
    x = torch.randn(2, 3, 224, 224)
    boxes, logits, obj = det_model(x)

    targets = [
        {"boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]), "labels": torch.tensor([0, 1])},
        {"boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]]), "labels": torch.tensor([1])}
    ]

    criterion = DetectionLoss()
    loss = criterion(boxes, logits, obj, targets)

    print("Detection loss:", loss.item())