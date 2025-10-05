import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from timm import create_model
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_iou, generalized_box_iou


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
# Detection Loss (GIoU + CE)
# -----------------------------
class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=1.0, lambda_cls=1.0, iou_threshold=0.5):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.iou_threshold = iou_threshold

    def forward(self, pred_boxes, pred_logits, targets):
        device = pred_boxes.device
        total_box_loss, total_cls_loss = 0.0, 0.0
        num_valid = 0

        for i, tgt in enumerate(targets):
            if len(tgt["boxes"]) == 0:
                continue

            gt_boxes = tgt["boxes"].to(device)
            gt_labels = tgt["labels"].to(device)
            pb = pred_boxes[i]
            pl = pred_logits[i]

            ious = box_iou(pb, gt_boxes)
            matched_pred_idx = ious.argmax(dim=0)  # best pred per GT
            matched_gt_idx = torch.arange(len(gt_boxes), device=device)

            if len(matched_pred_idx) == 0:
                continue

            matched_preds = pb[matched_pred_idx]
            matched_logits = pl[matched_pred_idx]
            matched_gt_boxes = gt_boxes[matched_gt_idx]
            matched_gt_labels = gt_labels[matched_gt_idx]

            iou_weights = ious[matched_pred_idx, matched_gt_idx].detach()
            iou_weights = iou_weights / (iou_weights.sum() + 1e-6)

            box_loss = (giou_loss(matched_preds, matched_gt_boxes, reduction='none') * iou_weights).sum()
            cls_loss = (F.cross_entropy(matched_logits, matched_gt_labels, reduction='none') * iou_weights).sum()

            total_box_loss += box_loss
            total_cls_loss += cls_loss
            num_valid += 1

        if num_valid == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = (self.lambda_box * total_box_loss + self.lambda_cls * total_cls_loss) / max(num_valid, 1)
        return torch.nan_to_num(loss)



# -----------------------------
# NMS + Postprocessing
# -----------------------------
@torch.no_grad()
def postprocess_detections(pred_boxes, pred_logits, conf_thresh=0.25, iou_thresh=0.45, img_size=224):
    probs = F.softmax(pred_logits, dim=-1)
    confs, labels = probs.max(dim=-1)

    mask = confs > conf_thresh
    boxes = pred_boxes[mask]
    confs = confs[mask]
    labels = labels[mask]

    if boxes.numel() == 0:
        return torch.empty((0, 6), device=pred_boxes.device)

    # Scale to image coordinates
    boxes = boxes.clamp(0, 1) * img_size

    # 🔹 Class-wise NMS (lightweight version)
    keep = []
    for c in labels.unique():
        mask_c = labels == c
        keep_c = torchvision.ops.nms(boxes[mask_c], confs[mask_c], iou_thresh)
        keep.append(torch.nonzero(mask_c)[keep_c])
    keep = torch.cat(keep).squeeze(1)

    boxes, confs, labels = boxes[keep], confs[keep], labels[keep]
    return torch.cat([boxes, confs.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)


# -----------------------------
# Detection Head
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
        self.box_pred = nn.Linear(in_channels, 4)
        self.cls_pred = nn.Linear(in_channels, num_classes)

    def forward(self, feat):
        feat = self.neck(feat)
        B, C, H, W = feat.shape
        x = feat.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        if x.size(1) > self.num_preds:
            x = x[:, :self.num_preds, :]
        pred_boxes = torch.sigmoid(self.box_pred(x)).clamp(0, 1)
        pred_logits = self.cls_pred(x)
        return pred_boxes, pred_logits


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
            s = self._swin_feats(x)
            return self.det_head(s)

    @torch.no_grad()
    def predict(self, x, conf_thresh=0.25, iou_thresh=0.45):
        self.eval()
        pred_boxes, pred_logits = self.forward(x)
        results = []
        for b in range(x.size(0)):
            dets = postprocess_detections(pred_boxes[b], pred_logits[b],
                                          conf_thresh=conf_thresh,
                                          iou_thresh=iou_thresh)
            results.append(dets.cpu())
        return results


# -----------------------------
# Test block
# -----------------------------
if __name__ == "__main__":
    det_model = DualBranchSwinCNNDetector(num_classes=2, task="detection")
    x = torch.randn(2, 3, 224, 224)
    boxes, logits = det_model(x)

    targets = [
        {"boxes": torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]]), "labels": torch.tensor([0, 1])},
        {"boxes": torch.tensor([[0.2, 0.2, 0.4, 0.4]]), "labels": torch.tensor([1])}
    ]

    criterion = DetectionLoss()
    loss = criterion(boxes, logits, targets)
    print("Detection loss:", loss.item())
