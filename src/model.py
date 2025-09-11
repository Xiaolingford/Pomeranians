import torch
import torch.nn as nn
import timm

class DualBranchSwinTinyClassifier(nn.Module):
    """
    Two inputs:
      - x_std: standard letterboxed RGB in [0,1], (B,3,224,224)
      - x_clahe: CLAHE-boosted version, same shape
    Each branch is a small Swin-T (no pretrained needed). Outputs are concatenated, then a small MLP head.
    """
    def __init__(self, num_classes: int = 2, freeze_backbone: bool = False, try_pretrained: bool = False):
        super().__init__()
        # Avoid internet weight downloads; allow opt-in via try_pretrained=True
        def make_swin(pretrained: bool):
            return timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                num_classes=0  # returns features (B, C)
            )

        if try_pretrained:
            try:
                self.branchA = make_swin(pretrained=True)
                self.branchB = make_swin(pretrained=True)
            except Exception:
                self.branchA = make_swin(pretrained=False)
                self.branchB = make_swin(pretrained=False)
        else:
            self.branchA = make_swin(pretrained=False)
            self.branchB = make_swin(pretrained=False)

        if freeze_backbone:
            for p in list(self.branchA.parameters()) + list(self.branchB.parameters()):
                p.requires_grad = False

        feat_dim = 768  # swin_tiny output dim
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim*2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_std: torch.Tensor, x_clahe: torch.Tensor) -> torch.Tensor:
        fA = self.branchA(x_std)   # (B,768)
        fB = self.branchB(x_clahe) # (B,768)
        f = torch.cat([fA, fB], dim=1)
        return self.fuse(f)
