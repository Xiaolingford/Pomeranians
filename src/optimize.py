from pathlib import Path
import torch
import torch.nn as nn
from model import DualBranchSwinTinyClassifier

ROOT = Path(__file__).resolve().parents[1]
CKPT_IN  = ROOT / "classifier.pth"
CKPT_OUT = ROOT / "classifier_optimized.pt"

def main():
    model = DualBranchSwinTinyClassifier(num_classes=2, freeze_backbone=False)
    if CKPT_IN.exists():
        model.load_state_dict(torch.load(CKPT_IN, map_location="cpu"))
        print(f"Loaded {CKPT_IN}")
    else:
        print("WARNING: classifier.pth not found; using random weights")

    model.eval()
    # Dynamic quantization for linear layers only (safe on CPU)
    q_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save(q_model.state_dict(), CKPT_OUT)
    print(f"Saved optimized (quantized) weights to {CKPT_OUT}")

if __name__ == "__main__":
    main()
