from pathlib import Path
import torch
import torch.nn as nn
from model import DualBranchSwinTinyClassifier

def export_torchscript_traced(model, out_path: Path):
    model.eval()
    ex_std = torch.randn(1,3,224,224)
    ex_cla = torch.randn(1,3,224,224)
    traced = torch.jit.trace(model, (ex_std, ex_cla))
    traced.save(str(out_path))
    print(f"TorchScript (traced) saved: {out_path}")

def export_onnx(model, onnx_path: Path):
    model.eval()
    ex_std = torch.randn(1,3,224,224)
    ex_cla = torch.randn(1,3,224,224)
    torch.onnx.export(
        model, (ex_std, ex_cla), str(onnx_path),
        input_names=["x_std","x_clahe"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes={"x_std":{0:"batch"}, "x_clahe":{0:"batch"}, "logits":{0:"batch"}}
    )
    print(f"ONNX (float) saved: {onnx_path}")

def main():
    ROOT = Path(__file__).resolve().parents[1]
    ckpt = ROOT / "classifier.pth"
    model = DualBranchSwinTinyClassifier(num_classes=2, freeze_backbone=False)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Loaded {ckpt}")
    else:
        print("WARNING: classifier.pth not found; exporting random weights")

    export_torchscript_traced(model, ROOT / "microplastic_classifier_scripted.pt")
    export_onnx(model, ROOT / "microplastic_classifier.onnx")

if __name__ == "__main__":
    main()
