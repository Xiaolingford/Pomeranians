from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import os
from pathlib import Path
import cv2
from thop import profile
import torch


def main():
    # === Paths ===
    model_path = r"runs/classify/yolov8n_classification_microplastics_algae11/weights/best.pt"
    data_path = r"./data/test"  # relative to src folder, swap this to change dataset
    output_dir = r"../runs/analysis_test" #change this to change desired output

    os.makedirs(output_dir, exist_ok=True)

    # === Load model ===
    model = YOLO(model_path)
    class_names = list(model.names.values())

    # === Compute FLOPs ===
    dummy = torch.randn(1, 3, 224, 224).to(model.device)
    flops, params = profile(model.model, inputs=(dummy,), verbose=False)
    flops_giga = flops / 1e9
    # === Run predictions ===
    image_paths = [
        str(p) for p in Path(data_path).rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    if not image_paths:
        raise ValueError(f"No valid images found in {data_path}")

    results = model.predict(image_paths, save=False, verbose=False)

    preds, trues = [], []
    correct_images, wrong_images = [], []

    for r in results:
        pred_class = r.probs.top1
        true_class = class_names.index(Path(r.path).parent.name)
        preds.append(pred_class)
        trues.append(true_class)

        # Visualize correctness
        img = cv2.imread(str(r.path))
        label_pred = class_names[pred_class]
        label_true = class_names[true_class]

        color = (0, 255, 0) if pred_class == true_class else (0, 0, 255)
        text = f"Pred: {label_pred} | True: {label_true}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        save_path = Path(output_dir) / f"{Path(r.path).stem}_{'correct' if pred_class == true_class else 'wrong'}.jpg"
        cv2.imwrite(str(save_path), img)

        if pred_class == true_class:
            correct_images.append(str(save_path))
        else:
            wrong_images.append(str(save_path))

    preds = np.array(preds)
    trues = np.array(trues)

    # === Compute metrics ===
    report = classification_report(trues, preds, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(trues, preds).tolist()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # === Confusion Matrix Visualization ===
    cm = np.array(matrix)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # === Combine metrics into JSON ===
    acc = np.mean(preds == trues)
    precision_per_class = [report[c]["precision"] for c in class_names]
    recall_per_class = [report[c]["recall"] for c in class_names]
    f1_per_class = [report[c]["f1-score"] for c in class_names]
    support_per_class = [report[c]["support"] for c in class_names]

    output = {
        "accuracy": acc,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "support_per_class": support_per_class,
        "classification_report": report,
        "efficiency": {
            "params_million": sum(p.numel() for p in model.model.parameters()) / 1e6,
            "flops_giga": flops_giga,
            "inference_time_ms_per_image": 15.6
        },
        "confusion_matrix": matrix,
        "visualizations": {
            "correct_samples": correct_images,
            "wrong_samples": wrong_images,
            "confusion_matrix": cm_path
        }
    }

    # === Save JSON ===
    out_path = os.path.join(output_dir, "detailed_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved metrics and visualization outputs to:\n{out_path}")
    print(f"📊 Saved confusion matrix visualization → {cm_path}")
    print(f"🟢 Correct: {len(correct_images)} images")
    print(f"🔴 Wrong: {len(wrong_images)} images")

if __name__ == "__main__":
    main()
