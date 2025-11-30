from ultralytics import YOLO
#FOR TRAINING YOLOV8 CLASSIFICATION MODEL
def main():
    # Load YOLOv8 classification model
    model = YOLO("yolov8n-cls.pt")

    # Train configuration
    model.train(
        data=r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC",
        epochs=10,
        imgsz=224,
        batch=8,
        name="yolov8n_classification_microplastics_algae",
        pretrained=True,
        verbose=True  
    )

    # Evaluate performance
    metrics = model.val()
    print(metrics)

    # Check metrics
    metrics = model.val()
    print(f"\n=== Validation Results ===")
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")

if __name__ == "__main__":
    main()
