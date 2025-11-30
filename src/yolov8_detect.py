from ultralytics import YOLO
#FOR TRAINING YOLOV8 DETECTION MODEL
def main():
    # Load YOLOv8n (nano) detection model
    model = YOLO("yolov8n.pt")  # You can later try yolov8s.pt or yolov8m.pt for larger models

    # Train configuration
    model.train(
        data=r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataD\data.yaml",  
        epochs=30,
        imgsz=224,         
        batch=8,
        name="yolov8n_detection_microplastics_algae",
        pretrained=True
    )

    # Evaluate model on validation set
    metrics = model.val()
    print(metrics)

    # Benchmark performance (inference speed, FLOPs, params, etc.)
    model.benchmark(data=r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataD\data.yaml")

if __name__ == "__main__":
    main()
