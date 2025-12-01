    # Microplastic Detection (Thesis)

    Lightweight **ResNet18 + Swin Transformer Tiny** hybrid classifier for **Algae vs Microplastics** detection.
    Pipeline: \*\*Balance → Preprocess → Train → Metrics → Detect → Export → Optimize.

    ---

    ## 📂 Project Directory

    microplastic-detection/
    │
    ├─ data/ # raw dataset (train/val split or unsplit)
    ├─ data_balanced/ # balanced dataset (after augment_balance.py)
    ├─ data_preprocessed/ # preprocessed dataset (resized & normalized)
    ├─ results/ # metrics reports, confusion matrix, detections
    ├─ src/ # source code
    │ ├─ augment_balance.py
    | | augmentations_util.py
    │ ├─ check.py
    │ ├─ detect.py
    │ ├─ dataset.py
    │ ├─ detect.py
    │ ├─ export.py
    │ ├─ metrics.py
    │ ├─ model.py
    │ ├─ optimize.py
    │ ├─ preprocess.py
    │ └─ train.py
    ├─ requirements.txt
    └─ README.md

    ---

    ## ⚙️ 1. Environment Setup (PowerShell)

    ```powershell

    Found 1140 images in data_yolo
    train: 798 images
    val: 228 images
    test: 114 images

    Found 2 classes: ['algae', 'microplastics']
    algae → train: 210, val: 60, test: 30
    microplastics → train: 588, val: 168, test: 84

    # Create virtual environment, if you already have skip and activate venv
    python -m venv .venv

    # Activate venv
    .\.venv\Scripts\Activate.ps1

    # Upgrade pip
    pip install --upgrade pip

    # Install dependencies
    pip install -r requirements.txtp

    #If you have Cuda nvida gputhen
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    #install for modle profiling
    pip install thop fvcore

    # Fix execution policy if activation fails
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

    #For testing the bounding boxes with the raw images or change data_yolo to data_yolo_balanced to test augmented set etc
    .venv\Scripts\python.exe src\visual_check.py --input data_yolo_preprocessed --num 5

    #split yolo dataset
    python src/split_yolo.py

    #split classification dataset
    python src/split_data.py

    #NOTE when you split it, can delete the originals if you want or not
    #Recommended approach: For classification

    --target for algae → 1050

    --target for microplastics → 1050

    to lessen the chance of overfitting for the algae
    common rule for balancing is <=5x per image

    2100 total, for training
    #Recommended approach for detection:

    Target: 4000–5000 images training only

    # Balance "algae" class (train only)
    python src/augment_balance.py --task classification --input ./data/train --outdir ./data_balanced/train --class algae --target 1050 --use_clahe

    # Balance "microplastics" class (train only)
    python src/augment_balance.py --task classification --input ./data/train --outdir ./data_balanced/train --class microplastics --target 1050 --use_clahe

    # Balance detection dataset to 4000 images
    python src/augment_balance.py --task detection --images "./data_yolo" --labels "./data_yolo" --outdir "./data_yolo_balanced" --target 4000 --use_clahe

    # Resize to 224x224, normalize, output into data_preprocessed/ for classification
    python src/preprocess.py --task classification --input ./data_balanced/train --outdir ./data_preprocessed/train --keep-structure --no-flatten

    # Resize to 224x224, normalize, output into ./data_yolo_preprocessed for detection
    python src/preprocess.py --task detection --input ./data_yolo_balanced --outdir ./data_yolo_preprocessed --no-flatten

    # Classification Training, Auto 70/15/15 split, 20 epochs, batch size 16, learning rate 2e-4 for unix
    python src/train.py --task classification \
        --data ./data_preprocessed \
        --auto_split \
        --epochs 20 \
        --batch_size 16 \
        --lr 2e-4 \
        --output_model classifier.pth
    # Classification Training, Auto 70/20/10 split, 20 epochs, batch size 16, learning rate 2e-4 for meow no need autosplit, I already split in the begining
        python src/train.py --task classification --data ./data_preprocessed --epochs 20 --batch_size 8 --lr 2e-4 --output_model classifier.pth

    # Detection Training, Auto 70/20/10 split, 20 epochs, batch size 16, learning rate 2e-4 for meow
        python src/train.py --task detection --data ./data_yolo_preprocessed --epochs 30 --batch_size 8 --lr 2e-4 --output_model detection.pth

    python src/train.py --task detection --data ./data_yolo_preprocessed --epochs 30 --batch_size 8 --lr 2e-4 --output_model detection.pth --conf_thresh 0.05 --nms_iou 0.5

python src/train.py --task detection --data ./data_yolo_preprocessed --epochs 30 --batch_size 8 --lr 2e-4 --conf_thresh 0.05 --nms_iou 0.5 --num_classes 2 --seed 42 --output_model detection.pth

# Generate accuracy, precision, recall, F1 + confusion matrix for classification models,change weights per model

python src/metrics.py --data ./data/val --weights classifier.pth --out ./results_classification_val --class_names algae microplastics --num_visuals 30

#metrics for our classification model

python src/metrics.py --data ./data/test --weights checkpoints/classifier_best_epoch?.pth --out results_classification_test --class_names algae microplastics --num_visuals 50 --num_classes 2 --batch_size 32
#metrics for our detection model
python src/metrics_detection.py --weights checkpoints/detector_best_epoch12.pth --images ./data_yolo/images/val --labels ./data_yolo/labels/val --out results_detection --img_size 224 --conf_thresh 0.05 --nms_iou 0.5 --num_classes 2 --class_names algae microplastics --visualize_n 30

    python src/metrics_detection.py --weights checkpoints/detector_best_epoch15.pth --images ./data_yolo/images/val --labels ./data_yolo/labels/val --out results_detection_final --img_size 224 --conf_thresh 0.05 --nms_iou 0.5 --num_classes 2 --class_names algae microplastics --visualize_n 30

    python src/metrics_detection.py --weights checkpoints/detector_best_epoch15.pth --images ./data_yolo/images/val --labels ./data_yolo/labels/val --out results_detection_final --img_size 224 --conf_thresh 0.25 --nms_iou 0.45 --num_classes 2 --class_names algae microplastics --visualize_n 30

#FOR TEST SET IT WORKED :DD
python src/metrics_detection.py --weights checkpoints/detector_best_epoch15.pth --images ./data_yolo/images/test --labels ./data_yolo/labels/test --out results_detection_test --img_size 224 --conf_thresh 0.25 --nms_iou 0.45 --num_classes 2 --class_names algae microplastics --visualize_n 30 # Single image detection
python src\detect.py --weights classifier.pth --source .\data\microplastics\1.jpg `
--output_dir .\results --bbox --cam

    # Folder detection
    python src\detect.py --weights classifier.pth --source .\data\microplastics `
        --output_dir .\results --bbox --cam

    # Export trained model to ONNX
    python src\export.py --weights classifier.pth --output model.onnx

    # Prune 20% of model weights, save optimized version
    python src\optimize.py --weights classifier.pth --output classifier_pruned.pth --prune_fraction 0.2

    #OTHER MODELSSS

    #Yolov8n Classification for training and val
    python src/yolov8_classify.py

    #Yolov8n Detection for training and val
    python src/yolov8_detect.py

    #Yolov8n analyze metrics of py w P R f1 type stuff for classification USE THIS instead opf classify its for yolov8n!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    python src/analyze_metrics.py

    #Yolov8n analyze metrics of py w P R f1 type stuff for detection USE THIS instead opf detection its for yolov8n !!!!!!!!!!!!!!!!!!!!!!
    python src/analyze_metrics_detection.py

    #Yolov8n analyze metrics of classification model with visualization same as ours
    python src/metrics.py --data ./dataC/val --weights ./runs/classify/yolov8n_classification_microplastics_algae14/weights/best.pt --out ./results_yolov8_classification --class_names algae microplastics --num_visuals 300

    .\.venv\Scripts\Activate.ps1

    #Yolov8n purely using Yolos built in stuff use the other above
    python src/analyze_metrics_detection.py --weights ./runs/detect/yolov8n_detection_microplastics_algae3/weights/best.pt --images ./dataD/images/val --labels ./dataD/labels/val --out ../results_yolo_detection --img_size 224


    #Resnet18 train model on classification data
    python src/resnet18_classify.py

    #autosplit to test if i properly split the set,
    python src/split_Classify.py

    #for metrics
    python src/resnet18_metrics.py
    ```

r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC\val" resnet18_best.pth
