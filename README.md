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



    #Revamped Preprocessing with online augmentation to fight against Overfitting, OVERFITTING DIEEE!!! for detection
    # Step 1: Analyze original data
    python src/balance_detection.py --input ./data_yolo --split train --analyze-only

    # Step 2: Balance training set
    python src/balance.py --task detection --input ./data_yolo --output ./data_yolo_balanced --split train --target 1500

    # Step 3: Light preprocess training set
    python src/preprocess_light.py --task detection --input ./data_yolo_balanced --output ./data_yolo_train_ready --size 224

    # Step 4: Train
    python src/train2.py --task detection --train_dir ./data_yolo_train_ready --val_dir ./data_yolo --num_classes 2 --epochs 50 --batch_size 8 --lr 1e-4 --fresh_start --output_model detector_v2.pth

    #Step 4: Train ( This w 100 epochs to try to increase accuracy) v3
    python src/train2.py --task detection --train_dir ./data_yolo_train_ready --val_dir ./data_yolo --num_classes 2 --epochs 100 --batch_size 8 --lr 1e-4 --fresh_start --output_model detector_v3.pth

    #Step 4: Train ( This w 150 epochs to try to increase accuracy) v4
    python src/train2.py --task detection --train_dir ./data_yolo_train_ready --val_dir ./data_yolo --num_classes 2 --epochs 150 --batch_size 8 --lr 1e-4 --fresh_start --output_model detector_v4.pth

#For detection test set v4
python src/evaluate.py --task detection --model detector_v4.pth --test_dir ./data_yolo_test_ready --num_classes 2 --conf_thresh 0.5
#For detection test set v2
python src/preprocess_light.py --task detection --input ./data_yolo/images/test --output ./data_yolo_test_ready/images --size 224

    python src/evaluate.py --task detection --model detector_v2.pth --test_dir ./data_yolo_test_ready --num_classes 2 --conf_thresh 0.5 #with conf thres 0.5

    #Revamped Preprocessing with online augmentation to fight against Overfitting, OVERFITTING DIEEE!!! for classification
    # Step 1: Analyze
    python src/balance.py --task classification --input ./data --analyze-only

    # Step 2: Balance training set (increase both classes to ~1500 each)
    python src/balance.py --task classification --input ./data/train --output ./data_train_balanced_classification --target 1500 --class algae
    python src/balance.py --task classification --input ./data_train_balanced_classification --output ./data_train_balanced_final --target 1500 --class microplastics

# Step 3: Light preprocess TRAINING

    # Preprocess training data
    python src/preprocess_light.py --task classification --input ./data_train_balanced_final --output ./data_train_ready --size 224

    # Preprocess validation data
    python src/preprocess_light.py --task classification --input ./data/val --output ./data_val_ready --size 224


    # Step 3b: Light preprocess VALIDATION
    python src/preprocess_light.py --task classification --input ./data/val --output ./data_val_ready --size 224

    # Step 4: Train for classification
    python src/train2.py --task classification --train_dir ./data_train_ready --val_dir ./data_val_ready --num_classes 2 --epochs 50 --batch_size 16 --lr 5e-5 --fresh_start --output_model classifier_v2.pth

    #For Test Set classification
    # Step 2: Preprocess test set
    python src/preprocess_light.py --task classification --input ./data/test --output ./data_test_ready --size 224

    # Step 3: Evaluate on test set v2
    python src/evaluate.py --task classification --model classifier_v2.pth --test_dir ./data_test_ready --num_classes 2

    #used this for training classification model v3
    python src/train2.py --task classification --train_dir ./data_train_ready --val_dir ./data_val_ready --num_classes 2 --epochs 50 --batch_size 16 --lr 5e-5 --fresh_start --output_model classifier_v3.pth

    #For Visualization our Model Classification and detection
    python src/visualize.py --task classification --model classifier_v3.pth --test_dir ./data_test_ready --output ./visuals_classification

    python src/visualize.py --task detection --model detector_v4.pth --test_dir ./data_yolo_test_ready --output ./visuals_detection --conf_thresh 0.5

    #For Benchamrking our model Classification and Detection
    python src/benchmark.py --model detector_v4.pth --task detection --num_runs 500 --warmup 50
    python src/benchmark.py --model classifier_v3.pth --task classification --num_runs 100

    #For training and evaluating classification baseline models
    python src/baseline_models.py --model resnet18 --train_dir ./data_train_ready --val_dir ./data_val_ready --test_dir ./data_test_ready --epochs 50 --batch_size 16 --lr 1e-4

    python src/baseline_models.py --model mobilenetv2 --train_dir ./data_train_ready --val_dir ./data_val_ready --test_dir ./data_test_ready --epochs 50 --batch_size 16 --lr 1e-4

    python src/baseline_models.py --model efficientnet_b0 --train_dir ./data_train_ready --val_dir ./data_val_ready --test_dir ./data_test_ready --epochs 50 --batch_size 16 --lr 1e-4

    #for training yolo baseline models v8 nano and v5 nano
    python src/yolo_baseline.py --model yolov8n --data_dir ./data_yolo --epochs 100 --imgsz 224 --batch_size 16 --conf_thresh 0.5

    python src/yolo_baseline.py --model yolov5n --data_dir ./data_yolo --epochs 100 --imgsz 224 --batch_size 16 --conf_thresh 0.5

python src/yolo_baseline.py --model yolov8n --benchmark_only --weights runs/detect/yolov8n_microplastic4/weights/best.pt
python src/yolo_baseline.py --model yolov5n --benchmark_only --weights runs/detect/yolov5n_microplastic2/weights/best.pt
#for training YOLOv8s and derty for comparing against bigger model

    python rtdetr_det.py --task both --data_dir ./data_yolo --epochs 100 --batch 16 --img_size 224 --conf 0.5

    #This is for benchmarking, no COCO weights

python src/yolo_scratch_baseline.py --model yolov8s --task both --data_dir ./data_yolo --img_size 224 --conf 0.5 --epochs 150
python src/yolo_scratch_baseline.py --model yolov8n --task both --data_dir ./data_yolo --img_size 224 --conf 0.5 --epochs 150
python src/yolo_scratch_baseline.py --model yolov5n --task both --data_dir ./data_yolo --img_size 224 --conf 0.5 --epochs 150

python src/yolo_scratch_baseline.py --model yolov8n --task benchmark --weights runs/detect/yolov8n_scratch/weights/best.pt --data_dir ./data_yolo

python src/yolo_scratch_baseline.py --model yolov8s --task benchmark --weights runs/detect/yolov8s_scratch/weights/best.pt --data_dir ./data_yolo

python src/yolo_scratch_baseline.py --model yolov5n --task benchmark --weights runs/detect/yolov5n_scratch/weights/best.pt --data_dir ./data_yolo
    #For Ablation Study
    python src/run_ablation.py --train_dir ./data_yolo_train_ready --val_dir ./data_yolo --num_classes 2 --epochs 150 --batch_size 8 --lr 1e-4 --fresh_start --variants swin_only cnn_only no_fusion
    #For Ablation Study if u wana check thebresults without training
    python src/run_ablation.py --val_dir ./data_yolo --num_classes 2 --skip_train --variants swin_only cnn_only no_fusion

python rtdetr_det.py --task benchmark --data_dir ./data_yolo --img_size 224 --conf 0.5 --weights runs/detect/rtdetr_l_microplastic/weights/best.pt

#Last for the Sensitivity run
python src/run_sensitivity.py --train_dir ./data_yolo_train_ready --val_dir ./data_yolo --num_classes 2 --epochs 150 --batch_size 8 --lr 1e-4 --fresh_start

python src/run_sensitivity.py --train_dir ./data_yolo_train_ready --val_dir ./data_yolo --num_classes 2 --epochs 150 --batch_size 8 --lr 1e-4 --fresh_start --runs heads_2 heads_8 window_4 window_14 lambdabox_1 lambdabox_10

#run this tomorrow on april 19,2026, the heads 2and 8 are finished, the other head didnt finish because it couldnt divide them evenly
python src/run_sensitivity.py --train_dir ./data_yolo_train_ready --val_dir ./data_yolo --num_classes 2 --epochs 150 --batch_size 8 --lr 1e-4 --fresh_start --runs window_2 window_14 lambdabox_1 lambdabox_10

#Rebenchmark w proper empty cache to see true vram
python src/run_sensitivity.py --val_dir ./data_yolo --num_classes 2 --skip_train --runs heads_2 heads_8

#rebenchmark ablation for proper vram 
python src/run_ablation.py --val_dir ./data_yolo --num_classes 2 --skip_train --variants swin_only cnn_only no_fusion

#for fair comparison run again
python src/rtdetr_scratch_baseline.py --task both --data_dir ./data_yolo --epochs 150 --batch 16 --img_size 224
    ```

r"C:\Users\User\Desktop\Programming languages for vs\T\Thesis_Microplastics\dataC\val" resnet18_best.pth

    .\.venv\Scripts\Activate.ps1
