# Microplastic Detection (Thesis)

Lightweight **ResNet18 + Swin Transformer Tiny** hybrid classifier for **Algae vs Microplastics** detection.  
Pipeline: **Balance → Preprocess → Train → Metrics → Detect → Export → Optimize.

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
# Create virtual environment, if you already have skip and activate venv
python -m venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Fix execution policy if activation fails
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

#For testing the bounding boxes with the raw images or change data_yolo to data_yolo_balanced to test augmented set etc
.venv\Scripts\python.exe src\visual_check.py --input data_yolo_preprocessed --num 5

# Balance "algae" class
python src/augment_balance.py --task classification --input ./data --outdir ./data_balanced --class algae --target 2000 --use_clahe

# Balance "microplastics" class
python src/augment_balance.py --task classification --input ./data --outdir ./data_balanced --class microplastics --target 2000 --use_clahe

# Balance detection dataset to 2000 images
python src/augment_balance.py --task detection --input ./data_yolo --outdir ./data_yolo_balanced --target 4000 --use_clahe

# Resize to 224x224, normalize, output into data_preprocessed/ for classification
python src/preprocess.py --task classification --input ./data_balanced --outdir ./data_preprocessed --keep-structure --no-flatten

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
# Classification Training, Auto 70/20/10 split, 20 epochs, batch size 16, learning rate 2e-4 for meow
    python src/train.py --task classification --data ./data_preprocessed --auto_split --epochs 20 --batch_size 16 --lr 2e-4 --output_model classifier.pth


# Detection Training, Auto 70/20/10 split, 20 epochs, batch size 16, learning rate 2e-4
python src/train.py --task classification \
    --data ./data_preprocessed \
    --auto_split \
    --epochs 20 \
    --batch_size 16 \
    --lr 2e-4 \
    --output_model classifier.pth


# Generate accuracy, precision, recall, F1 + confusion matrix
python src\metrics.py --data .\data_preprocessed --weights classifier.pth `
    --out .\results --class_names algae microplastics

# Single image detection
python src\detect.py --weights classifier.pth --source .\data\microplastics\1.jpg `
    --output_dir .\results --bbox --cam

# Folder detection
python src\detect.py --weights classifier.pth --source .\data\microplastics `
    --output_dir .\results --bbox --cam

# Export trained model to ONNX
python src\export.py --weights classifier.pth --output model.onnx

# Prune 20% of model weights, save optimized version
python src\optimize.py --weights classifier.pth --output classifier_pruned.pth --prune_fraction 0.2