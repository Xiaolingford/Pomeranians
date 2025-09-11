# Microplastic Detection using Lightweight Dual-Branch Swin Transformer

This repository implements a **dual-branch Swin Transformer classifier** for detecting **microplastics vs. algae** from microscopy images.  
It follows the thesis methodology: one branch processes standard RGB images, the other branch processes **CLAHE-enhanced images**, and their features are fused.

---

## 📂 Project Structure
microplastic-detection/
├── data/ # raw images (algae/ and microplastics/)
├── data_resized/ # (optional) preprocessed 224×224 images
├── results/ # outputs (detections, metrics, plots, CSVs)
├── src/ # source code
│ ├── dataset.py
│ ├── model.py
│ ├── preprocess.py
│ ├── train.py
│ ├── detect.py
│ ├── metrics.py
│ ├── optimize.py
│ └── export.py
└── requirements.txt


---

## ⚙️ Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # on Windows
   pip install -r requirements.txt

#1. PreProcess Images:
    python src/preprocess.py

#2. Train model
    python src/train.py

#3. Detection
    python src/detect.py --input ".\data\microplastics" --outdir ".\results"

#4. Evaluate Metrics
    python src/metrics.py --data ".\data_resized" --weights "classifier.pth" --csv

#5. Optimize
python src/optimize.py

#6. Export 
python src/export.py





