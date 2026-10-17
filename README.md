# 🐝 Bee Behavior Detection

A computer vision system that detects bees using **YOLOv8m**, tracks them using **ByteTrack**, and classifies their behavior using an **LSTM**.

### Pipeline

```text
Video
  ↓
YOLOv8m
  ↓
ByteTrack
  ↓
Feature Extraction
  ↓
10-Frame Sequence
  ↓
LSTM
  ↓
Behavior + Confidence
```

## Dataset

This project uses:

**Labeled dataset for bee detection and direction estimation on beehive landing boards**

Dataset:  
https://data.mendeley.com/datasets/8gb9r2yhfc/6

Version: **6**  
DOI: **10.17632/8gb9r2yhfc.6**

---

## 1. Download the Dataset

Download the dataset and use the detection images and annotations for YOLO training.

Required structure:

```text
dataset/
└── detection/
    ├── _bee_20230609a/
    ├── _bee_20230609b/
    ├── _bee_20230609c/
    ├── ...
    └── _bee_20230711c/
```

---

## 2. Prepare the YOLO Dataset

Organize the images and YOLO labels:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image should have its corresponding YOLO `.txt` annotation.

---

## 3. Train YOLOv8m

Train YOLOv8m using the prepared dataset.

```text
YOLO Dataset
     ↓
YOLOv8m Training
     ↓
best.pt
```

Place the trained model here:

```text
yolov8/
└── weights/
    └── best.pt
```

---

## 4. Track Bees with ByteTrack

The trained YOLOv8m model detects bees in each video frame.

ByteTrack assigns a unique ID to each detected bee.

```text
Video
  ↓
YOLOv8m
  ↓
Bee Detection
  ↓
ByteTrack
  ↓
Bee ID
```

---

## 5. Extract Features

For every tracked bee, the system extracts:

```text
x
y
width
height
distance
speed
direction_sin
direction_cos
acceleration
```

The features are grouped into sequences of **10 frames** for LSTM prediction.

---

## 6. Train the LSTM

The LSTM classifies bee behavior into four classes:

```text
Fanning
Defense
Foraging
Washboarding
```

Required LSTM files:

```text
LSTM WITH speed & distance/
├── best_bee_lstm.keras
├── bee_feature_scaler.pkl
└── bee_label_encoder.pkl
```

---

# 7. Run the Project

## Create Virtual Environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

## Install Dependencies

```bash
pip install ultralytics tensorflow opencv-python numpy joblib scikit-learn
```

## Project Structure

```text
prepared_detection/
│
├── run_bee_behavior.py
├── README.md
├── .gitignore
│
├── yolov8/
│   └── weights/
│       └── best.pt
│
├── LSTM WITH speed & distance/
│   ├── best_bee_lstm.keras
│   ├── bee_feature_scaler.pkl
│   └── bee_label_encoder.pkl
│
└── videos/
    └── test.mp4
```

## Run

### 1. DINOv2 ViT-S/14 + LSTM (Recommended — Visual Features)

Uses DINOv2 vision transformer embeddings (384-d) with full GPU acceleration:

```bash
python run_bee_vit_lstm.py
```

Optional arguments:
```bash
# Run with fused features (visual + kinematics)
python run_bee_vit_lstm.py --mode fused

# Run on a different video file
python run_bee_vit_lstm.py --video videos/20230609b-def.mp4

# Save annotated video output
python run_bee_vit_lstm.py --save-out videos/annotated_output.mp4
```

### 2. Baseline YOLOv8m + Kinematics LSTM (Legacy)

```bash
python run_bee_behavior.py
```

The system opens a live video preview with:

```text
ID + Behavior + Confidence
```

Example:

```text
ID 21 | Fanning | 96.4%
```

Press `Q` or `ESC` to stop the preview.
Press `P` to pause/resume.

---

# Final Pipeline

```text
Input Video
     ↓
YOLOv8m
     ↓
Bee Detection
     ↓
ByteTrack
     ↓
Bee Tracking
     ↓
9 Features
     ↓
10-Frame Sequence
     ↓
LSTM
     ↓
Behavior Classification
     ↓
Behavior + Confidence
```

## Behavior Classes

| Class | Description |
|---|---|
| Fanning | Bee fanning behavior |
| Defense | Defensive behavior |
| Foraging | Foraging activity |
| Washboarding | Washboarding behavior |

---

## Requirements

- Python 3.10+
- YOLOv8
- TensorFlow
- OpenCV
- NumPy
- Scikit-learn
- Joblib
- ByteTrack

---

## Notes

The trained model files are required to run the complete inference pipeline.

The dataset and training videos are not included in this repository.

For training, download the dataset from the Mendeley Data link above.
