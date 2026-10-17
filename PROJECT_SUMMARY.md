# Project Summary: Honeybee Behavior Detection & Classification

## 1. Overview
This project develops an automated computer vision and deep learning system to detect individual honeybees (*Apis mellifera*), track their trajectories across video frames, and classify distinct colony behaviors (Fanning, Defense, Foraging, Washboarding, and Other/Idle). The codebase incorporates two sequential generations of behavior classifiers:
1. **Legacy Pipeline (Keras LSTM)**: Combines YOLOv8m detection, ByteTrack tracking, and 7- or 9-feature handcrafted kinematic vectors fed into a lightweight 2-layer Keras LSTM over a 10-frame sliding window.
2. **Current Pipeline (PyTorch DINOv2 ViT-S/14 + LSTM)**: Extracts 384-dimensional deep visual embeddings per bee bounding box crop using Meta's pre-trained DINOv2 ViT-S/14 vision transformer, concatenates or compares against 7-dimensional smoothed kinematics, and classifies sequences across a causal 16-step temporal window ($K=6$ frame subsampling $\approx 1.92$ s) with entrance-zone polygon gating.

---

## 2. File Inventory

### 2.1 Repository Directory Structure
| Directory | File Count | Disk Size | Description |
|---|---|---|---|
| `.` (Repo Root) | 18 files | 95.83 MB | Main execution, training, reporting scripts, docx reports, configs, and base weights. |
| `Datasets/detection/` | 13,867 files | 6,422.40 MB | 6,658 raw frame images and YOLO txt annotation pairs across 8 camera sessions. |
| `Datasets/pose/` | 801 files | 356.08 MB | 400 images and 400 YOLO-format head/stinger keypoint annotations + `info.txt`. |
| `Datasets/ramp_detection/` | 312 files | 51.18 MB | 156 images and 156 label files annotating 4-corner entrance ramp polygons. |
| `Datasets/segmentation/` | 4,601 files | 4.86 MB | 2,300 cropped bee images with 3-point triangular body orientation polygons. |
| `Datasets/tracking_and_behavior/` | 20 files | 610.99 MB | 5 raw MP4 video clips, bounding-box track files, behavior txt files, entrance zones. |
| `dataset_yolo/` | 13,318 files | 5,948.14 MB | Verified session-split YOLO dataset: train (4,385), val (1,759), test (514). |
| `dinov2_features/` | 41 files | 3,334.80 MB | Extracted DINOv2 visual embeddings and kinematic NumPy `.npy` arrays and manifest. |
| `LSTM WITH speed & distance/` | 6 files | 0.83 MB | Legacy 9-feature Keras LSTM model, StandardScaler, label encoder, test metrics. |
| `LSTM WITHOUT speed & distance/` | 7 files | 0.41 MB | Legacy 7-feature Keras LSTM model, StandardScaler, label encoder, test metrics. |
| `runs/` | 36 files | 458.58 MB | YOLO training checkpoints (`yolov8m_960`), training log, curves, validation outputs. |
| `videos/` | 3 files | 131.59 MB | Benchmark video `20230711a-fan.mp4`, demo output clip, `.gitkeep`. |
| `vit_lstm_models/` | 4 files | 9.33 MB | PyTorch ViT-LSTM weights (`fused`, `visual_only`, `kinematics_only`) and benchmark JSON. |
| `weights/` | 1 file | 5.29 MB | Ultralytics nano model weight `yolo26n.pt`. |
| `yolov8/` | 21 files | 106.33 MB | Upstream baseline YOLOv8m weights (`best.pt`, `last.pt`), curves, and Kaggle results. |
| `.cache/` | 4 files | 0.83 MB | Local cache containing patched Keras model archives without `quantization_config`. |
| `.git/`, `.venv/`, `__pycache__/` | Skipped | Skipped | Virtual environment, git version control objects, and compiled bytecode. |

### 2.2 Inventory of Code, Config, Notebook, and Document Files
| Relative Path | Last Modified | One-Line Summary of Function |
|---|---|---|
| `Bee_Behavior_Classification_Report_Clean.docx` | 2026-10-17 22:14:16 | Formatted Word document presenting architecture comparison, quantitative benchmarks, confusion matrix, and 6,000-frame video benchmark. |
| `Bee_Behavior_Classification_Report_Final_Benchmark.docx` | 2026-10-17 22:12:06 | Prior version of the styled Word evaluation report. |
| `Bee_Behavior_Classification_Report_Pre_vs_Post.docx` | 2026-10-17 22:02:06 | Initial generation of the Pre vs. Post Word evaluation report. |
| `create_report_docx.py` | 2026-10-17 22:14:09 | Automated document generator using `python-docx` to format and output `Bee_Behavior_Classification_Report_Clean.docx`. |
| `data.yaml` | 2026-10-17 16:17:08 | Ultralytics dataset configuration specifying paths to `dataset_yolo` splits and 1 class (`bee`). |
| `download_dataset.py` | 2026-10-16 19:13:12 | Dataset verification, manifest verification, zip extraction, and YOLO directory builder for Mendeley Data v6. |
| `extract_dinov2_features.py` | 2026-10-17 21:32:20 | Extracts 384-d DINOv2 crop embeddings, 7-d smoothed kinematics, performs $K=6, T=16$ windowing, zone gating, and saves `.npy` splits. |
| `preview.py` | 2026-10-15 21:13:52 | Minimal OpenCV utility running YOLO detection on video frames with bounding box visualization. |
| `README.md` | 2026-10-17 21:56:14 | Project documentation detailing system architecture, setup commands, execution scripts, and benchmark results. |
| `requirements.txt` | 2026-10-16 19:38:08 | Pinned Python package dependencies including ultralytics, tensorflow, keras, opencv-python, torch, torchvision. |
| `run_bee_behavior.py` | 2026-10-17 16:51:21 | Real-time and offline inference script running YOLOv8m + ByteTrack + 9-feature (or 7-feature) Keras LSTM. |
| `run_bee_vit_lstm.py` | 2026-10-17 22:07:23 | Real-time inference application running YOLOv8m + ByteTrack + DINOv2 ViT-S/14 + PyTorch ViT-LSTM (5 classes). |
| `runs\train\yolov8m_960\args.yaml` | 2026-10-17 17:21:15 | Training arguments recorded for the 960px YOLOv8m retrain run (batch 8, imgsz 960, epochs 150). |
| `SESSION_EXPORT.md` | 2026-10-17 20:58:38 | Comprehensive development diary documenting environment setup, bug fixes, profiling benchmarks, and YOLO retraining. |
| `train_vit_lstm.py` | 2026-10-17 21:34:20 | PyTorch training suite for Fused ViT-LSTM, Visual-Only DINOv2, and Kinematics-Only control models. |
| `train_yolo.py` | 2026-10-17 17:20:57 | Training launcher script for fine-tuning YOLOv8m at imgsz=960 with data augmentation and cosine learning rate. |
| `yolov8\Result\args.yaml` | 2026-10-15 21:13:52 | Training arguments recorded from the original Kaggle baseline YOLOv8m training run (batch 32, imgsz 640, epochs 80). |
| `*.ipynb` | not found | Zero Jupyter Notebook files exist in this repository. |

### 2.3 Model Weights and Feature Caches
| File Path | Size (MB) | Last Modified | Producing Script | Loading Script |
|---|---|---|---|---|
| `yolov8\weights\best.pt` | 49.60 MB | 2026-10-15 21:13:52 | Upstream Kaggle training (`yolov8\Result\results.csv`) | `preview.py`, `run_bee_behavior.py`, `run_bee_vit_lstm.py` |
| `yolov8\weights\last.pt` | 49.60 MB | 2026-10-15 21:13:53 | Upstream Kaggle training | not referenced |
| `yolov8m.pt` | 49.72 MB | 2026-10-17 17:10:17 | Pretrained Ultralytics download | `train_yolo.py` |
| `weights\yolo26n.pt` | 5.29 MB | 2026-10-17 17:11:07 | Ultralytics download | not referenced |
| `runs\train\yolov8m_960\weights\best.pt` | 148.32 MB | 2026-10-17 17:31:08 | `train_yolo.py` (epoch 3) | not referenced |
| `runs\train\yolov8m_960\weights\epoch0.pt` | 148.32 MB | 2026-10-17 17:31:08 | `train_yolo.py` (epoch 1) | not referenced |
| `runs\train\yolov8m_960\weights\last.pt` | 148.32 MB | 2026-10-17 17:42:29 | `train_yolo.py` (stopped ep 3) | not referenced |
| `LSTM WITH speed & distance\best_bee_lstm.keras` | 0.41 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | `run_bee_behavior.py` |
| `LSTM WITH speed & distance\bee_behavior_lstm_final.keras` | 0.41 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | not referenced |
| `LSTM WITH speed & distance\bee_feature_scaler.pkl` | 0.001 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | `run_bee_behavior.py` |
| `LSTM WITH speed & distance\bee_label_encoder.pkl` | 0.001 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | `run_bee_behavior.py` |
| `LSTM WITH speed & distance\bee_behavior_confusion_matrix.npy` | 0.0003 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | not referenced |
| `LSTM WITHOUT speed & distance\bee_behavior_lstm_7features_final.keras` | 0.41 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | not referenced |
| `LSTM WITHOUT speed & distance\bee_feature_scaler_7features.pkl` | 0.001 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | not referenced |
| `LSTM WITHOUT speed & distance\bee_label_encoder_7features.pkl` | 0.0004 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | not referenced |
| `LSTM WITHOUT speed & distance\bee_behavior_confusion_matrix_7features.npy` | 0.0003 MB | 2026-10-15 21:13:52 | Upstream training (script not found) | not referenced |
| `vit_lstm_models\fused_dinov2_kinematics.pt` | 3.62 MB | 2026-10-17 21:46:53 | `train_vit_lstm.py` | `run_bee_vit_lstm.py` (mode=fused) |
| `vit_lstm_models\visual_only_dinov2.pt` | 3.59 MB | 2026-10-17 21:49:03 | `train_vit_lstm.py` | `run_bee_vit_lstm.py` (default mode) |
| `vit_lstm_models\kinematics_only_control.pt` | 2.11 MB | 2026-10-17 21:50:45 | `train_vit_lstm.py` | not referenced |
| `dinov2_features\X_train_visual.npy` | 1,578.89 MB | 2026-10-17 21:42:24 | `extract_dinov2_features.py` | `train_vit_lstm.py` |
| `dinov2_features\X_train_kinematics.npy` | 28.78 MB | 2026-10-17 21:42:24 | `extract_dinov2_features.py` | `train_vit_lstm.py` |
| `dinov2_features\y_train.npy` | 0.51 MB | 2026-10-17 21:42:24 | `extract_dinov2_features.py` | `train_vit_lstm.py` |
| `dinov2_features\X_val_visual.npy` | 8.88 MB | 2026-10-17 21:42:24 | `extract_dinov2_features.py` | `train_vit_lstm.py` |
| `dinov2_features\X_val_kinematics.npy` | 0.16 MB | 2026-10-17 21:42:24 | `extract_dinov2_features.py` | `train_vit_lstm.py` |
| `dinov2_features\y_val.npy` | 0.003 MB | 2026-10-17 21:42:24 | `extract_dinov2_features.py` | `train_vit_lstm.py` |
| `dinov2_features\{video_name}\*.npy` (30 files) | 1,718.57 MB total | 2026-10-17 21:35 - 21:42 | `extract_dinov2_features.py` | cached per-video splits |
| `*.engine`, `*.onnx`, `*.npz` | not found | — | Zero TensorRT engine, ONNX, or NPZ files exist. | — |

---

## 3. Data

### 3.1 Video Properties
Properties extracted via OpenCV `cv2.VideoCapture`:
| Video Identifier | File Location | Resolution | FPS | Frame Count | Duration |
|---|---|---|---|---|---|
| `20230609b-def` | `Datasets\tracking_and_behavior\20230609b-def.mp4` | 1920 x 1080 | 50.0 fps | 5,973 frames | 119.46 s (1.99 min) |
| `20230711a-fan` | `Datasets\tracking_and_behavior\20230711a-fan.mp4` | 1920 x 1080 | 50.0 fps | 6,000 frames | 120.00 s (2.00 min) |
| `20230711b-fan` | `Datasets\tracking_and_behavior\20230711b-fan.mp4` | 1920 x 1080 | 50.0 fps | 6,003 frames | 120.06 s (2.00 min) |
| `yt5` | `Datasets\tracking_and_behavior\yt5.mp4` | 1280 x 720 | 50.0 fps | 977 frames | 19.54 s (0.33 min) |
| `yt8` | `Datasets\tracking_and_behavior\yt8.mp4` | 1920 x 1080 | 50.0 fps | 3,000 frames | 60.00 s (1.00 min) |
| `videos/20230711a-fan.mp4` | `videos\20230711a-fan.mp4` | 1920 x 1080 | 50.0 fps | 6,000 frames | 120.00 s (2.00 min) |
| `demo_vit_lstm_output.mp4` | `videos\demo_vit_lstm_output.mp4` | 1920 x 1080 | 50.0 fps | 100 frames | 2.00 s (0.03 min) |

### 3.2 Annotation File Statistics
Each tracking video has a `tracks_*.txt` bounding box file (columns: `frame, track_id, x, y, w, h`) and a `tracks_and_behavior_classes_*.txt` file (columns: `frame, track_id, x, y, w, h, class0, class1, class2, class3`):
| Video File | Annotation File Name | Total Rows | Columns | Unique Tracks | Rows Labeled `[0,0,0,0]` |
|---|---|---|---|---|---|
| `20230609b-def` | `tracks_and_behavior_classes_20230609b-def.txt` | 30,379 | 10 | 2,138 | 16,145 (53.15%) |
| `20230711a-fan` | `tracks_and_behavior_classes_20230711a-fan.txt` | 127,452 | 10 | 1,674 | 94,508 (74.15%) |
| `20230711b-fan` | `tracks_and_behavior_classes_20230711b-fan.txt` | 72,802 | 10 | 3,508 | 50,156 (68.89%) |
| `yt5` | `tracks_and_behavior_classes__yt5.txt` | 63,834 | 10 | 887 | 39,634 (62.09%) |
| `yt8` | `tracks_and_behavior_classes_yt8.txt` | 129,139 | 10 | 3,781 | 102,907 (79.69%) |

### 3.3 Entrance Zone Polygons per Video
Directly parsed from `Datasets/tracking_and_behavior/entrance_zone_*.txt`:
- **`20230609b-def`**: `np.array([[245, 482], [1621, 482], [1617, 750], [255, 754]])`
- **`20230711a-fan`**: `np.array([[100, 220], [1860, 270], [1805, 590], [145, 545]])`
- **`20230711b-fan`**: `np.array([[165, 275], [1750, 255], [1690, 555], [230, 570]])`
- **`yt5`**: `np.array([[5, 65], [1274, 400], [1274, 714], [5, 714]])`
- **`yt8`**: `np.array([[210, 760], [1772, 792], [1884, 1018], [107, 990]])`

### 3.4 Behavior Breakdown per Video
Classes in raw annotation files (indices 0..3 correspond to `Foraging`, `Defense`, `Fanning`, `Washboarding`):
| Video | Foraging (Class 0) | Defense (Class 1) | Fanning (Class 2) | Washboarding (Class 3) | Total Labeled Rows |
|---|---|---|---|---|---|
| `20230609b-def` | 4,699 frames (344 tracks) | 9,190 frames (113 tracks) | 345 frames (1 track) | 0 frames (0 tracks) | 14,234 frames |
| `20230711a-fan` | 5,368 frames (341 tracks) | 0 frames (0 tracks) | 27,586 frames (107 tracks) | 0 frames (0 tracks) | 32,954 frames |
| `20230711b-fan` | 14,296 frames (1,157 tracks) | 0 frames (0 tracks) | 8,394 frames (104 tracks) | 0 frames (0 tracks) | 22,690 frames |
| `yt5` | 0 frames (0 tracks) | 0 frames (0 tracks) | 24,200 frames (131 tracks) | 0 frames (0 tracks) | 24,200 frames |
| `yt8` | 0 frames (0 tracks) | 0 frames (0 tracks) | 0 frames (0 tracks) | 26,232 frames (273 tracks) | 26,232 frames |

### 3.5 Pose Keypoints and Other Annotation Types
Verified presence across `Datasets/`:
1. **Pose Keypoints (Head / Stinger)**:
   - Location: `Datasets\pose\` (400 images, 400 label files + `info.txt`).
   - Format: `class x y w h head_x head_y stinger_x stinger_y` (YOLO pose keypoint format).
   - Documentation (`Datasets/pose/info.txt`): *"If bee is fully visible then first point marks a head, second point marks stinger. If only part of bee is visible then first point marks the front, second point marks the back."*
2. **Ramp Detection Polygons**:
   - Location: `Datasets\ramp_detection\` (156 images, 156 label files).
   - Format: 4-corner bounding polygon of the hive entrance ramp in normalized coordinates.
3. **Directional Triangle Segmentation**:
   - Location: `Datasets\segmentation\` (2,300 images, 2,300 label files + `info.txt`).
   - Format: 3-point triangle polygon (`x1 y1 x2 y2 x3 y3`) specifying orientation vector of individual cropped bees.
4. **General Object Detection**:
   - Location: `Datasets\detection\` (6,658 images across 8 session subdirectories).
   - Format: Standard 2D bounding boxes (`class x_center y_center width height`).

---

## 4. Pipeline as Actually Implemented

### 4.1 Detector
- **Model weights**: `yolov8\weights\best.pt` (deployed baseline) and `runs\train\yolov8m_960\weights\best.pt` (retrained checkpoint).
- **Inference resolution (`imgsz`)**: 640 px in `preview.py`, `run_bee_behavior.py`, and `run_bee_vit_lstm.py` (`imgsz=640`). In `train_yolo.py`, `imgsz=960`.
- **Thresholds**: Confidence threshold = `0.50`, IoU threshold = `0.50` (`run_bee_behavior.py` L728-L730; `run_bee_vit_lstm.py` L83-L84).
- **Baseline Training Configuration** (`yolov8\Result\args.yaml`): Epochs = 80 (early-stopped at epoch 46), batch = 32, imgsz = 640, optimizer = auto, seed = 0, lr0 = 0.01, lrf = 0.01.
- **Retraining Configuration** (`runs\train\yolov8m_960\args.yaml`): Epochs = 150 (stopped by user after epoch 3), batch = 8, imgsz = 960, patience = 50, cos_lr = True, mosaic = 1.0, fliplr = 0.5.
- **Dataset Split**: Session-level partition defined in `manifest.csv` and built into `dataset_yolo`:
  - Train (4,385 frames): Sessions `_bee_20230609a`, `b`, `c`, `d`, and `_bee_20230711a`.
  - Val (1,759 frames): Sessions `_bee_20230609e` and `_bee_20230711b`.
  - Test (514 frames): Session `_bee_20230711c`.

### 4.2 Tracker: ByteTrack Effective Parameters
`custom_bytetrack.yaml` was **not found** in the repository. Both `run_bee_behavior.py` (L726) and `run_bee_vit_lstm.py` (L175) pass `tracker="bytetrack.yaml"`, which invokes Ultralytics' built-in tracker defaults:
| Parameter | Ultralytics Default (`bytetrack.yaml`) | `custom_bytetrack.yaml` | Status |
|---|---|---|---|
| `tracker_type` | `bytetrack` | not found | Matches Ultralytics default |
| `track_high_thresh` | `0.25` | not found | Matches Ultralytics default |
| `track_low_thresh` | `0.10` | not found | Matches Ultralytics default |
| `new_track_thresh` | `0.25` | not found | Matches Ultralytics default |
| `track_buffer` | `30` | not found | Matches Ultralytics default |
| `match_thresh` | `0.80` | not found | Matches Ultralytics default |
| `fuse_score` | `True` | not found | Matches Ultralytics default |

### 4.3 Crop Generation
- **Padding**: 10% on all four sides (`pad_w = 0.10 * bw`, `pad_h = 0.10 * bh` in `extract_dinov2_features.py` L337-L338 and `run_bee_vit_lstm.py` L64-L65).
- **Output Size**: $224 \times 224$ pixels via BICUBIC interpolation (`transforms.Resize((224, 224), interpolation=BICUBIC)`).
- **Normalization**: ImageNet standard normalization: Mean = `[0.485, 0.456, 0.406]`, Std = `[0.229, 0.224, 0.225]`.

### 4.4 Backbone
- **Model**: Exact model loaded via TorchHub is Meta's `dinov2_vits14` (`torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`).
- **Weights State**: Frozen (`eval()` mode with zero gradient updates; acts as a fixed feature extractor).
- **Output Token**: 384-dimensional CLS token output (`embs = model(batch_tensor)`). Patch tokens are not retained.
- **Precision**: PyTorch FP16 automatic mixed precision on CUDA (`torch.autocast("cuda", dtype=torch.float16)`).

### 4.5 Kinematic Features
#### In Current ViT-LSTM (`extract_dinov2_features.py` & `run_bee_vit_lstm.py`):
7 features computed per step:
1. `norm_x`: Center $x$ normalized by frame width ($c_x / W$).
2. `norm_y`: Center $y$ normalized by frame height ($c_y / H$).
3. `norm_w`: Box width normalized by frame width ($w / W$).
4. `norm_h`: Box height normalized by frame height ($h / H$).
5. `sin_dir`: Direction sine ($\Delta y / \text{dist}$ or $0.0$ if $\text{dist} \le 10^{-4}$).
6. `cos_dir`: Direction cosine ($\Delta x / \text{dist}$ or $1.0$ if $\text{dist} \le 10^{-4}$).
7. `norm_speed`: Pixels per second normalized by frame diagonal: $[(\text{dist} / \Delta t) / \text{diag}] \times 50.0$.
*(Note: Acceleration was computed on L180 of `extract_dinov2_features.py` but dropped from the stored feature array on L183).*

#### In Legacy Pipeline (`run_bee_behavior.py`):
9 features computed at 50 fps:
`[x, y, box_width, box_height, distance, speed, direction_sin, direction_cos, acceleration]`
Normalized online using `StandardScaler` loaded from `LSTM WITH speed & distance/bee_feature_scaler.pkl`.

### 4.6 Windowing, Subsampling, and Gating
- **Sequence Length ($T$)**: 16 steps in ViT-LSTM (`extract_dinov2_features.py`); 10 steps in legacy LSTM (`run_bee_behavior.py`).
- **Subsample Factor ($K$)**: $K=6$ in ViT-LSTM (downsampling 50 fps $\to \approx 8.33$ fps; temporal observation window $= 16 \times 6 / 50 = 1.92$ s). $K=1$ (no subsampling, raw 50 fps) in legacy LSTM.
- **Stride**:
  - Training: Stride = 1 with 6 phase offsets (phases $0..5$).
  - Validation: Non-overlapping stride = 16 (single phase offset 0).
- **Label Rule**: A window is assigned class $C \in \{0, 1, 2, 3\}$ if $\ge 80\%$ of steps (13/16) have that class label and no other behavior is present. If all 16 steps have label `[0,0,0,0]`, it is assigned to class 4 (`Other`).
- **Dropped Windows**: Any window with mixed behaviors or $<80\%$ purity is dropped. Any window where any step exits the entrance polygon is dropped.
- **Entrance Zone Gating**: Windows must have all 16 bee centers strictly inside the entrance polygon (`cv2.pointPolygonTest >= 0`).
- **Minimum Track Length**: Track must span at least 16 frames within the split boundary.

### 4.7 Dataset Split Strategy
- **Method**: Time-blocked per video. First 70% of frames allocated to training, last 30% allocated to validation.
- **Buffer Gap**: A 250-frame temporal gap ($\approx 5.0$ s for long videos; 117 frames / 12% for `yt5`) is purged between train and val to prevent trajectory leakage.
- **Window Distribution**:
  - **Train Split (67,366 windows across 327 tracks)**: Foraging: 343 windows (4 tracks); Defense: 5,597 windows (16 tracks); Fanning: 20,743 windows (72 tracks); Washboarding: 3,854 windows (46 tracks); Other: 36,829 windows (389 tracks).
  - **Val Split (379 windows across 175 tracks)**: Foraging: 1 window (1 track); Defense: 1 window (1 track); Fanning: 135 windows (46 tracks); Washboarding: 22 windows (11 tracks); Other: 220 windows (116 tracks).

### 4.8 Classifier Architectures
#### PyTorch ViT-LSTM (`train_vit_lstm.py`):
- **Architecture**: `BeeLSTMClassifier`
  - Input LayerNorm (`nn.LayerNorm(input_dim)`: 384 for visual, 391 for fused, 7 for kinematics).
  - Input Dropout (0.2).
  - 2-layer Bidirectional LSTM (`hidden_size=128`, dropout=0.3).
  - Classification Head: Linear(256 $\to$ 64) $\to$ GELU $\to$ Dropout(0.15) $\to$ Linear(64 $\to$ 5).
- **Parameter Counts**:
  - Fused model (`fused_dinov2_kinematics.pt`): 946,323 parameters.
  - Visual-only model (`visual_only_dinov2.pt`): 939,141 parameters.
  - Kinematics-only model (`kinematics_only_control.pt`): 552,339 parameters.
- **Loss**: `nn.CrossEntropyLoss` with inverse square-root class frequency weights: `w = 1.0 / np.sqrt(counts)`.
- **Optimizer**: `AdamW(lr=1e-3, weight_decay=1e-4)` with `CosineAnnealingLR(T_max=40, eta_min=1e-5)`.
- **Epochs & Patience**: 40 max epochs, early stopping patience = 15 on validation Macro-F1.
- **Augmentation / Regularization**: Additive Gaussian feature noise during training (`torch.randn_like(x) * 0.02`), gradient clipping at `max_norm=1.0`.
- **Seeds**: not found (no random seed set in `train_vit_lstm.py`).

#### Legacy Keras LSTM (`best_bee_lstm.keras`):
- **Architecture**: Input(10, 9) $\to$ LSTM(64, return_sequences=True) $\to$ Dropout(0.3) $\to$ LSTM(32) $\to$ Dropout(0.3) $\to$ Dense(32) $\to$ Dropout(0.2) $\to$ Dense(4, softmax).
- **Parameter Count**: 32,548 parameters.

### 4.9 Live Inference Loop (`run_bee_vit_lstm.py`)
- **Ring Buffers**: `track_embeddings = defaultdict(lambda: deque(maxlen=16))`.
- **Prediction Timing**: Evaluated for active tracks inside the polygon whenever `len(track_embeddings[tid]) == 16`.
- **Smoothing**: none (raw argmax / softmax probability output).
- **Stale-Track Handling**: none (no pruning of inactive tracks; memory grows monotonically with track IDs).
- **Causal Issues**: Model forward pass uses `lstm_out[:, -1, :]` on a bidirectional LSTM, combining forward state at step 15 with reverse state from step 15 (which is the beginning of the reverse sequence). Furthermore, online kinematics are computed frame-to-frame without the convolution smoothing applied during offline training.

---

## 5. Results (with Sources)

### 5.1 Metrics Extracted Across Logs, JSONs, and Reports
| Model / Pipeline | Split & Support | Metrics Found | Source File | Status / Verification Note |
|---|---|---|---|---|
| **YOLOv8m Baseline** (640px) | Upstream Kaggle Val (1,759 images) | Epoch 46: P 0.9159, R 0.8790, mAP50 0.9260, mAP50-95 0.6983 | `yolov8\Result\results.csv` | Verified from CSV log |
| **YOLOv8m Baseline** (640px) | Local GPU Val (1,759 images) | P 0.9577, R 0.8734, mAP50 0.9367, mAP50-95 0.7078 | `SESSION_EXPORT.md` §5.1 | Verified log entry in markdown |
| **YOLOv8m Retrain** (960px) | Local GPU Val (1,759 images) | Ep 1: mAP50 0.9443; Ep 2: 0.9221; Ep 3: P 0.9520, R 0.8983, mAP50 0.9530, mAP50-95 0.7087 | `runs\train\yolov8m_960\results.csv` & `runs\train_log.txt` | Verified; stopped after epoch 3 |
| **Legacy 9-Feature LSTM** | Test split (unspecified random split) | Loss 0.0957, Accuracy 96.33%, Macro-F1 0.9418, Weighted-F1 0.9650 | `LSTM WITH speed & distance\bee_behavior_test_metrics.json` | Verified from JSON artifact |
| **Legacy 7-Feature LSTM** | Test split (11,075 windows; Defense: 695, Fan: 8164, Forage: 974, Wash: 1242) | Loss 0.0833, Accuracy 96.85%, Macro-F1 0.9416, Weighted-F1 0.9689 (Defense F1: 0.9324, Fan: 0.9820, Forage: 0.8740, Wash: 0.9780) | `LSTM WITHOUT speed & distance\bee_behavior_test_metrics_7features.json` | Verified from JSON artifact |
| **Kinematics-Only Control** | Time-blocked Val (379 windows; Forage: 1, Def: 1, Fan: 135, Wash: 22, Other: 220) | Accuracy 72.56%, Macro-F1 0.4574, Weighted-F1 0.7365 (Forage F1: 0.0, Def: 0.2222, Fan: 0.7009, Wash: 0.5846, Other: 0.7793) | `vit_lstm_models\benchmark_summary.json` | Verified. Matched to "Pre (Kinematics)" in report |
| **Fused ViT-LSTM** (DINOv2 + Kinematics) | Time-blocked Val (379 windows) | Accuracy 80.47%, Macro-F1 0.6365, Weighted-F1 0.8106 (Forage F1: 0.0, Def: 1.0000, Fan: 0.8197, Wash: 0.5263, Other: 0.8363) | `vit_lstm_models\benchmark_summary.json` | Verified from benchmark JSON |
| **Visual-Only DINOv2** | Time-blocked Val (379 windows) | Accuracy 83.11%, Macro-F1 0.6485, Weighted-F1 0.8333 (Forage F1: 0.0, Def: 1.0000, Fan: 0.8387, Wash: 0.5417, Other: 0.8621) | `vit_lstm_models\benchmark_summary.json` | Verified from benchmark JSON |
| **Baseline End-to-End Pipeline** | Full `20230711a-fan.mp4` (6,000 frames) | 6,000 frames, 3,788 tracks, 167,292 predictions, 405 s wall clock (14.8 FPS) | `SESSION_EXPORT.md` §5.3 | Verified from session export |
| **ViT-LSTM End-to-End Benchmark** | Full `20230711a-fan.mp4` (6,000 frames) | 236.8 s wall clock, 25.33 FPS, 295 tracks classified (Other: 274 / 92.9%, Fan: 17 / 5.8%, Forage: 4 / 1.4%, Def: 0, Wash: 0), 3.2 GB VRAM | `Bee_Behavior_Classification_Report_Clean.docx` Table 5 | **FLAGGED**: Hardcoded in `create_report_docx.py`. No execution log or raw output file exists in the repo. |

---

## 6. Issues and Gaps

### 6.1 Mismatched / Non-Causal Feature Smoothing
- **Reference**: `extract_dinov2_features.py` (Lines 148–156) vs. `run_bee_vit_lstm.py` (Lines 212–225).
- **Issue**: During offline feature extraction, 50 fps coordinates are smoothed using `np.convolve(xs, [0.25, 0.5, 0.25], mode="same")`, which accesses the future frame coordinate $x_{t+1}$ (non-causal). In online live inference (`run_bee_vit_lstm.py`), positions are unsmoothed frame-to-frame differences. The model is trained on smooth data but evaluated on noisy raw detections.

### 6.2 Speed Feature Bug in Legacy Pipeline
- **Reference**: `run_bee_behavior.py` (Line 926) vs. `LSTM WITH speed & distance\bee_feature_scaler.pkl`.
- **Issue**: In `run_bee_behavior.py`, `speed = distance * fps` ($50\times \text{distance}$). However, inspecting the serialized `StandardScaler` reveals that feature 4 (`distance`) and feature 5 (`speed`) have identical means (`0.0016126`) and standard deviations (`0.0031541`). The scaler was fit on unmultiplied distance/speed. In live inference, speed is normalized to $\sim 25$ standard deviations above the mean, distorting the input space.

### 6.3 Statistically Invalid Validation Support for Rare Classes
- **Reference**: `dinov2_features\manifest.json` (Lines 123–137) and `vit_lstm_models\benchmark_summary.json` (Lines 6–17).
- **Issue**: The time-blocked validation set across all five videos contains only **1 window of Foraging** and **1 window of Defense** (out of 379 total windows). As a consequence:
  - Defense F1 is reported as 1.0000 (1/1 correct).
  - Foraging F1 is reported as 0.0000 (0/1 correct).
  Reporting high confidence or severe failure based on a sample size of $n=1$ is statistically unreliable.

### 6.4 Mismatched Train vs. Val Stride and Subsampling Constraints
- **Reference**: `extract_dinov2_features.py` (Lines 242–251 vs. Lines 281–293).
- **Issue**: Training windows are generated using 6 phase offsets ($0..5$) with an overlapping stride of 1 frame ($67,366$ windows). Validation windows are strictly restricted to phase 0 with a non-overlapping stride of 16 frames ($379$ windows). Furthermore, offline extraction strictly rejects any frame sequence where the gap is not exactly $K=6$, whereas online inference (`run_bee_vit_lstm.py` L232) admits frames whenever $\Delta t \ge 6$.

### 6.5 Stale-Window Redundant Predictions in Live ViT-LSTM
- **Reference**: `run_bee_vit_lstm.py` (Lines 232, 256–273).
- **Issue**: In `run_bee_vit_lstm.py`, a new crop embedding is appended only every $K=6$ frames. However, behavior prediction is triggered on *every frame* where `len(track_embeddings[tid]) == window_length`. Once the deque reaches 16 items, the model runs forward inference on identical stale sequences 5 out of every 6 frames.

### 6.6 Unbounded Memory Leak in Live Inference Loop
- **Reference**: `run_bee_vit_lstm.py` (Lines 144–150, 220–225, 251, 272).
- **Issue**: Dictionaries `track_embeddings`, `track_last_sampled_frame`, `track_prev_data`, and `track_predictions` store history by `track_id` with no timeout or removal mechanism. As ByteTrack generates thousands of track IDs over time, Python process memory grows monotonically.

### 6.7 Reverse-Direction Information Mismatch in Bidirectional LSTM
- **Reference**: `train_vit_lstm.py` (Lines 52, 64, 84–87).
- **Issue**: `BeeLSTMClassifier` uses `bidirectional=True` and extracts `final_step = lstm_out[:, -1, :]`. In PyTorch, index `-1` on a bidirectional LSTM output gives the forward LSTM hidden state at step 15, but the backward LSTM hidden state at the final time index (which corresponds to input step 0 of the sequence). It does not provide the backward summary of the sequence.

### 6.8 Missing Random Seeds
- **Reference**: `train_vit_lstm.py` (Lines 165, 168).
- **Issue**: Neither `torch.manual_seed()` nor `np.random.seed()` is called. DataLoader batch shuffling and neural network layer initialization are non-deterministic, preventing exact reproduction of the reported F1 scores.

### 6.9 Hardcoded Environment Paths
- **Reference**: `data.yaml` (Line 1), `runs\train\yolov8m_960\args.yaml` (Lines 4, 15, 114), `yolov8\Result\args.yaml` (Line 4).
- **Issue**: Absolute Windows paths (`C:\Users\fizan\OneDrive\Documents\BEECLASSIFICATION\...`) and Kaggle virtual paths (`/kaggle/working/...`) are hardcoded.

### 6.10 Disconnected Baseline Attribution in Report
- **Reference**: `create_report_docx.py` (Table 2 & Table 3) and `Bee_Behavior_Classification_Report_Clean.docx`.
- **Issue**: The report labels the "Pre (Kinematics)" baseline as having Macro-F1 0.4574 and Accuracy 72.56%. These numbers come from `Kinematics_Only_Control` (a PyTorch model trained on the DINOv2 5-class dataset), not the original legacy Keras 4-class LSTM (`best_bee_lstm.keras`), which claimed 96.33% accuracy on its own split.

### 6.11 Untraced Benchmark Numbers in Final Report
- **Reference**: `create_report_docx.py` (Lines 314–326).
- **Issue**: The 6,000-frame benchmark results in Table 5 (236.8 s, 25.33 FPS, 295 tracks classified, 0 false alarms) are hardcoded strings in `create_report_docx.py`. No log file, console transcript, or JSON summary exists in the repo to verify that this benchmark was run and produced these values.

---

---

## 7. Phase 0 Correctness Fixes Resolution Summary

Phase 0 has been completed and verified across all criteria:
- **BiLSTM Pooling Fixed**: Resolved Issue 6.7 (`torch.cat([h_n[-2], h_n[-1]], dim=-1)` implemented).
- **Causal Relative Kinematics**: Resolved Issue 6.2 and 6.4 (`kinematics.py` enforces pure scale invariance, trailing EMA $\alpha=0.6$, gap reset rule, and shared offline/online parity).
- **Event-Driven Prediction & Pruning**: Resolved Issues 6.5 and 6.6 (`run_bee_vit_lstm.py` now runs inference strictly on sample arrival, holds predictions, and prunes tracks unseen > 2.0s).
- **Train/Tune/Test Protocol & Seed Logging**: Resolved Issues 6.8 and early-stopping contamination (`train_vit_lstm.py` implements track-grouped Tune splits, untouched Test evaluation, 5-seed mean $\pm$ sd, track-clustered bootstrap 95% CIs, and records every run to `results/experiments.csv`).
- **Relative Paths & Config**: Resolved Issue 6.9 (`config.py` centralizes repository paths and reproducibility seed).
- **Phase 0 Documentation**: Detailed report committed in `results/phase_0.md`.
