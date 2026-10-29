# Master Training Plan: Dual-PC Distributed Retraining
## Honeybee Behavior Classification & Tracking Pipeline
**Target Hardware:** 2x Systems with NVIDIA GeForce RTX 2070 Super (8 GB GDDR6 VRAM)  
**Operating System:** Windows 10 / 11  
**Frameworks:** PyTorch 2.1+, CUDA 12.x, Ultralytics 8.4+, DINOv2 ViT-S/14  
**Date:** October 2026  

---

## 1. Executive Summary & Distributed Architecture

### Why Asymmetric Task Parallelism?
On Windows systems, PyTorch lacks native support for NVIDIA's NCCL interconnect library (which is strictly Linux-only). When running PyTorch `DistributedDataParallel` (DDP) across multiple Windows PCs, the framework defaults to the CPU-based `gloo` communication backend over standard local area network (LAN) TCP sockets. Synchronizing 50–150 MB gradient tensors across Gigabit Wi-Fi or Ethernet via Gloo introduces significant CPU serialization and network latency, causing distributed training to run **2× to 5× slower** than running locally on a single GPU.

To maximize throughput across two **RTX 2070 Super (8 GB)** computers without network bottlenecks, this plan adopts **Asymmetric Task & Experiment Parallelism**:
- **PC 1** and **PC 2** execute independent, compute-heavy pipeline stages concurrently.
- No network synchronization during training; artifacts are consolidated at the end of execution.
- Workloads are partitioned 50/50 across video frame extraction, model architectures, and evaluation tasks.

```
┌────────────────────────────────────────────────────────┐
│               VIDEO & DATASET PARTITION                │
│             Mendeley Data v6 (21,953 frames)           │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│           PC 1            │ │           PC 2            │
│  (RTX 2070 Super - 8GB)   │ │  (RTX 2070 Super - 8GB)   │
├───────────────────────────┤ ├───────────────────────────┤
│ • Extract Videos:         │ │ • Extract Videos:         │
│   - 20230609b-def (5,973f)│ │   - 20230711a-fan (6,000f)│
│   - yt5           (977f)  │ │   - 20230711b-fan (6,003f)│
│   - yt8           (3,000f)│ │                           │
│   Subtotal: 9,950 frames  │ │   Subtotal: 12,003 frames │
├───────────────────────────┤ ├───────────────────────────┤
│ • Train Temporal Models:  │ │ • Train Temporal Models:  │
│   - TemporalConvNet (TCN) │ │   - Fused BiLSTM          │
│   - Fused BiGRU           │ │   - Causal UniLSTM        │
│   (5 seeds: 42-46)        │ │   (5 seeds: 42-46)        │
├───────────────────────────┤ ├───────────────────────────┤
│ • Optional Detector:      │ │ • Production Benchmark:   │
│   - YOLOv8m (640 / 960px) │ │   - 6,000-Frame Streaming │
│   - Detection mAP eval    │ │   - MOTA, IDF1, FPS test  │
└─────────────┬─────────────┘ └─────────────┬─────────────┘
              │                             │
              └─────────────┬───────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│            CONSOLIDATION & ARTIFACT MERGE              │
│  - Unified experiments.csv & JSON Benchmark Summaries  │
│  - Formatted Report: Word (.docx) & PDF (.pdf)         │
└────────────────────────────────────────────────────────┘
```

---

## 2. Workload Partition & Timing Matrix

### Workload Matrix

| Pipeline Component | PC 1 Assignment | PC 2 Assignment | Hardware Constraints & Profile |
|---|---|---|---|
| **DINOv2 Feature Extraction** | Videos `20230609b-def`, `yt5`, `yt8` (9,950 frames total) | Videos `20230711a-fan`, `20230711b-fan` (12,003 frames total) | Batch size 64; FP16 AMP; Peak VRAM ~1.8 GB. Takes **~3.5 min** in parallel (vs. 7.5 min on 1 PC). |
| **Window Configuration** | $T=6$, $K=6$, `half_inside` gating | $T=6$, $K=6$, `half_inside` gating | Recovers 156 Foraging tracks (182 windows); eliminates the legacy $T=16$ track discard. |
| **Temporal Models** | **TCN** (1D Conv + ResBlocks, 76.78% Acc) & **BiGRU** (76.56% Acc) | **BiLSTM** (Baseline, 75.64% Acc) & **UniLSTM** (Streaming, 74.60% Acc) | Batch size 32; 5 seeds (`42, 43, 44, 45, 46`); Peak VRAM ~450 MB. Takes **~4 min** in parallel. |
| **Video Benchmark** | Idle / Ready for Detector | Full 6,000-frame streaming on `20230711a-fan.mp4` | Live ByteTrack + DINOv2 + ViT-BiLSTM; Peak VRAM ~272 MB; 42.2 FPS sustained. Takes **~2.5 min**. |
| **Object Detector (Optional)** | YOLOv8m on `dataset_yolo` (4,385 train images) | Optional evaluation / inference serving | 640px: Batch 16, ~2.5 hrs.<br>960px: Batch 8, ~15–17 hrs (overnight run). |

---

## 3. Environment Setup & Prerequisites (Both Computers)

Perform these steps on **both PC 1 and PC 2**:

### Step 3.1: Python Virtual Environment
Open PowerShell inside the repository directory:
```powershell
# Navigate to workspace
cd c:\Users\<username>\...\BEECLASSIFICATION

# Create Python 3.10 virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 3.2: Install PyTorch with CUDA Acceleration
```powershell
# Install PyTorch with CUDA 12.1 / 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 3.3: Install Repository Dependencies
```powershell
pip install -r requirements.txt
pip install ultralytics docx2pdf python-docx reportlab opencv-python scikit-learn
```

### Step 3.4: Verify Hardware Detection
```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA Ready:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```
*Expected Output:* `CUDA Ready: True`, `GPU Name: NVIDIA GeForce RTX 2070 SUPER`.

---

## 4. Execution Protocols

We provide two execution scenarios based on whether you wish to retrain the behavior models only or also retrain the YOLOv8m detector.

### Scenario A: Fast ViT-Behavior Pipeline (Recommended: ~8–10 Minutes Total)
*Uses the existing high-accuracy YOLOv8m detector weights (`yolov8/weights/best.pt`) and retrains the entire vision-kinematics behavior pipeline with Foraging recovery.*

#### **Execution on PC 1:**
You can either double-click `run_pc1.bat` or run via PowerShell:
```powershell
.\.venv\Scripts\python.exe train_dual_pc.py --node pc1 --task all
```
*What this does on PC 1:*
1. Extracts DINOv2 visual features & kinematics for `20230609b-def`, `yt5`, and `yt8` ($T=6$, $K=6$, `half_inside`).
2. Trains **TemporalConvNet (TCN)** across 5 random seeds (42–46).
3. Trains **Fused BiGRU** across 5 random seeds (42–46).
4. Logs metrics to `results/experiments.csv`.

#### **Execution on PC 2 (Run Concurrently with PC 1):**
You can either double-click `run_pc2.bat` or run via PowerShell:
```powershell
.\.venv\Scripts\python.exe train_dual_pc.py --node pc2 --task all
```
*What this does on PC 2:*
1. Extracts DINOv2 visual features & kinematics for `20230711a-fan` and `20230711b-fan` ($T=6$, $K=6$, `half_inside`).
2. Trains **Fused BiLSTM** across 5 random seeds (42–46).
3. Trains **Fused Causal UniLSTM** across 5 random seeds (42–46).
4. Runs the **6,000-frame live video streaming benchmark** on `20230711a-fan.mp4` measuring FPS, latency, and MOTA tracking fidelity.

---

### Scenario B: Full Retrain with YOLO Detector Retraining
*Includes training YOLOv8m from scratch or fine-tuning at 640px or 960px.*

#### **On PC 1 (Dedicated Detector Node):**
- **Option 1: Standard Resolution (640px) — ~2.5 Hours**:
  ```powershell
  .\.venv\Scripts\python.exe train_dual_pc.py --node pc1 --task yolo --yolo-imgsz 640 --yolo-epochs 80
  ```
- **Option 2: High Resolution (960px) — ~15 Hours (Overnight Run)**:
  ```powershell
  .\.venv\Scripts\python.exe train_dual_pc.py --node pc1 --task yolo --yolo-imgsz 960 --yolo-epochs 150
  ```
  *(Note: At 960px, batch size is automatically constrained to 8 to avoid exceeding the 8 GB VRAM limit).*

#### **On PC 2:**
PC 2 proceeds with DINOv2 feature extraction and behavior model training uninterrupted:
```powershell
.\.venv\Scripts\python.exe train_dual_pc.py --node pc2 --task all
```

---

## 5. Artifact Consolidation & Report Generation

Once both PC 1 and PC 2 have finished their assigned tasks:

### Step 5.1: Copy Feature Caches and CSVs
Copy the generated artifacts from PC 1 into PC 2 (or vice versa):
1. **Per-Video Feature Directories:**  
   Copy `dinov2_features\20230609b-def\`, `dinov2_features\yt5\`, and `dinov2_features\yt8\` from PC 1 into the `dinov2_features\` folder on PC 2.
2. **Experiment Log:**  
   Append the newly generated rows from PC 1's `results\experiments.csv` into PC 2's `results\experiments.csv`.

### Step 5.2: Compile Clean Reports
Run the merge task on the primary machine:
```powershell
.\.venv\Scripts\python.exe train_dual_pc.py --task merge
```
This automatically updates and regenerates:
- [Bee_Behavior_Classification_Report_Clean_v2.docx](file:///c:/Users/fizan/OneDrive/Documents/BEECLASSIFICATION/Bee_Behavior_Classification_Report_Clean_v2.docx)
- [Bee_Behavior_Classification_Report_Clean_v2.pdf](file:///c:/Users/fizan/OneDrive/Documents/BEECLASSIFICATION/Bee_Behavior_Classification_Report_Clean_v2.pdf)

---

## 6. Verification & Monitoring Checklist

| Check | Tool / Command | Acceptance Criteria |
|---|---|---|
| **GPU Utilization** | `nvidia-smi -l 2` in separate terminal | GPU Compute > 85%, VRAM usage < 7.2 GB (no OOM). |
| **AMP Mixed Precision** | Console output | `amp=True`, FP16 active across all passes. |
| **Foraging Window Support** | Extracted manifest | Foraging track count $\ge 150$ tracks (validating $T=6$ recovery). |
| **Checkpoint Generation** | `dir vit_lstm_models/` | Model `.pt` weight files updated with recent timestamp. |
| **Live Streaming Latency** | `benchmark_live_6000frames.json` | Throughput $\ge 38$ FPS, mean latency $\le 25$ ms. |

---

## 7. Troubleshooting & Common Issues

1. **CUDA Out of Memory (OOM) on 8GB VRAM:**
   - For YOLO at 960px: Keep batch size at **8** (`--batch 8`).
   - For DINOv2 feature extraction: Default batch size is **64** (~1.8 GB VRAM); do not exceed 128.
2. **Windows File Lock (`PermissionError: [Errno 13]`):**
   - If Microsoft Word or Adobe Acrobat is open with the report, close Word/Acrobat before running the merge task. The script includes a safe fallback that outputs to `..._Clean_v2.docx`.
3. **PowerShell Script Execution Policy:**
   - If `.ps1` scripts fail with an execution policy error, run once in admin PowerShell:
     ```powershell
     Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     ```
