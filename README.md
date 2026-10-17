# 🐝 Automated Honeybee Behavior Detection & Classification

An end-to-end computer vision and deep learning system designed to detect individual honeybees (*Apis mellifera*), track their trajectories across video frames, and classify colony behaviors (Fanning, Defense, Foraging, Washboarding, and Other/Idle) at the beehive entrance.

---

## 📌 Architecture Overview

```text
Input Video (50 fps)
      │
      ▼
YOLOv8m Object Detection (imgsz=640 or 960)
      │
      ▼
ByteTrack Multi-Object Tracker (Track ID assignment)
      │
      ├─────────────────────────────────────────┐
      │ (RGB bee crop)                          │ (bounding box sequence)
      ▼                                         ▼
DINOv2 ViT-S/14 Embedder (384-d)          Causal Kinematics Tracker (7-d)
      │                                         │
      ▼                                         │
Balanced Linear Projection (384 → 64)           │
      │                                         │
      └───────────────────┬─────────────────────┘
                          │ (Fused 71-d sequence: T=6 or 16 steps, K=6 subsampling)
                          ▼
              Bidirectional PyTorch LSTM
                          │
                          ▼
            Temporal Smoothing & Hysteresis
                          │
                          ▼
           Behavior Classification & Confidence
     (Fanning, Defense, Foraging, Washboarding, Other)
```

---

## 📊 Dataset & Setup

This project uses the benchmark dataset:
- **Title:** Labeled dataset for bee detection and direction estimation on beehive landing boards
- **Repository:** [Mendeley Data v6 (DOI: 10.17632/8gb9r2yhfc.6)](https://data.mendeley.com/datasets/8gb9r2yhfc/6)
- **Sessions Annotated:** 5 video sessions (`20230609b-def`, `20230711a-fan`, `20230711b-fan`, `yt5`, `yt8`) featuring distinct camera angles, lighting conditions, and behavioral events.

### Installation

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
source .venv/bin/activate      # Linux / macOS

# 2. Install pinned dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics opencv-python numpy scikit-learn python-docx
```

---

## 🚀 Running Inference & Live Preview

### 1. DINOv2 ViT + LSTM (Production GPU Pipeline)
Runs YOLOv8m detection, ByteTrack tracking, causal kinematics extraction, DINOv2 visual embedding, and PyTorch BiLSTM classification:

```bash
# Run default visual/fused pipeline
python run_bee_vit_lstm.py

# Run with fused visual + kinematics representation
python run_bee_vit_lstm.py --mode fused

# Run on a custom video clip and export annotated video
python run_bee_vit_lstm.py --video videos/20230711a-fan.mp4 --save-out videos/annotated_output.mp4
```

**Key Inference Features:**
- **Event-Driven Execution:** Forward LSTM inference runs only when a new subsampled embedding is appended (every $K=6$ frames), holding prior predictions between samples to prevent redundant computation.
- **Track Dictionary Pruning:** Trajectories unseen for $>2.0$ seconds are purged from memory, preventing long-running memory leaks.
- **Temporal Hysteresis & Smoothing:** Exponential moving average probability smoothing with hysteresis margin ($0.10$) and confidence thresholding ($<0.45 \implies$ fallback to `"Other"`).

### 2. Legacy Kinematics Baseline (Reference Control)
```bash
python run_bee_behavior.py
```

---

## 🔬 Rigorous Evaluation Protocols (Phases 0–2)

This repository follows strict scientific evaluation rules:
1. **Multi-Seed Evaluation:** All benchmarks are executed over at least 5 seeds (`42, 43, 44, 45, 46`); all runs log configuration, split, and metrics directly to [`results/experiments.csv`](file:///results/experiments.csv).
2. **Dual Support Reporting:** Always reports support in both non-overlapping **windows** AND distinct **tracks**.
3. **Track-Clustered Bootstrap CIs:** Reports 95% confidence intervals clustered by track identity ($B=1000$).
4. **Untouched Test Splits:** Early stopping and hyperparameter tuning occur strictly on a track-grouped Tune set, never on the reported test blocks.
5. **Pre/Post Control Standard:** The Pre baseline is the **Kinematics-Only Control Model** evaluated on identical folds, not the legacy Keras model.

---

### Phase 1 Benchmark Summary: Blocked 5-Fold Cross-Validation

*Evaluated across 633 pooled out-of-fold test windows and 359 distinct tracks with $\ge 250$-frame temporal guard intervals (5 seeds, mean $\pm$ sd):*

| Model Architecture | Input Dim | Out-of-Fold Macro-F1 | Out-of-Fold Weighted-F1 | Out-of-Fold Accuracy | Seed 42 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kinematics Control (Pre)** | 7 | $0.2512 \pm 0.0129$ | $0.6545 \pm 0.0140$ | $0.6954 \pm 0.0157$ | $[0.2234, 0.2812]$ |
| **Visual-Only (DINOv2)** | 384 | $0.5327 \pm 0.0182$ | $0.7799 \pm 0.0154$ | $0.7826 \pm 0.0168$ | $[0.5000, 0.5988]$ |
| **Fused Model (Post: Proj 64 + Kin 7)** | 71 | **$0.5401 \pm 0.0126$** | **$0.7708 \pm 0.0060$** | **$0.7665 \pm 0.0062$** | **$[0.5089, 0.6028]$** |

**Per-Class Support & F1:**
- **Defense:** F1 = **$0.9054 \pm 0.0244$** | Support: **44 windows (14 tracks)** (resolved single-window failure in legacy splits).
- **Washboarding:** F1 = **$0.4296 \pm 0.0426$** | Support: **34 windows (30 tracks)** (kinematics-only completely fails at $0.0000$).
- **Fanning:** F1 = **$0.5274 \pm 0.0352$** | Support: **83 windows (31 tracks)**.
- **Other / Idle:** F1 = **$0.8380 \pm 0.0058$** | Support: **465 windows (280 tracks)**.
- **Foraging:** F1 = **$0.0000 \pm 0.0000$** | Support: **7 windows (4 tracks)** (identified window duration mismatch).

---

### Shortcut Diagnostics (Empirical Proof of Video Leakage)

To test whether deep representations rely on genuine insect motion or ambient video context, three diagnostic probes were evaluated:
1. **Position-Only Classifier ($16 \times 4$ coords):** Macro-F1 = **$0.4451 \pm 0.0316$**, Accuracy = $0.6667$. Spatial coordinates alone act as a strong prior (guards cluster along the slit; washboarders on the ramp).
2. **Background Patch Classifier (No Bee):** Macro-F1 = **$0.8298 \pm 0.0605$**, Accuracy = **$92.13\%$**! Background wood grain, lighting, and camera angle pseudo-classify behavior without the bee present.
3. **Video ID Linear Probe ($384 \to 5$):** Accuracy = **$100.0\% \pm 0.0\%$**. A single linear layer identifies the source video perfectly from visual embeddings, explaining why cross-video generalization drops in Leave-One-Video-Out (LOVO).

---

### Phase 2 Summary: Foraging & Window Horizon Ablations

*Factorial ablation over $T \in \{6, 8, 16\}$ and polygon gating rules (`all_inside`, `half_inside`, `endpoint_inside`):*

- **Root Cause Identified:** Across 1,842 raw Foraging tracks, median bout length is **3.0 frames ($0.06$ s)** and P90 is **20.0 frames ($0.40$ s)**. Requiring $T=16$ at $K=6$ ($91$ continuous frames / $1.82$ s) eliminates **$99.57\%$ of all Foraging tracks** ($1,842 \to 8$).
- **Optimal Horizon ($T=6$, `half_inside` at $K=6$):**
  - Out-of-Fold Macro-F1: **$0.3988 \pm 0.0094$** (Seed 42 95% Bootstrap CI: $[0.3720, 0.4252]$).
  - Out-of-Fold Accuracy: **$0.7299 \pm 0.0071$**.
  - Foraging F1: **$0.6700 \pm 0.0195$** (Support: **182 test windows across 156 distinct tracks** — a $52\times$ increase in track diversity).

---

## 🛠️ Repository Scripts & Tools

| Script | Purpose |
|---|---|
| `run_bee_vit_lstm.py` | Real-time GPU inference pipeline (YOLOv8m + ByteTrack + DINOv2 + PyTorch BiLSTM). |
| `train_vit_lstm.py` | Multi-seed trainer for Fused, Visual, and Kinematics models with track-clustered bootstrap CIs. |
| `train_phase1.py` | Complete Phase 1 suite: Blocked 5-Fold, LOVO cross-validation, and 3 shortcut diagnostic probes. |
| `audit_foraging_and_gating.py` | Granular audit script analyzing contiguous run lengths and track attrition funnels. |
| `train_phase2_kinematics.py` | Systematic $3 \times 3$ factorial ablation runner over $T \in \{6, 8, 16\}$ and polygon gating rules. |
| `create_report_docx.py` | Automated report generator building the Word `.docx` report directly from `results/*.json`. |
| `kinematics.py` | Shared causal scale-invariant kinematics tracker (enforces offline/online parity). |
| `config.py` | Centralized paths, video metadata, and reproducibility seed utilities. |

---

## 📄 License & Attribution

Developed for scientific honeybee behavioral research and automated apiary monitoring. Built on Meta DINOv2, Ultralytics YOLOv8, and OpenCV.
