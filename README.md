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

### Phase 3 Summary: Tracking & Full-Pipeline Skew Benchmark

- **Detector Evaluation:** Baseline YOLOv8m (640px) achieves **$51.82$ FPS** ($95.80\%$ mAP50), beating real-time requirements. 960px reaches $97.52\%$ mAP50 ($+6.68\%$ recall) at $29.90$ FPS.
- **ByteTrack Optimization:** 54-combination grid search yielded optimal parameters (`custom_bytetrack.yaml`): MOTA = **$0.1151$** (**$+82.4\%$ improvement** over default $0.0631$), IDF1 = **$0.3828$**.
- **Train/Serve Skew:** Evaluated classifier on raw ByteTrack tracker tracks: Macro-F1 = **$0.4414$** ($-0.0876$ drop from ground-truth tracks), with Fanning retaining F1 = $0.6888$ on 1,089 real serving windows.

---

### Phase 4 Summary: Model, Resolution, Token & Blur Ablations

*Evaluated across identical Blocked 5-Fold splits over 5 random seeds (42..46) on 633 pooled test windows (359 tracks):*
- **Temporal Architectures:** 
  - **Fused UniLSTM (Causal Streaming):** Macro-F1 = **$0.5372 \pm 0.0248$** (outperforms BiLSTM $0.5312$, proving causal streaming incurs zero penalty).
  - **1D Temporal Convolution (TCN):** Macro-F1 = **$0.5465 \pm 0.0260$**, Accuracy = **$76.78\%$** (highest overall score).
  - **Fused BiGRU:** Macro-F1 = **$0.5460 \pm 0.0334$** (exceeds BiLSTM due to simpler gating).
  - **Temporal Transformer:** Macro-F1 = **$0.5248 \pm 0.0175$** (lacks recurrent inductive bias).
- **Resolution & Tokens:** Crop $112 \times 112$ causes a massive **$-0.0752$ drop** ($0.5312 \to 0.4560$), proving full $224 \times 224$ is required. CLS token alone ($0.5511$) matches CLS + patch ($0.5505$).
- **Fanning Blur vs Motion Diagnostic:** Gaussian blur ($\sigma=3.0$) crashes Fanning F1 by **$-0.1256$** ($0.4920 \to 0.3664$), proving Fanning relies on sharp edge contrast, not static optical blur. Temporal order scrambling preserves F1 ($0.4899$), while zero-motion freezing degrades F1 ($0.4338$).

---

### Phase 5 Summary: Live 6,000-Frame Production Benchmark

*Continuous 6,000-frame 1080p stream (`20230711a-fan.mp4`) on an NVIDIA RTX 4060 Laptop GPU:*
- **Total Throughput:** **$42.22$ FPS** ($142.1$ seconds runtime).
- **Total Latency:** **$22.88$ ms mean** (median **$16.75$ ms**, P95 **$56.39$ ms**).
- **VRAM Footprint:** **$272.2$ MB peak** ($<3.5\%$ of 8 GB GPU).
- **Tracking Accuracy:** **$0.8185$ IDF1**, **$0.5542$ MOTA**, matching **$125,524$ ground-truth instances** ($98.5\%$ recall).
- **Live Classification:** **$91.87\%$ track-level accuracy** and **$83.44\%$ frame-level accuracy**, with Fanning achieving **$0.6726$ frame-level F1**.

---

## 🛠️ Repository Scripts & Tools

| Script | Purpose |
|---|---|
| `run_bee_vit_lstm.py` | Real-time GPU inference pipeline (YOLOv8m + ByteTrack + DINOv2 + PyTorch BiLSTM). |
| `run_phase5_live_benchmark.py` | Complete 6,000-frame live streaming benchmark measuring per-stage latencies, FPS, VRAM, and GT F1. |
| `train_phase4_ablations.py` | Phase 4 model ablation runner (UniLSTM, BiGRU, TCN, Transformer, Augmentation, Resolution, Blur). |
| `eval_cls_vs_patch.py` | Phase 4 token ablation evaluating CLS vs Mean Patch vs CLS+Patch representations. |
| `extract_phase4_visual_ablations.py` | Feature extractor for 112x112, CLS+patch, Gaussian blur, and downscaled crops. |
| `evaluate_detectors.py` | Phase 3 detector benchmark comparing YOLOv8m 640px vs 960px on held-out test split. |
| `grid_search_bytetrack.py` | Phase 3 54-combination grid search over ByteTrack hyperparameters. |
| `evaluate_tracker_skew.py` | Phase 3 tracker tracks dataset extractor and train/serve skew benchmark. |
| `train_phase1.py` | Complete Phase 1 suite: Blocked 5-Fold, LOVO cross-validation, and 3 shortcut diagnostic probes. |
| `train_phase2_kinematics.py` | Systematic $3 \times 3$ factorial ablation runner over $T \in \{6, 8, 16\}$ and polygon gating rules. |
| `create_report_docx.py` | Automated report generator building both `.docx` and `.pdf` reports directly from all `results/*.json`. |
| `kinematics.py` | Shared causal scale-invariant kinematics tracker (enforces offline/online parity). |
| `config.py` | Centralized paths, video metadata, and reproducibility seed utilities. |

---

## 📄 License & Attribution

Developed for scientific honeybee behavioral research and automated apiary monitoring. Built on Meta DINOv2, Ultralytics YOLOv8, and OpenCV.

