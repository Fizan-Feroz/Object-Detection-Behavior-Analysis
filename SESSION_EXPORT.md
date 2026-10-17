# Session Export — Bee Behavior Detection

**Project:** `C:\Users\fizan\OneDrive\Documents\BEECLASSIFICATION`
**Repository:** https://github.com/abdullasiyad/Object-Detection-Behavior-Analysis
**Branch:** `main` (uncommitted work at time of export)
**Date:** 2026-10-17
**Platform:** Windows 11, PowerShell 5.1, Python 3.10.0, RTX 4060 Laptop 8GB (driver 610.74)

---

## 1. Objective

Continue development of a bee detection + behavior classification system:

```
Video → YOLOv8m → ByteTrack → 9-feature extraction → 10-frame sequence → LSTM → Behavior
```

Behavior classes: `Fanning`, `Defense`, `Foraging`, `Washboarding`

---

## 2. Environment Setup

### 2.1 Clone

Repository cloned into a previously empty `BEECLASSIFICATION` folder. Upstream has 5 commits.

### 2.2 Virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

**Final working pins** (`requirements.txt`):

| Package | Version |
|---|---|
| ultralytics | 8.4.177 |
| tensorflow | 2.21.0 |
| keras | 3.12.4 |
| opencv-python | 5.0.0.93 |
| numpy | 2.2.6 |
| joblib | 1.6.0 |
| scikit-learn | 1.7.2 |
| torch | 2.14.1+cu126 |
| torchvision | 0.29.1+cu126 |
| lap | >=0.5.12 |

### 2.3 Dependency troubleshooting (important)

**Problem 1 — TensorFlow/Keras version conflict.**
`best_bee_lstm.keras` fails to load:

```
ValueError: Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

The saved `.keras` archives contain `"quantization_config": null` inside `Dense` layer configs, which Keras 3.12 rejects.

**Attempted fix that FAILED — do not repeat.** Downgrading to `tensorflow==2.18.0` + `keras==3.8.0` corrupted the installation:

```
ModuleNotFoundError: No module named 'tensorflow.python.trackable'
```

Recovery required:

```powershell
.\.venv\Scripts\pip.exe install --force-reinstall --no-cache-dir "tensorflow==2.21.0"
```

**Actual fix** — `load_lstm_compat()` in `run_bee_behavior.py`. On `TypeError` containing `quantization_config`, it copies the model to `.cache/` and strips the key from `config.json`:

```python
raw = raw.replace(', "quantization_config": null', "")
raw = raw.replace('"quantization_config": null, ', "")
raw = raw.replace('"quantization_config": null', "")
```

Verified: both models load — `(None,10,9)->(None,4)` and `(None,10,7)->(None,4)`.

**Problem 2 — GPU not available.**
Machine has an RTX 4060 but stock PyPI torch is CPU-only:

```
torch 2.14.1+cpu / cuda_available False
```

Fix:

```powershell
.\.venv\Scripts\pip.exe install --force-reinstall --no-cache-dir torch torchvision `
    --index-url https://download.pytorch.org/whl/cu126
```

Result: `torch 2.14.1+cu126`, CUDA 12.6, `cuda_available True`. (~2.6 GB download.)

Note: TensorFlow stays CPU-only — TF >= 2.11 has no native Windows GPU support. Not a problem, the LSTM is tiny relative to YOLO.

---

## 3. Datasets

### 3.1 Automatic download is impossible

`data.mendeley.com` sits behind Cloudflare. Every public-API endpoint returns:

```
403 — "Just a moment..." (Cloudflare bot challenge)
```

Tested and rejected:

- `data.mendeley.com/public-api/datasets/8gb9r2yhfc/versions/6`
- `data.mendeley.com/public-api/datasets/8gb9r2yhfc/files`
- `api.mendeley.com/datasets/8gb9r2yhfc/versions/6`
- `data.mendeley.com/datasets/8gb9r2yhfc/6/files.json`
- S3 cache path → `NoSuchBucket`

Manual download required from **https://doi.org/10.17632/8gb9r2yhfc.6** (v6, CC BY 4.0).

### 3.2 Contents received

User downloaded manually into `Datasets/` (note capital D; script accepts both cases):

| Folder | Contents |
|---|---|
| `detection/` | 8 sessions `_bee_20230609a-e`, `_bee_20230711a-c`, 13,867 files, `images/` + `labels/` |
| `tracking_and_behavior/` | `20230711a-fan.mp4` (128 MB), `20230711b-fan.mp4`, `20230609b-def.mp4`, `yt5.mp4`, `yt8.mp4`, `tracks_*.txt`, `tracks_and_behavior_classes_*.txt`, `entrance_zone_*.txt` |
| `pose/` | 400 frames |
| `segmentation/` | 2,300 crops |
| `ramp_detection/` | 156 images |

Dataset totals (v6): detection 7,200 frames · segmentation 2,300 · pose 400 · ramp 156 · tracking/behavior 17,162 tracks over 21,946 frames. Source resolution 1920x1080.

### 3.3 Verification

`download_dataset.py --check-only`:

```
Manifest rows : 6658
Found pairs   : 6658
Missing images: 0
Missing labels: 0
By split      : {'train': 4385, 'val': 1759, 'test': 514}
```

YOLO layout built via `--build-yolo` (13,316 files copied):

```
dataset_yolo/images/train  4385   labels/train  4385
dataset_yolo/images/val    1759   labels/val    1759
dataset_yolo/images/test    514   labels/test    514
```

### 3.4 Split integrity — CLEAN (verified)

Splits are **session-level**, not random frames. No leakage:

| Session | train | val | test |
|---|---|---|---|
| _bee_20230609a | 699 | | |
| _bee_20230609b | 963 | | |
| _bee_20230609c | 787 | | |
| _bee_20230609d | 616 | | |
| _bee_20230609e | | 903 | |
| _bee_20230711a | 1320 | | |
| _bee_20230711b | | 856 | |
| _bee_20230711c | | | 514 |

- 6658 unique image names of 6658 rows (no duplicates)
- Near-duplicate cross-split neighbour pairs (frame-id gap <= 6): **0**

Conclusion: the reported metrics are trustworthy, not inflated by frame leakage.

---

## 4. Performance Work

### 4.1 Critical bottleneck found

The inference loop ran at **0.50 fps → 199 minutes** for the 6,000-frame video.

Measured cause: 1,533 individual `lstm.predict()` calls per 60 frames. Keras 3's `predict()` rebuilds a data adapter and retraces on every call — ~75 ms even for this tiny model.

### 4.2 Benchmark

| Approach (26 tracks) | ms/frame |
|---|---|
| 26x `model.predict()` (original) | 1411.58 |
| 26x direct `__call__` | 758.56 |
| 26x `tf.function` (per item) | **19.38** |
| 1x `tf.function` (batched) | 1.32 |

**Correctness check: `max abs diff = 0.0`** — bit-identical to `predict()`.

### 4.3 Fix applied

Three surgical edits to `run_bee_behavior.py` (no restructuring of the ~1,400-line file):

1. After LSTM load, wrap in a cached function:
   ```python
   LSTM_FN = _tf.function(lambda t: lstm(t, training=False), reduce_retracing=True)
   ```
2. `predict_behavior()` uses `LSTM_FN(...)` when available, else falls back to `lstm.predict()`.
3. Pre-resolved `BEHAVIOR_CLASSES = list(encoder.classes_)` to avoid a per-track sklearn `inverse_transform` call.

**Verified identical over 200 random sequences before trusting it.**

### 4.4 Result

Full pipeline: **0.50 fps → 14.8 fps (405 s total)**. A 30x speedup with zero behavioural change.

---

## 5. Baselines

### 5.1 Detection — baseline (imgsz=640, 46 epochs)

Validated on GPU, 48 s:

| Metric | Value |
|---|---|
| precision | 0.9577 |
| recall | 0.8734 |
| mAP50 | 0.9367 |
| mAP50-95 | 0.7078 |

Consistent with upstream Kaggle run (`yolov8/Result/results.csv`, epoch 46): P 0.916 / R 0.879 / mAP50 0.926 / mAP50-95 0.698.

### 5.2 LSTM

| Model | Input | Test acc | macro-F1 | weighted-F1 |
|---|---|---|---|---|
| 9-feature | (10, 9) | 96.33% | 0.9418 | 0.9650 |
| 7-feature | (10, 7) | **96.85%** | 0.9416 | 0.9689 |

7-feature per-class F1: Defense 0.932 · Fanning 0.982 · Foraging 0.874 · Washboarding 0.978.

Note: dropping `speed`/`distance` **improved** accuracy — those features are not earning their place.

### 5.3 End-to-end pipeline (imgsz=640)

`videos/20230711a-fan.mp4` — 6000 frames, 50 fps, 1920x1080:

| Metric | Value |
|---|---|
| Frames processed | 6000 / 6000 |
| Unique tracks | **3788** |
| LSTM predictions | 167,292 |
| Wall clock | 405 s (~14.8 fps) |

---

## 6. Problems Identified

### 6.1 Track ID inflation — 3788 vs 1674 ground truth

ByteTrack fragments tracks **2.3x** more than the annotation. This directly harms the LSTM: a fragmented track restarts its 10-frame window and never produces a clean sequence.

Likely cause: `CONF_THRESHOLD = 0.50` combined with ByteTrack defaults dropping detections and breaking continuity. Tuning `track_high_thresh` / `track_low_thresh` / `new_track_thresh` is the lever.

### 6.2 Test-video labels are degenerate

Ground truth for `20230711a-fan.mp4` only contains two classes:

| Class | Rows | Share |
|---|---|---|
| Foraging | 99,876 | 78.36% |
| Fanning | 27,576 | 21.64% |
| Defense | 0 | 0.00% |
| Washboarding | 0 | 0.00% |

By track: Foraging 1,649 · Fanning 25 · Defense 0 · Washboarding 0.

So end-to-end accuracy on this video measures a **2-class problem**. Other videos carry the missing classes: `20230609b-def.mp4` (Defense), `20230711b-fan.mp4`.

Also: 99.6% of GT boxes fall inside the entrance-zone polygon, but the pipeline predicts on every detection:

```
polygon = np.array([[100, 220], [1860, 270], [1805, 590], [145, 545]])
```

GT track length: min 1, median 4, p90 181, max 5994 frames.

---

## 7. Retrain Attempt (960px) — STOPPED

### 7.1 Motivation

Bees are tiny. A sample GT label `0 0.344792 0.834722 0.033333 0.073148` at 1920x1080 = **~63x79 px**. At `imgsz=640` the 1920x1080 frame is letterboxed to 640x360, shrinking bees to **~21x26 px** — too small, which explains why recall is the weakest detection metric. At `imgsz=960` bees become ~32x40 px.

Secondary: the upstream run **early-stopped at epoch 46 of 80** (`patience=10`) with mAP still fluctuating — it was under-trained.

### 7.2 Configuration

Created `train_yolo.py`:

```python
imgsz=960, epochs=150, patience=50, batch=8, device=0, workers=2,
seed=0, deterministic=True, optimizer="auto",
cos_lr=True, close_mosaic=15,
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
flipud=0.0, fliplr=0.5, mosaic=1.0,
amp=True, plots=True, val=True, save_period=10,
project="runs/train", name="yolov8m_960"
```

Added `data.yaml`:

```yaml
path: C:/Users/fizan/OneDrive/Documents/BEECLASSIFICATION/dataset_yolo
train: images/train
val: images/val
test: images/test
names: {0: bee}
nc: 1
```

### 7.3 OOM at batch=16

First launch OOM'd on the 8 GB card:

```
[W CUDACachingAllocator.cpp:3934] memory allocation failed with OOM on device 0
while trying to allocate 536870912 bytes (free: 0, total: 8585216000)
```

Peak was 7831 / 8188 MiB. Reduced to `batch=8` → stable at ~7.8 GB, 100% GPU util.

### 7.4 Results before stopping

| Epoch | cum. time | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 1 | 583 s | 0.9050 | 0.8911 | 0.9443 | 0.7288 |
| 2 | 898 s | 0.8838 | 0.8847 | 0.9221 | 0.6424 |
| 3 | 1264 s | 0.9520 | 0.8983 | 0.9530 | 0.7087 |

Stopped by user after 3 epochs. ETA to 150 was ~15-17 h.

### 7.5 Honest assessment

**No improvement can be claimed from this run.** Three epochs is far less training than the baseline's 46, and early-epoch metrics at high resolution swing hard (epoch 2 fell to 0.9221 before rebounding). The apparent mAP50 gain (0.9530 vs 0.9367) is not a valid comparison.

Checkpoints preserved in `runs/train/yolov8m_960/weights/`: `best.pt`, `last.pt`, `epoch0.pt` (148.3 MB each). Original `yolov8/weights/best.pt` untouched.

---

## 8. Files Changed

### Modified
- `run_bee_behavior.py`
  - standalone-Keras import fallback
  - `load_lstm_compat()` — strips `quantization_config`
  - `DEVICE` auto-detection + `device=DEVICE` on `yolo.track()`
  - `.cache/` for the patched model (was system temp)
  - `LSTM_FN` tf.function fast path + `BEHAVIOR_CLASSES` pre-resolution

### Added
- `requirements.txt` — pinned deps
- `.gitignore` — `.venv/`, `.cache/`, `Datasets/`, `dataset/`, `downloads/`, `dataset_yolo/`, `videos/*.mp4`, `runs/`
- `data.yaml` — YOLO dataset config
- `download_dataset.py` — `--check-only` / `--extract` / `--build-yolo` / `--check-videos`
- `train_yolo.py` — 960px retrain config
- `videos/.gitkeep`

### Untracked at export
```
 M run_bee_behavior.py
?? .gitignore
?? data.yaml
?? download_dataset.py
?? requirements.txt
?? train_yolo.py
?? videos/
```

Nothing has been committed.

---

## 9. Local-Only Paths (gitignored)

`.venv/`, `.cache/`, `Datasets/`, `dataset/`, `downloads/`, `dataset_yolo/`, `videos/*.mp4`, `runs/`, `yolov8m.pt`

---

## 10. Recommended Next Steps

1. **ByteTrack tuning** (cheap, high value) — reduce 3788 tracks toward the 1674 ground truth. Fragmented tracks starve the LSTM.
2. **Fair 960px comparison** — `epochs=40, patience=15` at 960px ≈ 4 h, enough to compare against the 46-epoch baseline.
3. **LSTM class weighting** — Defense (695) and Foraging (974) are swamped by Fanning (8164); weighted loss targets macro-F1. Requires writing a training script (only `.keras` artifacts exist in repo).
4. **Adopt the 7-feature model** — it already beats the 9-feature one.
5. **Evaluate on `20230609b-def.mp4`** to actually exercise the Defense class.
6. **Consider entrance-zone filtering** — GT only annotates inside the polygon, but inference covers the whole frame.

---

## 11. Reproduction Commands

```powershell
# setup
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# verify dataset
.\.venv\Scripts\python.exe download_dataset.py --check-only

# rebuild YOLO splits
.\.venv\Scripts\python.exe download_dataset.py --build-yolo

# validate detector
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8/weights/best.pt').val(data='data.yaml', device=0, imgsz=640)"

# full pipeline
.\.venv\Scripts\python.exe run_bee_behavior.py

# retrain
.\.venv\Scripts\python.exe train_yolo.py
```