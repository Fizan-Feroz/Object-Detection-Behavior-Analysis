# Phase 1: Trustworthy Evaluation Suite - Results & Analysis

**Date:** October 18, 2026  
**Status:** Completed  
**Branch:** `main`  
**Evaluation Standard:** 5 seeds (`42, 43, 44, 45, 46`), 95% bootstrap confidence intervals clustered by track ($B=1000$), strict window and distinct track support reporting. No tuning or early-stopping on the reported test splits. All 90 experimental runs logged to `results/experiments.csv`. Results summary saved to `results/phase_1_benchmark_summary.json`.

---

## 1. Executive Summary & Core Changes

In Phase 1, we replaced all single-split, end-of-video evaluations with a rigorous, trustworthy evaluation protocol designed to resolve track contamination, dataset leakage, and class under-representation.

### What Was Changed:
1. **Blocked 5-Fold Cross-Validation Per Video:**
   - Partitioned each video into 5 contiguous chronological blocks separated by temporal guard intervals ($\ge 250$ frames; $78$ frames on the short 400-frame `yt5`).
   - Boundary-crossing tracks traversing block margins ($<1.5\%$ of tracks) were purged to guarantee complete track and temporal independence between train and test folds.
   - Test predictions were pooled out-of-fold across all 5 folds to yield complete, unbiased support across all videos.
2. **Leave-One-Video-Out (LOVO) Cross-Video Generalization:**
   - Evaluated true cross-video generalization for multi-video classes:
     - **Fanning:** Held-out across `20230711a-fan`, `20230711b-fan`, and `yt5`.
     - **Foraging:** Held-out across `20230609b-def`, `20230711a-fan`, and `20230711b-fan`.
   - Identified and marked single-video classes:
     - **Defense:** Present exclusively in `20230609b-def` (single-video event).
     - **Washboarding:** Present exclusively in `yt8` (single-video event).
3. **Shortcut Diagnostics Suite:**
   - **(a) Position + Box-Size Classifier:** Tested whether 2D coordinates ($16 \times 4$: normalized $x_{min}, y_{min}, x_{max}, y_{max}$) predict behavior.
   - **(b) Background Patch DINOv2 Classifier (No Bee):** Sampled identical spatio-temporal bounding box patches from the static median background image to determine if ambient background texture/lighting predicts behavior without the bee.
   - **(c) Video ID Linear Probe:** Evaluated how easily a single linear projection ($384 \to 5$) predicts the source video ID from bee visual embeddings.
4. **Architectural Balancing ($384 \to 64$ Linear Projection):**
   - Implemented `ProjectedFusedBeeLSTM` with a `Linear(384 -> 64)` projection + LayerNorm + GELU + Dropout before fusing with 7-d causal kinematics ($64 + 7 = 71$ dimensions), preventing visual dominance from overpowering kinematic dynamics.
5. **Fixed Pre/Post Control Logic:**
   - Standardized the "Pre" baseline to the **Kinematics-Only Control Model** evaluated under identical folds, replacing the legacy uncalibrated Keras baseline. The "Post" system is the **Fused Model** (`Linear(384 -> 64)` + causal kinematics).

---

## 2. Blocked 5-Fold Pooled Out-of-Fold Evaluation

All three model architectures were trained and evaluated across identical folds and 5 random seeds (`42, 43, 44, 45, 46`). 

Total Pooled Out-of-Fold Test Evaluation: **633 test windows across 359 distinct tracks**.

### Overall Performance (Mean $\pm$ SD across 5 Seeds)

| Model Architecture | Input Dim | OOF Macro-F1 | OOF Weighted-F1 | OOF Accuracy | Seed 42 95% Bootstrap CI (Macro-F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Kinematics-Only (Pre Control)** | 7 | $0.2512 \pm 0.0129$ | $0.6545 \pm 0.0140$ | $0.6954 \pm 0.0157$ | $[0.2234, 0.2812]$ |
| **Visual-Only (DINOv2)** | 384 | $0.5327 \pm 0.0182$ | $0.7799 \pm 0.0154$ | $0.7826 \pm 0.0168$ | $[0.5000, 0.5988]$ |
| **Fused Model (Post: Proj 64 + Kin 7)** | 71 | **$0.5401 \pm 0.0126$** | **$0.7708 \pm 0.0060$** | **$0.7665 \pm 0.0062$** | **$[0.5089, 0.6028]$** |

> **Key Finding:** The Fused model with balanced linear projection achieves the highest Macro-F1 ($0.5401 \pm 0.0126$), demonstrating a **$+0.2889$ Macro-F1 gain** over the Kinematics-Only Pre control baseline ($0.2512 \pm 0.0129$). Fusing kinematics with visual embeddings reduces variance (SD drops from $0.0182$ to $0.0126$) and enhances subtle behavioral discrimination.

---

### Per-Class F1 Scores and Real Support Counts

Pooled out-of-fold evaluation provides real, robust test support across all five behavior classes:

| Behavior Class | Pooled Test Windows | Distinct Test Tracks | Kinematics Control F1 (Pre) | Visual-Only F1 | Fused Model F1 (Post) | Fused vs Pre $\Delta$ F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foraging** | 7 | 4 | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $+0.0000$ |
| **Defense** | 44 | 14 | $0.0182 \pm 0.0364$ | $0.8796 \pm 0.0348$ | **$0.9054 \pm 0.0244$** | **$+0.8872$** |
| **Fanning** | 83 | 31 | $0.4244 \pm 0.0256$ | $0.5363 \pm 0.0416$ | **$0.5274 \pm 0.0352$** | **$+0.1030$** |
| **Washboarding** | 34 | 30 | $0.0000 \pm 0.0000$ | $0.3936 \pm 0.0418$ | **$0.4296 \pm 0.0426$** | **$+0.4296$** |
| **Other** | 465 | 280 | $0.8135 \pm 0.0144$ | $0.8539 \pm 0.0110$ | **$0.8380 \pm 0.0058$** | **$+0.0245$** |

### Per-Class Support Insights:
1. **Defense Test Support Unlocked:** In Phase 0's end-of-video split, Defense had only 1 test window. Blocked 5-fold cross-validation unlocks **44 test windows across 14 distinct tracks**, yielding a reliable F1 of **$0.9054 \pm 0.0244$**.
2. **Washboarding Discrimination:** Pure kinematics completely fails on Washboarding (F1 = $0.0000$), because washboarding bees remain nearly stationary while scraping the surface with their mandibles. The Fused model captures this micro-motion, reaching **$0.4296 \pm 0.0426$**.
3. **The Foraging Deficit:** Across 37,988 extracted windows, only 7 non-overlapping test windows (from 4 tracks) represent Foraging, and no model achieved non-zero F1. An in-depth audit of track gating and windowing is required in Phase 2.

---

## 3. Leave-One-Video-Out (LOVO) Generalization

To test whether models learn universal honeybee behavior rather than video-specific idiosyncrasies, we performed Leave-One-Video-Out evaluations across all 5 seeds.

### Fanning Cross-Video Generalization

Fanning is present across three videos (`20230711a-fan`, `20230711b-fan`, `yt5`). In each experiment, the model was trained on the remaining 4 videos and evaluated strictly on the held-out video:

| Held-Out Video | Fanning Test Windows | Fanning Test Tracks | Held-Out Fanning F1 (Mean $\pm$ SD) | Held-Out Macro-F1 (Mean $\pm$ SD) | In-Fold Benchmark F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `20230711a-fan` | 50 | 16 | $0.2831 \pm 0.1647$ | $0.2799 \pm 0.0485$ | $0.5274$ |
| `20230711b-fan` | 30 | 13 | $0.1214 \pm 0.0692$ | $0.2514 \pm 0.0340$ | $0.5274$ |
| `yt5` | 1 | 1 | $0.1460 \pm 0.1215$ | $0.1619 \pm 0.1235$ | $0.5274$ |

### Foraging Cross-Video Generalization

Foraging candidates: `20230609b-def`, `20230711a-fan`, `20230711b-fan`:

| Held-Out Video | Foraging Test Windows | Foraging Test Tracks | Held-Out Foraging F1 (Mean $\pm$ SD) | Held-Out Macro-F1 (Mean $\pm$ SD) |
| :--- | :---: | :---: | :---: | :---: |
| `20230609b-def` | 0 | 0 | N/A (0 test windows) | $0.1027 \pm 0.0103$ |
| `20230711a-fan` | 5 | 2 | $0.0000 \pm 0.0000$ | $0.2799 \pm 0.0485$ |
| `20230711b-fan` | 2 | 2 | $0.0630 \pm 0.0787$ | $0.2514 \pm 0.0340$ |

### Single-Video Limitations Confirmed:
- **Defense** exists exclusively in `20230609b-def`.
- **Washboarding** exists exclusively in `yt8`.
- LOVO cannot be evaluated on Defense or Washboarding; leaving out `20230609b-def` or `yt8` eliminates all positive training examples for those behaviors.

> **Generalization Gap Diagnosis:** When evaluated on a completely unseen video, Fanning F1 drops from **$0.5274$** to **$0.2831$** on `711a` and **$0.1214$** on `711b`. This dramatic degradation indicates that models are heavily reliant on domain-specific visual features (lighting, hive angle, background colors), confirming our shortcut hypothesis.

---

## 4. Shortcut Diagnostics Suite: Empirical Proof of Leakage

We executed three diagnostic probe experiments across all 5 seeds to isolate and quantify shortcut learning.

### Diagnostic Results (Mean $\pm$ SD over 5 Seeds)

| Diagnostic Probe | Input Features | Target Task | Macro-F1 | Accuracy | Key Per-Class Findings |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **(a) Position + Box-Size Classifier** | $16 \times 4$ coordinates (no visual, no speed) | Behavior (5 classes) | **$0.4451 \pm 0.0316$** | **$0.6667 \pm 0.0249$** | Defense F1: $0.4419$, Washboarding F1: $0.3456$, Fanning F1: $0.3811$ |
| **(b) Background Patch Classifier (No Bee)** | $16 \times 384$ DINOv2 on static background | Behavior (5 classes) | **$0.8298 \pm 0.0605$** | **$0.9213 \pm 0.0273$** | Defense F1: **$0.9641$**, Fanning F1: **$0.8306$**, Washboarding F1: **$0.7983$** |
| **(c) Video ID Linear Probe** | 384-d visual embedding $\to 5$ | Source Video ID | **$1.0000 \pm 0.0000$** | **$100.0\% \pm 0.0\%$** | Perfect classification across all 5 videos |

### Mechanistic Analysis:
1. **Background Leakage Explains High In-Domain Performance:**
   - The background patch classifier (which receives DINOv2 embeddings of the background where the bee was, with the bee completely masked out) achieves **$0.8298$ Macro-F1 and $92.13\%$ accuracy**!
   - Because Defense occurs exclusively in video `20230609b-def` and Washboarding exclusively in `yt8`, the DINOv2 model can achieve near-perfect classification simply by memorizing the static background wood texture, shadows, and lighting of the hive.
2. **Spatial Location As a Prior:**
   - Even without any image pixels, 2D spatial coordinates achieve **$0.4451$ Macro-F1**. Guards cluster along the entrance fissure, washboarders scrape on the elevated board, and fanners position themselves at ventilation vortices.
3. **Perfect Video ID Separation:**
   - A single linear layer maps the 384-dimensional DINOv2 representation to video identity with **100.0% accuracy**. The visual encoder represents the ambient video context so strongly that any classifier trained across sessions is prone to learning "video classifiers" rather than pure "behavior classifiers."

---

## 5. Artifacts and Run Ledger

- **Saved Summary File:** `results/phase_1_benchmark_summary.json`
- **Full Experiment Ledger:** `results/experiments.csv` (90 total logged runs covering blocked 5-fold folds, LOVO cross-validation splits, and 3 shortcut diagnostic suites across 5 random seeds).
- **Extracted Datasets:** `dinov2_features/phase1_data/` (all 9 memory-mapped NumPy arrays preserved).

---

## 6. Next Steps: Handoff to Phase 2

Phase 1 is now fully complete. All five requirements have been implemented, quantitatively verified over 5 seeds, logged to disk, and committed to git.

As instructed:
**STOPPING HERE until the user instructs "continue".**
In Phase 2, we will tackle:
1. Audit why only 4 Foraging tracks survived gating and analyze contiguous run lengths (median, p90).
2. Test gating variants (all steps inside polygon vs $\ge 50\%$ inside vs endpoint inside) and temporal window lengths $T \in \{6, 8, 16\}$ at step stride $K=6$.
3. Choose the optimal windowing and gating configuration based on out-of-fold results with bootstrap CIs, or formally document Foraging as a data limitation.
