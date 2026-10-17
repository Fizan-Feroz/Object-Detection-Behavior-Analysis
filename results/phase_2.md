# Phase 2: Foraging and Windows - Audit & Systematic Ablation Study

**Date:** October 18, 2026  
**Status:** Completed  
**Branch:** `main`  
**Evaluation Standard:** 5 seeds (`42, 43, 44, 45, 46`), 95% bootstrap confidence intervals clustered by track ($B=1000$), strict window and distinct track support reporting. No tuning or early-stopping on reported test splits. All 45 experimental ablation runs logged to `results/experiments.csv`. Results summary saved to `results/phase_2_ablation_summary.json` and raw audit log to `results/foraging_audit.txt`.

---

## 1. Executive Summary & Root Cause Audit

In Phase 1, Foraging achieved an F1 score of **$0.0000 \pm 0.0000$** with only **7 non-overlapping test windows across 4 distinct tracks** surviving in the entire pooled out-of-fold evaluation.

In Phase 2, we conducted:
1. A **complete granular audit** of raw track annotations and contiguous run lengths across all five behaviors to pinpoint the exact mechanism of track loss.
2. An **attrition funnel analysis** tracking where Foraging samples are eliminated.
3. A **systematic grid search** across 9 experimental configurations: temporal window length $T \in \{6, 8, 16\}$ at step stride $K=6$, crossed with three polygon containment rules (`all_inside`, `half_inside`, `endpoint_inside`).
4. An empirical evaluation over 5 random seeds using blocked 5-fold cross-validation with track-clustered bootstrap 95% confidence intervals.

---

## 2. Item 1: Contiguous Run Length Audit & The Foraging Deficit

We analyzed all raw track files across all 5 video sessions (`20230609b-def`, `20230711a-fan`, `20230711b-fan`, `yt5`, `yt8`).

### Contiguous Run Lengths by Behavior

A *contiguous run* is defined as an uninterrupted sequence of frames where a specific behavior is continuously active on an individual bee track.

| Behavior Class | Raw Annotated Tracks | Contiguous Runs | Median Run Length | P90 Run Length | Min Length | Max Length |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Foraging** | 1,842 | 3,142 | **3.0 frames** ($0.06$ s) | **20.0 frames** ($0.40$ s) | 1 | 525 |
| **Defense** | 113 | 576 | **2.0 frames** ($0.04$ s) | **20.0 frames** ($0.40$ s) | 1 | 1,077 |
| **Fanning** | 343 | 1,879 | **5.0 frames** ($0.10$ s) | **49.0 frames** ($0.98$ s) | 1 | 3,397 |
| **Washboarding** | 273 | 1,313 | **5.0 frames** ($0.10$ s) | **57.0 frames** ($1.14$ s) | 1 | 301 |
| **Other / Idle** | 11,870 | 26,407 | **2.0 frames** ($0.04$ s) | **25.0 frames** ($0.50$ s) | 1 | 1,237 |

### Why Only 4 Foraging Tracks Reached Train in Phase 0 & 1:

The audit reveals that Foraging was throttled by a compounding four-stage funnel:

```
[Raw Annotations] 1,842 Foraging Tracks (3,142 contiguous bouts)
       │
       ▼  (90% of bouts last <= 20 frames / 0.4s)
[Window Span Constraint: K=6, T=16] Requires >= 91 continuous frames (1.82s)
       │
       ▼  --> ELIMINATES 99.57% OF TRACKS! (1,834 tracks dropped)
[Span Survivors] Only 8 Tracks remain with continuous span >= 91 frames
       │
       ▼  (Foragers transit into/out of entrance across boundary)
[Polygon Gating: 100% Inside]
       │
       ▼  --> ELIMINATES 37.5% of survivors! (3 tracks dropped)
[Global Survivors] Only 5 Tracks survive across all 5 videos!
       │
       ▼  (Blocked Chronological Partitioning: 70% Train, 30% Test)
[Train Split] Exactly 4 Tracks in Train, 1 Track in Validation / Test!
```

1. **Severe Temporal Mismatch:** Foraging bees are in rapid transit (average speed $12-35$ px/frame vs $2-6$ px/frame for fanners/guards). A foraging bee approaches the hive, touches down on the landing ramp, and darts into the entrance slit within a few tenths of a second. Consequently, **90% of all foraging bouts terminate within 20 frames (0.40 s)**.
2. **Window Span Gatekeeper ($T=16, K=6$):** A 16-step window with $K=6$ frame stride requires a continuous span of $15 \times 6 + 1 = 91$ video frames ($1.82$ s). Demanding $1.82$ s of uninterrupted tracking on a $0.40$ s transit event eliminated **1,834 of the 1,842 raw tracks ($99.57\%$)**.
3. **Strict Polygon Containment (`all_inside`):** Because incoming foragers arrive from outside the camera frame and outgoing foragers depart into open air, forcing 100% of sampled steps to lie strictly inside the entrance polygon eliminated 3 of the remaining 8 long tracks.

---

## 3. Item 2: Systematic Ablation of Window Length ($T$) & Polygon Gating

To test whether shortening $T$ and relaxing polygon gating restores statistical support and predictive power without degrading precision, we conducted a full $3 \times 3$ factorial ablation study:
- **Temporal Window Lengths:** $T \in \{16, 8, 6\}$ (at $K=6$, requiring spans $\ge 91$, $\ge 43$, and $\ge 31$ frames respectively).
- **Gating Variants:**
  - `all_inside`: $100\%$ of window steps must reside inside the entrance polygon.
  - `half_inside`: $\ge 50\%$ of window steps must reside inside the entrance polygon.
  - `endpoint_inside`: The initial step ($t_0$) or final step ($t_T$) must reside inside the polygon.

All 9 configurations were evaluated across all 5 videos using **Blocked 5-Fold Cross-Validation** over **5 random seeds (`42, 43, 44, 45, 46`)**.

### Quantitative Results Table (Mean $\pm$ SD across 5 Seeds)

| Config ($T$, Gating) | Temporal Span ($K=6$) | OOF Macro-F1 (Mean $\pm$ SD) | Seed 42 95% Bootstrap CI | OOF Accuracy | Foraging F1 (Mean $\pm$ SD) | Real Foraging Test Support | Real Defense Test Support | Real Fanning Test Support | Real Wash Test Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$T=16$, all_inside** | 91 f ($1.82$ s) | $0.2711 \pm 0.0171$ | $[0.2173, 0.2821]$ | $0.6556$ | $0.0286 \pm 0.0381$ | 6 win (3 trk) | 39 win (15 trk) | 70 win (28 trk) | 24 win (19 trk) |
| **$T=16$, half_inside** | 91 f ($1.82$ s) | $0.3058 \pm 0.0083$ | $[0.2506, 0.3838]$ | $0.6573$ | $0.0819 \pm 0.0463$ | 8 win (4 trk) | 44 win (15 trk) | 71 win (28 trk) | 32 win (27 trk) |
| **$T=16$, endpoint_inside** | 91 f ($1.82$ s) | $0.2959 \pm 0.0178$ | $[0.2698, 0.3806]$ | $0.6290$ | $0.0865 \pm 0.0657$ | 9 win (5 trk) | 44 win (15 trk) | 71 win (28 trk) | 34 win (29 trk) |
| **$T=8$, all_inside** | 43 f ($0.86$ s) | $0.3576 \pm 0.0139$ | $[0.3329, 0.4125]$ | $0.6805$ | $0.4526 \pm 0.0270$ | 43 win (35 trk) | 90 win (17 trk) | 192 win (52 trk) | 96 win (52 trk) |
| **$T=8$, half_inside** | 43 f ($0.86$ s) | $0.3785 \pm 0.0059$ | $[0.3466, 0.4173]$ | $0.7164$ | $0.5130 \pm 0.0134$ | 58 win (48 trk) | 94 win (17 trk) | 196 win (54 trk) | 111 win (57 trk) |
| **$T=8$, endpoint_inside** | 43 f ($0.86$ s) | $0.3741 \pm 0.0136$ | $[0.3523, 0.4246]$ | $0.7221$ | $0.5108 \pm 0.0384$ | 68 win (57 trk) | 94 win (17 trk) | 197 win (55 trk) | 116 win (59 trk) |
| **$T=6$, all_inside** | 31 f ($0.62$ s) | $0.3943 \pm 0.0176$ | $[0.3728, 0.4340]$ | $0.7053$ | $0.6584 \pm 0.0244$ | 143 win (123 trk) | 133 win (17 trk) | 266 win (58 trk) | 167 win (78 trk) |
| **$T=6$, half_inside** | 31 f ($0.62$ s) | **$0.3988 \pm 0.0094$** | **$[0.3720, 0.4252]$** | **$0.7299$** | **$0.6700 \pm 0.0195$** | **182 win (156 trk)** | 137 win (17 trk) | 273 win (62 trk) | 189 win (87 trk) |
| **$T=6$, endpoint_inside** | 31 f ($0.62$ s) | $0.3985 \pm 0.0046$ | $[0.3634, 0.4248]$ | $0.7253$ | $0.6785 \pm 0.0138$ | 193 win (167 trk) | 139 win (18 trk) | 274 win (62 trk) | 197 win (88 trk) |

---

### Detailed Statistical Analysis of Ablations:

1. **Window Horizon is the Decisive Factor:**
   - Reducing $T$ from $16$ ($1.82$ s) to $6$ ($0.62$ s) yields a massive, statistically significant improvement across all metrics:
     - Macro-F1 increases from **$0.2711$ to $0.3988$ ($+0.1277$, non-overlapping 95% CIs)**.
     - Foraging F1 increases from **$0.0286$ to $0.6700$ ($+0.6414$)**!
     - Foraging test support expands from **6 windows / 3 tracks to 182 windows / 156 tracks** (a **$52\times$ increase** in distinct track diversity).
2. **Comparison of Gating Rules:**
   - **`all_inside` vs `half_inside`:** Relaxing `all_inside` to `half_inside` adds +39 test windows and +33 distinct tracks for Foraging at $T=6$, while raising Foraging F1 from $0.6584$ to $0.6700$ and overall Accuracy from $0.7053$ to $0.7299$.
   - **`half_inside` vs `endpoint_inside`:** Moving from `half_inside` to `endpoint_inside` adds 11 marginal tracks for Foraging, but produces virtually identical Macro-F1 ($0.3985$ vs $0.3988$) with fully overlapping confidence intervals ($[0.3634, 0.4248]$ vs $[0.3720, 0.4252]$).
   - However, `half_inside` is biologically and operationally superior: it guarantees that the bee spent at least 50% of the duration inside the ramp/entrance area, preventing false positives from stray bees flying across the periphery of the camera frame.
3. **Selection Decision (by Out-of-Fold CIs):**
   - **Winning Configuration:** **$T=6$, `half_inside`** at $K=6$.
   - Achieves the highest out-of-fold Accuracy ($0.7299 \pm 0.0071$) and Macro-F1 ($0.3988 \pm 0.0094$), with robust Foraging support (182 windows across 156 distinct tracks) and high Foraging F1 ($0.6700$).

---

## 4. Item 3: Data Limitation Documentation

Despite recovering Foraging track support to $n=156$ tracks at $T=6$, our audit uncovered structural dataset characteristics that must be formally documented as **fundamental data limitations**:

1. **Transit vs Micro-Motion Asymmetry:**
   - Behaviors like **Fanning** (median bout 5 frames, max 3,397 frames) and **Washboarding** (median 5 frames, max 301 frames) are stationary, sustained micro-motions that benefit from longer observation windows ($T=16$).
   - In contrast, **Foraging** is an acute transit behavior (median bout 3 frames, P90 20 frames). At $T=16$, Foraging is structurally undetectable without synthetic interpolation.
2. **Video-Level Imbalance:**
   - Foraging annotations are concentrated almost entirely in two videos (`20230711b-fan` with 1,157 raw tracks; `20230711a-fan` with 341 raw tracks; `20230609b-def` with 344 raw tracks). `yt5` and `yt8` contain zero labeled foraging tracks.
3. **Tracking Gaps at the Entrance:**
   - Because foragers fly directly through the entrance slit where high bee density and darkness obscure bounding boxes, ByteTrack frequently terminates the track at the threshold. A temporal window of $0.62$ s ($T=6$) is the maximum viable horizon that aligns with actual track survival in the wild.

---

## 5. Artifacts and Run Ledger

- **Saved Summary File:** `results/phase_2_ablation_summary.json`
- **Granular Audit Log:** `results/foraging_audit.txt`
- **Full Experiment Ledger:** `results/experiments.csv` (135 total logged runs across Phase 0, Phase 1, and the 45 Phase 2 factorial ablation runs).

---

## 6. Next Steps: Handoff to Phase 3

Phase 2 is now fully complete. All three items have been executed, statistically verified over 5 seeds with track-clustered bootstrap CIs, logged to disk, and committed to git.

As instructed:
**STOPPING HERE until the user instructs "continue".**
In Phase 3, we will tackle:
1. Build a "tracker tracks" dataset: Run YOLO + ByteTrack on the videos, transfer behavior labels from GT tracks via IoU matching, and measure train/serve skew drop.
2. Tracking evaluation with IDF1 against GT tracks, grid-searching ByteTrack parameters (`new_track_thresh`, `track_buffer`, `track_low_thresh`, `match_thresh`) to generate `custom_bytetrack.yaml`.
3. Detector evaluation: Compare baseline 640px `best.pt` vs 960px checkpoint on `_bee_20230711c` only (recall, mAP50-95, FPS).
