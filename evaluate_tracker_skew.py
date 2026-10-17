# ============================================================
# Phase 3 Item 1: Tracker Tracks Dataset & Train/Serve Skew Evaluation
#
# Protocol:
#   1. Runs YOLO + ByteTrack (using custom_bytetrack.yaml) to extract tracker tracks.
#   2. Transfers ground-truth behavior labels via spatio-temporal IoU matching (IoU >= 0.5).
#   3. Assembles T=16, K=6 causal sliding windows gated by entrance polygon.
#   4. Extracts 7-d causal kinematics & 384-d DINOv2 visual embeddings on CUDA.
#   5. Evaluates behavior classifier (Fused ViT-LSTM) on tracker tracks vs GT tracks.
#   6. Measures train/serve skew drop in Macro-F1 and per-class F1.
#   7. Reports dual support (windows and distinct tracks) and 95% bootstrap CIs.
#   8. Saves to results/tracker_skew_benchmark.json and logs to results/experiments.csv.
# ============================================================

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import classification_report, f1_score, accuracy_score
import yaml

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from config import (
    BASE_DIR,
    CLASS_NAMES,
    EXPERIMENTS_CSV,
    MODELS_DIR,
    RESULTS_DIR,
    TRACKING_DIR,
    VIDEOS,
    seed_everything,
)
from kinematics import CausalKinematicsTracker
from train_vit_lstm import BeeLSTMClassifier


TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DetectionBatch:
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __getitem__(self, mask: np.ndarray) -> DetectionBatch:
        return DetectionBatch(self.xywh[mask], self.conf[mask], self.cls[mask])

    def __len__(self) -> int:
        return len(self.conf)


def parse_polygon(zone_file: Path) -> np.ndarray:
    import re
    text = zone_file.read_text(encoding="utf-8").strip()
    pts = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not pts:
        raise ValueError(f"Could not parse polygon in {zone_file}")
    return np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)


def crop_bee_10pct_pad(frame_bgr: np.ndarray, cx: float, cy: float, bw: float, bh: float) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    pad_w = 0.10 * bw
    pad_h = 0.10 * bh

    x1 = max(0, int(cx - bw / 2.0 - pad_w))
    y1 = max(0, int(cy - bh / 2.0 - pad_h))
    x2 = min(W, int(cx + bw / 2.0 + pad_w))
    y2 = min(H, int(cy + bh / 2.0 + pad_h))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def load_gt_annotations_with_classes(ann_path: Path, video_w: int, video_h: int, max_frames: int | None = None) -> dict[int, list[dict]]:
    """
    Loads GT tracks with behavior classes from tracks_and_behavior_classes_*.txt.
    Coordinates: [tl_x, tl_y, w, h] normalized.
    Returns: frame_id -> list of dicts with keys: tid, box_tlwh_px, box_xyxy_px, classes.
    """
    gt_by_frame: dict[int, list[dict]] = defaultdict(list)
    with open(ann_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 10:
                continue
            fid = int(row[0])
            if max_frames and fid > max_frames:
                continue
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            classes = [int(row[6]), int(row[7]), int(row[8]), int(row[9])]

            tl_x = x * video_w
            tl_y = y * video_h
            bw = w * video_w
            bh = h * video_h
            gt_by_frame[fid].append({
                "tid": tid,
                "box_tlwh": [tl_x, tl_y, bw, bh],
                "box_xyxy": [tl_x, tl_y, tl_x + bw, tl_y + bh],
                "classes": classes,
            })
    return gt_by_frame


def compute_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    """Box format: [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def bootstrap_ci_clustered(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cluster_ids: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """Computes 95% bootstrap confidence intervals clustered by track."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    macro_f1s = []
    weighted_f1s = []
    accuracies = []

    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

    for _ in range(n_bootstraps):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])

        yt = y_true[sampled_indices]
        yp = y_pred[sampled_indices]

        macro_f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
        weighted_f1s.append(f1_score(yt, yp, average="weighted", zero_division=0))
        accuracies.append(accuracy_score(yt, yp))

    return {
        "macro_f1": [float(np.percentile(macro_f1s, 2.5)), float(np.percentile(macro_f1s, 97.5))],
        "weighted_f1": [float(np.percentile(weighted_f1s, 2.5)), float(np.percentile(weighted_f1s, 97.5))],
        "accuracy": [float(np.percentile(accuracies, 2.5)), float(np.percentile(accuracies, 97.5))],
    }


def main():
    seed_everything(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70, flush=True)
    print("PHASE 3 ITEM 1: TRACKER TRACKS DATASET & TRAIN/SERVE SKEW EVALUATION", flush=True)
    print("=" * 70, flush=True)

    # 1. Load ByteTrack Config
    custom_yaml = BASE_DIR / "custom_bytetrack.yaml"
    if custom_yaml.exists():
        with open(custom_yaml, "r", encoding="utf-8") as f:
            tracker_cfg = yaml.safe_load(f)
        print(f"Loaded custom tracker config from {custom_yaml}", flush=True)
    else:
        tracker_cfg = {
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.25,
            "track_buffer": 30,
            "match_thresh": 0.80,
            "fuse_score": True,
        }
        print("Using default ByteTrack config", flush=True)

    args = SimpleNamespace(
        track_high_thresh=float(tracker_cfg.get("track_high_thresh", 0.25)),
        track_low_thresh=float(tracker_cfg.get("track_low_thresh", 0.10)),
        new_track_thresh=float(tracker_cfg.get("new_track_thresh", 0.25)),
        track_buffer=int(tracker_cfg.get("track_buffer", 30)),
        match_thresh=float(tracker_cfg.get("match_thresh", 0.80)),
        fuse_score=bool(tracker_cfg.get("fuse_score", True)),
    )

    # 2. Load DINOv2 Feature Extractor
    print("Loading DINOv2 ViT-S/14 model on CUDA...", flush=True)
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dinov2.to(device).eval()

    # 3. Load Trained Behavior Classifier
    clf_path = MODELS_DIR / "fused_dinov2_kinematics_s42.pt"
    if not clf_path.exists():
        clf_path = MODELS_DIR / "fused_dinov2_kinematics.pt"
    print(f"Loading Behavior Classifier from {clf_path}...", flush=True)
    ckpt = torch.load(clf_path, map_location=device)
    is_bidi = any("reverse" in k for k in ckpt.keys())
    print(f"  Checkpoint architecture: bidirectional={is_bidi}", flush=True)
    clf = BeeLSTMClassifier(input_dim=391, hidden_dim=128, num_layers=2, bidirectional=is_bidi, num_classes=5)
    clf.load_state_dict(ckpt)
    clf.to(device).eval()

    # 4. First, evaluate classifier on Ground-Truth Tracks Test Set (Baseline)
    gt_vis_path = BASE_DIR / "dinov2_features" / "X_val_visual.npy"
    gt_kin_path = BASE_DIR / "dinov2_features" / "X_val_kinematics.npy"
    gt_y_path = BASE_DIR / "dinov2_features" / "y_val.npy"
    gt_tracks_path = BASE_DIR / "dinov2_features" / "val_tracks.npy"

    X_gt_vis = np.load(gt_vis_path)
    X_gt_kin = np.load(gt_kin_path)
    y_gt = np.load(gt_y_path)
    tracks_gt = np.load(gt_tracks_path, allow_pickle=True)

    X_gt = np.concatenate([X_gt_vis, X_gt_kin], axis=-1)  # (379, 16, 391)
    with torch.no_grad():
        gt_logits = clf(torch.tensor(X_gt, dtype=torch.float32, device=device))
        gt_preds = torch.argmax(gt_logits, dim=-1).cpu().numpy()

    gt_macro_f1 = f1_score(y_gt, gt_preds, average="macro", zero_division=0)
    gt_weighted_f1 = f1_score(y_gt, gt_preds, average="weighted", zero_division=0)
    gt_acc = accuracy_score(y_gt, gt_preds)
    gt_per_class_f1 = f1_score(y_gt, gt_preds, average=None, zero_division=0)
    gt_ci = bootstrap_ci_clustered(y_gt, gt_preds, tracks_gt, n_bootstraps=1000, seed=42)

    print("\n--- Ground-Truth Tracks Performance (Reference) ---", flush=True)
    print(f"Windows: {len(y_gt)}, Tracks: {len(np.unique(tracks_gt))}")
    print(f"Macro-F1: {gt_macro_f1:.4f} (95% CI: [{gt_ci['macro_f1'][0]:.4f}, {gt_ci['macro_f1'][1]:.4f}])")
    print(f"Weighted-F1: {gt_weighted_f1:.4f}, Accuracy: {gt_acc:.4f}")
    for c_idx, cname in enumerate(CLASS_NAMES):
        support_c = int(np.sum(y_gt == c_idx))
        print(f"  {cname}: F1={gt_per_class_f1[c_idx]:.4f} (support={support_c})")

    # 5. Extract Tracker Tracks & Transfer GT Labels on Evaluation Videos
    yolo_model = YOLO(str(BASE_DIR / "yolov8" / "weights" / "best.pt"))
    eval_videos = ["yt5", "yt8", "20230609b-def", "20230711a-fan", "20230711b-fan"]
    max_frames_map = {
        "yt5": 977,
        "yt8": 1500,
        "20230609b-def": 1500,
        "20230711a-fan": 1500,
        "20230711b-fan": 1500,
    }

    all_tracker_windows = []
    tracker_crops_to_extract = []
    tracker_keys = []

    print("\n--- Extracting Tracker Tracks and Transferring Labels via IoU Matching ---", flush=True)

    for vkey in eval_videos:
        vinfo = VIDEOS[vkey]
        vpath = TRACKING_DIR / vinfo["video"]
        ann_path = TRACKING_DIR / vinfo["ann"]
        zone_path = TRACKING_DIR / vinfo["zone"]
        polygon = parse_polygon(zone_path)
        vw, vh = vinfo["w"], vinfo["h"]
        mf = max_frames_map[vkey]

        print(f"Processing {vkey} (up to frame {mf})...", flush=True)
        gt_by_frame = load_gt_annotations_with_classes(ann_path, vw, vh, max_frames=mf)

        # Run ByteTrack on video frames
        tracker = BYTETracker(args)
        cache_file = RESULTS_DIR / "detection_cache" / f"dets_{vkey}_{mf}_640.npz"
        det_frames = []
        if cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            for i in range(int(data["num_frames"])):
                det_frames.append((data[f"xywh_{i}"], data[f"conf_{i}"], data[f"cls_{i}"]))
            print(f"  Loaded {len(det_frames)} cached detection frames for {vkey}", flush=True)
        else:
            print(f"  Extracting YOLO detections for {vkey} (up to {mf} frames)...", flush=True)
            for r in yolo_model.predict(source=str(vpath), imgsz=640, conf=0.10, device=0, stream=True, verbose=False):
                if len(det_frames) >= mf:
                    break
                b = r.boxes.cpu()
                det_frames.append((b.xywh.numpy(), b.conf.numpy(), b.cls.numpy()))
            save_d = {"num_frames": np.array(len(det_frames))}
            for i, (xywh, conf, cls) in enumerate(det_frames):
                save_d[f"xywh_{i}"] = xywh
                save_d[f"conf_{i}"] = conf
                save_d[f"cls_{i}"] = cls
            np.savez_compressed(cache_file, **save_d)

        tracks_obs = defaultdict(list)
        for frame_idx, (xywh, conf, cls) in enumerate(det_frames, start=1):
            batch = DetectionBatch(xywh, conf, cls)
            pred_tracks = tracker.update(batch)
            gt_items = gt_by_frame.get(frame_idx, [])

            for p_row in pred_tracks:
                px1, py1, px2, py2, tid, score = float(p_row[0]), float(p_row[1]), float(p_row[2]), float(p_row[3]), int(p_row[4]), float(p_row[5])
                p_box = [px1, py1, px2, py2]
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0
                pbw = max(1.0, px2 - px1)
                pbh = max(1.0, py2 - py1)

                # Spatio-temporal IoU matching with GT in current frame
                best_iou = 0.0
                matched_classes = [0, 0, 0, 0]  # default unlabeled (Other)
                for g_item in gt_items:
                    iou = compute_iou(p_box, g_item["box_xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        matched_classes = g_item["classes"]

                # Transfer label if max IoU >= 0.50
                final_classes = matched_classes if best_iou >= 0.50 else [0, 0, 0, 0]

                tracks_obs[tid].append({
                    "track_id": tid,
                    "frame_id": frame_idx,
                    "cx": pcx,
                    "cy": pcy,
                    "bw": pbw,
                    "bh": pbh,
                    "x": pcx / vw,
                    "y": pcy / vh,
                    "w": pbw / vw,
                    "h": pbh / vh,
                    "classes": final_classes,
                    "video": vkey,
                })

        print(f"  {vkey}: Generated {len(tracks_obs)} tracker tracks across {frame_idx} frames.", flush=True)

        # Assemble candidate windows (T=16, K=6)
        k_subsample = 6
        window_length = 16
        kin_tracker = CausalKinematicsTracker(polygon, vw, vh, fps=50.0)

        # Pre-enrich tracks with kinematics
        for tid, obs_list in tracks_obs.items():
            if len(obs_list) < window_length:
                continue

            # Sort by frame_id
            obs_list.sort(key=lambda o: o["frame_id"])
            frame_map = {o["frame_id"]: o for o in obs_list}

            # Enrich with causal kinematics
            kin_tracker_t = CausalKinematicsTracker(polygon, vw, vh, fps=50.0)
            for o in obs_list:
                o["kinematics"] = kin_tracker_t.update(tid, o["frame_id"], o["x"], o["y"], o["w"], o["h"])

            # Form causal windows of length 16 with step K=6
            min_f = obs_list[0]["frame_id"]
            max_f = obs_list[-1]["frame_id"]

            for start_idx in range(0, len(obs_list) - window_length * k_subsample + 1, k_subsample * 2):
                win_fids = [obs_list[start_idx + s * k_subsample]["frame_id"] for s in range(window_length) if start_idx + s * k_subsample < len(obs_list)]
                if len(win_fids) < window_length:
                    continue
                # Verify exact stride K=6
                if any(win_fids[s] - win_fids[s-1] != k_subsample for s in range(1, window_length)):
                    continue

                win_obs = [frame_map[f] for f in win_fids]

                # Entrance zone polygon gating: all 16 observations must be inside polygon
                if any(cv2.pointPolygonTest(polygon, (o["cx"], o["cy"]), False) < 0 for o in win_obs):
                    continue

                # 80% majority behavior rule
                class_counts = [sum(o["classes"][c] for o in win_obs) for c in range(4)]
                total_labeled = sum(class_counts)
                target_class = None
                if total_labeled == 0:
                    target_class = 4  # Other
                else:
                    for c_idx in range(4):
                        if class_counts[c_idx] >= int(0.80 * window_length) and total_labeled == class_counts[c_idx]:
                            target_class = c_idx
                            break

                if target_class is None:
                    continue

                all_tracker_windows.append({
                    "track_id": f"{vkey}_{tid}",
                    "raw_track_id": tid,
                    "class": target_class,
                    "obs": win_obs,
                    "video": vkey,
                })

    print(f"Total valid polygon-gated tracker windows extracted: {len(all_tracker_windows)}", flush=True)

    # 6. Extract DINOv2 embeddings for Tracker Windows
    # Stream video once per video to extract unique crops needed for tracker windows
    windows_by_video = defaultdict(list)
    for w in all_tracker_windows:
        windows_by_video[w["video"]].append(w)

    emb_cache: dict[tuple[str, int, int], np.ndarray] = {}

    print("Extracting DINOv2 visual features for tracker crops...", flush=True)
    batch_size = 64
    crop_queue = []
    key_queue = []

    def flush_crops():
        if not crop_queue:
            return
        batch_t = torch.stack(crop_queue).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            embs = dinov2(batch_t).cpu().numpy().astype(np.float32)
        for k, e in zip(key_queue, embs):
            emb_cache[k] = e
        crop_queue.clear()
        key_queue.clear()

    for vkey, vwindows in windows_by_video.items():
        vinfo = VIDEOS[vkey]
        vpath = TRACKING_DIR / vinfo["video"]

        # Collect needed (tid, fid) -> obs
        needed = defaultdict(dict)
        for w in vwindows:
            for o in w["obs"]:
                k = (vkey, o["track_id"], o["frame_id"])
                if k not in emb_cache:
                    needed[o["frame_id"]][o["track_id"]] = o

        if not needed:
            continue

        cap = cv2.VideoCapture(str(vpath))
        fid = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fid += 1
            if fid in needed:
                for tid, o in needed[fid].items():
                    crop_rgb = crop_bee_10pct_pad(frame, o["cx"], o["cy"], o["bw"], o["bh"])
                    crop_queue.append(TRANSFORM(crop_rgb))
                    key_queue.append((vkey, tid, fid))
                    if len(crop_queue) >= batch_size:
                        flush_crops()
        cap.release()

    flush_crops()
    print(f"Extracted {len(emb_cache)} unique DINOv2 crops for tracker tracks.", flush=True)

    # 7. Assemble Tensors for Tracker Windows
    n_w = len(all_tracker_windows)
    X_tr_vis = np.zeros((n_w, 16, 384), dtype=np.float32)
    X_tr_kin = np.zeros((n_w, 16, 7), dtype=np.float32)
    y_tracker = np.array([w["class"] for w in all_tracker_windows], dtype=np.int64)
    tracks_tracker = np.array([w["track_id"] for w in all_tracker_windows], dtype=object)

    for i, w in enumerate(all_tracker_windows):
        for s, o in enumerate(w["obs"]):
            k = (w["video"], o["track_id"], o["frame_id"])
            X_tr_vis[i, s] = emb_cache.get(k, np.zeros(384, dtype=np.float32))
            X_tr_kin[i, s] = o["kinematics"]

    X_tracker = np.concatenate([X_tr_vis, X_tr_kin], axis=-1)  # (N, 16, 391)

    # 8. Evaluate Behavior Classifier on Tracker Tracks
    with torch.no_grad():
        tr_logits = clf(torch.tensor(X_tracker, dtype=torch.float32, device=device))
        tr_preds = torch.argmax(tr_logits, dim=-1).cpu().numpy()

    tr_macro_f1 = f1_score(y_tracker, tr_preds, average="macro", zero_division=0)
    tr_weighted_f1 = f1_score(y_tracker, tr_preds, average="weighted", zero_division=0)
    tr_acc = accuracy_score(y_tracker, tr_preds)
    tr_per_class_f1 = f1_score(y_tracker, tr_preds, average=None, zero_division=0)
    tr_ci = bootstrap_ci_clustered(y_tracker, tr_preds, tracks_tracker, n_bootstraps=1000, seed=42)

    # Calculate Skew Drops
    delta_macro = tr_macro_f1 - gt_macro_f1
    delta_weighted = tr_weighted_f1 - gt_weighted_f1
    delta_acc = tr_acc - gt_acc
    delta_per_class = {
        cname: float(tr_per_class_f1[c_idx] - gt_per_class_f1[c_idx])
        for c_idx, cname in enumerate(CLASS_NAMES)
    }

    print("\n" + "=" * 70, flush=True)
    print("TRAIN / SERVE SKEW BENCHMARK RESULTS (GT TRACKS VS TRACKER TRACKS)")
    print("=" * 70, flush=True)
    print(f"GT Tracks:      Windows={len(y_gt):4d}, Tracks={len(np.unique(tracks_gt)):4d} | Macro-F1={gt_macro_f1:.4f} [{gt_ci['macro_f1'][0]:.4f}, {gt_ci['macro_f1'][1]:.4f}] | Acc={gt_acc:.4f}")
    print(f"Tracker Tracks: Windows={len(y_tracker):4d}, Tracks={len(np.unique(tracks_tracker)):4d} | Macro-F1={tr_macro_f1:.4f} [{tr_ci['macro_f1'][0]:.4f}, {tr_ci['macro_f1'][1]:.4f}] | Acc={tr_acc:.4f}")
    print(f"Train/Serve Skew Drop: ΔMacro-F1 = {delta_macro:+.4f}, ΔWeighted-F1 = {delta_weighted:+.4f}, ΔAccuracy = {delta_acc:+.4f}")
    print("\nPer-Class F1 Comparison:")
    for c_idx, cname in enumerate(CLASS_NAMES):
        gt_sup = int(np.sum(y_gt == c_idx))
        tr_sup = int(np.sum(y_tracker == c_idx))
        print(f"  {cname:<14}: GT F1 = {gt_per_class_f1[c_idx]:.4f} (n={gt_sup:3d}) | Tracker F1 = {tr_per_class_f1[c_idx]:.4f} (n={tr_sup:3d}) | Drop = {delta_per_class[cname]:+.4f}")
    print("=" * 70, flush=True)

    # 9. Save to results/tracker_skew_benchmark.json
    skew_data = {
        "tracker_config": vars(args),
        "gt_tracks_benchmark": {
            "windows": len(y_gt),
            "tracks": len(np.unique(tracks_gt)),
            "macro_f1": float(gt_macro_f1),
            "weighted_f1": float(gt_weighted_f1),
            "accuracy": float(gt_acc),
            "per_class_f1": {cname: float(gt_per_class_f1[i]) for i, cname in enumerate(CLASS_NAMES)},
            "support": {cname: int(np.sum(y_gt == i)) for i, cname in enumerate(CLASS_NAMES)},
            "bootstrap_ci_95": gt_ci,
        },
        "tracker_tracks_benchmark": {
            "windows": len(y_tracker),
            "tracks": len(np.unique(tracks_tracker)),
            "macro_f1": float(tr_macro_f1),
            "weighted_f1": float(tr_weighted_f1),
            "accuracy": float(tr_acc),
            "per_class_f1": {cname: float(tr_per_class_f1[i]) for i, cname in enumerate(CLASS_NAMES)},
            "support": {cname: int(np.sum(y_tracker == i)) for i, cname in enumerate(CLASS_NAMES)},
            "bootstrap_ci_95": tr_ci,
        },
        "train_serve_skew_drop": {
            "delta_macro_f1": float(delta_macro),
            "delta_weighted_f1": float(delta_weighted),
            "delta_accuracy": float(delta_acc),
            "per_class_drop": delta_per_class,
        },
    }

    out_json = RESULTS_DIR / "tracker_skew_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(skew_data, f, indent=2)
    print(f"Saved skew benchmark to {out_json}", flush=True)

    # 10. Log to results/experiments.csv
    now_str = datetime.now().isoformat()
    row = {
        "timestamp": now_str,
        "run_id": f"Tracker_Skew_Evaluation_{int(time.time())}",
        "experiment_name": "Tracker_Tracks_Train_Serve_Skew",
        "model_type": "Fused_DINOv2_Kinematics_TrackerTracks",
        "seed": "42",
        "split_type": "tracker_tracks_vs_gt_tracks",
        "train_windows": str(len(y_gt)),
        "train_tracks": str(len(np.unique(tracks_gt))),
        "tune_windows": "0",
        "tune_tracks": "0",
        "test_windows": str(len(y_tracker)),
        "test_tracks": str(len(np.unique(tracks_tracker))),
        "macro_f1": f"{tr_macro_f1:.4f}",
        "weighted_f1": f"{tr_weighted_f1:.4f}",
        "accuracy": f"{tr_acc:.4f}",
        "per_class_f1_json": json.dumps({cname: float(tr_per_class_f1[i]) for i, cname in enumerate(CLASS_NAMES)}),
        "config_json": json.dumps({
            "delta_macro_f1": float(delta_macro),
            "delta_weighted_f1": float(delta_weighted),
            "delta_accuracy": float(delta_acc),
            "tracker_cfg": vars(args),
            "bootstrap_ci_95": tr_ci,
        }),
    }

    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

    print("Logged tracker skew run to experiments.csv", flush=True)


if __name__ == "__main__":
    main()
