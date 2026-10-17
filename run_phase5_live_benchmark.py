# ============================================================
# Phase 5: End-to-End Live Video Benchmark (6,000 Frames)
#
# Pipeline:
#   1. Video Stream: 20230711a-fan.mp4 (6,000 frames, 1920x1080 @ 50 FPS)
#   2. Detection: YOLOv8m (weights: yolov8/weights/best.pt, imgsz=640)
#   3. Tracking: ByteTrack with custom tuned parameters
#   4. Causal Kinematics: 7-d scale-invariant causal tracker
#   5. Visual Embedding: DINOv2 ViT-S/14 (224x224, K=6 subsampling)
#   6. Temporal Classification: Fused Projected ViT-BiLSTM
#
# Metrics Evaluated:
#   - Latency breakdown per stage: Detection, Tracking, Kinematics, Visual Embedding, Classification
#   - Total throughput (FPS) and runtime
#   - GPU VRAM consumption (allocated, reserved, peak)
#   - Ground-Truth matching against tracks_and_behavior_classes_20230711a-fan.txt:
#       * Frame-level F1, Precision, Recall, Accuracy & Per-Class F1
#       * Track-level F1, Precision, Recall, Accuracy & Per-Class F1
#       * Tracking metrics: MOTA, IDF1, ID switches
#   - Saved to results/benchmark_live_6000frames.json
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torchvision import transforms
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
    get_device,
    seed_everything,
)
from kinematics import CausalKinematicsTracker
from train_phase4_ablations import ProjectedFusedBeeLSTM, log_to_experiments_csv

VIDEO_NAME = "20230711a-fan"
VIDEO_PATH = TRACKING_DIR / VIDEOS[VIDEO_NAME]["video"]
ANN_PATH = TRACKING_DIR / VIDEOS[VIDEO_NAME]["ann"]
ZONE_PATH = TRACKING_DIR / VIDEOS[VIDEO_NAME]["zone"]
YOLO_WEIGHTS = BASE_DIR / "yolov8" / "weights" / "best.pt"

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


def load_gt_annotations(ann_path: Path, video_w: int, video_h: int) -> dict[int, list[dict]]:
    gt_by_frame = defaultdict(list)
    with open(ann_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 10:
                continue
            fid = int(row[0])
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            classes = [int(row[6]), int(row[7]), int(row[8]), int(row[9])]
            
            # Ground truth label
            tot = sum(classes)
            if tot == 0:
                lbl = 4
            else:
                lbl = 4
                for c_idx in range(4):
                    if classes[c_idx] == 1 and tot == 1:
                        lbl = c_idx
                        break

            cx = (x + w / 2.0) * video_w
            cy = (y + h / 2.0) * video_h
            bw = w * video_w
            bh = h * video_h
            box_xyxy = [cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0]

            gt_by_frame[fid].append({
                "tid": tid,
                "box_xyxy": box_xyxy,
                "label": lbl,
            })
    return gt_by_frame


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / (union + 1e-6)


def run_live_benchmark(max_frames: int = 6000):
    device = get_device()
    print("=" * 75)
    print("PHASE 5: LIVE VIDEO BENCHMARK (6,000 FRAMES)")
    print(f"Video: {VIDEO_PATH}")
    print(f"Device: {device} | Max Frames: {max_frames}")
    print("=" * 75)

    # 1. Load Polygon & GT
    polygon = parse_polygon(ZONE_PATH)
    gt_annotations = load_gt_annotations(ANN_PATH, 1920, 1080)
    print(f"Loaded ground-truth annotations across {len(gt_annotations)} frames.")

    # 2. Load Models
    print("Loading YOLOv8m detector...")
    yolo_model = YOLO(str(YOLO_WEIGHTS))

    print("Loading DINOv2 ViT-S/14 feature extractor...")
    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).half().eval()

    print("Loading Fused ViT-BiLSTM Behavior Classifier...")
    classifier = ProjectedFusedBeeLSTM().to(device).eval()
    classifier_ckpt = MODELS_DIR / "projected_fused_bilstm_best.pt"
    if classifier_ckpt.exists():
        state = torch.load(classifier_ckpt, map_location=device)
        classifier.load_state_dict(state)
        print("  Loaded classifier checkpoint:", classifier_ckpt)
    else:
        print("  Using initialized classifier model.")

    # 3. Setup Tracker & Kinematics
    tracker_args = SimpleNamespace(
        track_high_thresh=0.25,
        track_low_thresh=0.15,
        new_track_thresh=0.30,
        track_buffer=15,
        match_thresh=0.70,
        fuse_score=True,
    )
    tracker = BYTETracker(tracker_args)
    kin_tracker = CausalKinematicsTracker(polygon=polygon, video_w=1920, video_h=1080, fps=50.0, alpha=0.6, max_gap_frames=5)

    # Rolling window buffers per active track
    track_vis_buffers = defaultdict(lambda: deque(maxlen=16))
    track_kin_buffers = defaultdict(lambda: deque(maxlen=16))
    track_last_pred = defaultdict(lambda: 4)  # Default Other
    track_pred_history = defaultdict(list)
    track_gt_history = defaultdict(list)

    # Benchmarking latency accumulators
    latencies = {
        "decode_ms": [],
        "detection_ms": [],
        "tracking_ms": [],
        "kinematics_ms": [],
        "visual_ms": [],
        "classification_ms": [],
        "total_frame_ms": [],
    }

    # Tracking metrics accumulators
    num_gt_total = 0
    num_pred_total = 0
    num_matches = 0
    id_switches = 0
    prev_matches = {}  # gt_tid -> tracker_tid

    frame_preds = []
    frame_targets = []

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames_in_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_run = min(max_frames, total_frames_in_vid)
    print(f"Beginning live stream processing of {frames_to_run} frames...\n")

    t_start_total = time.perf_counter()
    cur_fid = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    while cur_fid < frames_to_run:
        cur_fid += 1
        t_f0 = time.perf_counter()

        # 1. Decode Frame
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t_decode = (time.perf_counter() - t0) * 1000.0

        # 2. YOLO Detection
        t0 = time.perf_counter()
        yolo_res = yolo_model(frame, device=0 if torch.cuda.is_available() else "cpu", imgsz=640, conf=0.25, verbose=False)[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_det = (time.perf_counter() - t0) * 1000.0

        boxes_xyxy = yolo_res.boxes.xyxy.cpu().numpy()
        confs = yolo_res.boxes.conf.cpu().numpy()
        classes = yolo_res.boxes.cls.cpu().numpy()

        # Convert to xywh
        boxes_xywh = np.zeros_like(boxes_xyxy)
        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        # 3. ByteTrack Tracking
        t0 = time.perf_counter()
        det_batch = DetectionBatch(boxes_xywh, confs, classes)
        active_tracks = tracker.update(det_batch)
        t_track = (time.perf_counter() - t0) * 1000.0

        # 4. Kinematics Update
        t0 = time.perf_counter()
        active_track_info = []
        for t in active_tracks:
            # [x1, y1, x2, y2, track_id, score, cls, idx]
            tx1, ty1, tx2, ty2 = float(t[0]), float(t[1]), float(t[2]), float(t[3])
            tid = int(t[4])
            tcx = (tx1 + tx2) / 2.0
            tcy = (ty1 + ty2) / 2.0
            tbw = tx2 - tx1
            tbh = ty2 - ty1

            # Update causal kinematics
            kin_feat = kin_tracker.update(tid, cur_fid, tcx, tcy, tbw, tbh)
            inside = cv2.pointPolygonTest(polygon, (float(tcx), float(tcy)), False) >= 0

            active_track_info.append({
                "tid": tid,
                "box": [tx1, ty1, tx2, ty2],
                "cx": tcx, "cy": tcy, "bw": tbw, "bh": tbh,
                "kin_feat": kin_feat,
                "inside": inside,
            })
        t_kin = (time.perf_counter() - t0) * 1000.0

        # 5. Visual Feature Extraction (every K=6 frames for active inside bees)
        t0 = time.perf_counter()
        crops_to_embed = []
        embed_tids = []

        if cur_fid % 6 == 0:
            for info in active_track_info:
                if info["inside"]:
                    crop = crop_bee_10pct_pad(frame, info["cx"], info["cy"], info["bw"], info["bh"])
                    crops_to_embed.append(TRANSFORM(crop))
                    embed_tids.append(info["tid"])

        if crops_to_embed:
            batch_tensor = torch.stack(crops_to_embed).to(device).half()
            with torch.no_grad():
                embs = dinov2_model(batch_tensor).float().cpu().numpy()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for tid, emb in zip(embed_tids, embs):
                track_vis_buffers[tid].append(emb)

        for info in active_track_info:
            tid = info["tid"]
            track_kin_buffers[tid].append(info["kin_feat"])
        t_vis = (time.perf_counter() - t0) * 1000.0

        # 6. Temporal Classification
        t0 = time.perf_counter()
        windows_vis = []
        windows_kin = []
        pred_tids = []

        if cur_fid % 6 == 0:
            for info in active_track_info:
                tid = info["tid"]
                if len(track_vis_buffers[tid]) == 16 and len(track_kin_buffers[tid]) >= 16:
                    v_win = np.array(track_vis_buffers[tid])  # (16, 384)
                    k_win = np.array(list(track_kin_buffers[tid])[-16:])  # (16, 7)
                    windows_vis.append(v_win)
                    windows_kin.append(k_win)
                    pred_tids.append(tid)

        if windows_vis:
            t_v = torch.tensor(np.array(windows_vis), dtype=torch.float32, device=device)
            t_k = torch.tensor(np.array(windows_kin), dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = classifier(t_v, t_k)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
            for tid, p in zip(pred_tids, preds):
                track_last_pred[tid] = int(p)
        t_cls = (time.perf_counter() - t0) * 1000.0

        t_total_frame = (time.perf_counter() - t_f0) * 1000.0

        # Record latencies
        latencies["decode_ms"].append(t_decode)
        latencies["detection_ms"].append(t_det)
        latencies["tracking_ms"].append(t_track)
        latencies["kinematics_ms"].append(t_kin)
        latencies["visual_ms"].append(t_vis)
        latencies["classification_ms"].append(t_cls)
        latencies["total_frame_ms"].append(t_total_frame)

        # 7. Ground Truth Association & Evaluation
        gt_in_frame = gt_annotations.get(cur_fid, [])
        num_gt_total += len(gt_in_frame)
        num_pred_total += len(active_track_info)

        curr_matches = {}
        for gt_bee in gt_in_frame:
            best_iou = 0.0
            best_tid = None
            for p_info in active_track_info:
                iou = compute_iou(gt_bee["box_xyxy"], p_info["box"])
                if iou >= 0.50 and iou > best_iou:
                    best_iou = iou
                    best_tid = p_info["tid"]

            if best_tid is not None:
                num_matches += 1
                curr_matches[gt_bee["tid"]] = best_tid

                # Check ID switches
                if gt_bee["tid"] in prev_matches and prev_matches[gt_bee["tid"]] != best_tid:
                    id_switches += 1

                pred_behavior = track_last_pred[best_tid]
                frame_preds.append(pred_behavior)
                frame_targets.append(gt_bee["label"])

                track_pred_history[gt_bee["tid"]].append(pred_behavior)
                track_gt_history[gt_bee["tid"]].append(gt_bee["label"])

        prev_matches.update(curr_matches)

        if cur_fid % 500 == 0 or cur_fid == frames_to_run:
            fps_live = cur_fid / (time.perf_counter() - t_start_total)
            vram_live = torch.cuda.memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
            print(f"  Frame {cur_fid:4d}/{frames_to_run} | FPS: {fps_live:.1f} | Avg Frame Latency: {np.mean(latencies['total_frame_ms'][-500:]):.1f}ms | VRAM: {vram_live:.1f}MB | Matches: {len(frame_preds)}")

    cap.release()
    total_elapsed_sec = time.perf_counter() - t_start_total
    total_fps = frames_to_run / total_elapsed_sec

    # VRAM stats
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated(device) / (1024**2)
        vram_reserved = torch.cuda.memory_reserved(device) / (1024**2)
        vram_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        vram_alloc, vram_reserved, vram_peak = 0.0, 0.0, 0.0

    # Classification Metrics
    frame_preds = np.array(frame_preds)
    frame_targets = np.array(frame_targets)

    if len(frame_preds) > 0:
        f_macro = float(f1_score(frame_targets, frame_preds, average="macro", zero_division=0))
        f_weighted = float(f1_score(frame_targets, frame_preds, average="weighted", zero_division=0))
        f_acc = float(accuracy_score(frame_targets, frame_preds))
        rep = classification_report(frame_targets, frame_preds, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        per_class_frame_f1 = {c: rep[c]["f1-score"] for c in CLASS_NAMES}
    else:
        f_macro, f_weighted, f_acc = 0.0, 0.0, 0.0
        per_class_frame_f1 = {c: 0.0 for c in CLASS_NAMES}

    # Track-level majority voting
    track_preds = []
    track_targets = []
    for gtid in track_pred_history:
        preds_list = track_pred_history[gtid]
        tgts_list = track_gt_history[gtid]
        if preds_list and tgts_list:
            maj_pred = int(np.bincount(preds_list).argmax())
            maj_tgt = int(np.bincount(tgts_list).argmax())
            track_preds.append(maj_pred)
            track_targets.append(maj_tgt)

    if len(track_preds) > 0:
        t_macro = float(f1_score(track_targets, track_preds, average="macro", zero_division=0))
        t_weighted = float(f1_score(track_targets, track_preds, average="weighted", zero_division=0))
        t_acc = float(accuracy_score(track_targets, track_preds))
        t_rep = classification_report(track_targets, track_preds, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        per_class_track_f1 = {c: t_rep[c]["f1-score"] for c in CLASS_NAMES}
    else:
        t_macro, t_weighted, t_acc = 0.0, 0.0, 0.0
        per_class_track_f1 = {c: 0.0 for c in CLASS_NAMES}

    # Tracking Metrics
    mota = 1.0 - (num_gt_total - num_matches + (num_pred_total - num_matches) + id_switches) / max(1, num_gt_total)
    idf1 = (2.0 * num_matches) / max(1, (num_gt_total + num_pred_total))

    # Benchmark results dictionary
    benchmark_data = {
        "benchmark_name": "Phase5_Live_Video_Benchmark_6000Frames",
        "video_name": VIDEO_NAME,
        "total_frames_processed": frames_to_run,
        "total_runtime_seconds": float(total_elapsed_sec),
        "total_fps": float(total_fps),
        "gpu_device": str(device),
        "vram_mb": {
            "allocated": float(vram_alloc),
            "reserved": float(vram_reserved),
            "peak": float(vram_peak),
        },
        "latency_breakdown_ms": {
            stage: {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
            for stage, vals in latencies.items()
        },
        "frame_level_metrics": {
            "evaluated_samples": len(frame_preds),
            "macro_f1": f_macro,
            "weighted_f1": f_weighted,
            "accuracy": f_acc,
            "per_class_f1": per_class_frame_f1,
        },
        "track_level_metrics": {
            "evaluated_tracks": len(track_preds),
            "macro_f1": t_macro,
            "weighted_f1": t_weighted,
            "accuracy": t_acc,
            "per_class_f1": per_class_track_f1,
        },
        "tracking_metrics": {
            "mota": float(mota),
            "idf1": float(idf1),
            "id_switches": int(id_switches),
            "gt_instances": int(num_gt_total),
            "pred_instances": int(num_pred_total),
            "matches": int(num_matches),
        },
    }

    # Save to results/benchmark_live_6000frames.json
    out_file = RESULTS_DIR / "benchmark_live_6000frames.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    # Log to experiments.csv
    log_to_experiments_csv(
        experiment_name="Phase5_Live_Video_Benchmark_6000Frames",
        model_type="YOLOv8m_ByteTrack_DINOv2_BiLSTM",
        seed=42,
        split_type="live_video_stream",
        train_windows=0,
        train_tracks=0,
        tune_windows=0,
        tune_tracks=0,
        test_windows=len(frame_preds),
        test_tracks=len(track_preds),
        macro_f1=f_macro,
        weighted_f1=f_weighted,
        accuracy=f_acc,
        per_class_f1=per_class_frame_f1,
        config_dict={
            "total_fps": total_fps,
            "vram_peak_mb": vram_peak,
            "latency_p95_ms": benchmark_data["latency_breakdown_ms"]["total_frame_ms"]["p95"],
            "mota": mota,
            "idf1": idf1,
            "id_switches": id_switches,
        },
    )

    print("\n" + "=" * 75)
    print("PHASE 5 LIVE BENCHMARK COMPLETE!")
    print(f"Total Frames: {frames_to_run} | Runtime: {total_elapsed_sec:.1f}s | Speed: {total_fps:.1f} FPS")
    print(f"VRAM Peak: {vram_peak:.1f} MB | Latency (Mean): {benchmark_data['latency_breakdown_ms']['total_frame_ms']['mean']:.2f} ms | P95: {benchmark_data['latency_breakdown_ms']['total_frame_ms']['p95']:.2f} ms")
    print(f"Tracking IDF1: {idf1:.4f} | ID Switches: {id_switches}")
    print(f"Frame-Level Macro-F1: {f_macro:.4f} | Accuracy: {f_acc:.4f}")
    print(f"Track-Level Macro-F1: {t_macro:.4f} | Accuracy: {t_acc:.4f}")
    print(f"Results saved to: {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    run_live_benchmark(max_frames=6000)
