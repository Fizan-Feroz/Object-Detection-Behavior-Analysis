# ============================================================
# Real-Time Bee Behavior Detection & Classification
# Architecture: YOLOv8m + ByteTrack + DINOv2 ViT-S/14 + LSTM
#
# Improvements in Phase 0:
#   1. Causal Kinematics: Shared CausalKinematicsTracker from kinematics.py
#      with identical EMA smoothing and max_gap_frames=5.
#   2. Event-Driven Prediction: Predict only when a new embedding is
#      appended; hold last prediction between samples.
#   3. Track Pruning: Stale tracks unseen for > N seconds are pruned from all dicts.
#   4. Temporal Smoothing & Hysteresis: Probability EMA with hysteresis margin
#      to prevent rapid behavior flickering.
#   5. Confidence Threshold: Low-confidence predictions fall back to "Other".
# ============================================================

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO

from config import (
    BASE_DIR,
    CLASS_NAMES,
    MODELS_DIR,
    VIDEOS_DIR,
)
from kinematics import CausalKinematicsTracker
from train_vit_lstm import BeeLSTMClassifier


DEFAULT_YOLO = BASE_DIR / "yolov8" / "weights" / "best.pt"
DEFAULT_VIT_MODEL = MODELS_DIR / "visual_only_dinov2.pt"
DEFAULT_FUSED_MODEL = MODELS_DIR / "fused_dinov2_kinematics.pt"
DEFAULT_VIDEO = VIDEOS_DIR / "20230711a-fan.mp4"

# Entrance Zone Polygons for known videos (pixels in 1920x1080 / 1280x720)
DEFAULT_ENTRANCE_ZONE = np.array([
    [100, 220], [1860, 270], [1805, 590], [145, 545]
], dtype=np.int32)

# Color coding for behaviors (BGR format for OpenCV)
CLASS_COLORS = {
    "Foraging":     (0, 255, 0),       # Green
    "Defense":      (0, 0, 255),       # Red
    "Fanning":      (0, 255, 255),     # Yellow
    "Washboarding": (255, 140, 0),     # Cyan / Deep Sky Blue
    "Other":        (180, 180, 180),   # Subtle Gray
}

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_bee_10pct_pad(frame_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_w = int(0.10 * bw)
    pad_h = int(0.10 * bh)

    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(W, x2 + pad_w)
    cy2 = min(H, y2 + pad_h)

    crop = frame_rgb[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return crop


def run_pipeline(
    video_path: Path,
    yolo_path: Path,
    lstm_path: Path,
    mode: str = "visual",  # "visual" (384-d) or "fused" (391-d)
    conf_thresh: float = 0.50,
    iou_thresh: float = 0.50,
    k_subsample: int = 6,
    window_length: int = 16,
    behavior_conf_thresh: float = 0.45,
    hysteresis_margin: float = 0.10,
    prob_ema_alpha: float = 0.60,
    max_unseen_sec: float = 2.0,
    show_preview: bool = True,
    save_output: bool = False,
    output_path: Path | None = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)

    print("=" * 70)
    print("BEE BEHAVIOR REAL-TIME PIPELINE (DINOv2 ViT-S/14 + LSTM)")
    print("=" * 70)
    print(f"Device:                 {device}")
    print(f"Video:                  {video_path}")
    print(f"YOLO Detector:          {yolo_path}")
    print(f"LSTM Classifier:        {lstm_path} (mode: {mode})")
    print(f"Subsampling K:          {k_subsample} (~{50.0/k_subsample:.2f} fps)")
    print(f"Window length:          {window_length} steps (~{window_length*k_subsample/50.0:.2f} s)")
    print(f"Confidence Threshold:   {behavior_conf_thresh:.2f} (falls back to Other)")
    print(f"Hysteresis Margin:      {hysteresis_margin:.2f}")
    print(f"Track Pruning Timeout:  {max_unseen_sec:.1f} s")
    print("=" * 70 + "\n")

    # 1. Load YOLOv8
    print("Loading YOLOv8 detector...")
    yolo = YOLO(str(yolo_path))
    print("YOLOv8 loaded successfully.")

    # 2. Load DINOv2 ViT-S/14 Backbone
    print("Loading DINOv2 ViT-S/14 backbone...")
    dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    print("DINOv2 loaded successfully.")

    # 3. Load Trained LSTM Classifier
    print("Loading PyTorch LSTM classifier...")
    in_dim = 391 if mode == "fused" else 384

    # Check if checkpoint exists, or find latest seed checkpoint
    if not lstm_path.exists():
        candidates = list(MODELS_DIR.glob(f"{lstm_path.stem}*.pt"))
        if candidates:
            lstm_path = candidates[0]
            print(f"Using checkpoint: {lstm_path}")
        else:
            print(f"Warning: Checkpoint {lstm_path} not found. Running with uninitialized weights for testing.")

    is_bidi = True
    state_dict = None
    if lstm_path.exists():
        state_dict = torch.load(lstm_path, map_location=device)
        is_bidi = "lstm.weight_ih_l0_reverse" in state_dict

    lstm = BeeLSTMClassifier(input_dim=in_dim, num_classes=len(CLASS_NAMES), bidirectional=is_bidi).to(device)
    if state_dict is not None:
        lstm.load_state_dict(state_dict)
    lstm.eval()
    print(f"LSTM loaded successfully (Input dim: {in_dim}, Bidirectional: {is_bidi}, Classes: {CLASS_NAMES}).\n")

    # 4. Open Video Stream
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Stream: {video_w}x{video_h} @ {native_fps:.1f} fps ({total_frames} frames)")

    writer = None
    if save_output and output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, native_fps, (video_w, video_h))

    # Shared Causal Kinematics Tracker
    kin_tracker = CausalKinematicsTracker(
        polygon=DEFAULT_ENTRANCE_ZONE,
        video_w=video_w,
        video_h=video_h,
        fps=native_fps,
        alpha=0.6,
        max_gap_frames=5,
    )

    # Tracking & Temporal Buffers
    track_embeddings = defaultdict(lambda: deque(maxlen=window_length))
    track_last_sampled_frame = defaultdict(int)
    track_last_seen_frame = {}
    track_predictions = {}       # track_id -> (committed_class_name, confidence)
    track_smoothed_probs = {}    # track_id -> smoothed probability array (5,)

    max_unseen_frames = int(native_fps * max_unseen_sec)
    all_seen_tracks = set()

    frame_idx = 0
    fps_history = deque(maxlen=30)
    t_start = time.time()

    print("\nStarting real-time processing loop. Press 'Q' or 'ESC' to exit, 'P' to pause...\n")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video reached.")
                break
            frame_idx += 1
            t_frame_start = time.time()

            # ----------------------------------------------------
            # A. YOLO + ByteTrack
            # ----------------------------------------------------
            results = yolo.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=640,
                device=0 if device.type == "cuda" else "cpu",
                verbose=False,
            )

            result = results[0]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            active_tracks = []
            crops_to_embed = []
            crop_meta = []  # (track_id, kin_features)

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    all_seen_tracks.add(track_id)
                    track_last_seen_frame[track_id] = frame_idx

                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    # Entrance Polygon Check
                    inside_polygon = cv2.pointPolygonTest(DEFAULT_ENTRANCE_ZONE, (cx, cy), False) >= 0

                    # Causal Kinematics Update (shared module)
                    norm_x = cx / video_w
                    norm_y = cy / video_h
                    norm_w = (x2 - x1) / video_w
                    norm_h = (y2 - y1) / video_h

                    kin_vec = kin_tracker.update(
                        track_id=track_id,
                        frame_id=frame_idx,
                        norm_x=norm_x,
                        norm_y=norm_y,
                        norm_w=norm_w,
                        norm_h=norm_h,
                    )

                    active_tracks.append((track_id, (x1, y1, x2, y2), inside_polygon))

                    # Subsampling check: extract crop only every K frames per track inside polygon
                    last_f = track_last_sampled_frame[track_id]
                    if (frame_idx - last_f) >= k_subsample and inside_polygon:
                        crop = crop_bee_10pct_pad(rgb_frame, x1, y1, x2, y2)
                        crops_to_embed.append(TRANSFORM(crop))
                        crop_meta.append((track_id, kin_vec))
                        track_last_sampled_frame[track_id] = frame_idx

            # ----------------------------------------------------
            # B. Batched DINOv2 Feature Extraction & Buffer Update
            # ----------------------------------------------------
            updated_tids = []
            if crops_to_embed:
                batch_tensor = torch.stack(crops_to_embed).to(device, non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    embs = dinov2(batch_tensor).cpu().numpy().astype(np.float32)

                for (tid, kin_vec), emb in zip(crop_meta, embs):
                    if mode == "fused":
                        feat = np.concatenate([emb, kin_vec], axis=-1)
                    else:
                        feat = emb
                    track_embeddings[tid].append(feat)

                    # Mark tracks that have a full window and just received a new sample
                    if len(track_embeddings[tid]) == window_length:
                        updated_tids.append(tid)

            # ----------------------------------------------------
            # C. Event-Driven LSTM Behavior Prediction (Sample-Triggered)
            # ----------------------------------------------------
            if updated_tids:
                seqs = np.stack([np.array(track_embeddings[tid]) for tid in updated_tids])
                seqs_tensor = torch.tensor(seqs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits = lstm(seqs_tensor)
                    raw_probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for tid, p_dist in zip(updated_tids, raw_probs):
                    # 1. Temporal Smoothing (EMA on probability distribution)
                    if tid in track_smoothed_probs:
                        p_smooth = prob_ema_alpha * p_dist + (1.0 - prob_ema_alpha) * track_smoothed_probs[tid]
                    else:
                        p_smooth = p_dist
                    track_smoothed_probs[tid] = p_smooth

                    top_idx = int(np.argmax(p_smooth))
                    top_conf = float(p_smooth[top_idx])
                    top_name = CLASS_NAMES[top_idx]

                    # 2. Confidence Threshold Check (falls back to Other)
                    if top_conf < behavior_conf_thresh:
                        committed_name = "Other"
                        committed_conf = top_conf
                    else:
                        # 3. Temporal Hysteresis
                        curr_class, _ = track_predictions.get(tid, ("Other", 0.0))
                        if curr_class != "Other" and top_name != curr_class:
                            curr_idx = CLASS_NAMES.index(curr_class)
                            curr_prob = float(p_smooth[curr_idx])
                            if (top_conf - curr_prob) >= hysteresis_margin:
                                committed_name = top_name
                                committed_conf = top_conf
                            else:
                                committed_name = curr_class
                                committed_conf = curr_prob
                        else:
                            committed_name = top_name
                            committed_conf = top_conf

                    track_predictions[tid] = (committed_name, committed_conf)

            # ----------------------------------------------------
            # D. Track Dictionary Pruning (Unseen for > max_unseen_sec)
            # ----------------------------------------------------
            if frame_idx % 30 == 0:
                stale_tids = [
                    tid for tid, last_f in track_last_seen_frame.items()
                    if (frame_idx - last_f) > max_unseen_frames
                ]
                for tid in stale_tids:
                    track_embeddings.pop(tid, None)
                    track_last_sampled_frame.pop(tid, None)
                    track_predictions.pop(tid, None)
                    track_smoothed_probs.pop(tid, None)
                    track_last_seen_frame.pop(tid, None)
                    kin_tracker.reset_track(tid)

            # ----------------------------------------------------
            # E. Visualization & Overlay
            # ----------------------------------------------------
            if show_preview or writer is not None:
                # Draw entrance zone polygon boundary
                cv2.polylines(frame, [DEFAULT_ENTRANCE_ZONE], isClosed=True, color=(255, 200, 0), thickness=2)
                cv2.putText(
                    frame,
                    "Entrance Zone",
                    (DEFAULT_ENTRANCE_ZONE[0][0] + 10, DEFAULT_ENTRANCE_ZONE[0][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 0),
                    2,
                )

                # Draw bee bounding boxes and held predictions
                for (tid, (x1, y1, x2, y2), inside_polygon) in active_tracks:
                    pred_info = track_predictions.get(tid)
                    if pred_info is not None and inside_polygon:
                        c_name, conf = pred_info
                        color = CLASS_COLORS.get(c_name, (200, 200, 200))
                        label_text = f"ID {tid} | {c_name} {conf*100:.0f}%"
                    else:
                        color = (120, 120, 120)
                        label_text = f"ID {tid}" if inside_polygon else f"ID {tid} (outside)"

                    thickness = 2 if (pred_info and pred_info[0] != "Other") else 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    # Text background badge
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    text_color = (0, 0, 0) if (pred_info and pred_info[0] in ("Fanning", "Foraging")) else (255, 255, 255)
                    cv2.putText(
                        frame,
                        label_text,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        text_color,
                        1,
                        cv2.LINE_AA,
                    )

                # FPS and Status HUD
                fps = 1.0 / max(1e-4, (time.time() - t_frame_start))
                fps_history.append(fps)
                avg_fps = np.mean(fps_history)

                cv2.rectangle(frame, (10, 10), (420, 115), (0, 0, 0), -1)
                cv2.putText(frame, f"ViT-LSTM Behavior Pipeline ({mode.upper()})", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | FPS: {avg_fps:.1f}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(frame, f"Active Tracks: {len(active_tracks)} | Predictions Held: {len(track_predictions)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
                cv2.putText(frame, "Event-Driven LSTM | Hysteresis On | Keys: [Q] Quit [P] Pause", (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

                if writer is not None:
                    writer.write(frame)

                if show_preview:
                    cv2.imshow("Bee Behavior Detection (DINOv2 ViT-S/14 + LSTM)", frame)

        # Keyboard controls
        if show_preview:
            key = cv2.waitKey(1 if not paused else 50) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\nQuit requested by user.")
                break
            elif key in (ord("p"), ord("P")):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")

    cap.release()
    if writer is not None:
        writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_idx / (elapsed + 1e-5)

    pred_counts = defaultdict(int)
    for c_name, conf in track_predictions.values():
        pred_counts[c_name] += 1

    print("\n" + "=" * 70, flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Frames processed:       {frame_idx} / {total_frames}", flush=True)
    print(f"Total seen tracks:      {len(all_seen_tracks)}", flush=True)
    print(f"Tracks with prediction: {len(track_predictions)}", flush=True)
    print(f"Wall clock time:        {elapsed:.1f} s", flush=True)
    print(f"Average FPS:            {avg_fps:.2f} FPS", flush=True)
    print("\nBehavior Distribution (Active predicted tracks):", flush=True)
    for c_name in CLASS_NAMES:
        cnt = pred_counts[c_name]
        pct = (cnt / len(track_predictions) * 100) if track_predictions else 0.0
        print(f"  {c_name:<16}: {cnt:<5} ({pct:.1f}%)", flush=True)
    print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Live Bee Behavior Detection (DINOv2 + LSTM)")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO), help="Input video path")
    parser.add_argument("--yolo", type=str, default=str(DEFAULT_YOLO), help="YOLOv8 weights path")
    parser.add_argument("--lstm", type=str, default=str(DEFAULT_VIT_MODEL), help="ViT-LSTM weights path")
    parser.add_argument("--mode", type=str, choices=["visual", "fused"], default="visual", help="visual or fused")
    parser.add_argument("--conf", type=float, default=0.50, help="YOLO confidence threshold")
    parser.add_argument("--behavior-conf", type=float, default=0.45, help="Behavior confidence threshold fallback to Other")
    parser.add_argument("--hysteresis", type=float, default=0.10, help="Hysteresis margin to switch behavior")
    parser.add_argument("--prune-sec", type=float, default=2.0, help="Prune tracks unseen for > N seconds")
    parser.add_argument("--no-preview", action="store_true", help="Disable GUI window preview")
    parser.add_argument("--save-out", type=str, default="", help="Path to save annotated video output")
    args = parser.parse_args()

    lstm_file = Path(args.lstm)
    if args.mode == "fused" and args.lstm == str(DEFAULT_VIT_MODEL):
        lstm_file = DEFAULT_FUSED_MODEL

    run_pipeline(
        video_path=Path(args.video),
        yolo_path=Path(args.yolo),
        lstm_path=lstm_file,
        mode=args.mode,
        conf_thresh=args.conf,
        behavior_conf_thresh=args.behavior_conf,
        hysteresis_margin=args.hysteresis,
        max_unseen_sec=args.prune_sec,
        show_preview=not args.no_preview,
        save_output=bool(args.save_out),
        output_path=Path(args.save_out) if args.save_out else None,
    )


if __name__ == "__main__":
    main()
