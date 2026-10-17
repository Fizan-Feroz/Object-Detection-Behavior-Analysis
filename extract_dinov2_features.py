# ============================================================
# DINOv2 ViT-S/14 + Kinematics Feature Extractor (Optimized)
#
# Specifications:
#   1. Native 50 fps video rate subsampled by factor K=6 (~8.33 fps).
#   2. Sequence length T=16 steps (~1.92 s observation window).
#   3. Full 50 fps trajectory smoothing & kinematics computed
#      BEFORE subsampling, normalized by frame resolution.
#   4. 10% bounding box crop padding to preserve wings and legs.
#   5. Polygon entrance-zone gating: windows must remain strictly
#      inside the entrance zone throughout all 16 steps.
#   6. 5 Classes:
#        0: Foraging
#        1: Defense
#        2: Fanning
#        3: Washboarding
#        4: Other (strictly 100% unlabeled frames inside polygon)
#      Mixed/transition windows are dropped from training.
#   7. Split strategy:
#      - Time-blocked: first 70% train, last 30% val.
#      - Buffer gap: 250 frames (~5s) for long videos, or 10%
#        of frames for short videos, purged at the boundary.
#      - Multi-phase offsets (0..5) for 6x natural train augmentation.
#      - Single phase offset (0) with non-overlapping windows for val.
#   8. DINOv2 ViT-S/14 FP16 batched inference on CUDA:
#      Extracts each unique (track_id, frame_id) crop ONCE and caches
#      embeddings to eliminate all redundant ViT passes.
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from config import (
    BASE_DIR,
    CLASS_NAMES,
    DINOV2_FEATURES_DIR as OUTPUT_DIR,
    TRACKING_DIR as DATA_DIR,
    VIDEOS,
    seed_everything,
)
from kinematics import compute_causal_kinematics_offline



TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_polygon(zone_file: Path) -> np.ndarray:
    text = zone_file.read_text(encoding="utf-8").strip()
    import re
    pts = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not pts:
        raise ValueError(f"Could not parse polygon in {zone_file}")
    return np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)


def load_raw_tracks(ann_file: Path) -> dict[int, list[dict]]:
    tracks = defaultdict(list)
    with open(ann_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 10:
                continue
            frame_id = int(row[0])
            track_id = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            classes = [int(row[6]), int(row[7]), int(row[8]), int(row[9])]
            tracks[track_id].append({
                "track_id": track_id,
                "frame_id": frame_id,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "classes": classes,
            })

    for tid in tracks:
        tracks[tid].sort(key=lambda item: item["frame_id"])
    return tracks


def build_candidate_windows(
    enriched_tracks: dict[int, list[dict]],
    polygon: np.ndarray,
    video_w: int,
    video_h: int,
    total_frames: int,
    video_name: str = "",
    k_subsample: int = 6,
    window_length: int = 16,
    split_ratio: float = 0.70,
    buffer_gap: int = 250,
) -> tuple[list[dict], list[dict], dict]:
    # Ensure buffer gap doesn't exceed 15% of short videos
    actual_buffer = min(buffer_gap, int(total_frames * 0.12))
    train_boundary = int(total_frames * split_ratio)
    val_start = train_boundary + actual_buffer

    train_windows = []
    val_windows = []

    stats = {
        "train_by_class": defaultdict(int),
        "val_by_class": defaultdict(int),
        "train_tracks_by_class": defaultdict(set),
        "val_tracks_by_class": defaultdict(set),
        "dropped_mixed": 0,
        "dropped_outside_polygon": 0,
        "dropped_gap_crossing": 0,
        "actual_buffer": actual_buffer,
    }

    min_required_span = k_subsample * (window_length - 1)

    for track_id, obs_list in enriched_tracks.items():
        if len(obs_list) < window_length:
            continue

        train_segments = [o for o in obs_list if o["frame_id"] < train_boundary]
        val_segments = [o for o in obs_list if o["frame_id"] >= val_start]

        # Train: multi-phase offsets (0..K-1)
        if len(train_segments) >= window_length:
            train_frame_map = {o["frame_id"]: o for o in train_segments}
            min_train_f = train_segments[0]["frame_id"]
            max_train_f = train_segments[-1]["frame_id"]

            for phase in range(k_subsample):
                sampled_fids = [f for f in range(min_train_f + phase, max_train_f + 1, k_subsample) if f in train_frame_map]
                if len(sampled_fids) < window_length:
                    continue

                for i in range(0, len(sampled_fids) - window_length + 1):
                    win_fids = sampled_fids[i:i + window_length]
                    if any(win_fids[s] - win_fids[s-1] != k_subsample for s in range(1, window_length)):
                        continue

                    win_obs = [train_frame_map[f] for f in win_fids]
                    if any(cv2.pointPolygonTest(polygon, (o["x"] * video_w, o["y"] * video_h), False) < 0 for o in win_obs):
                        stats["dropped_outside_polygon"] += 1
                        continue

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
                        stats["dropped_mixed"] += 1
                        continue

                    train_windows.append({
                        "track_id": track_id,
                        "class": target_class,
                        "frame_ids": win_fids,
                        "obs": win_obs,
                        "video": video_name,
                    })
                    stats["train_by_class"][target_class] += 1
                    stats["train_tracks_by_class"][target_class].add(track_id)

        # Val: single phase offset 0, non-overlapping stride = window_length
        if len(val_segments) >= window_length:
            val_frame_map = {o["frame_id"]: o for o in val_segments}
            min_val_f = val_segments[0]["frame_id"]
            max_val_f = val_segments[-1]["frame_id"]

            sampled_fids = [f for f in range(min_val_f, max_val_f + 1, k_subsample) if f in val_frame_map]
            i = 0
            while i <= len(sampled_fids) - window_length:
                win_fids = sampled_fids[i:i + window_length]
                if any(win_fids[s] - win_fids[s-1] != k_subsample for s in range(1, window_length)):
                    i += 1
                    continue

                win_obs = [val_frame_map[f] for f in win_fids]
                if any(cv2.pointPolygonTest(polygon, (o["x"] * video_w, o["y"] * video_h), False) < 0 for o in win_obs):
                    stats["dropped_outside_polygon"] += 1
                    i += 1
                    continue

                class_counts = [sum(o["classes"][c] for o in win_obs) for c in range(4)]
                total_labeled = sum(class_counts)
                target_class = None
                if total_labeled == 0:
                    target_class = 4
                else:
                    for c_idx in range(4):
                        if class_counts[c_idx] >= int(0.80 * window_length) and total_labeled == class_counts[c_idx]:
                            target_class = c_idx
                            break

                if target_class is None:
                    stats["dropped_mixed"] += 1
                    i += 1
                    continue

                val_windows.append({
                    "track_id": track_id,
                    "class": target_class,
                    "frame_ids": win_fids,
                    "obs": win_obs,
                    "video": video_name,
                })
                stats["val_by_class"][target_class] += 1
                stats["val_tracks_by_class"][target_class].add(track_id)
                i += window_length

    return train_windows, val_windows, stats


def crop_bee_10pct_pad(frame: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = frame.shape[:2]
    cx = x * W
    cy = y * H
    bw = w * W
    bh = h * H

    pad_w = 0.10 * bw
    pad_h = 0.10 * bh

    x1 = max(0, int(cx - bw / 2.0 - pad_w))
    y1 = max(0, int(cy - bh / 2.0 - pad_h))
    x2 = min(W, int(cx + bw / 2.0 + pad_w))
    y2 = min(H, int(cy + bh / 2.0 + pad_h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return crop


def extract_features_optimized(
    video_path: Path,
    train_windows: list[dict],
    val_windows: list[dict],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extracts DINOv2 visual features by caching unique (track_id, frame_id) embeddings.
    Zero redundant ViT passes!
    """
    all_windows = train_windows + val_windows
    if not all_windows:
        empty_vis = np.empty((0, 16, 384), dtype=np.float32)
        empty_kin = np.empty((0, 16, 7), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        empty_tracks = np.empty((0,), dtype=np.int64)
        empty_vids = np.empty((0,), dtype=object)
        return (empty_vis, empty_kin, empty_y, empty_tracks, empty_vids), (empty_vis, empty_kin, empty_y, empty_tracks, empty_vids)

    # 1. Collect all unique (track_id, frame_id) -> observation dict
    needed_per_frame = defaultdict(dict)
    for win in all_windows:
        for obs in win["obs"]:
            tid = obs["track_id"]
            fid = obs["frame_id"]
            needed_per_frame[fid][tid] = obs

    total_unique_crops = sum(len(d) for d in needed_per_frame.values())
    print(f"  Unique crops to extract: {total_unique_crops} across {len(needed_per_frame)} frames (streaming video)...", flush=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    # Cache for 384-d embeddings
    emb_cache: dict[tuple[int, int], np.ndarray] = {}

    crops_queue = []
    keys_queue = []

    def flush_batch():
        if not crops_queue:
            return
        batch_tensor = torch.stack(crops_queue).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            embs = model(batch_tensor).cpu().numpy().astype(np.float32)
        for (tid, fid), emb in zip(keys_queue, embs):
            emb_cache[(tid, fid)] = emb
        crops_queue.clear()
        keys_queue.clear()

    cur_frame_id = 0
    t0 = time.time()
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_frame_id += 1

        if cur_frame_id in needed_per_frame:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for tid, obs in needed_per_frame[cur_frame_id].items():
                crop = crop_bee_10pct_pad(rgb_frame, obs["x"], obs["y"], obs["w"], obs["h"])
                crops_queue.append(TRANSFORM(crop))
                keys_queue.append((tid, cur_frame_id))
                extracted_count += 1

                if len(crops_queue) >= batch_size:
                    flush_batch()

    flush_batch()
    cap.release()
    t_extract = time.time() - t0
    print(f"  Extracted {len(emb_cache)} unique DINOv2 embeddings in {t_extract:.1f}s ({len(emb_cache)/(t_extract+1e-5):.1f} crops/s)", flush=True)

    # 2. Assemble Window Tensors
    def assemble(windows_list: list[dict]):
        n = len(windows_list)
        vis = np.zeros((n, 16, 384), dtype=np.float32)
        kin = np.zeros((n, 16, 7), dtype=np.float32)
        y = np.array([w["class"] for w in windows_list], dtype=np.int64)
        tracks = np.array([w["track_id"] for w in windows_list], dtype=np.int64)
        videos = np.array([w.get("video", "") for w in windows_list], dtype=object)

        for w_i, win in enumerate(windows_list):
            for s_i, obs in enumerate(win["obs"]):
                key = (obs["track_id"], obs["frame_id"])
                vis[w_i, s_i] = emb_cache.get(key, np.zeros(384, dtype=np.float32))
                kin[w_i, s_i] = obs["kinematics"]
        return vis, kin, y, tracks, videos

    train_data = assemble(train_windows)
    val_data = assemble(val_windows)
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 + Kinematics for Bee Behavior")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=64, help="DINOv2 batch size")
    parser.add_argument("--k-subsample", type=int, default=6, help="Subsample step (50fps / K)")
    parser.add_argument("--window-length", type=int, default=16, help="Window steps (16 steps = ~1.92s)")
    parser.add_argument("--buffer-gap", type=int, default=250, help="Purge gap between train and val")
    parser.add_argument("--single-video", type=str, default="", help="Extract single video (for testing)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70, flush=True)
    print("DINOv2 ViT-S/14 BEE FEATURE EXTRACTION (OPTIMIZED)", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {device}", flush=True)
    print(f"K-subsample: {args.k_subsample} (~{50.0/args.k_subsample:.2f} fps)", flush=True)
    print(f"Window length: {args.window_length} steps (~{args.window_length*args.k_subsample/50.0:.2f} s)", flush=True)
    print(f"Buffer gap: {args.buffer_gap} frames", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(flush=True)

    print("Loading DINOv2 ViT-S/14...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    print("DINOv2 loaded successfully.\n", flush=True)

    videos_to_process = {args.single_video: VIDEOS[args.single_video]} if args.single_video else VIDEOS

    dataset_manifest = {
        "k_subsample": args.k_subsample,
        "window_length": args.window_length,
        "classes": CLASS_NAMES,
        "videos": {},
    }

    all_train_vis, all_train_kin, all_train_y = [], [], []
    all_train_tracks, all_train_vids = [], []
    all_val_vis, all_val_kin, all_val_y = [], [], []
    all_val_tracks, all_val_vids = [], []

    for name, vinfo in videos_to_process.items():
        vpath = DATA_DIR / vinfo["video"]
        apath = DATA_DIR / vinfo["ann"]
        zpath = DATA_DIR / vinfo["zone"]

        if not vpath.exists() or not apath.exists() or not zpath.exists():
            print(f"Skipping {name}: file missing", flush=True)
            continue

        print(f"--- Processing {name} ---", flush=True)
        polygon = parse_polygon(zpath)
        tracks = load_raw_tracks(apath)
        print(f"  Loaded {len(tracks)} raw tracks from {apath.name}", flush=True)

        enriched = {}
        for tid, obs_list in tracks.items():
            enriched[tid] = compute_causal_kinematics_offline(
                obs_list,
                polygon=polygon,
                video_w=vinfo["w"],
                video_h=vinfo["h"],
                fps=50.0,
                alpha=0.6,
                max_gap_frames=5,
            )

        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        train_wins, val_wins, stats = build_candidate_windows(
            enriched,
            polygon,
            vinfo["w"],
            vinfo["h"],
            total_frames,
            video_name=name,
            k_subsample=args.k_subsample,
            window_length=args.window_length,
            buffer_gap=args.buffer_gap,
        )

        print(f"  Train windows: {len(train_wins)}, Val windows: {len(val_wins)} (actual buffer gap: {stats['actual_buffer']} frames)", flush=True)
        print("  Train distribution:", flush=True)
        for c_idx in sorted(stats["train_by_class"].keys()):
            count = stats["train_by_class"][c_idx]
            t_cnt = len(stats["train_tracks_by_class"][c_idx])
            print(f"    {CLASS_NAMES[c_idx]:<14}: {count:<6} windows ({t_cnt} distinct tracks)", flush=True)
        print("  Validation distribution:", flush=True)
        for c_idx in sorted(stats["val_by_class"].keys()):
            count = stats["val_by_class"][c_idx]
            t_cnt = len(stats["val_tracks_by_class"][c_idx])
            print(f"    {CLASS_NAMES[c_idx]:<14}: {count:<6} windows ({t_cnt} distinct tracks)", flush=True)
        print(f"  Dropped: {stats['dropped_mixed']} mixed/transition, {stats['dropped_outside_polygon']} outside polygon", flush=True)

        # Feature Extraction
        print("  Extracting unique DINOv2 crop embeddings...", flush=True)
        (tr_vis, tr_kin, tr_y, tr_tracks, tr_vids), (va_vis, va_kin, va_y, va_tracks, va_vids) = extract_features_optimized(
            vpath, train_wins, val_wins, model, device, args.batch_size
        )

        # Save per-video feature tensors for Leave-One-Video-Out checks
        vid_out_dir = OUTPUT_DIR / name
        vid_out_dir.mkdir(exist_ok=True)
        np.save(vid_out_dir / "train_visual.npy", tr_vis)
        np.save(vid_out_dir / "train_kinematics.npy", tr_kin)
        np.save(vid_out_dir / "train_labels.npy", tr_y)
        np.save(vid_out_dir / "train_tracks.npy", tr_tracks)
        np.save(vid_out_dir / "train_videos.npy", tr_vids)

        np.save(vid_out_dir / "val_visual.npy", va_vis)
        np.save(vid_out_dir / "val_kinematics.npy", va_kin)
        np.save(vid_out_dir / "val_labels.npy", va_y)
        np.save(vid_out_dir / "val_tracks.npy", va_tracks)
        np.save(vid_out_dir / "val_videos.npy", va_vids)

        dataset_manifest["videos"][name] = {
            "train_windows": len(train_wins),
            "val_windows": len(val_wins),
            "train_class_counts": {CLASS_NAMES[k]: v for k, v in stats["train_by_class"].items()},
            "val_class_counts": {CLASS_NAMES[k]: v for k, v in stats["val_by_class"].items()},
            "train_tracks_per_class": {CLASS_NAMES[k]: len(v) for k, v in stats["train_tracks_by_class"].items()},
            "val_tracks_per_class": {CLASS_NAMES[k]: len(v) for k, v in stats["val_tracks_by_class"].items()},
        }

        all_train_vis.append(tr_vis)
        all_train_kin.append(tr_kin)
        all_train_y.append(tr_y)
        all_train_tracks.append(tr_tracks)
        all_train_vids.append(tr_vids)

        all_val_vis.append(va_vis)
        all_val_kin.append(va_kin)
        all_val_y.append(va_y)
        all_val_tracks.append(va_tracks)
        all_val_vids.append(va_vids)
        print(flush=True)

    if all_train_vis:
        X_train_vis = np.concatenate(all_train_vis, axis=0)
        X_train_kin = np.concatenate(all_train_kin, axis=0)
        y_train = np.concatenate(all_train_y, axis=0)
        train_tracks = np.concatenate(all_train_tracks, axis=0)
        train_vids = np.concatenate(all_train_vids, axis=0)

        X_val_vis = np.concatenate(all_val_vis, axis=0)
        X_val_kin = np.concatenate(all_val_kin, axis=0)
        y_val = np.concatenate(all_val_y, axis=0)
        val_tracks = np.concatenate(all_val_tracks, axis=0)
        val_vids = np.concatenate(all_val_vids, axis=0)

        np.save(OUTPUT_DIR / "X_train_visual.npy", X_train_vis)
        np.save(OUTPUT_DIR / "X_train_kinematics.npy", X_train_kin)
        np.save(OUTPUT_DIR / "y_train.npy", y_train)
        np.save(OUTPUT_DIR / "train_tracks.npy", train_tracks)
        np.save(OUTPUT_DIR / "train_videos.npy", train_vids)

        np.save(OUTPUT_DIR / "X_val_visual.npy", X_val_vis)
        np.save(OUTPUT_DIR / "X_val_kinematics.npy", X_val_kin)
        np.save(OUTPUT_DIR / "y_val.npy", y_val)
        np.save(OUTPUT_DIR / "val_tracks.npy", val_tracks)
        np.save(OUTPUT_DIR / "val_videos.npy", val_vids)

        dataset_manifest["aggregated"] = {
            "train_total": len(y_train),
            "val_total": len(y_val),
            "train_class_distribution": {CLASS_NAMES[i]: int(np.sum(y_train == i)) for i in range(5)},
            "val_class_distribution": {CLASS_NAMES[i]: int(np.sum(y_val == i)) for i in range(5)},
        }

        with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(dataset_manifest, f, indent=2)

        print("=" * 70, flush=True)
        print("FEATURE EXTRACTION COMPLETE", flush=True)
        print("=" * 70, flush=True)
        print(f"Aggregated Train: {X_train_vis.shape[0]} windows (Visual: {X_train_vis.shape}, Kinematics: {X_train_kin.shape})", flush=True)
        print(f"Aggregated Val:   {X_val_vis.shape[0]} windows (Visual: {X_val_vis.shape}, Kinematics: {X_val_kin.shape})", flush=True)
        print("Train Distribution:", dataset_manifest["aggregated"]["train_class_distribution"], flush=True)
        print("Val Distribution:  ", dataset_manifest["aggregated"]["val_class_distribution"], flush=True)
        print(f"Artifacts saved to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
