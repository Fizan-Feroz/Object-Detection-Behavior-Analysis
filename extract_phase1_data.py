# ============================================================
# Phase 1 Data Extractor: Blocked 5-Fold, LOVO & Shortcut Features
#
# Specifications:
#   1. 5 Contiguous Blocks per video with >= 250-frame gaps
#      (proportional for short videos like yt5).
#   2. Boundary-crossing tracks dropped: No track crosses a boundary.
#   3. Dual stride:
#      - Non-overlapping evaluation windows (phase 0, stride=16 steps = 96 frames)
#      - Augmented training windows (multi-phase offsets 0..5, stride=1 step)
#   4. Causal Scale-Invariant Kinematics (7-d) from kinematics.py.
#   5. Bounding box coordinates (16, 4) for Shortcut Diagnostic (a).
#   6. Background patches cropped from median background frame
#      at exact (x, y, w, h) for Shortcut Diagnostic (b).
#   7. DINOv2 ViT-S/14 384-d visual embeddings extracted on GPU.
#   8. Video ID labels (0..4) for Shortcut Diagnostic (c).
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import re
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
    DINOV2_FEATURES_DIR,
    TRACKING_DIR as DATA_DIR,
    VIDEOS,
)
from kinematics import compute_causal_kinematics_offline


OUTPUT_DIR = DINOV2_FEATURES_DIR / "phase1_data"

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def parse_polygon(zone_file: Path) -> np.ndarray:
    text = zone_file.read_text(encoding="utf-8").strip()
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


def compute_median_background(video_path: Path, num_samples: int = 40) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    return median_frame


def crop_bee_10pct_pad(frame_bgr: np.ndarray, x: float, y: float, w: float, h: float) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
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

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def define_blocks(total_frames: int, num_blocks: int = 5) -> tuple[list[tuple[int, int]], int]:
    gap = 250 if total_frames > 1500 else int(total_frames * 0.08)
    avail = total_frames - (num_blocks - 1) * gap
    block_len = avail // num_blocks
    blocks = []
    start = 0
    for b in range(num_blocks):
        end = start + block_len if b < (num_blocks - 1) else total_frames
        blocks.append((start, end))
        start = end + gap
    return blocks, gap


def build_block_windows(
    enriched_tracks: dict[int, list[dict]],
    polygon: np.ndarray,
    video_w: int,
    video_h: int,
    blocks: list[tuple[int, int]],
    video_name: str,
    video_idx: int,
    k_subsample: int = 6,
    window_length: int = 16,
) -> tuple[list[dict], list[dict], dict]:
    """
    Builds both test windows (phase 0, non-overlapping stride 16)
    and train windows (multi-phase, stride 1) for each block.
    """
    test_windows = []
    train_windows = []

    stats = {
        "test_by_block_class": defaultdict(lambda: defaultdict(int)),
        "train_by_block_class": defaultdict(lambda: defaultdict(int)),
        "test_tracks_by_block": defaultdict(lambda: defaultdict(set)),
        "train_tracks_by_block": defaultdict(lambda: defaultdict(set)),
        "dropped_boundary_crossing": 0,
        "dropped_outside_polygon": 0,
        "dropped_mixed": 0,
    }

    # 1. Filter tracks that cross block boundaries
    valid_tracks = {}
    for tid, obs_list in enriched_tracks.items():
        f0, f1 = obs_list[0]["frame_id"], obs_list[-1]["frame_id"]
        touched = [b_idx for b_idx, (bs, be) in enumerate(blocks) if not (f1 < bs or f0 >= be)]
        if len(touched) == 1:
            # Track is strictly internal to one block!
            valid_tracks[tid] = (touched[0], obs_list)
        else:
            stats["dropped_boundary_crossing"] += 1

    # 2. Window formation per block
    for tid, (b_idx, obs_list) in valid_tracks.items():
        if len(obs_list) < window_length:
            continue

        bs, be = blocks[b_idx]
        b_obs = [o for o in obs_list if bs <= o["frame_id"] < be]
        if len(b_obs) < window_length:
            continue

        obs_map = {o["frame_id"]: o for o in b_obs}
        min_f, max_f = b_obs[0]["frame_id"], b_obs[-1]["frame_id"]

        # A. Non-Overlapping Test Windows (Phase 0, stride = window_length)
        sfids_test = [f for f in range(min_f, max_f + 1, k_subsample) if f in obs_map]
        i = 0
        while i <= len(sfids_test) - window_length:
            wfids = sfids_test[i:i + window_length]
            if any(wfids[s] - wfids[s-1] != k_subsample for s in range(1, window_length)):
                i += 1
                continue

            wobs = [obs_map[f] for f in wfids]
            if any(cv2.pointPolygonTest(polygon, (o["x"] * video_w, o["y"] * video_h), False) < 0 for o in wobs):
                stats["dropped_outside_polygon"] += 1
                i += 1
                continue

            cc = [sum(o["classes"][c] for o in wobs) for c in range(4)]
            tot = sum(cc)
            tgt = None
            if tot == 0:
                tgt = 4  # Other
            else:
                for c_idx in range(4):
                    if cc[c_idx] >= int(0.80 * window_length) and tot == cc[c_idx]:
                        tgt = c_idx
                        break

            if tgt is None:
                stats["dropped_mixed"] += 1
                i += 1
                continue

            test_windows.append({
                "video_name": video_name,
                "video_idx": video_idx,
                "track_id": tid,
                "block_id": b_idx,
                "class": tgt,
                "frame_ids": wfids,
                "obs": wobs,
                "is_test": True,
            })
            stats["test_by_block_class"][b_idx][tgt] += 1
            stats["test_tracks_by_block"][b_idx][tgt].add(tid)
            i += window_length

        # B. Dense Training Windows (Multi-phase offsets 0..K-1, stride = 1)
        for phase in range(k_subsample):
            sfids_tr = [f for f in range(min_f + phase, max_f + 1, k_subsample) if f in obs_map]
            if len(sfids_tr) < window_length:
                continue

            for idx in range(0, len(sfids_tr) - window_length + 1):
                wfids = sfids_tr[idx:idx + window_length]
                if any(wfids[s] - wfids[s-1] != k_subsample for s in range(1, window_length)):
                    continue

                wobs = [obs_map[f] for f in wfids]
                if any(cv2.pointPolygonTest(polygon, (o["x"] * video_w, o["y"] * video_h), False) < 0 for o in wobs):
                    continue

                cc = [sum(o["classes"][c] for o in wobs) for c in range(4)]
                tot = sum(cc)
                tgt = None
                if tot == 0:
                    tgt = 4
                else:
                    for c_idx in range(4):
                        if cc[c_idx] >= int(0.80 * window_length) and tot == cc[c_idx]:
                            tgt = c_idx
                            break

                if tgt is None:
                    continue

                train_windows.append({
                    "video_name": video_name,
                    "video_idx": video_idx,
                    "track_id": tid,
                    "block_id": b_idx,
                    "class": tgt,
                    "frame_ids": wfids,
                    "obs": wobs,
                    "is_test": False,
                })
                stats["train_by_block_class"][b_idx][tgt] += 1
                stats["train_tracks_by_block"][b_idx][tgt].add(tid)

    return test_windows, train_windows, stats


def extract_embeddings_and_background(
    video_path: Path,
    all_windows: list[dict],
    median_bg: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """
    Extracts DINOv2 embeddings for both bee crops and background patches.
    """
    needed_per_frame = defaultdict(dict)
    for win in all_windows:
        for obs in win["obs"]:
            needed_per_frame[obs["frame_id"]][obs["track_id"]] = obs

    cap = cv2.VideoCapture(str(video_path))
    bee_cache = {}
    bg_cache = {}

    crops_queue = []
    keys_queue = []

    bg_crops_queue = []
    bg_keys_queue = []

    def flush_bee_batch():
        if not crops_queue:
            return
        batch_tensor = torch.stack(crops_queue).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            embs = model(batch_tensor).cpu().numpy().astype(np.float32)
        for key, emb in zip(keys_queue, embs):
            bee_cache[key] = emb
        crops_queue.clear()
        keys_queue.clear()

    def flush_bg_batch():
        if not bg_crops_queue:
            return
        batch_tensor = torch.stack(bg_crops_queue).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            embs = model(batch_tensor).cpu().numpy().astype(np.float32)
        for key, emb in zip(bg_keys_queue, embs):
            bg_cache[key] = emb
        bg_crops_queue.clear()
        bg_keys_queue.clear()

    cur_fid = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_fid += 1

        if cur_fid in needed_per_frame:
            for tid, obs in needed_per_frame[cur_fid].items():
                # 1. Bee Crop
                crop = crop_bee_10pct_pad(frame, obs["x"], obs["y"], obs["w"], obs["h"])
                crops_queue.append(TRANSFORM(crop))
                keys_queue.append((tid, cur_fid))

                # 2. Background Patch at exact same position (from median background)
                bg_crop = crop_bee_10pct_pad(median_bg, obs["x"], obs["y"], obs["w"], obs["h"])
                bg_crops_queue.append(TRANSFORM(bg_crop))
                bg_keys_queue.append((tid, cur_fid))

                if len(crops_queue) >= batch_size:
                    flush_bee_batch()
                if len(bg_crops_queue) >= batch_size:
                    flush_bg_batch()

    flush_bee_batch()
    flush_bg_batch()
    cap.release()

    elapsed = time.time() - t0
    print(f"  Extracted {len(bee_cache)} bee crops & {len(bg_cache)} bg patches in {elapsed:.1f}s ({len(bee_cache)/(elapsed+1e-5):.1f} pairs/s)", flush=True)
    return bee_cache, bg_cache


def assemble_tensors(
    windows: list[dict],
    bee_cache: dict[tuple[int, int], np.ndarray],
    bg_cache: dict[tuple[int, int], np.ndarray],
) -> dict[str, np.ndarray]:
    n = len(windows)
    vis = np.zeros((n, 16, 384), dtype=np.float32)
    bg_vis = np.zeros((n, 16, 384), dtype=np.float32)
    kin = np.zeros((n, 16, 7), dtype=np.float32)
    coords = np.zeros((n, 16, 4), dtype=np.float32)  # [x, y, w, h]
    y = np.array([w["class"] for w in windows], dtype=np.int64)
    tids = np.array([w["track_id"] for w in windows], dtype=np.int64)
    vids = np.array([w["video_idx"] for w in windows], dtype=np.int64)
    bids = np.array([w["block_id"] for w in windows], dtype=np.int64)
    is_test = np.array([w["is_test"] for w in windows], dtype=bool)

    for w_i, win in enumerate(windows):
        for s_i, obs in enumerate(win["obs"]):
            key = (obs["track_id"], obs["frame_id"])
            vis[w_i, s_i] = bee_cache.get(key, np.zeros(384, dtype=np.float32))
            bg_vis[w_i, s_i] = bg_cache.get(key, np.zeros(384, dtype=np.float32))
            kin[w_i, s_i] = obs["kinematics"]
            coords[w_i, s_i] = [obs["x"], obs["y"], obs["w"], obs["h"]]

    return {
        "visual": vis,
        "bg_visual": bg_vis,
        "kinematics": kin,
        "coords": coords,
        "labels": y,
        "track_ids": tids,
        "video_ids": vids,
        "block_ids": bids,
        "is_test": is_test,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract Phase 1 Blocked K-Fold and Shortcut Features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70, flush=True)
    print("PHASE 1 DATA EXTRACTION: BLOCKED 5-FOLD & SHORTCUT PROBES", flush=True)
    print("=" * 70, flush=True)
    print(f"Device: {device} | Batch Size: {args.batch_size}", flush=True)
    print(f"Output Dir: {OUTPUT_DIR}\n", flush=True)

    print("Loading DINOv2 ViT-S/14...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    print("DINOv2 loaded successfully.\n", flush=True)

    all_tensors = defaultdict(list)
    manifest = {
        "videos": {},
        "classes": CLASS_NAMES,
    }

    for v_idx, (v_name, vinfo) in enumerate(VIDEOS.items()):
        vpath = DATA_DIR / vinfo["video"]
        apath = DATA_DIR / vinfo["ann"]
        zpath = DATA_DIR / vinfo["zone"]

        print(f"--- Processing {v_name} ({v_idx+1}/5) ---", flush=True)
        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        polygon = parse_polygon(zpath)
        tracks = load_raw_tracks(apath)
        print(f"  Loaded {len(tracks)} raw tracks across {total_frames} frames", flush=True)

        blocks, gap = define_blocks(total_frames, num_blocks=5)
        print(f"  5 Contiguous Blocks (gap={gap} frames):", flush=True)
        for b_i, (bs, be) in enumerate(blocks):
            print(f"    Block {b_i}: [{bs}, {be}) ({be-bs} frames)", flush=True)

        # Compute causal kinematics
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

        # Build block windows
        test_wins, train_wins, stats = build_block_windows(
            enriched,
            polygon,
            vinfo["w"],
            vinfo["h"],
            blocks,
            video_name=v_name,
            video_idx=v_idx,
            k_subsample=6,
            window_length=16,
        )

        print(f"  Generated {len(test_wins)} test windows (non-overlapping) and {len(train_wins)} train windows (dense)", flush=True)
        print("  Test Window Distribution by Block:", flush=True)
        for b_i in range(5):
            counts_str = ", ".join([f"{CLASS_NAMES[c]}: {stats['test_by_block_class'][b_i][c]} ({len(stats['test_tracks_by_block'][b_i][c])} trk)" for c in range(5) if stats['test_by_block_class'][b_i][c] > 0])
            print(f"    Block {b_i}: {counts_str if counts_str else 'empty'}", flush=True)

        # Compute median background frame
        print("  Computing median background frame...", flush=True)
        median_bg = compute_median_background(vpath, num_samples=35)

        # Extract embeddings for both bee crops and background patches
        all_wins = test_wins + train_wins
        bee_cache, bg_cache = extract_embeddings_and_background(
            vpath, all_wins, median_bg, model, device, args.batch_size
        )

        # Assemble video tensors
        vid_tensors = assemble_tensors(all_wins, bee_cache, bg_cache)

        # Save per-video arrays
        vid_dir = OUTPUT_DIR / v_name
        vid_dir.mkdir(exist_ok=True)
        for k, arr in vid_tensors.items():
            np.save(vid_dir / f"{k}.npy", arr)
            all_tensors[k].append(arr)

        manifest["videos"][v_name] = {
            "total_frames": total_frames,
            "blocks": blocks,
            "gap": gap,
            "test_windows": len(test_wins),
            "train_windows": len(train_wins),
            "test_class_counts": {
                b_i: {CLASS_NAMES[c]: stats["test_by_block_class"][b_i][c] for c in range(5)}
                for b_i in range(5)
            },
        }
        print(flush=True)

    # Concatenate aggregated arrays
    print("=" * 70, flush=True)
    print("AGGREGATING PHASE 1 DATASETS", flush=True)
    print("=" * 70, flush=True)
    for k in all_tensors:
        agg = np.concatenate(all_tensors[k], axis=0)
        np.save(OUTPUT_DIR / f"{k}.npy", agg)
        print(f"  {k:<12}: shape={agg.shape}, dtype={agg.dtype}", flush=True)

    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nPhase 1 datasets saved to: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
