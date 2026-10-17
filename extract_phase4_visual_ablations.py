# ============================================================
# Phase 4 Visual Ablations Feature Extractor
# Extracts for the 633 Blocked 5-Fold Test Windows:
#   1. CLS + Mean Patch Tokens (768-d)
#   2. Crop 112x112 (384-d)
#   3. Gaussian Blurred Crops (384-d, sigma=3.0)
#   4. Downscaled 4x Crops (384-d)
# ============================================================

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import (
    BASE_DIR,
    CLASS_NAMES,
    DINOV2_FEATURES_DIR,
    TRACKING_DIR,
    VIDEOS,
    get_device,
)
from extract_phase1_data import (
    crop_bee_10pct_pad,
    define_blocks,
    load_raw_tracks,
    parse_polygon,
    build_block_windows,
)
from kinematics import compute_causal_kinematics_offline

OUTPUT_DIR = DINOV2_FEATURES_DIR / "phase4_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORM_224 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM_112 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_visual_ablations(device: torch.device, batch_size: int = 64):
    print("=" * 75)
    print("EXTRACTING PHASE 4 VISUAL ABLATION FEATURES FOR TEST WINDOWS")
    print(f"Device: {device} | Batch Size: {batch_size}")
    print("=" * 75)

    print("Loading DINOv2 ViT-S/14 model...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).half().eval()

    all_test_wins = []
    
    # Rebuild test windows exactly identical to Phase 1
    for v_idx, (v_name, vinfo) in enumerate(VIDEOS.items()):
        vpath = TRACKING_DIR / vinfo["video"]
        ann_path = TRACKING_DIR / vinfo["ann"]
        zone_path = TRACKING_DIR / vinfo["zone"]

        polygon = parse_polygon(zone_path)
        raw_tracks = load_raw_tracks(ann_path)

        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        blocks, gap = define_blocks(total_frames)

        enriched = {}
        for tid, obs_list in raw_tracks.items():
            enriched[tid] = compute_causal_kinematics_offline(
                obs_list,
                polygon=polygon,
                video_w=vinfo["w"],
                video_h=vinfo["h"],
                fps=50.0,
                alpha=0.6,
                max_gap_frames=5,
            )

        test_wins, _, _ = build_block_windows(
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
        print(f"  {v_name}: {len(test_wins)} test windows")
        all_test_wins.extend(test_wins)

    print(f"\nTotal test windows: {len(all_test_wins)} (Expected 633)")
    assert len(all_test_wins) == 633, f"Expected 633 test windows, got {len(all_test_wins)}"

    # Map needed frames per video
    wins_by_video = defaultdict(list)
    for win in all_test_wins:
        wins_by_video[win["video_name"]].append(win)

    # Output arrays
    n_test = len(all_test_wins)
    cls_patch_vis = np.zeros((n_test, 16, 768), dtype=np.float32)
    crop112_vis = np.zeros((n_test, 16, 384), dtype=np.float32)
    blur_vis = np.zeros((n_test, 16, 384), dtype=np.float32)
    downscale_vis = np.zeros((n_test, 16, 384), dtype=np.float32)

    global_win_idx = 0

    for v_name, vwins in wins_by_video.items():
        vinfo = VIDEOS[v_name]
        vpath = TRACKING_DIR / vinfo["video"]
        print(f"\nProcessing {v_name} ({len(vwins)} windows)...")

        # Gather required frames
        needed_per_frame = defaultdict(dict)
        for w_rel_idx, win in enumerate(vwins):
            w_abs_idx = global_win_idx + w_rel_idx
            for s_idx, obs in enumerate(win["obs"]):
                fid = obs["frame_id"]
                tid = obs["track_id"]
                needed_per_frame[fid][tid] = (w_abs_idx, s_idx, obs)

        cap = cv2.VideoCapture(str(vpath))
        cur_fid = 0

        crops_raw = []
        crop_meta = []

        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cur_fid += 1

            if cur_fid in needed_per_frame:
                for tid, (w_abs, s_idx, obs) in needed_per_frame[cur_fid].items():
                    crop = crop_bee_10pct_pad(frame, obs["x"], obs["y"], obs["w"], obs["h"])
                    crops_raw.append(crop)
                    crop_meta.append((w_abs, s_idx))
        cap.release()

        print(f"  Extracted {len(crops_raw)} raw crops from {v_name} in {time.time() - t0:.1f}s. Computing DINOv2 embeddings...")

        # Batch inference
        t_infer = time.time()
        for i in range(0, len(crops_raw), batch_size):
            batch_crops = crops_raw[i:i + batch_size]
            batch_meta = crop_meta[i:i + batch_size]

            # 1. 224x224 standard -> CLS + patch
            tensors_224 = torch.stack([TRANSFORM_224(c) for c in batch_crops]).to(device).half()
            with torch.no_grad():
                res = model.forward_features(tensors_224)
                cls_tok = res["x_norm_clstoken"]  # (B, 384)
                patch_tok = res["x_norm_patchtokens"].mean(dim=1)  # (B, 384)
                fused_cp = torch.cat([cls_tok, patch_tok], dim=-1).float().cpu().numpy()  # (B, 768)

            # 2. 112x112 crops
            tensors_112 = torch.stack([TRANSFORM_112(c) for c in batch_crops]).to(device).half()
            with torch.no_grad():
                emb_112 = model(tensors_112).float().cpu().numpy()  # (B, 384)

            # 3. Gaussian Blur (sigma=3.0, 9x9 kernel)
            blurred_crops = [cv2.GaussianBlur(c, (9, 9), 3.0) for c in batch_crops]
            tensors_blur = torch.stack([TRANSFORM_224(c) for c in blurred_crops]).to(device).half()
            with torch.no_grad():
                emb_blur = model(tensors_blur).float().cpu().numpy()  # (B, 384)

            # 4. Downscaled 4x then upscaled
            down_crops = []
            for c in batch_crops:
                h, w = c.shape[:2]
                small = cv2.resize(c, (max(2, w // 4), max(2, h // 4)), interpolation=cv2.INTER_AREA)
                up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                down_crops.append(up)
            tensors_down = torch.stack([TRANSFORM_224(c) for c in down_crops]).to(device).half()
            with torch.no_grad():
                emb_down = model(tensors_down).float().cpu().numpy()  # (B, 384)

            # Assign to arrays
            for k, (w_abs, s_idx) in enumerate(batch_meta):
                cls_patch_vis[w_abs, s_idx] = fused_cp[k]
                crop112_vis[w_abs, s_idx] = emb_112[k]
                blur_vis[w_abs, s_idx] = emb_blur[k]
                downscale_vis[w_abs, s_idx] = emb_down[k]

        print(f"  Computed embeddings in {time.time() - t_infer:.1f}s")
        global_win_idx += len(vwins)

    # Save all arrays
    np.save(OUTPUT_DIR / "test_visual_cls_patch.npy", cls_patch_vis)
    np.save(OUTPUT_DIR / "test_visual_112.npy", crop112_vis)
    np.save(OUTPUT_DIR / "test_visual_blur.npy", blur_vis)
    np.save(OUTPUT_DIR / "test_visual_downscale.npy", downscale_vis)

    print("\nSaved all Phase 4 visual ablation test arrays to:", OUTPUT_DIR)
    print("  test_visual_cls_patch.npy:", cls_patch_vis.shape)
    print("  test_visual_112.npy:      ", crop112_vis.shape)
    print("  test_visual_blur.npy:     ", blur_vis.shape)
    print("  test_visual_downscale.npy:", downscale_vis.shape)


if __name__ == "__main__":
    device = get_device()
    extract_visual_ablations(device=device, batch_size=64)
