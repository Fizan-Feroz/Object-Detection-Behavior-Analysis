# ============================================================
# Phase 2: Foraging & Gating Comprehensive Audit
#
# Answers:
# 1. Contiguous run lengths per behavior (median, p90, min, max)
# 2. Track attrition funnel through each gating rule
# 3. Why only 4 Foraging tracks reached train
# ============================================================

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

from config import (
    BASE_DIR,
    CLASS_NAMES,
    RESULTS_DIR,
    TRACKING_DIR,
    VIDEOS,
)


def load_raw_tracks(filepath: Path) -> dict[int, list[dict]]:
    tracks = defaultdict(list)
    with open(filepath, "r", encoding="utf-8") as f:
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
        tracks[tid].sort(key=lambda o: o["frame_id"])
    return tracks


def parse_polygon(zone_file: Path) -> np.ndarray:
    text = zone_file.read_text(encoding="utf-8").strip()
    import re
    pts = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if not pts:
        raise ValueError(f"Could not parse polygon in {zone_file}")
    return np.array([[int(x), int(y)] for x, y in pts], dtype=np.int32)


def run_audit():
    import sys
    out_file = RESULTS_DIR / "foraging_audit.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    f_out = open(out_file, "w", encoding="utf-8")

    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = Tee(sys.stdout, f_out)
    print("=" * 80)
    print("PHASE 2 ITEM 1: FORAGING & GATING AUDIT")
    print("=" * 80)

    # 1. Collect all tracks across all 5 videos
    video_tracks = {}
    video_polygons = {}
    video_dims = {}

    for v_name, v_info in VIDEOS.items():
        tr_file = TRACKING_DIR / v_info["ann"]
        poly_file = TRACKING_DIR / v_info["zone"]
        if not tr_file.exists() or not poly_file.exists():
            print(f"Skipping {v_name}: missing files.")
            continue

        raw_tracks = load_raw_tracks(tr_file)
        polygon = parse_polygon(poly_file)

        vw = v_info["w"]
        vh = v_info["h"]
        mp4_path = TRACKING_DIR / v_info["video"]
        cap = cv2.VideoCapture(str(mp4_path))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        video_tracks[v_name] = raw_tracks
        video_polygons[v_name] = polygon
        video_dims[v_name] = (vw, vh, total_f)

    # ------------------------------------------------------------
    # A. Contiguous Run Length Analysis by Behavior
    # ------------------------------------------------------------
    # For each behavior (0: Foraging, 1: Defense, 2: Fanning, 3: Washboarding, 4: Other/None)
    # A contiguous run is an uninterrupted sequence of frames where the behavior is active.
    run_lengths = defaultdict(list)
    track_total_lengths = defaultdict(list)
    tracks_with_behavior = defaultdict(lambda: defaultdict(set))  # class -> video -> set(tids)

    for v_name, tracks in video_tracks.items():
        for tid, obs_list in tracks.items():
            fids = [o["frame_id"] for o in obs_list]
            track_len = len(obs_list)

            # Check each of the 4 active behaviors
            for c_idx in range(4):
                c_name = CLASS_NAMES[c_idx]
                has_c = any(o["classes"][c_idx] == 1 for o in obs_list)
                if has_c:
                    tracks_with_behavior[c_name][v_name].add(tid)
                    track_total_lengths[c_name].append(track_len)

                    # Compute contiguous runs of active c_idx
                    current_run = 0
                    prev_f = None
                    for o in obs_list:
                        is_active = (o["classes"][c_idx] == 1)
                        if is_active:
                            if prev_f is not None and o["frame_id"] == prev_f + 1:
                                current_run += 1
                            else:
                                if current_run > 0:
                                    run_lengths[c_name].append(current_run)
                                current_run = 1
                            prev_f = o["frame_id"]
                        else:
                            if current_run > 0:
                                run_lengths[c_name].append(current_run)
                                current_run = 0
                            prev_f = None
                    if current_run > 0:
                        run_lengths[c_name].append(current_run)

            # Other / None (all 4 classes == 0)
            has_other = any(sum(o["classes"]) == 0 for o in obs_list)
            if has_other:
                tracks_with_behavior["Other"][v_name].add(tid)
                track_total_lengths["Other"].append(track_len)
                current_run = 0
                prev_f = None
                for o in obs_list:
                    is_other = (sum(o["classes"]) == 0)
                    if is_other:
                        if prev_f is not None and o["frame_id"] == prev_f + 1:
                            current_run += 1
                        else:
                            if current_run > 0:
                                run_lengths["Other"].append(current_run)
                            current_run = 1
                        prev_f = o["frame_id"]
                    else:
                        if current_run > 0:
                            run_lengths["Other"].append(current_run)
                            current_run = 0
                        prev_f = None
                if current_run > 0:
                    run_lengths["Other"].append(current_run)

    print("\n--- 1. Contiguous Run Lengths & Raw Track Counts by Behavior ---")
    print(f"{'Behavior':<14} | {'Raw Tracks':<12} | {'Contig Runs':<12} | {'Median Len':<12} | {'P90 Len':<10} | {'Min Len':<10} | {'Max Len':<10}")
    print("-" * 85)
    for c_name in CLASS_NAMES:
        total_trks = sum(len(tids) for tids in tracks_with_behavior[c_name].values())
        runs = run_lengths[c_name]
        if runs:
            med_len = float(np.median(runs))
            p90_len = float(np.percentile(runs, 90))
            min_l = int(np.min(runs))
            max_l = int(np.max(runs))
            print(f"{c_name:<14} | {total_trks:<12} | {len(runs):<12} | {med_len:<12.1f} | {p90_len:<10.1f} | {min_l:<10} | {max_l:<10}")
        else:
            print(f"{c_name:<14} | {total_trks:<12} | 0            | N/A          | N/A        | N/A        | N/A")

    print("\nRaw Tracks Breakdown by Video:")
    for c_name in CLASS_NAMES:
        breakdown = {v: len(tids) for v, tids in tracks_with_behavior[c_name].items() if len(tids) > 0}
        print(f"  {c_name:<14}: Total {sum(breakdown.values())} tracks across videos: {breakdown}")

    # ------------------------------------------------------------
    # B. Track Gating Attrition Funnel
    # ------------------------------------------------------------
    # Test how many tracks survive each rule:
    # Rule 1: Contiguous track frames span >= required_span = K * (T - 1) + 1
    # Rule 2: Temporal sampling gap == K exactly
    # Rule 3: Spatial polygon gating:
    #         - all_inside (100% steps in polygon)
    #         - half_inside (>= 50% steps in polygon)
    #         - endpoint_inside (step 0 or step -1 in polygon)
    #         - any_location (no polygon filter)
    # Rule 4: Behavior label purity: >= 80% behavior in window
    print("\n" + "=" * 80)
    print("--- 2. Track Attrition Funnel Across Gating Rules & Temporal Horizons ---")
    print("=" * 80)

    for T in [16, 8, 6]:
        K = 6
        req_span = K * (T - 1) + 1
        sec = req_span / 50.0
        print(f"\n>>> CONFIGURATION: Window T = {T} steps at K = {K} (Requires >= {req_span} frames / {sec:.2f} s contiguous span)")
        print(f"{'Behavior':<14} | {'Raw Trk':<8} | {'Span>=' + str(req_span):<10} | {'All Inside':<12} | {'>=50% Inside':<12} | {'Endpoint In':<12} | {'Any Location':<12}")
        print("-" * 85)

        for c_idx in range(4):
            c_name = CLASS_NAMES[c_idx]
            raw_cnt = sum(len(tids) for tids in tracks_with_behavior[c_name].values())

            span_survivors = set()
            all_in_survivors = set()
            half_in_survivors = set()
            end_in_survivors = set()
            any_loc_survivors = set()

            for v_name, tracks in video_tracks.items():
                vw, vh, _ = video_dims[v_name]
                poly = video_polygons[v_name]

                for tid in tracks_with_behavior[c_name][v_name]:
                    obs_list = tracks[tid]
                    # Filter observations that have behavior c_idx
                    c_obs = [o for o in obs_list if o["classes"][c_idx] == 1]
                    if len(c_obs) < T:
                        continue

                    # Check if there is any contiguous block of length >= req_span with gap K
                    fids = [o["frame_id"] for o in obs_list]
                    frame_map = {o["frame_id"]: o for o in obs_list}
                    min_f, max_f = fids[0], fids[-1]

                    # Check sampleable windows
                    has_span = False
                    has_all_in = False
                    has_half_in = False
                    has_end_in = False
                    has_any_loc = False

                    for phase in range(K):
                        sampled_fids = [f for f in range(min_f + phase, max_f + 1, K) if f in frame_map]
                        if len(sampled_fids) < T:
                            continue

                        for i in range(0, len(sampled_fids) - T + 1):
                            win_fids = sampled_fids[i:i + T]
                            if any(win_fids[s] - win_fids[s-1] != K for s in range(1, T)):
                                continue  # gap violation

                            win_obs = [frame_map[f] for f in win_fids]
                            # Check behavior purity
                            c_count = sum(o["classes"][c_idx] for o in win_obs)
                            tot_labeled = sum(sum(o["classes"]) for o in win_obs)
                            if c_count < int(0.80 * T) or tot_labeled != c_count:
                                continue  # purity failure

                            has_span = True
                            has_any_loc = True

                            # Check polygon containment
                            in_poly = [cv2.pointPolygonTest(poly, (o["x"] * vw, o["y"] * vh), False) >= 0 for o in win_obs]
                            frac_in = sum(in_poly) / float(T)

                            if frac_in == 1.0:
                                has_all_in = True
                            if frac_in >= 0.5:
                                has_half_in = True
                            if in_poly[0] or in_poly[-1]:
                                has_end_in = True

                    unique_trk_key = f"{v_name}_{tid}"
                    if has_span:
                        span_survivors.add(unique_trk_key)
                    if has_all_in:
                        all_in_survivors.add(unique_trk_key)
                    if has_half_in:
                        half_in_survivors.add(unique_trk_key)
                    if has_end_in:
                        end_in_survivors.add(unique_trk_key)
                    if has_any_loc:
                        any_loc_survivors.add(unique_trk_key)

            print(f"{c_name:<14} | {raw_cnt:<8} | {len(span_survivors):<10} | {len(all_in_survivors):<12} | {len(half_in_survivors):<12} | {len(end_in_survivors):<12} | {len(any_loc_survivors):<12}")

    # ------------------------------------------------------------
    # C. Deep Dive on Foraging Tracks
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("--- 3. Granular Inspection of Every Raw Foraging Track ---")
    print("=" * 80)

    for v_name, tracks in video_tracks.items():
        vw, vh, _ = video_dims[v_name]
        poly = video_polygons[v_name]
        for tid in sorted(tracks_with_behavior["Foraging"][v_name]):
            obs = tracks[tid]
            tot_len = len(obs)
            forage_obs = [o for o in obs if o["classes"][0] == 1]
            forage_len = len(forage_obs)
            min_f = obs[0]["frame_id"]
            max_f = obs[-1]["frame_id"]
            in_poly_cnt = sum(cv2.pointPolygonTest(poly, (o["x"] * vw, o["y"] * vh), False) >= 0 for o in obs)
            poly_pct = (in_poly_cnt / tot_len) * 100.0

            # Calculate speed in pixels/frame
            dists = []
            for s in range(1, tot_len):
                dx = (obs[s]["x"] - obs[s-1]["x"]) * vw
                dy = (obs[s]["y"] - obs[s-1]["y"]) * vh
                dists.append(np.hypot(dx, dy))
            avg_speed_px = float(np.mean(dists)) if dists else 0.0

            print(f"Video: {v_name:<16} | Track ID: {tid:<4} | Total Frames: {tot_len:<4} | Foraging Frames: {forage_len:<4} | Span: [{min_f}..{max_f}] | In Polygon: {poly_pct:5.1f}% | Avg Speed: {avg_speed_px:4.1f} px/f")


if __name__ == "__main__":
    run_audit()
