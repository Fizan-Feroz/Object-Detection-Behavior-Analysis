# ============================================================
# Inspect Real Support Counts across 9 (T, Gating) Configurations
# ============================================================

import csv
from collections import defaultdict
import cv2
import numpy as np

from config import CLASS_NAMES, TRACKING_DIR, VIDEOS
from extract_dinov2_features import load_raw_tracks, parse_polygon


def count_support_for_config(T: int, gating_mode: str, k_subsample: int = 6):
    """
    Simulates 5-block partitioning per video and counts test support (windows & tracks)
    and train support for given T and gating_mode.
    """
    total_test_windows = 0
    test_counts = defaultdict(int)
    test_tracks = defaultdict(set)
    train_counts = defaultdict(int)
    train_tracks = defaultdict(set)

    for v_name, v_info in VIDEOS.items():
        tr_file = TRACKING_DIR / v_info["ann"]
        poly_file = TRACKING_DIR / v_info["zone"]
        vw, vh = v_info["w"], v_info["h"]

        tracks = load_raw_tracks(tr_file)
        poly = parse_polygon(poly_file)

        mp4_path = TRACKING_DIR / v_info["video"]
        cap = cv2.VideoCapture(str(mp4_path))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 5 contiguous blocks with gap
        gap = 78 if v_name == "yt5" else 250
        num_blocks = 5
        avail_frames = total_f - (num_blocks - 1) * gap
        block_len = avail_frames // num_blocks

        blocks = []
        cur_f = 0
        for b_i in range(num_blocks):
            b_start = cur_f
            b_end = cur_f + block_len
            blocks.append((b_start, b_end))
            cur_f = b_end + gap

        # Drop boundary crossing tracks
        for tid, obs_list in tracks.items():
            fids = [o["frame_id"] for o in obs_list]
            track_blocks = set()
            for f in fids:
                for b_i, (b_start, b_end) in enumerate(blocks):
                    if b_start <= f < b_end:
                        track_blocks.add(b_i)
                        break
            if len(track_blocks) > 1:
                continue  # boundary crossing track dropped
            if len(track_blocks) == 0:
                continue
            b_idx = list(track_blocks)[0]

            frame_map = {o["frame_id"]: o for o in obs_list}
            min_f, max_f = fids[0], fids[-1]

            # Block observations
            b_start, b_end = blocks[b_idx]
            block_fids = [f for f in fids if b_start <= f < b_end]
            if len(block_fids) < T:
                continue

            # In each block, extract test windows (stride T, phase 0)
            # and train windows (stride 1 for behavior, stride T for other, phases 0..5)
            # Test windows
            test_sampled = [f for f in range(b_start, b_end, k_subsample) if f in frame_map]
            i = 0
            while i <= len(test_sampled) - T:
                win_fids = test_sampled[i:i + T]
                if any(win_fids[s] - win_fids[s-1] != k_subsample for s in range(1, T)):
                    i += 1
                    continue

                win_obs = [frame_map[f] for f in win_fids]
                # Check gating
                in_poly = [cv2.pointPolygonTest(poly, (o["x"] * vw, o["y"] * vh), False) >= 0 for o in win_obs]
                frac_in = sum(in_poly) / float(T)

                keep = False
                if gating_mode == "all_inside" and frac_in == 1.0:
                    keep = True
                elif gating_mode == "half_inside" and frac_in >= 0.5:
                    keep = True
                elif gating_mode == "endpoint_inside" and (in_poly[0] or in_poly[-1]):
                    keep = True

                if keep:
                    class_counts = [sum(o["classes"][c] for o in win_obs) for c in range(4)]
                    total_labeled = sum(class_counts)
                    target_class = 4
                    if total_labeled > 0:
                        target_class = None
                        for c_idx in range(4):
                            if class_counts[c_idx] >= int(0.80 * T) and total_labeled == class_counts[c_idx]:
                                target_class = c_idx
                                break
                    if target_class is not None:
                        c_name = CLASS_NAMES[target_class]
                        test_counts[c_name] += 1
                        test_tracks[c_name].add(f"{v_name}_{tid}")
                        total_test_windows += 1
                i += T  # non-overlapping

    return total_test_windows, test_counts, {c: len(test_tracks[c]) for c in CLASS_NAMES}


def main():
    print("=" * 80)
    print("PHASE 2: REAL TEST SUPPORT COUNTS ACROSS 9 (T, GATING) VARIANTS")
    print("=" * 80)

    configs = [
        (16, "all_inside"),
        (16, "half_inside"),
        (16, "endpoint_inside"),
        (8, "all_inside"),
        (8, "half_inside"),
        (8, "endpoint_inside"),
        (6, "all_inside"),
        (6, "half_inside"),
        (6, "endpoint_inside"),
    ]

    print(f"{'Config (T, Gating)':<24} | {'Total Test Win':<14} | {'Foraging Win (Trk)':<20} | {'Defense Win (Trk)':<18} | {'Fanning Win (Trk)':<18} | {'Wash Win (Trk)':<16}")
    print("-" * 115)

    for T, gating in configs:
        tot_win, counts, trks = count_support_for_config(T, gating)
        cfg_str = f"T={T}, {gating}"
        forage_str = f"{counts['Foraging']} win ({trks['Foraging']} trk)"
        def_str = f"{counts['Defense']} win ({trks['Defense']} trk)"
        fan_str = f"{counts['Fanning']} win ({trks['Fanning']} trk)"
        wash_str = f"{counts['Washboarding']} win ({trks['Washboarding']} trk)"
        print(f"{cfg_str:<24} | {tot_win:<14} | {forage_str:<20} | {def_str:<18} | {fan_str:<18} | {wash_str:<16}")


if __name__ == "__main__":
    main()
