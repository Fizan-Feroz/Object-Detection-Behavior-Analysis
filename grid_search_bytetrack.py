# ============================================================
# Phase 3 Item 2: Tracking Evaluation & ByteTrack Grid Search
# Evaluates ByteTrack against ground-truth tracks using IDF1 and MOTA.
# Grid-searches new_track_thresh, track_buffer, track_low_thresh, match_thresh.
# Saves winning configuration to custom_bytetrack.yaml,
# writes benchmark to results/tracking_benchmark.json,
# and logs to results/experiments.csv.
# ============================================================

from __future__ import annotations

import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# Patch numpy 2.0 compatibility for motmetrics
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, *args, **kwargs: np.asarray(a, dtype=float)

import motmetrics as mm
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from config import (
    BASE_DIR,
    EXPERIMENTS_CSV,
    RESULTS_DIR,
    TRACKING_DIR,
    VIDEOS,
    seed_everything,
)


class DetectionBatch:
    """Lightweight adapter exposing xywh, conf, cls with slicing for BYTETracker."""
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __getitem__(self, mask: np.ndarray) -> DetectionBatch:
        return DetectionBatch(self.xywh[mask], self.conf[mask], self.cls[mask])

    def __len__(self) -> int:
        return len(self.conf)


def load_gt_tracks(ann_path: Path, video_w: int, video_h: int, max_frames: int | None = None) -> dict[int, list[tuple[int, list[float]]]]:
    gt_by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    with open(ann_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            fid = int(row[0])
            if max_frames and fid > max_frames:
                continue
            tid = int(row[1])
            x, y, w, h = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            tl_x = x * video_w
            tl_y = y * video_h
            bw = w * video_w
            bh = h * video_h
            if fid not in gt_by_frame:
                gt_by_frame[fid] = []
            gt_by_frame[fid].append((tid, [tl_x, tl_y, bw, bh]))
    return gt_by_frame


def extract_or_load_detections(
    video_key: str,
    video_path: Path,
    model_path: Path,
    cache_dir: Path,
    max_frames: int = 1000,
    conf_thresh: float = 0.05,
    imgsz: int = 640,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"dets_{video_key}_{max_frames}_{imgsz}.npz"

    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        num_frames = int(data["num_frames"])
        frames_data = []
        for i in range(num_frames):
            xywh = data[f"xywh_{i}"]
            conf = data[f"conf_{i}"]
            cls = data[f"cls_{i}"]
            frames_data.append((xywh, conf, cls))
        print(f"Loaded {len(frames_data)} cached detection frames for {video_key}", flush=True)
        return frames_data

    print(f"Extracting detections for {video_key} ({video_path.name}) up to frame {max_frames}...", flush=True)
    model = YOLO(str(model_path))
    frames_data = []
    frame_idx = 0
    t0 = time.time()

    for r in model.predict(source=str(video_path), imgsz=imgsz, conf=conf_thresh, device=0, stream=True, verbose=False):
        frame_idx += 1
        if frame_idx > max_frames:
            break
        b = r.boxes.cpu()
        frames_data.append((b.xywh.numpy(), b.conf.numpy(), b.cls.numpy()))

    t_elapsed = time.time() - t0
    print(f"Extracted {len(frames_data)} frames in {t_elapsed:.1f}s ({len(frames_data)/max(t_elapsed, 1e-4):.1f} FPS)", flush=True)

    save_dict: dict[str, np.ndarray] = {"num_frames": np.array(len(frames_data))}
    for i, (xywh, conf, cls) in enumerate(frames_data):
        save_dict[f"xywh_{i}"] = xywh
        save_dict[f"conf_{i}"] = conf
        save_dict[f"cls_{i}"] = cls
    np.savez_compressed(cache_file, **save_dict)
    print(f"Cached detections to {cache_file}", flush=True)

    return frames_data


def evaluate_tracker_config(
    eval_videos: dict[str, dict],
    detections_by_video: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    gt_by_video: dict[str, dict[int, list[tuple[int, list[float]]]]],
    config: dict,
) -> dict:
    args = SimpleNamespace(
        track_high_thresh=float(config.get("track_high_thresh", 0.25)),
        track_low_thresh=float(config.get("track_low_thresh", 0.10)),
        new_track_thresh=float(config.get("new_track_thresh", 0.25)),
        track_buffer=int(config.get("track_buffer", 30)),
        match_thresh=float(config.get("match_thresh", 0.80)),
        fuse_score=bool(config.get("fuse_score", True)),
    )

    acc = mm.MOTAccumulator(auto_id=True)
    total_tracks_count = 0
    t0 = time.time()

    for vid_key, det_frames in detections_by_video.items():
        tracker = BYTETracker(args)
        gt_by_frame = gt_by_video[vid_key]

        for frame_idx, (xywh, conf, cls) in enumerate(det_frames, start=1):
            batch = DetectionBatch(xywh, conf, cls)
            pred_tracks = tracker.update(batch)
            total_tracks_count += len(pred_tracks)

            gt_items = gt_by_frame.get(frame_idx, [])
            gt_ids = [item[0] for item in gt_items]
            gt_boxes = [item[1] for item in gt_items]

            pred_ids = [int(row[4]) for row in pred_tracks]
            pred_boxes = [[float(row[0]), float(row[1]), float(row[2] - row[0]), float(row[3] - row[1])] for row in pred_tracks]

            dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            acc.update(gt_ids, pred_ids, dists)

    eval_time = time.time() - t0
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "num_frames",
            "mota",
            "motp",
            "idf1",
            "idp",
            "idr",
            "mostly_tracked",
            "mostly_lost",
            "num_false_positives",
            "num_misses",
            "num_switches",
            "num_fragmentations",
        ],
        name="eval",
    )

    row = summary.iloc[0]
    mota = float(row.get("mota", 0.0))
    idf1 = float(row.get("idf1", 0.0))
    motp = float(row.get("motp", 0.0))
    idp = float(row.get("idp", 0.0))
    idr = float(row.get("idr", 0.0))
    num_switches = int(row.get("num_switches", 0))
    num_fps = int(row.get("num_false_positives", 0))
    num_misses = int(row.get("num_misses", 0))
    num_frames = int(row.get("num_frames", 0))

    composite_score = (idf1 + max(0.0, mota)) / 2.0

    # Explicit memory cleanup
    del acc
    del mh
    del summary
    gc.collect()

    return {
        "config": config,
        "mota": mota,
        "idf1": idf1,
        "motp": motp,
        "idp": idp,
        "idr": idr,
        "num_switches": num_switches,
        "num_false_positives": num_fps,
        "num_misses": num_misses,
        "num_frames": num_frames,
        "eval_time_sec": eval_time,
        "composite_score": composite_score,
    }


def main():
    seed_everything(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = RESULTS_DIR / "detection_cache"

    yolo_weights = BASE_DIR / "yolov8" / "weights" / "best.pt"

    eval_video_keys = ["yt5", "yt8", "20230609b-def", "20230711a-fan"]
    max_frames_per_video = {
        "yt5": 977,
        "yt8": 1000,
        "20230609b-def": 1000,
        "20230711a-fan": 1000,
    }

    detections_by_video = {}
    gt_by_video = {}

    print("=" * 70, flush=True)
    print("PHASE 3 ITEM 2: BYTETRACK HYPERPARAMETER GRID SEARCH & MOT BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    for vkey in eval_video_keys:
        vinfo = VIDEOS[vkey]
        vpath = TRACKING_DIR / vinfo["video"]
        ann_path = TRACKING_DIR / vinfo["ann"]
        mf = max_frames_per_video[vkey]

        dets = extract_or_load_detections(
            video_key=vkey,
            video_path=vpath,
            model_path=yolo_weights,
            cache_dir=cache_dir,
            max_frames=mf,
            conf_thresh=0.05,
            imgsz=640,
        )
        detections_by_video[vkey] = dets

        gt = load_gt_tracks(ann_path, vinfo["w"], vinfo["h"], max_frames=mf)
        gt_by_video[vkey] = gt
        print(f"Loaded {len(gt)} GT frames for {vkey}", flush=True)

    # 1. Baseline configuration (Ultralytics default)
    baseline_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.25,
        "track_buffer": 30,
        "match_thresh": 0.80,
        "fuse_score": True,
    }

    print("\n--- Evaluating Baseline ByteTrack Configuration ---", flush=True)
    baseline_res = evaluate_tracker_config(VIDEOS, detections_by_video, gt_by_video, baseline_cfg)
    print(f"Baseline -> IDF1: {baseline_res['idf1']:.4f}, MOTA: {baseline_res['mota']:.4f}, Score: {baseline_res['composite_score']:.4f}, IDSW: {baseline_res['num_switches']}", flush=True)

    # 2. Targeted Grid Search across the 4 specified hyperparameters:
    # new_track_thresh, track_buffer, track_low_thresh, match_thresh
    new_track_thresh_vals = [0.20, 0.25, 0.30]
    track_buffer_vals = [15, 30, 50]
    track_low_thresh_vals = [0.10, 0.15]
    match_thresh_vals = [0.70, 0.80]

    grid_configs = []
    # Core factorial grid: 3 x 3 x 2 x 2 = 36 configs
    for ntt in new_track_thresh_vals:
        for tb in track_buffer_vals:
            for tlt in track_low_thresh_vals:
                for mt in match_thresh_vals:
                    grid_configs.append({
                        "tracker_type": "bytetrack",
                        "track_high_thresh": 0.25,
                        "track_low_thresh": tlt,
                        "new_track_thresh": ntt,
                        "track_buffer": tb,
                        "match_thresh": mt,
                        "fuse_score": True,
                    })

    # Add specific exploratory boundary conditions:
    boundary_configs = [
        {"tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.15, "new_track_thresh": 0.20, "track_buffer": 15, "match_thresh": 0.65, "fuse_score": True},
        {"tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.10, "new_track_thresh": 0.20, "track_buffer": 15, "match_thresh": 0.65, "fuse_score": True},
        {"tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.15, "new_track_thresh": 0.25, "track_buffer": 20, "match_thresh": 0.70, "fuse_score": True},
        {"tracker_type": "bytetrack", "track_high_thresh": 0.25, "track_low_thresh": 0.15, "new_track_thresh": 0.20, "track_buffer": 20, "match_thresh": 0.70, "fuse_score": True},
    ]
    grid_configs.extend(boundary_configs)

    print(f"\nRunning Grid Search across {len(grid_configs)} ByteTrack hyperparameter combinations...", flush=True)

    all_results = [baseline_res]
    best_res = baseline_res
    t_start = time.time()

    for idx, cfg in enumerate(grid_configs, start=1):
        res = evaluate_tracker_config(VIDEOS, detections_by_video, gt_by_video, cfg)
        all_results.append(res)

        if res["composite_score"] > best_res["composite_score"]:
            best_res = res
            print(f"[{idx}/{len(grid_configs)}] *NEW BEST* -> IDF1: {res['idf1']:.4f}, MOTA: {res['mota']:.4f}, Score: {res['composite_score']:.4f} | cfg: ntt={cfg['new_track_thresh']}, tb={cfg['track_buffer']}, tlt={cfg['track_low_thresh']}, mt={cfg['match_thresh']}", flush=True)
        elif idx % 5 == 0 or idx == len(grid_configs):
            print(f"[{idx}/{len(grid_configs)}] Current: IDF1={res['idf1']:.4f}, MOTA={res['mota']:.4f} | Best: IDF1={best_res['idf1']:.4f}, MOTA={best_res['mota']:.4f} (Score={best_res['composite_score']:.4f})", flush=True)

    total_grid_time = time.time() - t_start
    print(f"\nGrid search completed in {total_grid_time:.1f}s ({total_grid_time/len(grid_configs):.2f}s per config)", flush=True)

    sorted_results = sorted(all_results, key=lambda x: x["composite_score"], reverse=True)
    best_cfg = best_res["config"]

    print("=" * 70, flush=True)
    print("BEST BYTETRACK CONFIGURATION:")
    print(json.dumps(best_cfg, indent=2))
    print(f"Metrics -> IDF1: {best_res['idf1']:.4f}, MOTA: {best_res['mota']:.4f}, MOTP: {best_res['motp']:.4f}, IDP: {best_res['idp']:.4f}, IDR: {best_res['idr']:.4f}, ID Switches: {best_res['num_switches']}")
    print(f"Improvement over baseline -> ΔIDF1: {best_res['idf1'] - baseline_res['idf1']:+.4f}, ΔMOTA: {best_res['mota'] - baseline_res['mota']:+.4f}, ΔIDSW: {best_res['num_switches'] - baseline_res['num_switches']}")
    print("=" * 70, flush=True)

    # 3. Save custom_bytetrack.yaml
    custom_yaml_path = BASE_DIR / "custom_bytetrack.yaml"
    yaml_content = f"""# ============================================================
# Optimized ByteTrack Configuration for Honeybee Multi-Object Tracking
# Generated via Phase 3 Grid Search on Honeybee Tracking Dataset
# Evaluated on {best_res['num_frames']} frames across multi-behavior videos
# Metrics: IDF1 = {best_res['idf1']:.4f}, MOTA = {best_res['mota']:.4f}, ID Switches = {best_res['num_switches']}
# ============================================================

tracker_type: bytetrack
track_high_thresh: {best_cfg['track_high_thresh']}
track_low_thresh: {best_cfg['track_low_thresh']}
new_track_thresh: {best_cfg['new_track_thresh']}
track_buffer: {best_cfg['track_buffer']}
match_thresh: {best_cfg['match_thresh']}
fuse_score: {best_cfg['fuse_score']}
"""
    with open(custom_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"Saved custom tracker configuration to {custom_yaml_path}", flush=True)

    # 4. Save results/tracking_benchmark.json
    benchmark_data = {
        "evaluation_dataset": {
            "videos": eval_video_keys,
            "total_frames_evaluated": best_res["num_frames"],
            "detector_weights": str(yolo_weights),
            "imgsz": 640,
        },
        "baseline_default_bytetrack": baseline_res,
        "best_custom_bytetrack": best_res,
        "delta": {
            "idf1": best_res["idf1"] - baseline_res["idf1"],
            "mota": best_res["mota"] - baseline_res["mota"],
            "id_switches": best_res["num_switches"] - baseline_res["num_switches"],
            "false_positives": best_res["num_false_positives"] - baseline_res["num_false_positives"],
            "misses": best_res["num_misses"] - baseline_res["num_misses"],
        },
        "top_10_configurations": sorted_results[:10],
        "parameter_ablation_summary": {
            "new_track_thresh_sensitivity": {
                str(ntt): float(np.mean([r["idf1"] for r in all_results if r["config"]["new_track_thresh"] == ntt]))
                for ntt in new_track_thresh_vals
            },
            "track_buffer_sensitivity": {
                str(tb): float(np.mean([r["idf1"] for r in all_results if r["config"]["track_buffer"] == tb]))
                for tb in track_buffer_vals
            },
            "track_low_thresh_sensitivity": {
                str(tlt): float(np.mean([r["idf1"] for r in all_results if r["config"]["track_low_thresh"] == tlt]))
                for tlt in track_low_thresh_vals
            },
            "match_thresh_sensitivity": {
                str(mt): float(np.mean([r["idf1"] for r in all_results if r["config"]["match_thresh"] == mt]))
                for mt in match_thresh_vals
            },
        },
    }

    out_json = RESULTS_DIR / "tracking_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Saved tracking benchmark to {out_json}", flush=True)

    # 5. Log to results/experiments.csv
    now_str = datetime.now().isoformat()
    rows = [
        {
            "timestamp": now_str,
            "run_id": f"ByteTrack_Baseline_{int(time.time())}",
            "experiment_name": "ByteTrack_Default_Benchmark",
            "model_type": "ByteTrack_Default",
            "seed": "42",
            "split_type": "multi_video_tracking_subset",
            "train_windows": "0",
            "train_tracks": "0",
            "tune_windows": "0",
            "tune_tracks": "0",
            "test_windows": str(baseline_res["num_frames"]),
            "test_tracks": str(baseline_res["num_switches"]),
            "macro_f1": f"{baseline_res['idf1']:.4f}",
            "weighted_f1": f"{baseline_res['mota']:.4f}",
            "accuracy": f"{baseline_res['motp']:.4f}",
            "per_class_f1_json": json.dumps({
                "IDF1": baseline_res["idf1"],
                "MOTA": baseline_res["mota"],
                "IDP": baseline_res["idp"],
                "IDR": baseline_res["idr"],
                "IDSW": baseline_res["num_switches"],
            }),
            "config_json": json.dumps(baseline_cfg),
        },
        {
            "timestamp": now_str,
            "run_id": f"ByteTrack_Optimized_{int(time.time()) + 1}",
            "experiment_name": "ByteTrack_GridSearch_Best",
            "model_type": "ByteTrack_Optimized",
            "seed": "42",
            "split_type": "multi_video_tracking_subset",
            "train_windows": "0",
            "train_tracks": "0",
            "tune_windows": "0",
            "tune_tracks": "0",
            "test_windows": str(best_res["num_frames"]),
            "test_tracks": str(best_res["num_switches"]),
            "macro_f1": f"{best_res['idf1']:.4f}",
            "weighted_f1": f"{best_res['mota']:.4f}",
            "accuracy": f"{best_res['motp']:.4f}",
            "per_class_f1_json": json.dumps({
                "IDF1": best_res["idf1"],
                "MOTA": best_res["mota"],
                "IDP": best_res["idp"],
                "IDR": best_res["idr"],
                "IDSW": best_res["num_switches"],
            }),
            "config_json": json.dumps(best_cfg),
        },
    ]

    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        for r in rows:
            writer.writerow(r)

    print("Logged tracking runs to experiments.csv", flush=True)


if __name__ == "__main__":
    main()
