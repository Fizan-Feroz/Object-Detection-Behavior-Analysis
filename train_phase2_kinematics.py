# ============================================================
# Phase 2: Systematic Evaluation of Gating Variants & Window Lengths (T in {6, 8, 16})
#
# Protocols Implemented:
#   1. 9 Configurations:
#      T in {16, 8, 6} x Gating in {"all_inside", "half_inside", "endpoint_inside"}
#   2. Blocked 5-Fold Cross-Validation Per Video
#      - Purges boundary-crossing tracks
#      - Pools out-of-fold predictions
#   3. Multi-seed Evaluation: 5 seeds (42, 43, 44, 45, 46)
#   4. Track-clustered Bootstrap 95% Confidence Intervals (B=1000)
#   5. Strict support reporting: Windows AND Distinct Tracks
#   6. Full experiment logging to results/experiments.csv
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BASE_DIR,
    CLASS_NAMES,
    EXPERIMENTS_CSV,
    RESULTS_DIR,
    TRACKING_DIR,
    VIDEOS,
    seed_everything,
)
from extract_dinov2_features import load_raw_tracks, parse_polygon
from kinematics import compute_causal_kinematics_offline


# ------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------

class KinematicsBeeLSTM(nn.Module):
    """
    BiLSTM Classifier over temporal sequence of 7-d causal kinematic vectors.
    Dynamically accepts any sequence length T.
    """
    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        lstm_out, (h_n, _) = self.lstm(x_norm)
        if self.bidirectional:
            pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            pooled = h_n[-1]
        return self.head(pooled)


# ------------------------------------------------------------
# Data Building for (T, Gating)
# ------------------------------------------------------------

def build_dataset_for_config(
    all_video_tracks: dict,
    all_video_polygons: dict,
    all_video_blocks: dict,
    T: int,
    gating_mode: str,
    k_subsample: int = 6,
) -> dict[str, np.ndarray]:
    """
    Assembles causal kinematics tensors, labels, track_ids, video_ids, block_ids, and is_test
    for a given (T, gating_mode) configuration across all 5 videos.
    """
    kin_list = []
    labels_list = []
    track_ids_list = []
    video_ids_list = []
    block_ids_list = []
    is_test_list = []

    vid_names = list(VIDEOS.keys())

    for v_idx, v_name in enumerate(vid_names):
        v_info = VIDEOS[v_name]
        vw, vh = v_info["w"], v_info["h"]
        poly = all_video_polygons[v_name]
        blocks = all_video_blocks[v_name]
        tracks = all_video_tracks[v_name]

        for tid, obs_list in tracks.items():
            fids = [o["frame_id"] for o in obs_list]

            # Block membership & boundary purge
            track_blocks = set()
            for f in fids:
                for b_i, (b_start, b_end) in enumerate(blocks):
                    if b_start <= f < b_end:
                        track_blocks.add(b_i)
                        break
            if len(track_blocks) != 1:
                continue  # Drop boundary crossing or gap tracks
            b_idx = list(track_blocks)[0]

            frame_map = {o["frame_id"]: o for o in obs_list}
            b_start, b_end = blocks[b_idx]
            block_fids = [f for f in fids if b_start <= f < b_end]
            if len(block_fids) < T:
                continue

            # Check if track contains any labeled behavior
            has_behavior = any(sum(o["classes"]) > 0 for o in obs_list)

            # 1. Non-overlapping test windows (phase 0, stride T)
            test_fids = [f for f in range(b_start, b_end, k_subsample) if f in frame_map]
            i = 0
            while i <= len(test_fids) - T:
                win_fids = test_fids[i:i + T]
                if any(win_fids[s] - win_fids[s-1] != k_subsample for s in range(1, T)):
                    i += 1
                    continue

                win_obs = [frame_map[f] for f in win_fids]
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
                        kin_mat = np.array([o["kinematics"] for o in win_obs], dtype=np.float32)
                        kin_list.append(kin_mat)
                        labels_list.append(target_class)
                        track_ids_list.append(v_idx * 100000 + tid)
                        video_ids_list.append(v_idx)
                        block_ids_list.append(b_idx)
                        is_test_list.append(True)
                i += T  # non-overlapping test

            # 2. Dense training windows (phases 0..K-1, stride 1 for behavior, stride T for other)
            min_f, max_f = block_fids[0], block_fids[-1]
            phases = range(k_subsample) if has_behavior else [0]

            for phase in phases:
                sampled = [f for f in range(min_f + phase, max_f + 1, k_subsample) if f in frame_map]
                step_size = 1 if has_behavior else T
                for j in range(0, len(sampled) - T + 1, step_size):
                    win_fids = sampled[j:j + T]
                    if any(win_fids[s] - win_fids[s-1] != k_subsample for s in range(1, T)):
                        continue

                    win_obs = [frame_map[f] for f in win_fids]
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
                            kin_mat = np.array([o["kinematics"] for o in win_obs], dtype=np.float32)
                            kin_list.append(kin_mat)
                            labels_list.append(target_class)
                            track_ids_list.append(v_idx * 100000 + tid)
                            video_ids_list.append(v_idx)
                            block_ids_list.append(b_idx)
                            is_test_list.append(False)

    return {
        "kinematics": np.array(kin_list, dtype=np.float32),
        "labels": np.array(labels_list, dtype=np.int64),
        "track_ids": np.array(track_ids_list, dtype=np.int64),
        "video_ids": np.array(video_ids_list, dtype=np.int64),
        "block_ids": np.array(block_ids_list, dtype=np.int64),
        "is_test": np.array(is_test_list, dtype=bool),
    }


# ------------------------------------------------------------
# Helper: Clustered Bootstrap CIs
# ------------------------------------------------------------

def track_clustered_bootstrap_ci(
    targets: np.ndarray,
    preds: np.ndarray,
    track_ids: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    unique_tracks = np.unique(track_ids)
    n_tracks = len(unique_tracks)
    if n_tracks == 0 or len(targets) == 0:
        return {"macro_f1": (0.0, 0.0), "weighted_f1": (0.0, 0.0), "accuracy": (0.0, 0.0)}

    track_to_indices = defaultdict(list)
    for idx, tid in enumerate(track_ids):
        track_to_indices[tid].append(idx)

    boot_macro_f1s = []
    boot_weighted_f1s = []
    boot_accs = []

    for _ in range(n_bootstraps):
        sampled_tracks = rng.choice(unique_tracks, size=n_tracks, replace=True)
        sampled_indices = []
        for tid in sampled_tracks:
            sampled_indices.extend(track_to_indices[tid])

        b_targets = targets[sampled_indices]
        b_preds = preds[sampled_indices]
        boot_macro_f1s.append(f1_score(b_targets, b_preds, average="macro", zero_division=0))
        boot_weighted_f1s.append(f1_score(b_targets, b_preds, average="weighted", zero_division=0))
        boot_accs.append(float(np.mean(b_targets == b_preds)))

    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0
    return {
        "macro_f1": (float(np.percentile(boot_macro_f1s, lower_p)), float(np.percentile(boot_macro_f1s, upper_p))),
        "weighted_f1": (float(np.percentile(boot_weighted_f1s, lower_p)), float(np.percentile(boot_weighted_f1s, upper_p))),
        "accuracy": (float(np.percentile(boot_accs, lower_p)), float(np.percentile(boot_accs, upper_p))),
    }


def compute_class_weights(y: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / np.sum(weights) * float(num_classes)
    return torch.tensor(weights, dtype=torch.float32)


def log_to_experiments_csv(
    experiment_name: str,
    model_type: str,
    seed: int,
    split_type: str,
    train_windows: int,
    train_tracks: int,
    tune_windows: int,
    tune_tracks: int,
    test_windows: int,
    test_tracks: int,
    macro_f1: float,
    weighted_f1: float,
    accuracy: float,
    per_class_f1: dict[str, float],
    config_dict: dict,
) -> None:
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(),
        "run_id": f"{experiment_name}_s{seed}_{int(time.time())}",
        "experiment_name": experiment_name,
        "model_type": model_type,
        "seed": seed,
        "split_type": split_type,
        "train_windows": train_windows,
        "train_tracks": train_tracks,
        "tune_windows": tune_windows,
        "tune_tracks": tune_tracks,
        "test_windows": test_windows,
        "test_tracks": test_tracks,
        "macro_f1": f"{macro_f1:.4f}",
        "weighted_f1": f"{weighted_f1:.4f}",
        "accuracy": f"{accuracy:.4f}",
        "per_class_f1_json": json.dumps(per_class_f1),
        "config_json": json.dumps(config_dict),
    }
    file_exists = EXPERIMENTS_CSV.exists() and EXPERIMENTS_CSV.stat().st_size > 0
    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------------
# Blocked 5-Fold Evaluator for a Configuration
# ------------------------------------------------------------

def evaluate_config(
    data: dict[str, np.ndarray],
    device: torch.device,
    T: int,
    gating: str,
    seeds: list[int] = [42, 43, 44, 45, 46],
    batch_size: int = 128,
) -> dict:
    is_test_mask = data["is_test"]
    test_indices = np.where(is_test_mask)[0]
    train_indices = np.where(~is_test_mask)[0]

    if len(test_indices) == 0 or len(train_indices) == 0:
        return {"error": "no data"}

    oof_targets = data["labels"][test_indices]
    oof_tracks = data["track_ids"][test_indices]
    oof_blocks = data["block_ids"][test_indices]

    seed_results = []

    for seed in seeds:
        seed_everything(seed)
        oof_probs = np.zeros((len(test_indices), 5), dtype=np.float32)

        for fold_k in range(5):
            fold_test_sub = np.where(oof_blocks == fold_k)[0]
            if len(fold_test_sub) == 0:
                continue
            fold_test_global = test_indices[fold_test_sub]

            # Train pool: all blocks != fold_k
            fold_tr_mask = (data["block_ids"] != fold_k)
            fold_tr_global = np.where(~data["is_test"] & fold_tr_mask)[0]

            # Partition into Train (80%) and Tune (20%) grouped by track
            tr_tids = data["track_ids"][fold_tr_global]
            unique_tr_tids = np.unique(tr_tids)
            rng = np.random.default_rng(seed + fold_k)
            shuffled_tids = set(rng.choice(unique_tr_tids, size=max(1, int(len(unique_tr_tids) * 0.20)), replace=False))

            tune_sub = np.array([tid in shuffled_tids for tid in tr_tids])
            train_sub = ~tune_sub

            idx_train = fold_tr_global[train_sub]
            idx_tune = fold_tr_global[tune_sub]

            weights = compute_class_weights(data["labels"][idx_train], num_classes=5).to(device)

            model = KinematicsBeeLSTM(input_dim=7, hidden_dim=64).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
            criterion = nn.CrossEntropyLoss(weight=weights)

            ds_train = TensorDataset(torch.tensor(data["kinematics"][idx_train]), torch.tensor(data["labels"][idx_train], dtype=torch.long))
            ds_tune = TensorDataset(torch.tensor(data["kinematics"][idx_tune]), torch.tensor(data["labels"][idx_tune], dtype=torch.long))
            ds_test = TensorDataset(torch.tensor(data["kinematics"][fold_test_global]), torch.tensor(data["labels"][fold_test_global], dtype=torch.long))

            tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            tu_loader = DataLoader(ds_tune, batch_size=batch_size, shuffle=False)
            te_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

            best_macro = -1.0
            best_state = None
            patience_cnt = 0

            for ep in range(1, 21):
                model.train()
                for x_b, y_b in tr_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    optimizer.zero_grad()
                    logits = model(x_b)
                    loss = criterion(logits, y_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

                model.eval()
                t_preds, t_targets = [], []
                with torch.no_grad():
                    for x_b, y_b in tu_loader:
                        logits = model(x_b.to(device))
                        preds = torch.argmax(logits, dim=-1).cpu().numpy()
                        t_preds.extend(preds)
                        t_targets.extend(y_b.numpy())
                tu_macro = float(f1_score(t_targets, t_preds, average="macro", zero_division=0))
                if tu_macro > best_macro:
                    best_macro = tu_macro
                    patience_cnt = 0
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_cnt += 1
                if patience_cnt >= 6:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            fold_probs = []
            with torch.no_grad():
                for x_b, _ in te_loader:
                    logits = model(x_b.to(device))
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    fold_probs.append(probs)
            oof_probs[fold_test_sub] = np.concatenate(fold_probs, axis=0)

        # Pooled metrics for this seed
        oof_preds = np.argmax(oof_probs, axis=-1)
        macro_f1 = float(f1_score(oof_targets, oof_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(oof_targets, oof_preds, average="weighted", zero_division=0))
        acc = float(np.mean(oof_preds == oof_targets))
        cis = track_clustered_bootstrap_ci(oof_targets, oof_preds, oof_tracks, seed=seed)

        rep = classification_report(
            oof_targets, oof_preds, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0
        )
        per_class_f1 = {c: rep[c]["f1-score"] for c in CLASS_NAMES}

        log_to_experiments_csv(
            experiment_name=f"Phase2_Kinematics_T{T}_{gating}",
            model_type=f"Kinematics_T{T}_{gating}",
            seed=seed,
            split_type="blocked_5fold_pooled_oof",
            train_windows=len(train_indices),
            train_tracks=len(np.unique(data["track_ids"][train_indices])),
            tune_windows=0,
            tune_tracks=0,
            test_windows=len(test_indices),
            test_tracks=len(np.unique(oof_tracks)),
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            accuracy=acc,
            per_class_f1=per_class_f1,
            config_dict={"T": T, "gating": gating, "bootstrap_ci_95": cis},
        )

        seed_results.append({
            "seed": seed,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": acc,
            "bootstrap_ci_95": cis,
            "per_class_f1": per_class_f1,
        })

    # Summary across seeds
    macros = [r["macro_f1"] for r in seed_results]
    wf1s = [r["weighted_f1"] for r in seed_results]
    accs = [r["accuracy"] for r in seed_results]

    mean_macro, sd_macro = float(np.mean(macros)), float(np.std(macros))
    mean_wf1, sd_wf1 = float(np.mean(wf1s)), float(np.std(wf1s))
    mean_acc, sd_acc = float(np.mean(accs)), float(np.std(accs))

    per_class_agg = {}
    support_counts = {}
    for c_i, c_name in enumerate(CLASS_NAMES):
        c_f1s = [r["per_class_f1"][c_name] for r in seed_results]
        per_class_agg[c_name] = {"mean": float(np.mean(c_f1s)), "sd": float(np.std(c_f1s))}
        mask_c = (oof_targets == c_i)
        win_c = int(np.sum(mask_c))
        trk_c = len(np.unique(oof_tracks[mask_c])) if win_c > 0 else 0
        support_counts[c_name] = {"windows": win_c, "tracks": trk_c}

    # Extract seed 42 CI for reference
    s42_ci = seed_results[0]["bootstrap_ci_95"]["macro_f1"]

    return {
        "T": T,
        "gating": gating,
        "mean_macro_f1": mean_macro,
        "sd_macro_f1": sd_macro,
        "mean_weighted_f1": mean_wf1,
        "sd_weighted_f1": sd_wf1,
        "mean_accuracy": mean_acc,
        "sd_accuracy": sd_acc,
        "per_class_f1": per_class_agg,
        "support": support_counts,
        "seed42_macro_ci": s42_ci,
        "seed_results": seed_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Gating & Window Lengths Evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print("PHASE 2: GATING & WINDOW LENGTH ABLATION STUDY")
    print(f"Testing T in {{16, 8, 6}} x Gating in {{all_inside, half_inside, endpoint_inside}}")
    print(f"Seeds: {args.seeds} | Device: {device} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # 1. Pre-load raw tracks and enriched causal kinematics once
    print("Loading and enriching causal kinematics for all videos...")
    all_video_tracks = {}
    all_video_polygons = {}
    all_video_blocks = {}

    for v_name, v_info in VIDEOS.items():
        tr_file = TRACKING_DIR / v_info["ann"]
        poly_file = TRACKING_DIR / v_info["zone"]
        vw, vh = v_info["w"], v_info["h"]

        tracks = load_raw_tracks(tr_file)
        polygon = parse_polygon(poly_file)

        # Enriched causal kinematics
        enriched = {}
        for tid, obs_list in tracks.items():
            enriched[tid] = compute_causal_kinematics_offline(
                obs_list, polygon=polygon, video_w=vw, video_h=vh, fps=50.0, alpha=0.6, max_gap_frames=5
            )

        mp4_path = TRACKING_DIR / v_info["video"]
        cap = cv2.VideoCapture(str(mp4_path))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

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

        all_video_tracks[v_name] = enriched
        all_video_polygons[v_name] = polygon
        all_video_blocks[v_name] = blocks

    print("All videos enriched with causal kinematics.")

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

    phase2_results = {}

    print("\n" + "=" * 110)
    print(f"{'Config (T, Gating)':<24} | {'Macro-F1 (Mean +/- SD)':<24} | {'95% CI (Seed 42)':<22} | {'Accuracy':<12} | {'Forage F1':<12} | {'Forage Support'}")
    print("=" * 110)

    for T, gating in configs:
        cfg_key = f"T{T}_{gating}"
        data = build_dataset_for_config(
            all_video_tracks, all_video_polygons, all_video_blocks, T=T, gating_mode=gating, k_subsample=6
        )
        res = evaluate_config(data, device, T=T, gating=gating, seeds=args.seeds, batch_size=args.batch_size)
        phase2_results[cfg_key] = res

        f_sup = res["support"]["Foraging"]
        f_f1 = res["per_class_f1"]["Foraging"]["mean"]
        ci_str = f"[{res['seed42_macro_ci'][0]:.4f}, {res['seed42_macro_ci'][1]:.4f}]"

        print(f"{cfg_key:<24} | {res['mean_macro_f1']:.4f} +/- {res['sd_macro_f1']:.4f}       | {ci_str:<22} | {res['mean_accuracy']:.4f}     | {f_f1:.4f}     | {f_sup['windows']} win ({f_sup['tracks']} trk)")

    # Save summary JSON
    summary_path = RESULTS_DIR / "phase_2_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(phase2_results, f, indent=2)
    print(f"\nPhase 2 ablation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
