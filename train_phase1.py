# ============================================================
# Phase 1: Trustworthy Evaluation Suite
#
# Protocols Implemented:
#   1. Blocked 5-Fold Cross-Validation Per Video:
#      - 5 contiguous blocks per video with >= 250-frame gaps.
#      - Boundary-crossing tracks dropped to ensure block independence.
#      - Out-of-fold predictions pooled to give real support for Defense & Washboarding.
#   2. Leave-One-Video-Out (LOVO):
#      - Fanning: held-out across 711a, 711b, yt5.
#      - Foraging: held-out across 609b, 711a, 711b.
#      - Defense & Washboarding marked as single-video.
#   3. Shortcut Diagnostics:
#      (a) Position + box size classifier (16x4 coords)
#      (b) Background patch DINOv2 classifier (16x384 bg embeddings)
#      (c) Linear probe predicting video ID from visual embeddings (384 -> 5)
#   4. Model Comparison on Identical Folds:
#      - Kinematics-Only (7-d)
#      - Visual-Only (384-d)
#      - Fused with Linear(384 -> 64) projection before concatenation (64 + 7 = 71-d)
#   5. Pre/Post Comparison:
#      - Pre = Kinematics-Only Control Baseline.
#      - Post = Fused Model (with Linear 384->64 projection).
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BASE_DIR,
    CLASS_NAMES,
    DINOV2_FEATURES_DIR,
    EXPERIMENTS_CSV,
    MODELS_DIR,
    RESULTS_DIR,
    VIDEOS,
    seed_everything,
)


DATA_DIR = DINOV2_FEATURES_DIR / "phase1_data"
VID_NAMES = list(VIDEOS.keys())


# ------------------------------------------------------------
# Model Architectures
# ------------------------------------------------------------

class ProjectedFusedBeeLSTM(nn.Module):
    """
    Fused Model with Linear(384 -> 64) projection before concatenation.
    Balancing: 64 projected visual dims + 7 kinematic dims = 71 fused dims.
    """
    def __init__(
        self,
        vis_dim: int = 384,
        kin_dim: int = 7,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.kin_norm = nn.LayerNorm(kin_dim)
        fused_dim = proj_dim + kin_dim  # 64 + 7 = 71

        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_hidden = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_vis: torch.Tensor, x_kin: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_vis = x_vis + torch.randn_like(x_vis) * 0.02
        z_vis = self.vis_proj(x_vis)
        z_kin = self.kin_norm(x_kin)
        x_fused = torch.cat([z_vis, z_kin], dim=-1)

        lstm_out, (h_n, c_n) = self.lstm(x_fused)
        if self.bidirectional:
            final_step = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final_step = h_n[-1]
        return self.head(final_step)


class SingleStreamBeeLSTM(nn.Module):
    """
    Generic 2-layer BiLSTM for single-stream inputs:
    - Visual-Only (384-d)
    - Kinematics-Only (7-d)
    - Position-Only (4-d coords)
    - Background-Only (384-d bg patches)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_hidden = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        if self.training and x.shape[-1] >= 64:
            x = x + torch.randn_like(x) * 0.02
        x = self.input_dropout(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            final_step = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final_step = h_n[-1]
        return self.head(final_step)


class VideoIDLinearProbe(nn.Module):
    """
    Linear Probe predicting Video ID (0..4) from 384-d DINOv2 embedding.
    """
    def __init__(self, input_dim: int = 384, num_videos: int = 5):
        super().__init__()
        self.probe = nn.Linear(input_dim, num_videos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Sequence, 384) -> mean pool over sequence
        pooled = torch.mean(x, dim=1)
        return self.probe(pooled)


# ------------------------------------------------------------
# Training & Evaluation Helpers
# ------------------------------------------------------------

def compute_class_weights(y: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / np.sum(weights) * float(num_classes)
    return torch.tensor(weights, dtype=torch.float32)


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
        return {
            "macro_f1": (0.0, 0.0),
            "weighted_f1": (0.0, 0.0),
            "accuracy": (0.0, 0.0),
        }

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


def train_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    tune_loader: DataLoader,
    device: torch.device,
    weights: torch.Tensor,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 8,
    is_fused: bool = False,
) -> tuple[nn.Module, int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_macro = -1.0
    best_ep = 0
    best_state = None
    patience_cnt = 0

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if is_fused:
                x_vis, x_kin, y_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                logits = model(x_vis, x_kin)
            else:
                x_b, y_b = batch[0].to(device), batch[1].to(device)
                logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Evaluate on Tune
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for batch in tune_loader:
                if is_fused:
                    x_vis, x_kin, y_b = batch[0].to(device), batch[1].to(device), batch[2]
                    logits = model(x_vis, x_kin)
                else:
                    x_b, y_b = batch[0].to(device), batch[1]
                    logits = model(x_b)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                t_preds.extend(preds)
                t_targets.extend(y_b.numpy())

        tune_macro = float(f1_score(t_targets, t_preds, average="macro", zero_division=0))
        if tune_macro > best_macro:
            best_macro = tune_macro
            best_ep = ep
            patience_cnt = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best_macro


def predict_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_fused: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            if is_fused:
                x_vis, x_kin, y_b = batch[0].to(device), batch[1].to(device), batch[2]
                logits = model(x_vis, x_kin)
            else:
                x_b, y_b = batch[0].to(device), batch[1]
                logits = model(x_b)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_targets.extend(y_b.numpy())
    if all_probs:
        return np.concatenate(all_probs, axis=0), np.array(all_targets)
    return np.empty((0, 5)), np.empty((0,), dtype=int)


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
    fieldnames = list(row.keys())

    with open(EXPERIMENTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ------------------------------------------------------------
# Protocol 1: Blocked 5-Fold Cross Validation Per Video
# ------------------------------------------------------------

def run_blocked_kfold(
    data: dict[str, np.ndarray],
    device: torch.device,
    model_mode: str = "fused_proj",  # "fused_proj", "visual", "kinematics"
    seeds: list[int] = [42, 43, 44, 45, 46],
    batch_size: int = 64,
) -> dict:
    """
    Executes Blocked 5-Fold per video. Pools out-of-fold predictions.
    Gives Defense and Washboarding real support.
    """
    print(f"\n{'='*75}")
    print(f"PROTOCOL 1: BLOCKED 5-FOLD CROSS VALIDATION ({model_mode.upper()})")
    print(f"Seeds: {seeds} | Batch Size: {batch_size}")
    print(f"{'='*75}")

    is_test_mask = data["is_test"]
    test_indices = np.where(is_test_mask)[0]
    train_indices = np.where(~is_test_mask)[0]

    seed_results = []

    for seed in seeds:
        seed_everything(seed)
        oof_probs = np.zeros((len(test_indices), 5), dtype=np.float32)
        oof_targets = data["labels"][test_indices]
        oof_tracks = data["track_ids"][test_indices]
        oof_blocks = data["block_ids"][test_indices]

        # 5 contiguous folds: fold_k is held-out block across all videos
        for fold_k in range(5):
            fold_test_sub = np.where(oof_blocks == fold_k)[0]
            if len(fold_test_sub) == 0:
                continue
            fold_test_global = test_indices[fold_test_sub]

            # Train pool: all other blocks (j != fold_k) across all videos
            fold_tr_mask = (data["block_ids"] != fold_k)
            fold_tr_global = np.where(~data["is_test"] & fold_tr_mask)[0]

            # Partition train pool into Train (80%) and Tune (20%) grouped strictly by track
            tr_tids = data["track_ids"][fold_tr_global]
            unique_tr_tids = np.unique(tr_tids)
            rng = np.random.default_rng(seed + fold_k)
            shuffled_tids = set(rng.choice(unique_tr_tids, size=max(1, int(len(unique_tr_tids)*0.20)), replace=False))

            tune_sub = np.array([tid in shuffled_tids for tid in tr_tids])
            train_sub = ~tune_sub

            idx_train = fold_tr_global[train_sub]
            idx_tune = fold_tr_global[tune_sub]

            weights = compute_class_weights(data["labels"][idx_train], num_classes=5).to(device)

            # Initialize model
            is_fused = (model_mode == "fused_proj")
            if model_mode == "fused_proj":
                model = ProjectedFusedBeeLSTM().to(device)
                ds_train = TensorDataset(
                    torch.tensor(data["visual"][idx_train]),
                    torch.tensor(data["kinematics"][idx_train]),
                    torch.tensor(data["labels"][idx_train], dtype=torch.long),
                )
                ds_tune = TensorDataset(
                    torch.tensor(data["visual"][idx_tune]),
                    torch.tensor(data["kinematics"][idx_tune]),
                    torch.tensor(data["labels"][idx_tune], dtype=torch.long),
                )
                ds_test = TensorDataset(
                    torch.tensor(data["visual"][fold_test_global]),
                    torch.tensor(data["kinematics"][fold_test_global]),
                    torch.tensor(data["labels"][fold_test_global], dtype=torch.long),
                )
            elif model_mode == "visual":
                model = SingleStreamBeeLSTM(input_dim=384).to(device)
                ds_train = TensorDataset(torch.tensor(data["visual"][idx_train]), torch.tensor(data["labels"][idx_train], dtype=torch.long))
                ds_tune = TensorDataset(torch.tensor(data["visual"][idx_tune]), torch.tensor(data["labels"][idx_tune], dtype=torch.long))
                ds_test = TensorDataset(torch.tensor(data["visual"][fold_test_global]), torch.tensor(data["labels"][fold_test_global], dtype=torch.long))
            elif model_mode == "kinematics":
                model = SingleStreamBeeLSTM(input_dim=7).to(device)
                ds_train = TensorDataset(torch.tensor(data["kinematics"][idx_train]), torch.tensor(data["labels"][idx_train], dtype=torch.long))
                ds_tune = TensorDataset(torch.tensor(data["kinematics"][idx_tune]), torch.tensor(data["labels"][idx_tune], dtype=torch.long))
                ds_test = TensorDataset(torch.tensor(data["kinematics"][fold_test_global]), torch.tensor(data["labels"][fold_test_global], dtype=torch.long))

            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            tune_loader = DataLoader(ds_tune, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

            trained_model, best_ep, _ = train_single_model(
                model, train_loader, tune_loader, device, weights,
                epochs=20, lr=1e-3, patience=6, is_fused=is_fused,
            )

            # Predict out-of-fold on block_k
            probs, _ = predict_model(trained_model, test_loader, device, is_fused=is_fused)
            oof_probs[fold_test_sub] = probs

        # Compute pooled out-of-fold metrics for this seed
        oof_preds = np.argmax(oof_probs, axis=-1)
        macro_f1 = float(f1_score(oof_targets, oof_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(oof_targets, oof_preds, average="weighted", zero_division=0))
        acc = float(np.mean(oof_preds == oof_targets))

        cis = track_clustered_bootstrap_ci(oof_targets, oof_preds, oof_tracks, seed=seed)

        rep = classification_report(
            oof_targets, oof_preds, labels=list(range(5)),
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        per_class_f1 = {c: rep[c]["f1-score"] for c in CLASS_NAMES}

        print(f"  Seed {seed:2d} | Out-of-Fold Macro-F1: {macro_f1:.4f} (95% CI: [{cis['macro_f1'][0]:.4f}, {cis['macro_f1'][1]:.4f}]) | Acc: {acc:.4f} | W-F1: {weighted_f1:.4f}")

        # Log run
        log_to_experiments_csv(
            experiment_name=f"Phase1_Blocked5Fold_{model_mode}",
            model_type=model_mode,
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
            config_dict={"bootstrap_ci_95": cis, "oof_total": len(oof_targets)},
        )

        seed_results.append({
            "seed": seed,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": acc,
            "bootstrap_ci_95": cis,
            "per_class_f1": per_class_f1,
            "confusion_matrix": confusion_matrix(oof_targets, oof_preds, labels=list(range(5))).tolist(),
        })

    # Summary over seeds
    macros = [r["macro_f1"] for r in seed_results]
    weights_f1 = [r["weighted_f1"] for r in seed_results]
    accs = [r["accuracy"] for r in seed_results]

    mean_macro, sd_macro = float(np.mean(macros)), float(np.std(macros))
    mean_w, sd_w = float(np.mean(weights_f1)), float(np.std(weights_f1))
    mean_a, sd_a = float(np.mean(accs)), float(np.std(accs))

    per_class_agg = {}
    for c in CLASS_NAMES:
        cf1s = [r["per_class_f1"][c] for r in seed_results]
        per_class_agg[c] = {"mean": float(np.mean(cf1s)), "sd": float(np.std(cf1s))}

    # Support counts by class on pooled OOF test set
    support_counts = {}
    for c_i, c_name in enumerate(CLASS_NAMES):
        mask = (oof_targets == c_i)
        win_cnt = int(np.sum(mask))
        trk_cnt = len(np.unique(oof_tracks[mask])) if win_cnt > 0 else 0
        support_counts[c_name] = {"windows": win_cnt, "tracks": trk_cnt}

    print(f"\n  -> Pooled OOF Summary ({model_mode.upper()}):")
    print(f"     Macro-F1:    {mean_macro:.4f} +/- {sd_macro:.4f}")
    print(f"     Weighted-F1: {mean_w:.4f} +/- {sd_w:.4f}")
    print(f"     Accuracy:    {mean_a:.4f} +/- {sd_a:.4f}")
    print("     Per-Class F1 (mean +/- sd) & Real Support:")
    for c in CLASS_NAMES:
        sup = support_counts[c]
        print(f"       {c:<14}: {per_class_agg[c]['mean']:.4f} +/- {per_class_agg[c]['sd']:.4f} | Support: {sup['windows']} win ({sup['tracks']} trk)")

    return {
        "experiment": f"Blocked5Fold_{model_mode}",
        "model_mode": model_mode,
        "mean_macro_f1": mean_macro,
        "sd_macro_f1": sd_macro,
        "mean_weighted_f1": mean_w,
        "sd_weighted_f1": sd_w,
        "mean_accuracy": mean_a,
        "sd_accuracy": sd_a,
        "per_class_f1": per_class_agg,
        "support": support_counts,
        "seed_results": seed_results,
    }


# ------------------------------------------------------------
# Protocol 2: Leave-One-Video-Out (LOVO) Cross-Validation
# ------------------------------------------------------------

def run_lovo_evaluation(
    data: dict[str, np.ndarray],
    device: torch.device,
    target_behavior: str,  # "Fanning" or "Foraging"
    candidate_videos: list[str],
    seeds: list[int] = [42, 43, 44, 45, 46],
    batch_size: int = 64,
) -> dict:
    """
    Evaluates cross-video generalization for target behavior across multiple seeds.
    Trains on all other videos, reports on the held-out video.
    """
    target_cls_idx = CLASS_NAMES.index(target_behavior)
    print(f"\n{'='*75}")
    print(f"PROTOCOL 2: LEAVE-ONE-VIDEO-OUT (LOVO) FOR {target_behavior.upper()}")
    print(f"Target Class: {target_behavior} (idx={target_cls_idx}) | Videos: {candidate_videos} | Seeds: {seeds}")
    print(f"{'='*75}")

    lovo_results = {}

    for held_out_name in candidate_videos:
        held_out_idx = VID_NAMES.index(held_out_name)
        test_mask = data["is_test"] & (data["video_ids"] == held_out_idx)
        idx_test = np.where(test_mask)[0]

        train_mask = (~data["is_test"]) & (data["video_ids"] != held_out_idx)
        idx_train = np.where(train_mask)[0]

        if len(idx_test) == 0:
            print(f"  Held-out {held_out_name}: no test windows. Skipping.")
            continue

        seed_target_f1s = []
        seed_macros = []
        seed_accuracies = []
        target_win_support = int(np.sum(data["labels"][idx_test] == target_cls_idx))
        target_trk_support = len(np.unique(data["track_ids"][idx_test][data["labels"][idx_test] == target_cls_idx])) if target_win_support > 0 else 0

        present_classes = sorted(list(set(data["labels"][idx_test])))
        present_names = [CLASS_NAMES[c] for c in present_classes]

        for seed in seeds:
            seed_everything(seed)

            # Partition train into Train (80%) and Tune (20%) grouped by track
            tr_tids = data["track_ids"][idx_train]
            unique_tr_tids = np.unique(tr_tids)
            rng = np.random.default_rng(seed)
            shuffled_tids = set(rng.choice(unique_tr_tids, size=max(1, int(len(unique_tr_tids)*0.20)), replace=False))

            tune_sub = np.array([tid in shuffled_tids for tid in tr_tids])
            train_sub = ~tune_sub

            i_tr = idx_train[train_sub]
            i_tu = idx_train[tune_sub]

            weights = compute_class_weights(data["labels"][i_tr], num_classes=5).to(device)

            model = ProjectedFusedBeeLSTM().to(device)

            ds_train = TensorDataset(
                torch.tensor(data["visual"][i_tr]),
                torch.tensor(data["kinematics"][i_tr]),
                torch.tensor(data["labels"][i_tr], dtype=torch.long),
            )
            ds_tune = TensorDataset(
                torch.tensor(data["visual"][i_tu]),
                torch.tensor(data["kinematics"][i_tu]),
                torch.tensor(data["labels"][i_tu], dtype=torch.long),
            )
            ds_test = TensorDataset(
                torch.tensor(data["visual"][idx_test]),
                torch.tensor(data["kinematics"][idx_test]),
                torch.tensor(data["labels"][idx_test], dtype=torch.long),
            )

            tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            tu_loader = DataLoader(ds_tune, batch_size=batch_size, shuffle=False)
            te_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

            trained_model, best_ep, _ = train_single_model(
                model, tr_loader, tu_loader, device, weights,
                epochs=20, lr=1e-3, patience=6, is_fused=True,
            )

            probs, targets = predict_model(trained_model, te_loader, device, is_fused=True)
            preds = np.argmax(probs, axis=-1)

            rep = classification_report(
                targets, preds, labels=present_classes,
                target_names=present_names, output_dict=True, zero_division=0,
            )

            target_f1 = float(rep.get(target_behavior, {}).get("f1-score", 0.0))
            macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
            acc = float(np.mean(preds == targets))

            seed_target_f1s.append(target_f1)
            seed_macros.append(macro_f1)
            seed_accuracies.append(acc)

            log_to_experiments_csv(
                experiment_name=f"Phase1_LOVO_{target_behavior}_{held_out_name}",
                model_type="Fused_Proj_BiLSTM",
                seed=seed,
                split_type=f"lovo_held_out_{held_out_name}",
                train_windows=len(i_tr),
                train_tracks=len(np.unique(data["track_ids"][i_tr])),
                tune_windows=len(i_tu),
                tune_tracks=len(np.unique(data["track_ids"][i_tu])),
                test_windows=len(targets),
                test_tracks=len(np.unique(data["track_ids"][idx_test])),
                macro_f1=macro_f1,
                weighted_f1=float(f1_score(targets, preds, average="weighted", zero_division=0)),
                accuracy=acc,
                per_class_f1={c: rep.get(c, {}).get("f1-score", 0.0) for c in CLASS_NAMES},
                config_dict={"target_behavior": target_behavior, "held_out_video": held_out_name},
            )

        mean_tf1, sd_tf1 = float(np.mean(seed_target_f1s)), float(np.std(seed_target_f1s))
        mean_mf1, sd_mf1 = float(np.mean(seed_macros)), float(np.std(seed_macros))
        mean_acc, sd_acc = float(np.mean(seed_accuracies)), float(np.std(seed_accuracies))

        print(f"  Held-out {held_out_name:<16} | {target_behavior} F1: {mean_tf1:.4f} +/- {sd_tf1:.4f} (Support: {target_win_support} win / {target_trk_support} trk) | Macro-F1: {mean_mf1:.4f} +/- {sd_mf1:.4f} | Present: {present_names}")

        lovo_results[held_out_name] = {
            "target_behavior": target_behavior,
            "target_f1_mean": mean_tf1,
            "target_f1_sd": sd_tf1,
            "macro_f1_mean": mean_mf1,
            "macro_f1_sd": sd_mf1,
            "accuracy_mean": mean_acc,
            "accuracy_sd": sd_acc,
            "target_win_support": target_win_support,
            "target_trk_support": target_trk_support,
            "present_classes": present_names,
            "seed_target_f1s": seed_target_f1s,
        }

    return lovo_results


# ------------------------------------------------------------
# Protocol 3: Shortcut Diagnostics
# ------------------------------------------------------------

def run_shortcut_diagnostics(
    data: dict[str, np.ndarray],
    device: torch.device,
    seeds: list[int] = [42, 43, 44, 45, 46],
    batch_size: int = 64,
) -> dict:
    """
    Executes 3 diagnostic shortcut checks across multiple seeds:
    (a) Classifier on position + box size only (16x4 coords)
    (b) Classifier on background patch DINOv2 embeddings (16x384)
    (c) Linear probe predicting video ID from embeddings (384 -> 5)
    """
    print(f"\n{'='*75}")
    print("PROTOCOL 3: SHORTCUT DIAGNOSTICS")
    print(f"Testing (a) Position, (b) Background Patch, (c) Video ID Probe | Seeds: {seeds}")
    print(f"{'='*75}")

    is_test_mask = data["is_test"]
    idx_test = np.where(is_test_mask)[0]
    idx_train = np.where(~is_test_mask)[0]

    pos_macros, pos_accs = [], []
    pos_per_class = defaultdict(list)

    bg_macros, bg_accs = [], []
    bg_per_class = defaultdict(list)

    vid_accs, vid_macros = [], []

    for seed in seeds:
        seed_everything(seed)

        # Partition train into train (80%) and tune (20%) grouped by track
        tr_tids = data["track_ids"][idx_train]
        unique_tr_tids = np.unique(tr_tids)
        rng = np.random.default_rng(seed)
        shuffled_tids = set(rng.choice(unique_tr_tids, size=max(1, int(len(unique_tr_tids)*0.20)), replace=False))

        tune_sub = np.array([tid in shuffled_tids for tid in tr_tids])
        train_sub = ~tune_sub

        i_tr = idx_train[train_sub]
        i_tu = idx_train[tune_sub]

        weights = compute_class_weights(data["labels"][i_tr], num_classes=5).to(device)

        # (a) Position
        pos_model = SingleStreamBeeLSTM(input_dim=4, hidden_dim=64).to(device)
        ds_tr_pos = TensorDataset(torch.tensor(data["coords"][i_tr]), torch.tensor(data["labels"][i_tr], dtype=torch.long))
        ds_tu_pos = TensorDataset(torch.tensor(data["coords"][i_tu]), torch.tensor(data["labels"][i_tu], dtype=torch.long))
        ds_te_pos = TensorDataset(torch.tensor(data["coords"][idx_test]), torch.tensor(data["labels"][idx_test], dtype=torch.long))

        l_tr_pos = DataLoader(ds_tr_pos, batch_size=batch_size, shuffle=True)
        l_tu_pos = DataLoader(ds_tu_pos, batch_size=batch_size, shuffle=False)
        l_te_pos = DataLoader(ds_te_pos, batch_size=batch_size, shuffle=False)

        trained_pos, _, _ = train_single_model(pos_model, l_tr_pos, l_tu_pos, device, weights, epochs=20, lr=1e-3, patience=6)
        probs_pos, targets_pos = predict_model(trained_pos, l_te_pos, device)
        preds_pos = np.argmax(probs_pos, axis=-1)
        macro_pos = float(f1_score(targets_pos, preds_pos, average="macro", zero_division=0))
        acc_pos = float(np.mean(preds_pos == targets_pos))
        rep_pos = classification_report(targets_pos, preds_pos, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)

        pos_macros.append(macro_pos)
        pos_accs.append(acc_pos)
        for c in CLASS_NAMES:
            pos_per_class[c].append(rep_pos[c]["f1-score"])

        log_to_experiments_csv(
            experiment_name="Phase1_Shortcut_PositionOnly",
            model_type="PositionOnly_BiLSTM",
            seed=seed,
            split_type="shortcut_diagnostic_a",
            train_windows=len(i_tr),
            train_tracks=len(np.unique(data["track_ids"][i_tr])),
            tune_windows=len(i_tu),
            tune_tracks=len(np.unique(data["track_ids"][i_tu])),
            test_windows=len(targets_pos),
            test_tracks=len(np.unique(data["track_ids"][idx_test])),
            macro_f1=macro_pos,
            weighted_f1=float(f1_score(targets_pos, preds_pos, average="weighted", zero_division=0)),
            accuracy=acc_pos,
            per_class_f1={c: rep_pos[c]["f1-score"] for c in CLASS_NAMES},
            config_dict={"input_dim": 4},
        )

        # (b) Background
        bg_model = SingleStreamBeeLSTM(input_dim=384, hidden_dim=128).to(device)
        ds_tr_bg = TensorDataset(torch.tensor(data["bg_visual"][i_tr]), torch.tensor(data["labels"][i_tr], dtype=torch.long))
        ds_tu_bg = TensorDataset(torch.tensor(data["bg_visual"][i_tu]), torch.tensor(data["labels"][i_tu], dtype=torch.long))
        ds_te_bg = TensorDataset(torch.tensor(data["bg_visual"][idx_test]), torch.tensor(data["labels"][idx_test], dtype=torch.long))

        l_tr_bg = DataLoader(ds_tr_bg, batch_size=batch_size, shuffle=True)
        l_tu_bg = DataLoader(ds_tu_bg, batch_size=batch_size, shuffle=False)
        l_te_bg = DataLoader(ds_te_bg, batch_size=batch_size, shuffle=False)

        trained_bg, _, _ = train_single_model(bg_model, l_tr_bg, l_tu_bg, device, weights, epochs=20, lr=1e-3, patience=6)
        probs_bg, targets_bg = predict_model(trained_bg, l_te_bg, device)
        preds_bg = np.argmax(probs_bg, axis=-1)
        macro_bg = float(f1_score(targets_bg, preds_bg, average="macro", zero_division=0))
        acc_bg = float(np.mean(preds_bg == targets_bg))
        rep_bg = classification_report(targets_bg, preds_bg, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)

        bg_macros.append(macro_bg)
        bg_accs.append(acc_bg)
        for c in CLASS_NAMES:
            bg_per_class[c].append(rep_bg[c]["f1-score"])

        log_to_experiments_csv(
            experiment_name="Phase1_Shortcut_BackgroundOnly",
            model_type="BackgroundOnly_BiLSTM",
            seed=seed,
            split_type="shortcut_diagnostic_b",
            train_windows=len(i_tr),
            train_tracks=len(np.unique(data["track_ids"][i_tr])),
            tune_windows=len(i_tu),
            tune_tracks=len(np.unique(data["track_ids"][i_tu])),
            test_windows=len(targets_bg),
            test_tracks=len(np.unique(data["track_ids"][idx_test])),
            macro_f1=macro_bg,
            weighted_f1=float(f1_score(targets_bg, preds_bg, average="weighted", zero_division=0)),
            accuracy=acc_bg,
            per_class_f1={c: rep_bg[c]["f1-score"] for c in CLASS_NAMES},
            config_dict={"input_dim": 384},
        )

        # (c) Video ID Probe
        vid_probe = VideoIDLinearProbe(input_dim=384, num_videos=5).to(device)
        crit_vid = nn.CrossEntropyLoss()
        opt_vid = torch.optim.AdamW(vid_probe.parameters(), lr=1e-2, weight_decay=1e-4)

        ds_tr_vid = TensorDataset(torch.tensor(data["visual"][idx_train]), torch.tensor(data["video_ids"][idx_train], dtype=torch.long))
        ds_te_vid = TensorDataset(torch.tensor(data["visual"][idx_test]), torch.tensor(data["video_ids"][idx_test], dtype=torch.long))

        l_tr_vid = DataLoader(ds_tr_vid, batch_size=batch_size, shuffle=True)
        l_te_vid = DataLoader(ds_te_vid, batch_size=batch_size, shuffle=False)

        for _ in range(15):
            vid_probe.train()
            for x_b, y_b in l_tr_vid:
                x_b, y_b = x_b.to(device), y_b.to(device)
                opt_vid.zero_grad()
                logits = vid_probe(x_b)
                loss = crit_vid(logits, y_b)
                loss.backward()
                opt_vid.step()

        vid_probe.eval()
        v_preds, v_targets = [], []
        with torch.no_grad():
            for x_b, y_b in l_te_vid:
                logits = vid_probe(x_b.to(device))
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                v_preds.extend(preds)
                v_targets.extend(y_b.numpy())

        vp_arr = np.array(v_preds)
        vt_arr = np.array(v_targets)
        v_acc = float(np.mean(vp_arr == vt_arr))
        v_macro = float(f1_score(vt_arr, vp_arr, average="macro", zero_division=0))
        vid_accs.append(v_acc)
        vid_macros.append(v_macro)

        log_to_experiments_csv(
            experiment_name="Phase1_Shortcut_VideoIDProbe",
            model_type="VideoID_LinearProbe",
            seed=seed,
            split_type="shortcut_diagnostic_c",
            train_windows=len(idx_train),
            train_tracks=len(np.unique(data["track_ids"][idx_train])),
            tune_windows=0,
            tune_tracks=0,
            test_windows=len(vt_arr),
            test_tracks=len(np.unique(data["track_ids"][idx_test])),
            macro_f1=v_macro,
            weighted_f1=v_macro,
            accuracy=v_acc,
            per_class_f1={},
            config_dict={"task": "predict_video_id", "accuracy": v_acc},
        )

    diag_results = {
        "position_shortcut": {
            "macro_f1_mean": float(np.mean(pos_macros)),
            "macro_f1_sd": float(np.std(pos_macros)),
            "accuracy_mean": float(np.mean(pos_accs)),
            "accuracy_sd": float(np.std(pos_accs)),
            "per_class_f1": {c: {"mean": float(np.mean(pos_per_class[c])), "sd": float(np.std(pos_per_class[c]))} for c in CLASS_NAMES},
        },
        "background_shortcut": {
            "macro_f1_mean": float(np.mean(bg_macros)),
            "macro_f1_sd": float(np.std(bg_macros)),
            "accuracy_mean": float(np.mean(bg_accs)),
            "accuracy_sd": float(np.std(bg_accs)),
            "per_class_f1": {c: {"mean": float(np.mean(bg_per_class[c])), "sd": float(np.std(bg_per_class[c]))} for c in CLASS_NAMES},
        },
        "video_id_probe": {
            "accuracy_mean": float(np.mean(vid_accs)),
            "accuracy_sd": float(np.std(vid_accs)),
            "macro_f1_mean": float(np.mean(vid_macros)),
            "macro_f1_sd": float(np.std(vid_macros)),
        },
    }

    print("\n--- Diagnostic Results (Mean +/- SD over seeds) ---")
    print(f"  (a) Position-Only Classifier:      Macro-F1 = {diag_results['position_shortcut']['macro_f1_mean']:.4f} +/- {diag_results['position_shortcut']['macro_f1_sd']:.4f} | Acc = {diag_results['position_shortcut']['accuracy_mean']:.4f}")
    print(f"  (b) Background-Only (No Bee):      Macro-F1 = {diag_results['background_shortcut']['macro_f1_mean']:.4f} +/- {diag_results['background_shortcut']['macro_f1_sd']:.4f} | Acc = {diag_results['background_shortcut']['accuracy_mean']:.4f}")
    print(f"  (c) Video ID Linear Probe:         Probe Accuracy = {diag_results['video_id_probe']['accuracy_mean']*100:.1f}% +/- {diag_results['video_id_probe']['accuracy_sd']*100:.1f}%")

    return diag_results


# ------------------------------------------------------------
# Main Launcher
# ------------------------------------------------------------

def main():
    import sys

    parser = argparse.ArgumentParser(description="Phase 1 Trustworthy Evaluation Suite")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Check phase 1 dataset
    req_tensors = [
        "visual.npy",
        "bg_visual.npy",
        "kinematics.npy",
        "coords.npy",
        "labels.npy",
        "track_ids.npy",
        "video_ids.npy",
        "block_ids.npy",
        "is_test.npy",
    ]
    missing = [f for f in req_tensors if not (DATA_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing Phase 1 data tensors in {DATA_DIR}: {missing}")
        print("Please run extract_phase1_data.py first.")
        return

    print("Loading Phase 1 dataset arrays from disk...")
    data = {k.replace(".npy", ""): np.load(DATA_DIR / k) for k in req_tensors}
    print(f"Loaded: Total {len(data['labels'])} windows ({len(np.unique(data['track_ids']))} distinct tracks across {len(VID_NAMES)} videos).")

    phase1_summary = {}

    # 1. Blocked 5-Fold Evaluation for all 3 models (Item 1 & Item 4)
    res_fused = run_blocked_kfold(data, device, model_mode="fused_proj", seeds=args.seeds, batch_size=args.batch_size)
    res_vis = run_blocked_kfold(data, device, model_mode="visual", seeds=args.seeds, batch_size=args.batch_size)
    res_kin = run_blocked_kfold(data, device, model_mode="kinematics", seeds=args.seeds, batch_size=args.batch_size)

    phase1_summary["blocked_5fold"] = {
        "fused_proj": res_fused,
        "visual_only": res_vis,
        "kinematics_control": res_kin,
    }

    # 2. Leave-One-Video-Out (LOVO) (Item 2)
    # Fanning videos: 711a, 711b, yt5
    fanning_videos = ["20230711a-fan", "20230711b-fan", "yt5"]
    res_lovo_fanning = run_lovo_evaluation(data, device, target_behavior="Fanning", candidate_videos=fanning_videos, seeds=args.seeds, batch_size=args.batch_size)

    # Foraging videos: 609b, 711a, 711b
    foraging_videos = ["20230609b-def", "20230711a-fan", "20230711b-fan"]
    res_lovo_foraging = run_lovo_evaluation(data, device, target_behavior="Foraging", candidate_videos=foraging_videos, seeds=args.seeds, batch_size=args.batch_size)

    phase1_summary["lovo"] = {
        "fanning": res_lovo_fanning,
        "foraging": res_lovo_foraging,
        "single_video_classes": ["Defense (20230609b-def only)", "Washboarding (yt8 only)"],
    }

    # 3. Shortcut Diagnostics (Item 3)
    diag_results = run_shortcut_diagnostics(data, device, seeds=args.seeds, batch_size=args.batch_size)
    phase1_summary["shortcut_diagnostics"] = diag_results

    # Summary Table
    print("\n" + "=" * 80)
    print("PHASE 1 TRUSTWORTHY EVALUATION: FINAL SUMMARY (Mean +/- SD over 5 Seeds)")
    print("=" * 80)
    print(f"{'Model / Experiment':<32} | {'OOF Macro-F1':<18} | {'OOF Weighted-F1':<18} | {'OOF Accuracy':<14}")
    print("-" * 80)
    print(f"{'Fused (Proj 384->64 + Kin 7)':<32} | {res_fused['mean_macro_f1']:.4f} +/- {res_fused['sd_macro_f1']:.4f} | {res_fused['mean_weighted_f1']:.4f} +/- {res_fused['sd_weighted_f1']:.4f} | {res_fused['mean_accuracy']:.4f} +/- {res_fused['sd_accuracy']:.4f}")
    print(f"{'Visual-Only (DINOv2 384)':<32} | {res_vis['mean_macro_f1']:.4f} +/- {res_vis['sd_macro_f1']:.4f} | {res_vis['mean_weighted_f1']:.4f} +/- {res_vis['sd_weighted_f1']:.4f} | {res_vis['mean_accuracy']:.4f} +/- {res_vis['sd_accuracy']:.4f}")
    print(f"{'Kinematics Control (Pre)':<32} | {res_kin['mean_macro_f1']:.4f} +/- {res_kin['sd_macro_f1']:.4f} | {res_kin['mean_weighted_f1']:.4f} +/- {res_kin['sd_weighted_f1']:.4f} | {res_kin['mean_accuracy']:.4f} +/- {res_kin['sd_accuracy']:.4f}")
    print("-" * 80)
    print("SHORTCUT DIAGNOSTICS:")
    print(f"  (a) Position-Only Classifier:      Macro-F1 = {diag_results['position_shortcut']['macro_f1_mean']:.4f} +/- {diag_results['position_shortcut']['macro_f1_sd']:.4f} | Accuracy = {diag_results['position_shortcut']['accuracy_mean']:.4f}")
    print(f"  (b) Background-Only (No Bee):      Macro-F1 = {diag_results['background_shortcut']['macro_f1_mean']:.4f} +/- {diag_results['background_shortcut']['macro_f1_sd']:.4f} | Accuracy = {diag_results['background_shortcut']['accuracy_mean']:.4f}")
    print(f"  (c) Video ID Linear Probe:         Probe Accuracy = {diag_results['video_id_probe']['accuracy_mean']*100:.1f}% +/- {diag_results['video_id_probe']['accuracy_sd']*100:.1f}%")
    print("=" * 80)

    # Save summary JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "phase_1_benchmark_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(phase1_summary, f, indent=2)
    print(f"\nPhase 1 benchmark summary saved to: {summary_file}")
    sys.exit(0)


if __name__ == "__main__":
    main()
