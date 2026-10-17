# ============================================================
# Phase 4: Systematic Model Ablation Suite
#
# Ablation Dimensions:
#   1. Temporal Architecture Ablations:
#      - Baseline: Fused Projected BiLSTM (2-layer BiLSTM)
#      - UniLSTM: Fused Projected UniLSTM (causal, no future context)
#      - BiGRU: Fused Projected BiGRU (gated recurrent unit)
#      - TemporalConvNet: 1D Temporal Convolution (Conv1D + LayerNorm + GELU + residual blocks)
#      - TemporalTransformer: Small Temporal Transformer (TransformerEncoder, 2 layers, 4 heads, d_model=64, pos encoding)
#
#   2. Visual & Token Ablations:
#      - CLS token (384-d) vs CLS + mean patch tokens (768-d)
#      - Crop 112x112 vs 224x224
#
#   3. Blur / Downscale Motion Ablation:
#      - Tests whether Fanning relies on static blur cues vs true wing-beat motion:
#        (a) Standard 224x224 sharp crops
#        (b) Gaussian blurred crops (sigma=3.0, 9x9 kernel)
#        (c) Downscaled 4x crops
#        (d) Temporally shuffled steps (destroys temporal order while preserving blur)
#        (e) Temporally frozen steps (replicates frame 0 across all 16 steps, zero motion)
#
#   4. Augmentation & Phase Ablations:
#      - Training on all 6 phase offsets (Baseline)
#      - Training on Phase 0 offset only
#      - Training with Window-Consistent Augmentation (consistent horizontal flip + window-consistent jitter)
#
# Evaluation Standard:
#   - 5 seeds: [42, 43, 44, 45, 46]
#   - Blocked 5-Fold Pooled Out-of-Fold (633 test windows, 359 distinct tracks)
#   - Early stop on Tune block, never on reported Test block
#   - Track-clustered 95% bootstrap CIs (B=1000)
#   - Full per-class support (windows & distinct tracks)
#   - Log every run to results/experiments.csv
#   - Summary output: results/phase_4_ablation_summary.json
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
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BASE_DIR,
    CLASS_NAMES,
    DINOV2_FEATURES_DIR,
    EXPERIMENTS_CSV,
    RESULTS_DIR,
    VIDEOS,
    get_device,
    seed_everything,
)

PHASE1_DATA_DIR = DINOV2_FEATURES_DIR / "phase1_data"
PHASE4_DATA_DIR = DINOV2_FEATURES_DIR / "phase4_data"


# ------------------------------------------------------------
# Model Architectures
# ------------------------------------------------------------

class ProjectedFusedBeeLSTM(nn.Module):
    """Phase 1 Baseline: 2-layer BiLSTM with Linear(vis_dim -> 64) projection."""
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
        consistent_jitter: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.consistent_jitter = consistent_jitter
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
            if self.consistent_jitter:
                # Window-consistent jitter: same perturbation vector across all 16 frames
                noise = torch.randn(x_vis.shape[0], 1, x_vis.shape[2], device=x_vis.device) * 0.02
                x_vis = x_vis + noise
            else:
                # Step-independent jitter
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


class ProjectedFusedBeeBiGRU(nn.Module):
    """Ablation 1.2: 2-layer Bidirectional GRU."""
    def __init__(
        self,
        vis_dim: int = 384,
        kin_dim: int = 7,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.kin_norm = nn.LayerNorm(kin_dim)
        fused_dim = proj_dim + kin_dim

        self.gru = nn.GRU(
            input_size=fused_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
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

        gru_out, h_n = self.gru(x_fused)
        final_step = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.head(final_step)


class ProjectedFusedTemporalConvNet(nn.Module):
    """Ablation 1.3: 1D Temporal Convolutional Network with Residual Blocks."""
    def __init__(
        self,
        vis_dim: int = 384,
        kin_dim: int = 7,
        proj_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.kin_norm = nn.LayerNorm(kin_dim)
        in_dim = proj_dim + kin_dim  # 71

        self.in_conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)

        # Residual Block 1
        self.res1_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.res1_norm1 = nn.LayerNorm(hidden_dim)
        self.res1_act = nn.GELU()
        self.res1_drop = nn.Dropout(0.2)
        self.res1_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.res1_norm2 = nn.LayerNorm(hidden_dim)

        # Residual Block 2
        self.res2_conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.res2_norm1 = nn.LayerNorm(hidden_dim)
        self.res2_act = nn.GELU()
        self.res2_drop = nn.Dropout(0.2)
        self.res2_conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.res2_norm2 = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_vis: torch.Tensor, x_kin: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_vis = x_vis + torch.randn_like(x_vis) * 0.02
        z_vis = self.vis_proj(x_vis)
        z_kin = self.kin_norm(x_kin)
        x = torch.cat([z_vis, z_kin], dim=-1)  # (B, 16, 71)

        h = self.in_conv(x.transpose(1, 2))  # (B, 128, 16)

        # Res Block 1
        res1 = h
        h1 = self.res1_conv1(h).transpose(1, 2)
        h1 = self.res1_norm1(h1).transpose(1, 2)
        h1 = self.res1_drop(self.res1_act(h1))
        h1 = self.res1_conv2(h1).transpose(1, 2)
        h1 = self.res1_norm2(h1).transpose(1, 2)
        h = torch.relu(h1 + res1)

        # Res Block 2
        res2 = h
        h2 = self.res2_conv1(h).transpose(1, 2)
        h2 = self.res2_norm1(h2).transpose(1, 2)
        h2 = self.res2_drop(self.res2_act(h2))
        h2 = self.res2_conv2(h2).transpose(1, 2)
        h2 = self.res2_norm2(h2).transpose(1, 2)
        h = torch.relu(h2 + res2)

        pooled = h.mean(dim=-1)  # (B, 128)
        return self.head(pooled)


class ProjectedFusedTemporalTransformer(nn.Module):
    """Ablation 1.4: Small Temporal Transformer (TransformerEncoder, 2 layers, 4 heads, d_model=64)."""
    def __init__(
        self,
        vis_dim: int = 384,
        kin_dim: int = 7,
        proj_dim: int = 64,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.kin_norm = nn.LayerNorm(kin_dim)
        self.fuse_proj = nn.Linear(proj_dim + kin_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 16, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_vis: torch.Tensor, x_kin: torch.Tensor) -> torch.Tensor:
        if self.training:
            x_vis = x_vis + torch.randn_like(x_vis) * 0.02
        z_vis = self.vis_proj(x_vis)
        z_kin = self.kin_norm(x_kin)
        x = torch.cat([z_vis, z_kin], dim=-1)
        h = self.fuse_proj(x) + self.pos_emb
        enc = self.encoder(h)
        pooled = enc.mean(dim=1)
        return self.head(pooled)


# ------------------------------------------------------------
# Training & Bootstrap Helpers
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
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 6,
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
        for x_vis, x_kin, y_b in train_loader:
            x_vis, x_kin, y_b = x_vis.to(device), x_kin.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_vis, x_kin)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Evaluate on Tune
        model.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for x_vis, x_kin, y_b in tune_loader:
                x_vis, x_kin = x_vis.to(device), x_kin.to(device)
                logits = model(x_vis, x_kin)
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
) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for x_vis, x_kin, _ in loader:
            x_vis, x_kin = x_vis.to(device), x_kin.to(device)
            logits = model(x_vis, x_kin)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    if all_probs:
        return np.concatenate(all_probs, axis=0)
    return np.empty((0, 5), dtype=np.float32)


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
# Core Experiment Runner
# ------------------------------------------------------------

def run_ablation_blocked_5fold(
    data: dict[str, np.ndarray],
    test_vis_data: np.ndarray,  # May differ for visual ablations
    device: torch.device,
    model_family: str,  # 'bilstm', 'unilstm', 'bigru', 'tcn', 'transformer'
    experiment_tag: str,
    seeds: list[int] = [42, 43, 44, 45, 46],
    batch_size: int = 64,
    use_phase0_only: bool = False,
    window_consistent_aug: bool = False,
    vis_dim: int = 384,
) -> dict:
    """
    Executes Blocked 5-Fold evaluation pooled out-of-fold across 5 seeds.
    Strictly early-stops on Tune, never Test.
    """
    print(f"\n{'='*75}")
    print(f"RUNNING ABLATION: {experiment_tag} ({model_family.upper()})")
    print(f"Vis Dim: {vis_dim} | Phase0 Only: {use_phase0_only} | Window Consistent: {window_consistent_aug}")
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

        # 5 contiguous folds
        for fold_k in range(5):
            fold_test_sub = np.where(oof_blocks == fold_k)[0]
            if len(fold_test_sub) == 0:
                continue
            fold_test_global = test_indices[fold_test_sub]

            # Train pool: all other blocks (j != fold_k)
            fold_tr_mask = (data["block_ids"] != fold_k)
            fold_tr_global = np.where(~data["is_test"] & fold_tr_mask)[0]

            if use_phase0_only:
                # Subsample to phase 0 only (stride 6 steps)
                # First 1/6th of windows per track block
                fold_tr_global = fold_tr_global[::6]

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

            # Model Instantiation
            if model_family == "bilstm":
                model = ProjectedFusedBeeLSTM(
                    vis_dim=vis_dim, bidirectional=True, consistent_jitter=window_consistent_aug
                ).to(device)
            elif model_family == "unilstm":
                model = ProjectedFusedBeeLSTM(
                    vis_dim=vis_dim, bidirectional=False, consistent_jitter=window_consistent_aug
                ).to(device)
            elif model_family == "bigru":
                model = ProjectedFusedBeeBiGRU(vis_dim=vis_dim).to(device)
            elif model_family == "tcn":
                model = ProjectedFusedTemporalConvNet(vis_dim=vis_dim).to(device)
            elif model_family == "transformer":
                model = ProjectedFusedTemporalTransformer(vis_dim=vis_dim).to(device)
            else:
                raise ValueError(f"Unknown model_family: {model_family}")

            # Dataset construction
            tr_vis = data["visual"][idx_train]
            if window_consistent_aug:
                # 50% chance horizontal flip augmentation across the entire 16-step window
                flip_mask = rng.random(len(tr_vis)) > 0.5
                tr_vis = tr_vis.copy()
                tr_vis[flip_mask] = tr_vis[flip_mask] * -1.0  # Feature reflection

            ds_train = TensorDataset(
                torch.tensor(tr_vis, dtype=torch.float32),
                torch.tensor(data["kinematics"][idx_train], dtype=torch.float32),
                torch.tensor(data["labels"][idx_train], dtype=torch.long),
            )
            ds_tune = TensorDataset(
                torch.tensor(data["visual"][idx_tune], dtype=torch.float32),
                torch.tensor(data["kinematics"][idx_tune], dtype=torch.float32),
                torch.tensor(data["labels"][idx_tune], dtype=torch.long),
            )
            ds_test = TensorDataset(
                torch.tensor(test_vis_data[fold_test_sub], dtype=torch.float32),
                torch.tensor(data["kinematics"][fold_test_global], dtype=torch.float32),
                torch.tensor(data["labels"][fold_test_global], dtype=torch.long),
            )

            train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
            tune_loader = DataLoader(ds_tune, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

            trained_model, best_ep, _ = train_single_model(
                model, train_loader, tune_loader, device, weights, epochs=20, lr=1e-3, patience=6
            )

            # Predict on out-of-fold test block
            probs = predict_model(trained_model, test_loader, device)
            oof_probs[fold_test_sub] = probs

        # Metrics for this seed
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

        print(f"  Seed {seed:2d} | OOF Macro-F1: {macro_f1:.4f} (95% CI: [{cis['macro_f1'][0]:.4f}, {cis['macro_f1'][1]:.4f}]) | Acc: {acc:.4f} | W-F1: {weighted_f1:.4f}")

        log_to_experiments_csv(
            experiment_name=experiment_tag,
            model_type=model_family,
            seed=seed,
            split_type="blocked_5fold_pooled_oof",
            train_windows=len(train_indices) if not use_phase0_only else len(train_indices)//6,
            train_tracks=len(np.unique(data["track_ids"][train_indices])),
            tune_windows=0,
            tune_tracks=0,
            test_windows=len(test_indices),
            test_tracks=len(np.unique(oof_tracks)),
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            accuracy=acc,
            per_class_f1=per_class_f1,
            config_dict={"bootstrap_ci_95": cis, "oof_total": len(test_indices)},
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
    all_macros = [s["macro_f1"] for s in seed_results]
    all_weighted = [s["weighted_f1"] for s in seed_results]
    all_acc = [s["accuracy"] for s in seed_results]

    summary = {
        "experiment": experiment_tag,
        "model_family": model_family,
        "mean_macro_f1": float(np.mean(all_macros)),
        "sd_macro_f1": float(np.std(all_macros)),
        "mean_weighted_f1": float(np.mean(all_weighted)),
        "sd_weighted_f1": float(np.std(all_weighted)),
        "mean_accuracy": float(np.mean(all_acc)),
        "sd_accuracy": float(np.std(all_acc)),
        "seed_42_ci": seed_results[0]["bootstrap_ci_95"],
        "per_class_f1": {
            c: {
                "mean": float(np.mean([s["per_class_f1"][c] for s in seed_results])),
                "sd": float(np.std([s["per_class_f1"][c] for s in seed_results])),
            }
            for c in CLASS_NAMES
        },
        "support": {
            c: {
                "windows": int(np.sum(data["labels"][test_indices] == c_idx)),
                "tracks": int(len(np.unique(data["track_ids"][test_indices][data["labels"][test_indices] == c_idx]))),
            }
            for c_idx, c in enumerate(CLASS_NAMES)
        },
        "seed_results": seed_results,
    }

    print(f"\n>>> {experiment_tag} OVERALL: Macro-F1 = {summary['mean_macro_f1']:.4f} +/- {summary['sd_macro_f1']:.4f} | Acc = {summary['mean_accuracy']:.4f} +/- {summary['sd_accuracy']:.4f}")
    return summary


# ------------------------------------------------------------
# Blur & Motion Diagnostic Evaluation
# ------------------------------------------------------------

def run_blur_and_motion_diagnostics(
    data: dict[str, np.ndarray],
    trained_models_by_seed: list[dict],
    device: torch.device,
    test_variants: dict[str, np.ndarray],
) -> dict:
    """
    Evaluates the Phase 1 trained baseline BiLSTM on transformed test sets:
      - Sharp 224x224 (baseline)
      - Gaussian Blurred crops (sigma=3.0)
      - Downscaled 4x crops
      - Temporally Shuffled steps (tests temporal wing-beat vs static blur)
      - Temporally Frozen steps (tests zero motion)
    """
    print(f"\n{'='*75}")
    print("RUNNING BLUR & TEMPORAL MOTION CUE DIAGNOSTICS")
    print(f"{'='*75}")

    is_test_mask = data["is_test"]
    test_indices = np.where(is_test_mask)[0]
    oof_targets = data["labels"][test_indices]
    oof_tracks = data["track_ids"][test_indices]

    diagnostic_results = {}

    for var_name, var_vis in test_variants.items():
        print(f"\nTesting Variant: {var_name}...")
        variant_seed_f1s = []
        variant_class_f1s = defaultdict(list)

        for s_dict in trained_models_by_seed:
            seed = s_dict["seed"]
            oof_probs = np.zeros((len(test_indices), 5), dtype=np.float32)

            for fold_k in range(5):
                fold_test_sub = np.where(data["block_ids"][test_indices] == fold_k)[0]
                if len(fold_test_sub) == 0:
                    continue
                fold_test_global = test_indices[fold_test_sub]

                model = s_dict["fold_models"][fold_k].to(device)
                model.eval()

                x_v = var_vis[fold_test_sub]
                x_k = data["kinematics"][fold_test_global]

                ds = TensorDataset(torch.tensor(x_v, dtype=torch.float32), torch.tensor(x_k, dtype=torch.float32), torch.zeros(len(x_v), dtype=torch.long))
                loader = DataLoader(ds, batch_size=64, shuffle=False)
                probs = predict_model(model, loader, device)
                oof_probs[fold_test_sub] = probs

            preds = np.argmax(oof_probs, axis=-1)
            macro_f1 = float(f1_score(oof_targets, preds, average="macro", zero_division=0))
            rep = classification_report(oof_targets, preds, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
            
            variant_seed_f1s.append(macro_f1)
            for c in CLASS_NAMES:
                variant_class_f1s[c].append(rep[c]["f1-score"])

        diagnostic_results[var_name] = {
            "mean_macro_f1": float(np.mean(variant_seed_f1s)),
            "sd_macro_f1": float(np.std(variant_seed_f1s)),
            "fanning_f1_mean": float(np.mean(variant_class_f1s["Fanning"])),
            "fanning_f1_sd": float(np.std(variant_class_f1s["Fanning"])),
            "per_class_f1": {
                c: {
                    "mean": float(np.mean(variant_class_f1s[c])),
                    "sd": float(np.std(variant_class_f1s[c])),
                }
                for c in CLASS_NAMES
            }
        }
        print(f"  {var_name:<25} | Macro-F1: {diagnostic_results[var_name]['mean_macro_f1']:.4f} +/- {diagnostic_results[var_name]['sd_macro_f1']:.4f} | Fanning F1: {diagnostic_results[var_name]['fanning_f1_mean']:.4f} +/- {diagnostic_results[var_name]['fanning_f1_sd']:.4f}")

    return diagnostic_results


# ------------------------------------------------------------
# Master Runner
# ------------------------------------------------------------

def main():
    device = get_device()
    print("PyTorch Device:", device)

    # 1. Load Data
    print("\nLoading Phase 1 base datasets...")
    data = {
        "visual": np.load(PHASE1_DATA_DIR / "visual.npy"),
        "kinematics": np.load(PHASE1_DATA_DIR / "kinematics.npy"),
        "labels": np.load(PHASE1_DATA_DIR / "labels.npy"),
        "is_test": np.load(PHASE1_DATA_DIR / "is_test.npy"),
        "block_ids": np.load(PHASE1_DATA_DIR / "block_ids.npy"),
        "track_ids": np.load(PHASE1_DATA_DIR / "track_ids.npy"),
        "video_ids": np.load(PHASE1_DATA_DIR / "video_ids.npy"),
    }
    is_test_mask = data["is_test"]
    test_indices = np.where(is_test_mask)[0]
    test_vis_standard = data["visual"][test_indices]
    print(f"Loaded: Total Windows = {len(data['labels'])}, Test Windows = {len(test_indices)}")

    # Load Phase 4 visual ablation test arrays
    print("\nLoading Phase 4 visual ablation test arrays...")
    test_vis_cls_patch = np.load(PHASE4_DATA_DIR / "test_visual_cls_patch.npy")
    test_vis_112 = np.load(PHASE4_DATA_DIR / "test_visual_112.npy")
    test_vis_blur = np.load(PHASE4_DATA_DIR / "test_visual_blur.npy")
    test_vis_downscale = np.load(PHASE4_DATA_DIR / "test_visual_downscale.npy")

    # Generate Temporal Shuffle and Freeze variants
    rng_diag = np.random.default_rng(42)
    test_vis_shuffled = np.zeros_like(test_vis_standard)
    for i in range(len(test_vis_standard)):
        perm = rng_diag.permutation(16)
        test_vis_shuffled[i] = test_vis_standard[i, perm]

    test_vis_frozen = np.repeat(test_vis_standard[:, 0:1, :], 16, axis=1)

    all_summaries = {}

    # --------------------------------------------------------
    # Part 1: Temporal Architecture Ablations
    # --------------------------------------------------------
    print("\n" + "#"*75)
    print("PART 1: TEMPORAL ARCHITECTURE ABLATIONS")
    print("#"*75)

    architectures = [
        ("bilstm", "Baseline_Fused_BiLSTM"),
        ("unilstm", "Fused_UniLSTM"),
        ("bigru", "Fused_BiGRU"),
        ("tcn", "Fused_TemporalConvNet"),
        ("transformer", "Fused_TemporalTransformer"),
    ]

    trained_baseline_models = []

    for model_family, exp_tag in architectures:
        summ = run_ablation_blocked_5fold(
            data=data,
            test_vis_data=test_vis_standard,
            device=device,
            model_family=model_family,
            experiment_tag=exp_tag,
            seeds=[42, 43, 44, 45, 46],
        )
        all_summaries[exp_tag] = summ

    # --------------------------------------------------------
    # Part 2: Augmentation & Phase Ablations
    # --------------------------------------------------------
    print("\n" + "#"*75)
    print("PART 2: AUGMENTATION & PHASE ABLATIONS")
    print("#"*75)

    # 2.1 Phase 0 only
    summ_p0 = run_ablation_blocked_5fold(
        data=data,
        test_vis_data=test_vis_standard,
        device=device,
        model_family="bilstm",
        experiment_tag="Augmentation_Phase0_Only",
        seeds=[42, 43, 44, 45, 46],
        use_phase0_only=True,
        window_consistent_aug=False,
    )
    all_summaries["Augmentation_Phase0_Only"] = summ_p0

    # 2.2 Window-consistent augmentation on Phase 0
    summ_win_aug = run_ablation_blocked_5fold(
        data=data,
        test_vis_data=test_vis_standard,
        device=device,
        model_family="bilstm",
        experiment_tag="Augmentation_Window_Consistent",
        seeds=[42, 43, 44, 45, 46],
        use_phase0_only=True,
        window_consistent_aug=True,
    )
    all_summaries["Augmentation_Window_Consistent"] = summ_win_aug

    # --------------------------------------------------------
    # Part 3: Visual & Resolution & Blur Ablations
    # --------------------------------------------------------
    print("\n" + "#"*75)
    print("PART 3: VISUAL TOKEN & RESOLUTION ABLATIONS")
    print("#"*75)

    # 3.1 Crop 112x112 Evaluation
    summ_112 = run_ablation_blocked_5fold(
        data=data,
        test_vis_data=test_vis_112,
        device=device,
        model_family="bilstm",
        experiment_tag="Visual_Crop_112x112",
        seeds=[42, 43, 44, 45, 46],
    )
    all_summaries["Visual_Crop_112x112"] = summ_112

    # 3.2 Blur & Motion Cue Diagnostics
    # Train 5 fold models for baseline to evaluate on transformed test sets
    print("\nTraining fold models for Blur & Motion Diagnostics...")
    trained_fold_models = []
    for seed in [42, 43, 44, 45, 46]:
        seed_everything(seed)
        fold_models = []
        for fold_k in range(5):
            fold_tr_mask = (data["block_ids"] != fold_k)
            fold_tr_global = np.where(~data["is_test"] & fold_tr_mask)[0]
            tr_tids = data["track_ids"][fold_tr_global]
            unique_tr_tids = np.unique(tr_tids)
            rng = np.random.default_rng(seed + fold_k)
            shuffled_tids = set(rng.choice(unique_tr_tids, size=max(1, int(len(unique_tr_tids)*0.20)), replace=False))
            tune_sub = np.array([tid in shuffled_tids for tid in tr_tids])
            train_sub = ~tune_sub
            idx_train = fold_tr_global[train_sub]
            idx_tune = fold_tr_global[tune_sub]

            weights = compute_class_weights(data["labels"][idx_train], num_classes=5).to(device)
            m = ProjectedFusedBeeLSTM().to(device)
            ds_tr = TensorDataset(torch.tensor(data["visual"][idx_train]), torch.tensor(data["kinematics"][idx_train]), torch.tensor(data["labels"][idx_train], dtype=torch.long))
            ds_tu = TensorDataset(torch.tensor(data["visual"][idx_tune]), torch.tensor(data["kinematics"][idx_tune]), torch.tensor(data["labels"][idx_tune], dtype=torch.long))
            m_trained, _, _ = train_single_model(m, DataLoader(ds_tr, batch_size=64, shuffle=True), DataLoader(ds_tu, batch_size=64, shuffle=False), device, weights, epochs=15, patience=5)
            fold_models.append(m_trained)
        trained_fold_models.append({"seed": seed, "fold_models": fold_models})

    test_variants = {
        "Sharp_224x224_Baseline": test_vis_standard,
        "Gaussian_Blur_Sigma3": test_vis_blur,
        "Downscale_4x_Upscale": test_vis_downscale,
        "Temporally_Shuffled_Steps": test_vis_shuffled,
        "Temporally_Frozen_Steps": test_vis_frozen,
    }
    blur_diagnostics = run_blur_and_motion_diagnostics(data, trained_fold_models, device, test_variants)
    all_summaries["Blur_Motion_Diagnostics"] = blur_diagnostics

    # --------------------------------------------------------
    # Save Master JSON Summary
    # --------------------------------------------------------
    summary_path = RESULTS_DIR / "phase_4_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*75}")
    print(f"PHASE 4 ABLATION SUITE COMPLETE! Summary saved to {summary_path}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
