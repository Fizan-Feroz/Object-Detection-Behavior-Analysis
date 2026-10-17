# ============================================================
# PyTorch ViT-LSTM & Kinematics Classifier Suite
#
# Improvements in Phase 0:
#   1. BiLSTM Pooling Fix: Uses top-layer concat([h_n[-2], h_n[-1]], dim=-1)
#      instead of lstm_out[:, -1, :]. Supports --bidirectional flag.
#   2. Train/Tune/Test Protocol: Strict separation where early stopping
#      monitors Tune block only, and Test block is evaluated once.
#   3. Multi-Seed Runner: Evaluates over >= 5 seeds (mean +/- sd).
#   4. Clustered Bootstrap CIs: 95% confidence intervals clustered by track_id.
#   5. Dual Support Reporting: Reports both window count AND distinct tracks.
#   6. Experiment Logging: Appends full run records to results/experiments.csv.
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
    DINOV2_FEATURES_DIR as FEATURES_DIR,
    EXPERIMENTS_CSV,
    MODELS_DIR,
    RESULTS_DIR,
    seed_everything,
)


class BeeLSTMClassifier(nn.Module):
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
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
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
        # x: (Batch, Sequence=16, Features)
        x = self.input_norm(x)
        if self.training:
            # Subtle feature noise as regularization against background memorization
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        x = self.input_dropout(x)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Correct pooling:
        # For bidirectional: last layer forward is h_n[-2], backward is h_n[-1]
        if self.bidirectional:
            final_step = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            final_step = h_n[-1]
        return self.head(final_step)


def compute_class_weights(y: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    # Inverse square-root frequency weighting (gentler than pure inverse)
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
    """
    Computes 95% bootstrap confidence intervals clustered by track_id.
    Resamples unique tracks with replacement, pools all windows belonging
    to sampled tracks, and calculates metrics.
    """
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


def create_track_grouped_tune_split(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    tracks_full: np.ndarray,
    tune_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partitions the training pool into Train and Tune blocks clustered by track_id.
    Zero track overlap between Train and Tune.
    Stratifies across classes so each class with multiple tracks has representation in Tune.
    """
    rng = np.random.default_rng(seed)
    track_to_classes = defaultdict(list)
    for tid, y in zip(tracks_full, y_train_full):
        track_to_classes[tid].append(y)

    track_dominant_class = {
        tid: int(np.bincount(cls_list).argmax())
        for tid, cls_list in track_to_classes.items()
    }

    class_to_tracks = defaultdict(list)
    for tid, c in track_dominant_class.items():
        class_to_tracks[c].append(tid)

    tune_tracks = set()
    for c, t_list in class_to_tracks.items():
        shuffled = list(rng.permutation(t_list))
        # At least 1 track in tune if >= 4 tracks, up to tune_ratio
        n_tune = max(1 if len(shuffled) >= 4 else 0, int(round(len(shuffled) * tune_ratio)))
        # Keep at least 1 track in train
        if n_tune >= len(shuffled):
            n_tune = len(shuffled) - 1
        for tid in shuffled[:n_tune]:
            tune_tracks.add(tid)

    train_mask = np.array([tid not in tune_tracks for tid in tracks_full])
    tune_mask = ~train_mask

    if tune_mask.sum() == 0:
        unique_tids = list(track_dominant_class.keys())
        shuffled = list(rng.permutation(unique_tids))
        n_tune = max(1, int(len(shuffled) * tune_ratio))
        tune_tracks = set(shuffled[:n_tune])
        train_mask = np.array([tid not in tune_tracks for tid in tracks_full])
        tune_mask = ~train_mask

    return (
        X_train_full[train_mask],
        y_train_full[train_mask],
        tracks_full[train_mask],
        X_train_full[tune_mask],
        y_train_full[tune_mask],
        tracks_full[tune_mask],
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    macro_f1 = float(f1_score(targets_arr, preds_arr, average="macro", zero_division=0))
    return macro_f1, preds_arr, targets_arr


def log_experiment_to_csv(
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


def run_single_seed_experiment(
    exp_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    tr_tracks: np.ndarray,
    X_tu: np.ndarray,
    y_tu: np.ndarray,
    tu_tracks: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    te_tracks: np.ndarray,
    device: torch.device,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    bidirectional: bool = True,
) -> dict:
    seed_everything(seed)

    weights = compute_class_weights(y_tr, num_classes=5).to(device)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    tune_ds = TensorDataset(torch.tensor(X_tu, dtype=torch.float32), torch.tensor(y_tu, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    tune_loader = DataLoader(tune_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BeeLSTMClassifier(
        input_dim=X_tr.shape[-1],
        hidden_dim=128,
        num_layers=2,
        num_classes=5,
        bidirectional=bidirectional,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_tune_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        tune_macro_f1, _, _ = evaluate(model, tune_loader, device)

        if tune_macro_f1 > best_tune_macro_f1:
            best_tune_macro_f1 = tune_macro_f1
            best_epoch = ep
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if ep % 10 == 0 or ep == epochs or patience_counter == 0:
            pass  # Keep concise during multi-seed

        if patience_counter >= patience:
            break

    elapsed = time.time() - t0

    # Load best checkpoint selected by Tune block
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model weights
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_save_path = MODELS_DIR / f"{exp_name.lower().replace(' ', '_')}_s{seed}.pt"
    torch.save(best_state, model_save_path)

    # FINAL EVALUATION ON UNTOUCHED TEST BLOCK
    test_macro_f1, test_preds, test_targets = evaluate(model, test_loader, device)
    test_weighted_f1 = float(f1_score(test_targets, test_preds, average="weighted", zero_division=0))
    test_acc = float(np.mean(test_preds == test_targets))

    # Bootstrap CIs clustered by track
    cis = track_clustered_bootstrap_ci(test_targets, test_preds, te_tracks, n_bootstraps=1000, seed=seed)

    # Per-class report
    present_classes = sorted(list(set(test_targets) | set(test_preds)))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    report = classification_report(
        test_targets,
        test_preds,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(test_targets, test_preds, labels=present_classes).tolist()

    per_class_f1 = {
        cls_name: report.get(cls_name, {}).get("f1-score", 0.0)
        for cls_name in CLASS_NAMES
    }

    # Support counts (windows AND distinct tracks)
    train_distinct_tracks = len(np.unique(tr_tracks))
    tune_distinct_tracks = len(np.unique(tu_tracks))
    test_distinct_tracks = len(np.unique(te_tracks))

    # Log to CSV
    log_experiment_to_csv(
        experiment_name=exp_name,
        model_type=f"LSTM_{'Bi' if bidirectional else 'Uni'}",
        seed=seed,
        split_type="train_tune_test_track_grouped",
        train_windows=len(y_tr),
        train_tracks=train_distinct_tracks,
        tune_windows=len(y_tu),
        tune_tracks=tune_distinct_tracks,
        test_windows=len(y_te),
        test_tracks=test_distinct_tracks,
        macro_f1=test_macro_f1,
        weighted_f1=test_weighted_f1,
        accuracy=test_acc,
        per_class_f1=per_class_f1,
        config_dict={
            "epochs": epochs,
            "best_epoch": best_epoch,
            "lr": lr,
            "bidirectional": bidirectional,
            "patience": patience,
            "bootstrap_ci_95": cis,
        },
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "tune_macro_f1": best_tune_macro_f1,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "test_accuracy": test_acc,
        "bootstrap_ci_95": cis,
        "per_class_f1": per_class_f1,
        "report": report,
        "confusion_matrix": cm,
        "support": {
            "train_windows": len(y_tr),
            "train_tracks": train_distinct_tracks,
            "tune_windows": len(y_tu),
            "tune_tracks": tune_distinct_tracks,
            "test_windows": len(y_te),
            "test_tracks": test_distinct_tracks,
        },
    }


def run_multi_seed_suite(
    exp_name: str,
    X_tr_full: np.ndarray,
    y_tr_full: np.ndarray,
    tr_tracks_full: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    te_tracks: np.ndarray,
    device: torch.device,
    seeds: list[int],
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    bidirectional: bool = True,
) -> dict:
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name} | {len(seeds)} Seeds: {seeds} | BiDirectional={bidirectional}")
    print(f"{'='*70}")
    print(f"Full Train Pool: {X_tr_full.shape[0]} windows ({len(np.unique(tr_tracks_full))} tracks)")
    print(f"Untouched Test:  {X_te.shape[0]} windows ({len(np.unique(te_tracks))} tracks)")

    seed_runs = []
    for s in seeds:
        # Create seed-specific track-grouped Tune split from the training pool
        X_tr, y_tr, tr_tids, X_tu, y_tu, tu_tids = create_track_grouped_tune_split(
            X_tr_full, y_tr_full, tr_tracks_full, tune_ratio=0.20, seed=s
        )

        res = run_single_seed_experiment(
            exp_name=exp_name,
            X_tr=X_tr,
            y_tr=y_tr,
            tr_tracks=tr_tids,
            X_tu=X_tu,
            y_tu=y_tu,
            tu_tracks=tu_tids,
            X_te=X_te,
            y_te=y_te,
            te_tracks=te_tracks,
            device=device,
            seed=s,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            bidirectional=bidirectional,
        )
        ci_m = res["bootstrap_ci_95"]["macro_f1"]
        print(f"  Seed {s:2d} | Best Ep {res['best_epoch']:2d} | Tune Macro-F1: {res['tune_macro_f1']:.4f} | Test Macro-F1: {res['test_macro_f1']:.4f} (95% CI: [{ci_m[0]:.4f}, {ci_m[1]:.4f}]) | Acc: {res['test_accuracy']:.4f}")
        seed_runs.append(res)

    macro_f1s = [r["test_macro_f1"] for r in seed_runs]
    weighted_f1s = [r["test_weighted_f1"] for r in seed_runs]
    accs = [r["test_accuracy"] for r in seed_runs]

    mean_macro = float(np.mean(macro_f1s))
    sd_macro = float(np.std(macro_f1s))
    mean_weighted = float(np.mean(weighted_f1s))
    sd_weighted = float(np.std(weighted_f1s))
    mean_acc = float(np.mean(accs))
    sd_acc = float(np.std(accs))

    per_class_summary = {}
    for c in CLASS_NAMES:
        c_f1s = [r["per_class_f1"][c] for r in seed_runs]
        per_class_summary[c] = {
            "mean": float(np.mean(c_f1s)),
            "sd": float(np.std(c_f1s)),
        }

    print(f"\n  -> {exp_name} Summary ({len(seeds)} Seeds):")
    print(f"     Test Macro-F1:    {mean_macro:.4f} +/- {sd_macro:.4f}")
    print(f"     Test Weighted-F1: {mean_weighted:.4f} +/- {sd_weighted:.4f}")
    print(f"     Test Accuracy:    {mean_acc:.4f} +/- {sd_acc:.4f}")
    print("     Per-Class F1 (mean +/- sd):")
    for c in CLASS_NAMES:
        print(f"       {c:<14}: {per_class_summary[c]['mean']:.4f} +/- {per_class_summary[c]['sd']:.4f}")

    suite_summary = {
        "experiment_name": exp_name,
        "bidirectional": bidirectional,
        "seeds": seeds,
        "mean_test_macro_f1": mean_macro,
        "sd_test_macro_f1": sd_macro,
        "mean_test_weighted_f1": mean_weighted,
        "sd_test_weighted_f1": sd_weighted,
        "mean_test_accuracy": mean_acc,
        "sd_test_accuracy": sd_acc,
        "per_class_f1": per_class_summary,
        "seed_runs": seed_runs,
    }

    # Save to results/
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / f"{exp_name.lower().replace(' ', '_')}_multiseed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    return suite_summary


def main():
    parser = argparse.ArgumentParser(description="Train Bee Behavior LSTM on DINOv2 + Kinematics with Multi-Seed Evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46], help="List of >= 5 seeds")
    parser.add_argument("--bidirectional", action="store_true", default=True, help="Use BiLSTM (default True)")
    parser.add_argument("--unidirectional", dest="bidirectional", action="store_false", help="Use unidirectional LSTM")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("BEE BEHAVIOR PYTORCH VI-LSTM SUITE (PHASE 0)")
    print(f"Device: {device} | Seeds: {args.seeds} | BiLSTM: {args.bidirectional}")
    print("=" * 70)

    # Check extracted tensors
    req_files = [
        "X_train_visual.npy",
        "X_train_kinematics.npy",
        "y_train.npy",
        "train_tracks.npy",
        "X_val_visual.npy",
        "X_val_kinematics.npy",
        "y_val.npy",
        "val_tracks.npy",
    ]
    missing = [f for f in req_files if not (FEATURES_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing extracted tensors in {FEATURES_DIR}: {missing}")
        print("Please run extract_dinov2_features.py first.")
        return

    print("Loading extracted feature arrays...")
    tr_vis = np.load(FEATURES_DIR / "X_train_visual.npy")
    tr_kin = np.load(FEATURES_DIR / "X_train_kinematics.npy")
    y_tr = np.load(FEATURES_DIR / "y_train.npy")
    tr_tracks = np.load(FEATURES_DIR / "train_tracks.npy")

    te_vis = np.load(FEATURES_DIR / "X_val_visual.npy")
    te_kin = np.load(FEATURES_DIR / "X_val_kinematics.npy")
    y_te = np.load(FEATURES_DIR / "y_val.npy")
    te_tracks = np.load(FEATURES_DIR / "val_tracks.npy")

    print(f"Loaded: Train pool {len(y_tr)} windows ({len(np.unique(tr_tracks))} tracks), Test {len(y_te)} windows ({len(np.unique(te_tracks))} tracks).")

    # Fused features
    tr_fused = np.concatenate([tr_vis, tr_kin], axis=-1)
    te_fused = np.concatenate([te_vis, te_kin], axis=-1)

    all_summaries = []

    # 1. Fused Model (DINOv2 + Causal Kinematics)
    res_fused = run_multi_seed_suite(
        "Fused_DINOv2_Kinematics",
        tr_fused, y_tr, tr_tracks,
        te_fused, y_te, te_tracks,
        device=device,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        bidirectional=args.bidirectional,
    )
    all_summaries.append(res_fused)

    # 2. Visual-Only Model (DINOv2)
    res_vis = run_multi_seed_suite(
        "Visual_Only_DINOv2",
        tr_vis, y_tr, tr_tracks,
        te_vis, y_te, te_tracks,
        device=device,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        bidirectional=args.bidirectional,
    )
    all_summaries.append(res_vis)

    # 3. Kinematics-Only Control
    res_kin = run_multi_seed_suite(
        "Kinematics_Only_Control",
        tr_kin, y_tr, tr_tracks,
        te_kin, y_te, te_tracks,
        device=device,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        bidirectional=args.bidirectional,
    )
    all_summaries.append(res_kin)

    # Summary Comparison Table
    print("\n" + "=" * 80)
    print("PHASE 0 MULTI-SEED BENCHMARK COMPARISON (MEAN +/- SD OVER 5 SEEDS)")
    print("=" * 80)
    print(f"{'Experiment':<28} | {'Test Macro-F1':<18} | {'Test Weighted-F1':<18} | {'Test Accuracy':<15}")
    print("-" * 80)
    for s in all_summaries:
        m_str = f"{s['mean_test_macro_f1']:.4f} +/- {s['sd_test_macro_f1']:.4f}"
        w_str = f"{s['mean_test_weighted_f1']:.4f} +/- {s['sd_test_weighted_f1']:.4f}"
        a_str = f"{s['mean_test_accuracy']:.4f} +/- {s['sd_test_accuracy']:.4f}"
        print(f"{s['experiment_name']:<28} | {m_str:<18} | {w_str:<18} | {a_str:<15}")
    print("=" * 80)

    # Save overall summary JSON
    summary_path = RESULTS_DIR / "phase_0_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nPhase 0 multi-seed summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
