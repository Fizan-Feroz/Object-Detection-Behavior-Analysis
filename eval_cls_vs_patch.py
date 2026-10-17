# ============================================================
# CLS Token vs CLS + Mean Patch Tokens Ablation
# Evaluates on identical Blocked 5-Fold splits across 5 seeds:
#   1. CLS Token Only (384-d)
#   2. Mean Patch Token Only (384-d)
#   3. CLS + Mean Patch Tokens (768-d)
# ============================================================

import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config import CLASS_NAMES, EXPERIMENTS_CSV, RESULTS_DIR, get_device, seed_everything
from train_phase4_ablations import (
    compute_class_weights,
    log_to_experiments_csv,
    predict_model,
    track_clustered_bootstrap_ci,
    train_single_model,
    ProjectedFusedBeeLSTM,
)

PHASE1_DATA_DIR = RESULTS_DIR.parent / "dinov2_features" / "phase1_data"
PHASE4_DATA_DIR = RESULTS_DIR.parent / "dinov2_features" / "phase4_data"


def evaluate_token_variants(device: torch.device):
    print("=" * 75)
    print("EVALUATING TOKEN ABLATION: CLS vs PATCH vs CLS+PATCH")
    print("=" * 75)

    is_t = np.load(PHASE1_DATA_DIR / "is_test.npy")
    test_idx = np.where(is_t)[0]

    y_test = np.load(PHASE1_DATA_DIR / "labels.npy")[test_idx]
    b_test = np.load(PHASE1_DATA_DIR / "block_ids.npy")[test_idx]
    t_test = np.load(PHASE1_DATA_DIR / "track_ids.npy")[test_idx]
    kin_test = np.load(PHASE1_DATA_DIR / "kinematics.npy")[test_idx]

    cls_patch = np.load(PHASE4_DATA_DIR / "test_visual_cls_patch.npy")
    cls_only = cls_patch[:, :, :384]
    patch_only = cls_patch[:, :, 384:]

    variants = {
        "Visual_CLS_Token_Only": (cls_only, 384),
        "Visual_Mean_Patch_Only": (patch_only, 384),
        "Visual_CLS_Plus_Patch": (cls_patch, 768),
    }

    seeds = [42, 43, 44, 45, 46]
    variant_summaries = {}

    for var_name, (vis_arr, vis_dim) in variants.items():
        print(f"\nRunning {var_name} (vis_dim={vis_dim})...")
        seed_results = []

        for seed in seeds:
            seed_everything(seed)
            oof_probs = np.zeros((len(y_test), 5), dtype=np.float32)

            for fold_k in range(5):
                test_fold_idx = np.where(b_test == fold_k)[0]
                train_fold_idx = np.where(b_test != fold_k)[0]

                if len(test_fold_idx) == 0:
                    continue

                # 80/20 train/tune split grouped by track
                tr_tids = t_test[train_fold_idx]
                unique_tids = np.unique(tr_tids)
                rng = np.random.default_rng(seed + fold_k)
                shuffled = set(rng.choice(unique_tids, size=max(1, int(len(unique_tids) * 0.20)), replace=False))
                tune_mask = np.array([tid in shuffled for tid in tr_tids])
                train_mask = ~tune_mask

                idx_tr = train_fold_idx[train_mask]
                idx_tu = train_fold_idx[tune_mask]

                weights = compute_class_weights(y_test[idx_tr], num_classes=5).to(device)

                model = ProjectedFusedBeeLSTM(vis_dim=vis_dim, bidirectional=True).to(device)

                ds_tr = TensorDataset(torch.tensor(vis_arr[idx_tr]), torch.tensor(kin_test[idx_tr]), torch.tensor(y_test[idx_tr], dtype=torch.long))
                ds_tu = TensorDataset(torch.tensor(vis_arr[idx_tu]), torch.tensor(kin_test[idx_tu]), torch.tensor(y_test[idx_tu], dtype=torch.long))
                ds_te = TensorDataset(torch.tensor(vis_arr[test_fold_idx]), torch.tensor(kin_test[test_fold_idx]), torch.tensor(y_test[test_fold_idx], dtype=torch.long))

                trained_m, _, _ = train_single_model(
                    model, DataLoader(ds_tr, batch_size=32, shuffle=True),
                    DataLoader(ds_tu, batch_size=32, shuffle=False),
                    device, weights, epochs=25, lr=1e-3, patience=8
                )

                probs = predict_model(trained_m, DataLoader(ds_te, batch_size=32, shuffle=False), device)
                oof_probs[test_fold_idx] = probs

            preds = np.argmax(oof_probs, axis=-1)
            macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
            weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
            acc = float(np.mean(preds == y_test))

            cis = track_clustered_bootstrap_ci(y_test, preds, t_test, seed=seed)
            rep = classification_report(y_test, preds, labels=list(range(5)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
            per_class_f1 = {c: rep[c]["f1-score"] for c in CLASS_NAMES}

            print(f"  Seed {seed} | Macro-F1: {macro_f1:.4f} (95% CI: [{cis['macro_f1'][0]:.4f}, {cis['macro_f1'][1]:.4f}]) | Acc: {acc:.4f}")

            log_to_experiments_csv(
                experiment_name=var_name,
                model_type="bilstm",
                seed=seed,
                split_type="blocked_5fold_pooled_oof",
                train_windows=len(y_test) * 4 // 5,
                train_tracks=len(np.unique(t_test)) * 4 // 5,
                tune_windows=0,
                tune_tracks=0,
                test_windows=len(y_test),
                test_tracks=len(np.unique(t_test)),
                macro_f1=macro_f1,
                weighted_f1=weighted_f1,
                accuracy=acc,
                per_class_f1=per_class_f1,
                config_dict={"bootstrap_ci_95": cis, "token_dim": vis_dim},
            )

            seed_results.append({
                "seed": seed,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "accuracy": acc,
                "bootstrap_ci_95": cis,
                "per_class_f1": per_class_f1,
            })

        macros = [s["macro_f1"] for s in seed_results]
        variant_summaries[var_name] = {
            "experiment": var_name,
            "mean_macro_f1": float(np.mean(macros)),
            "sd_macro_f1": float(np.std(macros)),
            "mean_accuracy": float(np.mean([s["accuracy"] for s in seed_results])),
            "sd_accuracy": float(np.std([s["accuracy"] for s in seed_results])),
            "per_class_f1": {
                c: {
                    "mean": float(np.mean([s["per_class_f1"][c] for s in seed_results])),
                    "sd": float(np.std([s["per_class_f1"][c] for s in seed_results])),
                }
                for c in CLASS_NAMES
            },
            "seed_results": seed_results,
        }
        print(f"  --> {var_name} Overall: Macro-F1 = {variant_summaries[var_name]['mean_macro_f1']:.4f} +/- {variant_summaries[var_name]['sd_macro_f1']:.4f}")

    # Update phase_4_ablation_summary.json
    summary_path = RESULTS_DIR / "phase_4_ablation_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing.update(variant_summaries)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print("\nUpdated summary saved to:", summary_path)


if __name__ == "__main__":
    device = get_device()
    evaluate_token_variants(device)
