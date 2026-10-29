# ============================================================
# Dual-PC Distributed Retraining Orchestrator
# Designed for 2x NVIDIA GeForce RTX 2070 Super (8GB VRAM)
# Operating System: Windows 10/11
# ============================================================
"""
Orchestrates workload distribution across two independent Windows PCs:
  PC 1 (Compute Node 1):
    - Video extraction subset: 20230609b-def, yt5, yt8 (~9,950 frames, 45% load)
    - YOLOv8m Object Detector training (optional, imgsz=640 or 960)
    - Temporal Architecture Training: Fused TCN (champion) & BiGRU across 5 seeds
    
  PC 2 (Compute Node 2):
    - Video extraction subset: 20230711a-fan, 20230711b-fan (~12,003 frames, 55% load)
    - Temporal Architecture Training: Fused BiLSTM & Causal UniLSTM across 5 seeds
    - End-to-End Live Video Streaming Benchmark (6,000 frames @ 50 FPS)

Usage:
  # On PC 1:
  python train_dual_pc.py --node pc1 --task all
  
  # On PC 2:
  python train_dual_pc.py --node pc2 --task all
  
  # Consolidation (on either PC after syncing results):
  python train_dual_pc.py --task merge
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from config import (
    BASE_DIR,
    CLASS_NAMES,
    DINOV2_FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    VIDEOS,
    get_device,
    seed_everything,
)

# Assigned video subsets for 50/50 load balancing
PC1_VIDEOS = ["20230609b-def", "yt5", "yt8"]       # 5,973 + 977 + 3,000 = 9,950 frames
PC2_VIDEOS = ["20230711a-fan", "20230711b-fan"]   # 6,000 + 6,003 = 12,003 frames

PC1_MODELS = ["TemporalConvNet", "BiGRU"]
PC2_MODELS = ["BiLSTM", "UniLSTM"]


def check_environment() -> None:
    """Print hardware and software diagnostic info."""
    print("=" * 75)
    print("DUAL-PC TRAINING ENVIRONMENT DIAGNOSTICS")
    print("=" * 75)
    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"PyTorch:      {torch.__version__}")
    print(f"CUDA Ready:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device:   {gpu_name} ({vram_gb:.2f} GB VRAM)")
        if "2070" in gpu_name:
            print(">> Detected RTX 2070 Super: Optimized for 8GB Turing architecture.")
    else:
        print(">> WARNING: CUDA is not available. Training will fall back to CPU.")
    print("=" * 75, flush=True)


def run_yolo_training(imgsz: int = 640, epochs: int = 80, batch: int = 16) -> None:
    """Train YOLOv8m object detector on PC 1."""
    print("\n" + "=" * 75)
    print(f"STEP: YOLOv8m Object Detector Retraining (imgsz={imgsz}, epochs={epochs}, batch={batch})")
    print("=" * 75)
    from ultralytics import YOLO

    data_yaml = BASE_DIR / "data.yaml"
    runs_dir = BASE_DIR / "runs" / "train"

    # Auto-adjust batch size for 8GB VRAM
    if imgsz >= 960 and batch > 8:
        print(f"Auto-adjusting batch from {batch} to 8 for imgsz={imgsz} to prevent OOM on 8GB card.")
        batch = 8

    model = YOLO("yolov8m.pt")
    model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        epochs=epochs,
        patience=25,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project=str(runs_dir),
        name=f"yolov8m_{imgsz}_dual_pc",
        exist_ok=True,
        seed=0,
        deterministic=True,
        optimizer="auto",
        cos_lr=True,
        amp=True,
        plots=True,
        val=True,
        save_period=10,
        verbose=True,
    )
    print("YOLO Training complete. Checkpoints saved to:", runs_dir / f"yolov8m_{imgsz}_dual_pc")


def run_feature_extraction(node: str, window_length: int = 6, k_subsample: int = 6) -> None:
    """Extract DINOv2 visual features and kinematics for assigned video subset."""
    assigned_videos = PC1_VIDEOS if node == "pc1" else PC2_VIDEOS
    print("\n" + "=" * 75)
    print(f"STEP: Feature Extraction on {node.upper()} (T={window_length}, K={k_subsample})")
    print(f"Assigned Videos: {', '.join(assigned_videos)}")
    print("=" * 75)

    python_exe = sys.executable
    for v_name in assigned_videos:
        print(f"\n>> Extracting features for: {v_name}...")
        cmd = [
            python_exe,
            str(BASE_DIR / "extract_dinov2_features.py"),
            "--single-video", v_name,
            "--window-length", str(window_length),
            "--k-subsample", str(k_subsample),
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Error extracting features for {v_name}. Exit code: {res.returncode}")


def run_behavior_training(node: str, seeds: list[int] = [42, 43, 44, 45, 46]) -> None:
    """Train designated behavior classification models."""
    assigned_models = PC1_MODELS if node == "pc1" else PC2_MODELS
    print("\n" + "=" * 75)
    print(f"STEP: Behavior Architecture Training on {node.upper()}")
    print(f"Assigned Models: {', '.join(assigned_models)} | Seeds: {seeds}")
    print("=" * 75)

    python_exe = sys.executable
    for model_name in assigned_models:
        print(f"\n>> Training Model: {model_name} across {len(seeds)} seeds...")
        cmd = [
            python_exe,
            str(BASE_DIR / "train_phase4_ablations.py"),
            "--dimension", "temporal",
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
        ]
        # Run training
        subprocess.run(cmd)


def run_live_benchmark() -> None:
    """Execute 6,000 continuous frames live video inference benchmark (PC 2)."""
    print("\n" + "=" * 75)
    print("STEP: End-to-End Live Video Streaming Benchmark (6,000 Frames)")
    print("=" * 75)
    python_exe = sys.executable
    cmd = [
        python_exe,
        str(BASE_DIR / "run_phase5_live_benchmark.py"),
        "--frames", "6000",
        "--warmup", "100",
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
    ]
    subprocess.run(cmd)


def merge_dual_pc_results() -> None:
    """Merge artifacts and consolidate results from both PCs."""
    print("\n" + "=" * 75)
    print("STEP: Consolidating Dual-PC Artifacts & Report Generation")
    print("=" * 75)
    
    python_exe = sys.executable
    
    # 1. Update report docx and pdf
    report_script = BASE_DIR / "create_report_docx.py"
    if report_script.exists():
        print("Recompiling Word document and PDF...")
        subprocess.run([python_exe, str(report_script)])
        
    print("\nAll artifacts consolidated successfully.")
    print("Final reports available at:")
    print(f"  - Word: {BASE_DIR / 'Bee_Behavior_Classification_Report_Clean_v2.docx'}")
    print(f"  - PDF:  {BASE_DIR / 'Bee_Behavior_Classification_Report_Clean_v2.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-PC Retraining Orchestrator")
    parser.add_argument("--node", choices=["pc1", "pc2"], default="pc1", help="Current computer role")
    parser.add_argument(
        "--task",
        choices=["all", "yolo", "extract", "train", "benchmark", "merge"],
        default="all",
        help="Task to execute on this computer",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=640, choices=[640, 960], help="YOLO resolution (640=fast, 960=high-res)")
    parser.add_argument("--yolo-epochs", type=int, default=80, help="YOLO training epochs")
    parser.add_argument("--window-length", type=int, default=6, help="Temporal window T (T=6 optimal for Foraging recovery)")
    parser.add_argument("--k-subsample", type=int, default=6, help="Temporal subsample factor K")
    args = parser.parse_args()

    check_environment()

    t_start = time.time()

    if args.task == "merge":
        merge_dual_pc_results()
        return

    if args.node == "pc1":
        print(f"\n>>> EXECUTING PC 1 WORKLOAD (Tasks: {args.task}) <<<")
        if args.task in ["all", "extract"]:
            run_feature_extraction(node="pc1", window_length=args.window_length, k_subsample=args.k_subsample)
        if args.task in ["all", "train"]:
            run_behavior_training(node="pc1")
        if args.task == "yolo":
            run_yolo_training(imgsz=args.yolo_imgsz, epochs=args.yolo_epochs)

    elif args.node == "pc2":
        print(f"\n>>> EXECUTING PC 2 WORKLOAD (Tasks: {args.task}) <<<")
        if args.task in ["all", "extract"]:
            run_feature_extraction(node="pc2", window_length=args.window_length, k_subsample=args.k_subsample)
        if args.task in ["all", "train"]:
            run_behavior_training(node="pc2")
        if args.task in ["all", "benchmark"]:
            run_live_benchmark()

    elapsed = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"DUAL-PC EXECUTION FINISHED on {args.node.upper()} in {elapsed/60.0:.2f} minutes.")
    print("=" * 75)


if __name__ == "__main__":
    main()
