# ============================================================
# Central Project Configuration & Path Resolver
# All paths resolved relative to repository root.
# ============================================================

from __future__ import annotations

import os
import random
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent

# Datasets
DATASETS_ROOT = BASE_DIR / "Datasets"
TRACKING_DIR = DATASETS_ROOT / "tracking_and_behavior"
DETECTION_DIR = DATASETS_ROOT / "detection"
YOLO_DIR = BASE_DIR / "dataset_yolo"
DATA_YAML = BASE_DIR / "data.yaml"

# Artifacts & Output Dirs
DINOV2_FEATURES_DIR = BASE_DIR / "dinov2_features"
MODELS_DIR = BASE_DIR / "vit_lstm_models"
RESULTS_DIR = BASE_DIR / "results"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
VIDEOS_DIR = BASE_DIR / "videos"

CLASS_NAMES = [
    "Foraging",
    "Defense",
    "Fanning",
    "Washboarding",
    "Other",
]

VIDEOS = {
    "20230609b-def": {
        "video": "20230609b-def.mp4",
        "ann": "tracks_and_behavior_classes_20230609b-def.txt",
        "zone": "entrance_zone_20230609b-def.txt",
        "w": 1920,
        "h": 1080,
    },
    "20230711a-fan": {
        "video": "20230711a-fan.mp4",
        "ann": "tracks_and_behavior_classes_20230711a-fan.txt",
        "zone": "entrance_zone_20230711a-fan.txt",
        "w": 1920,
        "h": 1080,
    },
    "20230711b-fan": {
        "video": "20230711b-fan.mp4",
        "ann": "tracks_and_behavior_classes_20230711b-fan.txt",
        "zone": "entrance_zone_20230711b-fan.txt",
        "w": 1920,
        "h": 1080,
    },
    "yt5": {
        "video": "yt5.mp4",
        "ann": "tracks_and_behavior_classes__yt5.txt",
        "zone": "entrance_zone_yt5.txt",
        "w": 1280,
        "h": 720,
    },
    "yt8": {
        "video": "yt8.mp4",
        "ann": "tracks_and_behavior_classes_yt8.txt",
        "zone": "entrance_zone_yt8.txt",
        "w": 1920,
        "h": 1080,
    },
}


def seed_everything(seed: int = 42) -> None:
    """Ensure full reproducibility across random, numpy, and torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_arg: str | None = None) -> torch.device:
    """Return PyTorch device, prioritizing CUDA if available."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
