# ============================================================
# YOLOv8m retrain - improved config
#
# Improvements over the baseline (mAP50 0.9367 / mAP50-95 0.7078):
#   1. imgsz 640 -> 960
#      Bees are ~63x79 px in the 1920x1080 source. Letterboxing
#      to 640x360 shrinks them to ~21x26 px, which is why recall
#      is the weakest detection metric. At 960 they are ~32x40 px.
#   2. epochs 80 -> 150, patience 10 -> 50
#      The baseline stopped at epoch 46 of 80 (early stopping)
#      with mAP still fluctuating, so it was under-trained.
#   3. batch 32 -> 8
#      imgsz=960 at batch=16 peaked at 7831/8188 MiB and OOM'd
#      on this 8 GB card, so batch is halved again.
#
# Everything stays inside the project folder.
# ============================================================

from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

DATA_YAML = BASE_DIR / "data.yaml"

RUNS_DIR = BASE_DIR / "runs" / "train"

BASE_WEIGHTS = "yolov8m.pt"


def main() -> None:
    model = YOLO(BASE_WEIGHTS)

    model.train(
        data=str(DATA_YAML),

        # Resolution: the single biggest lever here.
        imgsz=960,

        # Disable premature early stopping.
        epochs=150,
        patience=50,

        # Small batch: imgsz=960 needs ~4 GB per 8 images on an
        # 8 GB card. batch=16 peaked at 7831 MiB and OOM'd.
        batch=8,

        device=0,
        workers=2,

        # Keep all output in the project folder.
        project=str(RUNS_DIR),
        name="yolov8m_960",
        exist_ok=True,

        # Match the baseline run for a fair comparison.
        seed=0,
        deterministic=True,

        optimizer="auto",
        cos_lr=True,
        close_mosaic=15,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,

        amp=True,
        plots=True,
        val=True,
        save_period=10,
        verbose=True,
    )


if __name__ == "__main__":
    main()