@echo off
TITLE Honeybee Retraining - PC 1 (RTX 2070 Super)
echo =======================================================================
echo          HONEYBEE CLASSIFICATION - DUAL-PC RETRAINING (PC 1)
echo =======================================================================
echo Hardware: NVIDIA GeForce RTX 2070 Super (8GB VRAM)
echo Assigned Tasks:
echo   1. DINOv2 Feature Extraction for Videos: 20230609b-def, yt5, yt8
echo   2. Architecture Retraining: TemporalConvNet (TCN) + BiGRU (5 seeds)
echo   3. Optional: YOLOv8m Object Detector Retraining
echo =======================================================================
echo.

IF EXIST .venv\Scripts\python.exe (
    set PYTHON_EXEC=.venv\Scripts\python.exe
) ELSE (
    set PYTHON_EXEC=python
)

echo Using Python: %PYTHON_EXEC%
echo.
echo Select Run Mode:
echo   [1] Fast ViT-Behavior Pipeline (Extract assigned videos + Train TCN & BiGRU) ~8 mins
echo   [2] Full Retrain with YOLO Detector (640px) ~2.5 hrs
echo   [3] High-Res YOLO Detector Only (960px) ~15 hrs
echo   [4] Custom / Manual CLI
echo.
set /p choice="Enter choice [1-4] (default=1): "
if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    echo Starting Fast ViT-Behavior Pipeline on PC 1...
    %PYTHON_EXEC% train_dual_pc.py --node pc1 --task all
)
if "%choice%"=="2" (
    echo Starting Full Retrain with YOLO 640px on PC 1...
    %PYTHON_EXEC% train_dual_pc.py --node pc1 --task all
    %PYTHON_EXEC% train_dual_pc.py --node pc1 --task yolo --yolo-imgsz 640
)
if "%choice%"=="3" (
    echo Starting YOLO 960px Retraining on PC 1...
    %PYTHON_EXEC% train_dual_pc.py --node pc1 --task yolo --yolo-imgsz 960
)
if "%choice%"=="4" (
    cmd /k
)

echo.
echo =======================================================================
echo PC 1 Tasks Completed! Sync output files to PC 2 or shared storage.
echo =======================================================================
pause
