@echo off
TITLE Honeybee Retraining - PC 2 (RTX 2070 Super)
echo =======================================================================
echo          HONEYBEE CLASSIFICATION - DUAL-PC RETRAINING (PC 2)
echo =======================================================================
echo Hardware: NVIDIA GeForce RTX 2070 Super (8GB VRAM)
echo Assigned Tasks:
echo   1. DINOv2 Feature Extraction for Videos: 20230711a-fan, 20230711b-fan
echo   2. Architecture Retraining: BiLSTM + Causal UniLSTM (5 seeds)
echo   3. 6,000-Frame Live Streaming & FPS Throughput Benchmark
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
echo   [1] Run All PC 2 Tasks (Extract + Train BiLSTM/UniLSTM + 6,000-frame Live Test) ~10 mins
echo   [2] Extract Features Only (Videos 711a + 711b) ~4 mins
echo   [3] Train Models Only (BiLSTM + UniLSTM) ~3 mins
echo   [4] Run 6,000-Frame Live Video Benchmark Only ~3 mins
echo   [5] Merge Results from Both PCs (Generates Final Word + PDF Report)
echo.
set /p choice="Enter choice [1-5] (default=1): "
if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    echo Starting Full PC 2 Pipeline...
    %PYTHON_EXEC% train_dual_pc.py --node pc2 --task all
)
if "%choice%"=="2" (
    echo Extracting Features for 711a and 711b...
    %PYTHON_EXEC% train_dual_pc.py --node pc2 --task extract
)
if "%choice%"=="3" (
    echo Training BiLSTM and UniLSTM Models...
    %PYTHON_EXEC% train_dual_pc.py --node pc2 --task train
)
if "%choice%"=="4" (
    echo Running Live Streaming Benchmark...
    %PYTHON_EXEC% train_dual_pc.py --node pc2 --task benchmark
)
if "%choice%"=="5" (
    echo Merging Results and Compiling Clean Document...
    %PYTHON_EXEC% train_dual_pc.py --task merge
)

echo.
echo =======================================================================
echo PC 2 Tasks Completed!
echo =======================================================================
pause
