# Honeybee Retraining Launcher - PC 1 (PowerShell)
# Hardware: NVIDIA GeForce RTX 2070 Super (8GB VRAM)

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "         HONEYBEE CLASSIFICATION - DUAL-PC RETRAINING (PC 1)" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Hardware: NVIDIA GeForce RTX 2070 Super (8GB VRAM)"
Write-Host "Assigned Workload:"
Write-Host "  1. Video Extraction: 20230609b-def, yt5, yt8 (~9,950 frames)"
Write-Host "  2. Temporal Models:  TemporalConvNet (TCN) + BiGRU (5 seeds)"
Write-Host "  3. Optional Task:    YOLOv8m Object Detector Retraining"
Write-Host "======================================================================="

$PythonExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
Write-Host "Using Python: $PythonExe`n"

Write-Host "Select Execution Mode:"
Write-Host "  [1] Fast ViT-Behavior Pipeline (Extract videos + Train TCN & BiGRU) ~8 mins"
Write-Host "  [2] Full Retrain with YOLO 640px ~2.5 hrs"
Write-Host "  [3] High-Res YOLO 960px Retraining ~15 hrs"
Write-Host "  [4] Extract Assigned Videos Only ~3.5 mins"
Write-Host "  [5] Train Assigned Models Only ~4 mins"

$choice = Read-Host "Enter selection [1-5] (default=1)"
if (-not $choice) { $choice = "1" }

switch ($choice) {
    "1" { & $PythonExe train_dual_pc.py --node pc1 --task all }
    "2" {
        & $PythonExe train_dual_pc.py --node pc1 --task all
        & $PythonExe train_dual_pc.py --node pc1 --task yolo --yolo-imgsz 640
    }
    "3" { & $PythonExe train_dual_pc.py --node pc1 --task yolo --yolo-imgsz 960 }
    "4" { & $PythonExe train_dual_pc.py --node pc1 --task extract }
    "5" { & $PythonExe train_dual_pc.py --node pc1 --task train }
    default { & $PythonExe train_dual_pc.py --node pc1 --task all }
}

Write-Host "`nPC 1 Workload Completed!" -ForegroundColor Green
