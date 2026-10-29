# Honeybee Retraining Launcher - PC 2 (PowerShell)
# Hardware: NVIDIA GeForce RTX 2070 Super (8GB VRAM)

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "         HONEYBEE CLASSIFICATION - DUAL-PC RETRAINING (PC 2)" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "Hardware: NVIDIA GeForce RTX 2070 Super (8GB VRAM)"
Write-Host "Assigned Workload:"
Write-Host "  1. Video Extraction: 20230711a-fan, 20230711b-fan (~12,003 frames)"
Write-Host "  2. Temporal Models:  Fused BiLSTM + Causal UniLSTM (5 seeds)"
Write-Host "  3. Live Benchmark:   6,000-Frame Streaming Latency & MOTA Profiling"
Write-Host "======================================================================="

$PythonExe = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
Write-Host "Using Python: $PythonExe`n"

Write-Host "Select Execution Mode:"
Write-Host "  [1] Run All PC 2 Tasks (Extract + Train BiLSTM/UniLSTM + 6,000-Frame Live Test) ~10 mins"
Write-Host "  [2] Extract Features Only (711a + 711b) ~4 mins"
Write-Host "  [3] Train Models Only (BiLSTM + UniLSTM) ~3 mins"
Write-Host "  [4] Run 6,000-Frame Live Streaming Benchmark Only ~3 mins"
Write-Host "  [5] Merge Results from Both PCs (Recompiles Word + PDF Reports)"

$choice = Read-Host "Enter selection [1-5] (default=1)"
if (-not $choice) { $choice = "1" }

switch ($choice) {
    "1" { & $PythonExe train_dual_pc.py --node pc2 --task all }
    "2" { & $PythonExe train_dual_pc.py --node pc2 --task extract }
    "3" { & $PythonExe train_dual_pc.py --node pc2 --task train }
    "4" { & $PythonExe train_dual_pc.py --node pc2 --task benchmark }
    "5" { & $PythonExe train_dual_pc.py --task merge }
    default { & $PythonExe train_dual_pc.py --node pc2 --task all }
}

Write-Host "`nPC 2 Workload Completed!" -ForegroundColor Green
