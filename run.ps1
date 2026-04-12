# run.ps1 — Correct launcher for AI Email Agent
# Fixes the "Email_Env vs EmailEnv" venv path mismatch.
# Usage: .\run.ps1
#   or:  .\run.ps1 inference
#   or:  .\run.ps1 test

param([string]$Mode = "app")

$PY = "venv\Scripts\python.exe"

switch ($Mode) {
    "app"  { & $PY -m streamlit run app.py --server.port=8501 --server.headless=false }
    "run"  { & $PY inference.py }
    "test" { & $PY -m unittest discover tests -v }
    default { Write-Host "Usage: .\run.ps1 [app|run|test]" }
}
