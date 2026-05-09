# Waits for Stage 1 training to finish (models/book-rec-stage1/ appears),
# then kicks off Stage 2 personal fine-tune.

$RepoRoot   = "D:\repos\Lavely-LLM"
$Python     = "$RepoRoot\.venv\Scripts\python.exe"
$Stage1Out  = "$RepoRoot\models\book-rec-stage1"
$ReadyFile  = "$Stage1Out\adapter_config.json"   # written last by PEFT save

Write-Host "[Stage2-Watcher] Waiting for Stage 1 to finish ($ReadyFile) ..."

while (-not (Test-Path $ReadyFile)) {
    Start-Sleep -Seconds 60
}

Write-Host "[Stage2-Watcher] Stage 1 complete. Starting Stage 2 fine-tune..." -ForegroundColor Cyan

$Stage2Args = @(
    "scripts\train_qlora.py",
    "--model-id",         "$RepoRoot\models\qwen3.5-2b",
    "--data",             "datasets\books\train.jsonl",
    "--val-data",         "datasets\books\val.jsonl",
    "--output-dir",       "models\book-rec-lora",
    "--epochs",           "5",
    "--lr",               "5e-5",
    "--resume-from-lora", "$Stage1Out"
)

Set-Location $RepoRoot
& $Python @Stage2Args

if ($LASTEXITCODE -eq 0) {
    Write-Host "[Stage2-Watcher] Stage 2 complete. LoRA saved to models/book-rec-lora" -ForegroundColor Green
} else {
    Write-Host "[Stage2-Watcher] Stage 2 exited with code $LASTEXITCODE" -ForegroundColor Red
}
