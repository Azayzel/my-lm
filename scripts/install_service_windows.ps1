#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install (or remove) the BookMind Open Library ingest crawler as a
    Windows Scheduled Task.

.DESCRIPTION
    Creates a Task Scheduler task called "BookMind-OLIngest" that runs
    ol_ingest.py once per day at 03:00 using the repo's virtual environment.

    The task runs with the SYSTEM account so it works even when no user
    is logged in, and is set to run whether on battery or AC power.

.PARAMETER Uninstall
    Remove the scheduled task instead of creating it.

.PARAMETER RunNow
    Trigger the task immediately after registering it.

.PARAMETER SleepHours
    Hours between crawl passes if you prefer daemon mode (default: 0 = use
    Task Scheduler for scheduling, not internal daemon loop).

.EXAMPLE
    # Install (run from repo root as Administrator)
    powershell -ExecutionPolicy Bypass -File scripts\install_service_windows.ps1

.EXAMPLE
    # Remove the task
    powershell -ExecutionPolicy Bypass -File scripts\install_service_windows.ps1 -Uninstall
#>

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$RunNow,
    [int]$SleepHours = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TaskName   = "BookMind-OLIngest"
$RepoRoot   = (Resolve-Path "$PSScriptRoot\..").Path
$PythonExe  = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Script     = Join-Path $RepoRoot "scripts\ol_ingest.py"
$LogDir     = Join-Path $RepoRoot "logs"
$LogFile    = Join-Path $LogDir   "ol_ingest.log"

# ─── Uninstall ────────────────────────────────────────────────────────────────

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[BookMind] Scheduled task '$TaskName' removed." -ForegroundColor Green
    } else {
        Write-Host "[BookMind] Task '$TaskName' not found — nothing to remove."
    }
    exit 0
}

# ─── Pre-flight checks ────────────────────────────────────────────────────────

if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment not found at: $PythonExe`nRun setup.bat first."
}
if (-not (Test-Path $Script)) {
    Write-Error "Ingest script not found at: $Script"
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# ─── Build the command ────────────────────────────────────────────────────────

$Arguments = "`"$Script`""
if ($SleepHours -gt 0) {
    $Arguments += " --daemon --sleep-hours $SleepHours"
}

# Redirect stdout+stderr to log file via cmd.exe wrapper
$CmdWrapper  = "cmd.exe"
$CmdArgs     = "/C `"`"$PythonExe`" $Arguments >> `"$LogFile`" 2>&1`""

# ─── Register the task ───────────────────────────────────────────────────────

$Action  = New-ScheduledTaskAction `
    -Execute  $CmdWrapper `
    -Argument $CmdArgs `
    -WorkingDirectory $RepoRoot

# Daily at 03:00; if the task was missed (machine off) run it within 1 hour
$Trigger = New-ScheduledTaskTrigger -Daily -At "03:00"

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit      (New-TimeSpan -Hours 6) `
    -StartWhenAvailable      `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances       IgnoreNew `
    -DisallowHardTerminate:$false

# Run as SYSTEM — no password needed, works headless
$Principal = New-ScheduledTaskPrincipal `
    -UserId    "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel  Highest

# Remove existing task if present
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[BookMind] Replaced existing task '$TaskName'."
}

Register-ScheduledTask `
    -TaskName  $TaskName `
    -Action    $Action `
    -Trigger   $Trigger `
    -Settings  $Settings `
    -Principal $Principal `
    -Description "BookMind: crawl Open Library and ingest books into MongoDB Atlas." | Out-Null

Write-Host ""
Write-Host "[BookMind] Scheduled task '$TaskName' registered successfully." -ForegroundColor Green
Write-Host "  Schedule : Daily at 03:00 (runs on next wake if missed)"
Write-Host "  Command  : $PythonExe $Arguments"
Write-Host "  Log file : $LogFile"
Write-Host ""

if ($RunNow) {
    Write-Host "[BookMind] Starting task immediately..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "[BookMind] Task started. Tail the log:"
    Write-Host "  Get-Content '$LogFile' -Wait"
}

Write-Host "To remove this task later:"
Write-Host "  powershell -File scripts\install_service_windows.ps1 -Uninstall"
