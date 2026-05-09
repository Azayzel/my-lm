<#
.SYNOPSIS
    Install / uninstall a Windows service that runs the OpenLibrary ingest
    daemon (scripts/ol_ingest.py --daemon) and auto-restarts on crash.

.DESCRIPTION
    Uses NSSM (the Non-Sucking Service Manager, https://nssm.cc) because plain
    sc.exe can't host a Python script as a service. The script:
      - Resolves the project root, .venv Python, and ol_ingest.py paths.
      - Registers the service with appropriate working dir, log files,
        startup type, and restart policy.
      - Streams stdout/stderr to logs/ol_ingest.{out,err}.log with rotation.

    Run from an elevated PowerShell prompt.

.PARAMETER Action
    install   — register the service (default)
    uninstall — stop and remove the service
    start     — start the service
    stop      — stop the service
    status    — print current state

.PARAMETER ServiceName
    Service name (default: BookMindOLIngest)

.PARAMETER NssmPath
    Path to nssm.exe. If omitted, looks on PATH then aborts with install
    instructions.

.PARAMETER SleepHours
    Hours between daemon passes (default 12).

.PARAMETER PerSubject
    Max new books per subject per pass (default 200).

.PARAMETER RefreshAfterDays
    Re-crawl subjects whose state is older than this (default 30).

.EXAMPLE
    pwsh -File scripts\install_ol_ingest_service.ps1 -Action install

.EXAMPLE
    pwsh -File scripts\install_ol_ingest_service.ps1 -Action status
#>

[CmdletBinding()]
param(
    [ValidateSet("install", "uninstall", "start", "stop", "status")]
    [string]$Action = "install",

    [string]$ServiceName = "BookMindOLIngest",

    [string]$NssmPath = "",

    [double]$SleepHours = 12,

    [int]$PerSubject = 200,

    [double]$RefreshAfterDays = 30
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ──────────────────────────────────────────────────────────
$RepoRoot   = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Script     = Join-Path $RepoRoot "scripts\ol_ingest.py"
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$LogsDir    = Join-Path $RepoRoot "logs"

if (-not (Test-Path $Script))     { throw "Cannot find $Script" }
if (-not (Test-Path $VenvPython)) { throw "Cannot find $VenvPython — create .venv first" }

if (-not $NssmPath) {
    $found = Get-Command nssm.exe -ErrorAction SilentlyContinue
    if ($found) { $NssmPath = $found.Source }
}

function Require-Nssm {
    if (-not $NssmPath -or -not (Test-Path $NssmPath)) {
        Write-Error @"
nssm.exe not found.

Install options:
  winget install nssm                         # via winget
  choco install nssm                          # via chocolatey
  scoop install nssm                          # via scoop
  Or download zip:  https://nssm.cc/download

Then re-run this script, or pass -NssmPath C:\path\to\nssm.exe
"@
        exit 1
    }
}

function Require-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $pr = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $pr.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Error "This action requires an elevated PowerShell (Run as Administrator)."
        exit 1
    }
}

function Service-Exists {
    return [bool](Get-Service -Name $ServiceName -ErrorAction SilentlyContinue)
}

# ── Actions ────────────────────────────────────────────────────────────────
switch ($Action) {

    "install" {
        Require-Admin
        Require-Nssm

        if (Service-Exists) {
            Write-Host "Service '$ServiceName' already exists. Removing first…"
            & $NssmPath stop   $ServiceName confirm | Out-Null
            & $NssmPath remove $ServiceName confirm | Out-Null
        }

        New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

        $args = @(
            $Script,
            "--daemon",
            "--sleep-hours",       $SleepHours,
            "--per-subject",       $PerSubject,
            "--refresh-after-days", $RefreshAfterDays
        ) -join " "

        Write-Host "Installing '$ServiceName'…"
        & $NssmPath install $ServiceName $VenvPython $args | Out-Null

        & $NssmPath set $ServiceName AppDirectory   $RepoRoot                                        | Out-Null
        & $NssmPath set $ServiceName DisplayName    "BookMind — OpenLibrary Ingest"                  | Out-Null
        & $NssmPath set $ServiceName Description    "Continuous Open Library → MongoDB book ingest." | Out-Null
        & $NssmPath set $ServiceName Start          SERVICE_AUTO_START                               | Out-Null

        # Restart on crash with backoff (5s → 60s → 300s)
        & $NssmPath set $ServiceName AppExit Default Restart    | Out-Null
        & $NssmPath set $ServiceName AppRestartDelay 5000       | Out-Null
        & $NssmPath set $ServiceName AppThrottle     30000      | Out-Null

        # Log rotation: 10 MB per file, keep on rotate
        & $NssmPath set $ServiceName AppStdout       (Join-Path $LogsDir "ol_ingest.out.log") | Out-Null
        & $NssmPath set $ServiceName AppStderr       (Join-Path $LogsDir "ol_ingest.err.log") | Out-Null
        & $NssmPath set $ServiceName AppRotateFiles  1                                       | Out-Null
        & $NssmPath set $ServiceName AppRotateOnline 1                                       | Out-Null
        & $NssmPath set $ServiceName AppRotateBytes  10485760                                | Out-Null

        # Graceful shutdown — give the daemon 30s after Ctrl+Break
        & $NssmPath set $ServiceName AppStopMethodSkip 0     | Out-Null
        & $NssmPath set $ServiceName AppStopMethodConsole 30000 | Out-Null
        & $NssmPath set $ServiceName AppStopMethodWindow  10000 | Out-Null
        & $NssmPath set $ServiceName AppStopMethodThreads 10000 | Out-Null

        Write-Host "Installed. Start with:  pwsh -File $PSCommandPath -Action start"
    }

    "uninstall" {
        Require-Admin
        Require-Nssm
        if (-not (Service-Exists)) { Write-Host "Service '$ServiceName' not found."; return }
        & $NssmPath stop   $ServiceName confirm | Out-Null
        & $NssmPath remove $ServiceName confirm | Out-Null
        Write-Host "Removed '$ServiceName'."
    }

    "start" {
        Require-Admin
        Start-Service -Name $ServiceName
        Get-Service -Name $ServiceName | Format-Table -AutoSize
    }

    "stop" {
        Require-Admin
        Stop-Service -Name $ServiceName
        Get-Service -Name $ServiceName | Format-Table -AutoSize
    }

    "status" {
        if (Service-Exists) {
            Get-Service -Name $ServiceName | Format-Table -AutoSize
            $heartbeat = Join-Path $RepoRoot "data\ol_ingest_status.json"
            if (Test-Path $heartbeat) {
                Write-Host "`nHeartbeat ($heartbeat):"
                Get-Content $heartbeat
            } else {
                Write-Host "`nNo heartbeat file yet at $heartbeat"
            }
        } else {
            Write-Host "Service '$ServiceName' is not installed."
        }
    }
}
