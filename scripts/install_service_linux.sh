#!/usr/bin/env bash
# Install (or remove) the BookMind Open Library ingest service on Linux
# using systemd.  Creates both a .service and a .timer unit so the crawl
# runs daily without needing --daemon mode in the script.
#
# Usage:
#   sudo bash scripts/install_service_linux.sh           # install
#   sudo bash scripts/install_service_linux.sh --remove  # uninstall
#   sudo bash scripts/install_service_linux.sh --run-now # install + start immediately
#
# The service runs as the current user ($SUDO_USER or $USER).

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
INGEST_SCRIPT="$REPO_ROOT/scripts/ol_ingest.py"
LOG_DIR="$REPO_ROOT/logs"
LOG_FILE="$LOG_DIR/ol_ingest.log"

SERVICE_NAME="bookmind-ol-ingest"
SYSTEMD_DIR="/etc/systemd/system"
SERVICE_FILE="$SYSTEMD_DIR/$SERVICE_NAME.service"
TIMER_FILE="$SYSTEMD_DIR/$SERVICE_NAME.timer"

# Run as the real user (not root) when called via sudo
RUN_USER="${SUDO_USER:-$USER}"
RUN_GROUP="$(id -gn "$RUN_USER")"

# ── Helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\033[32m[BookMind]\033[0m $*"; }
warn()  { echo -e "\033[33m[BookMind]\033[0m $*"; }
error() { echo -e "\033[31m[BookMind] ERROR:\033[0m $*" >&2; exit 1; }

require_root() {
    [[ "$(id -u)" -eq 0 ]] || error "Please run with sudo."
}

# ── Remove ─────────────────────────────────────────────────────────────────────
do_remove() {
    require_root
    for unit in "$SERVICE_NAME.timer" "$SERVICE_NAME.service"; do
        if systemctl is-active --quiet "$unit" 2>/dev/null; then
            systemctl stop "$unit"
            info "Stopped $unit"
        fi
        if systemctl is-enabled --quiet "$unit" 2>/dev/null; then
            systemctl disable "$unit"
            info "Disabled $unit"
        fi
    done
    rm -f "$TIMER_FILE" "$SERVICE_FILE"
    systemctl daemon-reload
    info "Service and timer removed."
    exit 0
}

# ── Parse args ─────────────────────────────────────────────────────────────────
REMOVE=false
RUN_NOW=false
for arg in "$@"; do
    case "$arg" in
        --remove)  REMOVE=true ;;
        --run-now) RUN_NOW=true ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

$REMOVE && do_remove
require_root

# ── Pre-flight checks ──────────────────────────────────────────────────────────
[[ -f "$PYTHON" ]]        || error "Python venv not found at $PYTHON — run setup.sh first."
[[ -f "$INGEST_SCRIPT" ]] || error "Ingest script not found at $INGEST_SCRIPT"

mkdir -p "$LOG_DIR"
chown "$RUN_USER:$RUN_GROUP" "$LOG_DIR"

# ── Write the .service unit ────────────────────────────────────────────────────
cat > "$SERVICE_FILE" << EOF
[Unit]
Description=BookMind Open Library ingest crawler
Documentation=https://github.com/Azayzel/my-lm
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
Group=$RUN_GROUP
WorkingDirectory=$REPO_ROOT
ExecStart=$PYTHON $INGEST_SCRIPT
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE
# Restart on failure, back off to avoid hammering OL on repeated errors
Restart=on-failure
RestartSec=60
# Give the crawl up to 6 hours before it's killed
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

# ── Write the .timer unit (runs daily at 03:00) ────────────────────────────────
cat > "$TIMER_FILE" << EOF
[Unit]
Description=Run BookMind OL ingest daily
Requires=$SERVICE_NAME.service

[Timer]
# Run at 03:00 local time every day
OnCalendar=*-*-* 03:00:00
# If the system was off at 03:00, run shortly after boot
Persistent=true
# Randomise start within ±10 min to avoid thundering-herd if many hosts
RandomizedDelaySec=600

[Install]
WantedBy=timers.target
EOF

# ── Enable & start the timer ───────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable "$SERVICE_NAME.timer"
systemctl start  "$SERVICE_NAME.timer"

info ""
info "Service installed successfully."
info "  Service unit : $SERVICE_FILE"
info "  Timer unit   : $TIMER_FILE"
info "  Runs as user : $RUN_USER"
info "  Schedule     : daily at 03:00 (persistent — runs on boot if missed)"
info "  Log file     : $LOG_FILE"
info ""

if $RUN_NOW; then
    info "Starting crawler now (one-shot)..."
    systemctl start "$SERVICE_NAME.service"
    info "Running. Tail the log with:"
    info "  journalctl -fu $SERVICE_NAME.service"
    info "  # or: tail -f $LOG_FILE"
fi

echo ""
echo "Useful commands:"
echo "  sudo systemctl status  $SERVICE_NAME.timer"
echo "  sudo systemctl start   $SERVICE_NAME.service   # run now"
echo "  sudo journalctl -fu    $SERVICE_NAME.service   # follow log"
echo "  sudo bash $SCRIPT_DIR/install_service_linux.sh --remove"
