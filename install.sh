#!/usr/bin/env bash
# Paperless AI Classifier — install
#
# Run from the git checkout you want to serve from (recommended: /opt/paperless-classifier):
#   sudo ./install.sh                 guided: asks for missing settings, tests connections
#   sudo ./install.sh --non-interactive   for scripted installs; .env must already hold the credentials
#
# Idempotent: re-running updates packages, the systemd unit and asks only for what is missing.
# Later updates: sudo ./update.sh  (or the Update button in the dashboard).
set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE=paperless-classifier
VENV="$INSTALL_DIR/venv"
NONINTERACTIVE=0
[[ "${1:-}" == "--non-interactive" ]] && NONINTERACTIVE=1

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Run as root: sudo ./install.sh"
[[ -f "$INSTALL_DIR/app/main.py" ]] || die "app/ not found next to install.sh. Run this from the git checkout."

# --- 1. system packages -------------------------------------------------------------
say "Installing system packages (python3, venv, git, poppler-utils)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip git poppler-utils >/dev/null

# --- 2. import settings from an old copy-based installation -----------------------------
import_from() {
  local old="$1"
  [[ -d "$old" && "$old" != "$INSTALL_DIR" ]] || return 0
  [[ -f "$old/.env" || -f "$old/classifier_audit.db" ]] || return 0
  [[ -f "$INSTALL_DIR/.env" ]] && return 0
  say "Found an earlier installation in $old"
  local answer=y
  if [[ $NONINTERACTIVE -eq 0 ]]; then
    read -r -p "Import its .env and audit database? [Y/n] " answer
    answer=${answer:-y}
  fi
  if [[ "$answer" =~ ^[Yy] ]]; then
    [[ -f "$old/.env" ]] && cp "$old/.env" "$INSTALL_DIR/.env" && say "Imported .env"
    [[ -f "$old/classifier_audit.db" && ! -f "$INSTALL_DIR/classifier_audit.db" ]] \
      && cp "$old/classifier_audit.db" "$INSTALL_DIR/classifier_audit.db" && say "Imported classifier_audit.db"
  fi
}
import_from "${INSTALL_DIR}.old"
import_from "/opt/paperless-classifier"

# --- 3. python environment ------------------------------------------------------------
say "Creating virtual environment in $VENV"
[[ -d "$VENV" ]] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$INSTALL_DIR/requirements.txt"

# --- 4. guided settings ----------------------------------------------------------------
say "Checking settings"
if [[ $NONINTERACTIVE -eq 1 ]]; then
  (cd "$INSTALL_DIR" && "$VENV/bin/python" -m app.setup --non-interactive) || die "Settings incomplete. Edit $INSTALL_DIR/.env and re-run."
else
  (cd "$INSTALL_DIR" && "$VENV/bin/python" -m app.setup)
fi
chmod 600 "$INSTALL_DIR/.env" 2>/dev/null || true

# --- 5. systemd ----------------------------------------------------------------------------
say "Installing systemd unit $SERVICE"
cat > "/etc/systemd/system/$SERVICE.service" <<EOF
[Unit]
Description=Paperless AI Classifier
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV/bin/python -m app
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null
systemctl restart "$SERVICE"

# --- 6. done ----------------------------------------------------------------------------------
PORT=$(grep -E '^API_PORT=' "$INSTALL_DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d ' ' || true)
PORT=${PORT:-8001}
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
sleep 2
if systemctl is-active --quiet "$SERVICE"; then
  say "Service running. Dashboard: http://${IP:-localhost}:${PORT}/"
else
  warn "Service did not start. Check: journalctl -u $SERVICE -n 50"
fi
echo "Config: $INSTALL_DIR/.env   Logs: journalctl -u $SERVICE -f   Update: sudo $INSTALL_DIR/update.sh"
