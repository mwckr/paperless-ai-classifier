#!/usr/bin/env bash
# Paperless AI Classifier — update the git checkout and restart.
#   sudo ./update.sh              pull, install requirements, check settings, restart
#   sudo ./update.sh --no-restart
# Also started by the dashboard's Update button (then it logs to logs/update.log).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE=paperless-classifier
VENV="$DIR/venv"
RESTART=1
[[ "${1:-}" == "--no-restart" ]] && RESTART=0

if [[ -n "${PCLS_UPDATE_LOG:-}" ]]; then
  exec >>"$PCLS_UPDATE_LOG" 2>&1
fi
say() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

cd "$DIR"
[[ -x "$VENV/bin/python" ]] || { echo "No venv in $VENV. Run install.sh first."; exit 1; }

before=$(git rev-parse --short HEAD 2>/dev/null || echo "?")
say "Pulling updates (current: $before)"
git pull --ff-only
after=$(git rev-parse --short HEAD 2>/dev/null || echo "?")

say "Installing requirements"
"$VENV/bin/pip" install -q -r requirements.txt

say "Checking settings"
if [[ -t 0 ]]; then
  "$VENV/bin/python" -m app.setup --check
else
  "$VENV/bin/python" -m app.setup --check --non-interactive || echo "Open the dashboard Config tab to complete the settings."
fi

if [[ $RESTART -eq 1 ]]; then
  say "Restarting $SERVICE"
  systemctl restart "$SERVICE"
  sleep 2
  if systemctl is-active --quiet "$SERVICE"; then
    say "Updated $before -> $after, service running"
  else
    echo "Service failed to start. Check: journalctl -u $SERVICE -n 50"
    exit 1
  fi
else
  say "Updated $before -> $after (not restarted)"
fi
