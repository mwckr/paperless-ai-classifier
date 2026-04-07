#!/bin/bash
# deploy.sh — Copy updated files from git repo to /opt/paperless-classifier/ and restart service
# Usage: cd /root/paperless-ai-classifier && git pull && ./deploy.sh

set -e

INSTALL_DIR="/opt/paperless-classifier"
SERVICE_NAME="paperless-classifier"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Paperless AI Classifier — Deploy ===${NC}"

# Verify install dir exists
if [ ! -d "$INSTALL_DIR" ]; then
    echo -e "${RED}Error: $INSTALL_DIR not found. Run install.sh first.${NC}"
    exit 1
fi

# Backup current files
BACKUP_DIR="${INSTALL_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
echo -e "${YELLOW}Backing up current files to ${BACKUP_DIR}${NC}"
mkdir -p "$BACKUP_DIR"
for f in classifier_api_v2.py gemma4.py learning.py; do
    [ -f "${INSTALL_DIR}/${f}" ] && cp "${INSTALL_DIR}/${f}" "${BACKUP_DIR}/"
done

# Copy application files
echo "Copying application files..."
cp "${SCRIPT_DIR}/classifier_api_v2.py" "$INSTALL_DIR/"
cp "${SCRIPT_DIR}/gemma4.py" "$INSTALL_DIR/"
cp "${SCRIPT_DIR}/learning.py" "$INSTALL_DIR/"

# Clear Python cache
echo "Clearing __pycache__..."
rm -rf "${INSTALL_DIR}/__pycache__"

# Restart service
echo "Restarting ${SERVICE_NAME}..."
systemctl restart "$SERVICE_NAME"

# Wait and check status
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${GREEN}✓ Service restarted successfully${NC}"
    echo -e "${GREEN}  Dashboard: http://$(hostname -I | awk '{print $1}'):8001${NC}"
else
    echo -e "${RED}✗ Service failed to start. Check: journalctl -u ${SERVICE_NAME} -n 20${NC}"
    exit 1
fi
