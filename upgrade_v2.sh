#!/bin/bash
#
# Paperless AI Classifier - Upgrade to v2
# ========================================
# Features:
#   - Gemma 4 vision model support (3-4x faster than Ministral)
#   - Learning layer (term mappings, few-shot examples)
#   - Dashboard training interface
#
# Usage:
#   cd /opt/paperless-classifier
#   wget https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main/upgrade_v2.sh
#   chmod +x upgrade_v2.sh
#   ./upgrade_v2.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="${SCRIPT_DIR}/backup_$(date +%Y%m%d_%H%M%S)"
REPO_URL="https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Paperless AI Classifier - Upgrade to v2${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "This upgrade includes:"
echo "  • Gemma 4 vision model support (3-4x faster)"
echo "  • Learning layer for continuous improvement"
echo "  • Dashboard training interface"
echo ""

# Check we're in the right directory
if [[ ! -f "${SCRIPT_DIR}/classifier_api.py" ]]; then
    echo -e "${RED}Error: classifier_api.py not found in current directory${NC}"
    echo "Please run this script from the paperless-classifier directory"
    echo "  cd /opt/paperless-classifier"
    echo "  ./upgrade_v2.sh"
    exit 1
fi

# Check for root
if [[ $EUID -ne 0 ]]; then
    echo -e "${YELLOW}Note: Not running as root. Some operations may fail.${NC}"
fi

echo -e "${YELLOW}Creating backup in: ${BACKUP_DIR}${NC}"
mkdir -p "${BACKUP_DIR}"

# Backup existing files
for file in classifier_api.py ministral.py .env classifier_audit.db; do
    if [[ -f "${SCRIPT_DIR}/${file}" ]]; then
        cp "${SCRIPT_DIR}/${file}" "${BACKUP_DIR}/"
        echo "  Backed up: ${file}"
    fi
done

echo ""
echo -e "${BLUE}Downloading new files...${NC}"

# Download new modules
download_file() {
    local filename="$1"
    local url="${REPO_URL}/${filename}"
    
    echo -n "  ${filename}... "
    if curl -fsSL "${url}" -o "${SCRIPT_DIR}/${filename}" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    elif wget -qO "${SCRIPT_DIR}/${filename}" "${url}" 2>/dev/null; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

# Download files (or copy from local if available)
FILES_TO_DOWNLOAD=(
    "gemma4.py"
    "learning.py"
    "classifier_api_v2.py"
)

DOWNLOAD_FAILED=0
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    if [[ -f "${SCRIPT_DIR}/${file}" ]]; then
        echo -e "  ${file}... ${GREEN}EXISTS${NC}"
    else
        if ! download_file "${file}"; then
            DOWNLOAD_FAILED=1
        fi
    fi
done

# If classifier_api_v2.py was downloaded, rename it
if [[ -f "${SCRIPT_DIR}/classifier_api_v2.py" ]]; then
    mv "${SCRIPT_DIR}/classifier_api_v2.py" "${SCRIPT_DIR}/classifier_api.py"
    echo -e "  Renamed classifier_api_v2.py → classifier_api.py"
fi

echo ""
echo -e "${BLUE}Updating database schema...${NC}"

# Initialize learning tables via Python
python3 << 'PYEOF'
import sqlite3
from pathlib import Path

db_path = Path("/opt/paperless-classifier/classifier_audit.db")
if not db_path.exists():
    db_path = Path("classifier_audit.db")

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Term mappings table
c.execute('''
    CREATE TABLE IF NOT EXISTS term_mappings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        term_type TEXT NOT NULL,
        ai_term TEXT NOT NULL,
        approved_term TEXT NOT NULL,
        times_used INTEGER DEFAULT 1,
        created_at INTEGER DEFAULT (strftime('%s', 'now')),
        last_used INTEGER DEFAULT (strftime('%s', 'now')),
        UNIQUE(term_type, ai_term)
    )
''')

# Classification examples table
c.execute('''
    CREATE TABLE IF NOT EXISTS classification_examples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        document_title TEXT,
        document_type TEXT,
        correspondent TEXT,
        tags TEXT,
        confidence REAL,
        user_verified INTEGER DEFAULT 0,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
    )
''')

# Add confidence column to audit_log if missing
try:
    c.execute('ALTER TABLE audit_log ADD COLUMN confidence REAL')
    print("  Added confidence column to audit_log")
except:
    pass

# Add ai_raw column to audit_log for storing original AI response
try:
    c.execute('ALTER TABLE audit_log ADD COLUMN ai_raw TEXT')
    print("  Added ai_raw column to audit_log")
except:
    pass

# Indexes
c.execute('CREATE INDEX IF NOT EXISTS idx_mappings_lookup ON term_mappings(term_type, ai_term)')
c.execute('CREATE INDEX IF NOT EXISTS idx_mappings_approved ON term_mappings(term_type, approved_term)')
c.execute('CREATE INDEX IF NOT EXISTS idx_examples_verified ON classification_examples(user_verified, confidence DESC)')

conn.commit()
conn.close()
print("  Database schema updated")
PYEOF

echo ""
echo -e "${BLUE}Updating .env configuration...${NC}"

# Add new config options to .env if not present
ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # Add OLLAMA_MODEL if not present or still set to ministral
    if ! grep -q "^OLLAMA_MODEL=" "${ENV_FILE}"; then
        echo "OLLAMA_MODEL=gemma4:e4b" >> "${ENV_FILE}"
        echo "  Added: OLLAMA_MODEL=gemma4:e4b"
    else
        CURRENT_MODEL=$(grep "^OLLAMA_MODEL=" "${ENV_FILE}" | cut -d'=' -f2)
        echo "  Current model: ${CURRENT_MODEL}"
        echo ""
        echo -e "${YELLOW}Do you want to switch to Gemma 4? (recommended, 3-4x faster)${NC}"
        read -p "  Switch to gemma4:e4b? [Y/n]: " switch_model
        if [[ "${switch_model}" != "n" && "${switch_model}" != "N" ]]; then
            sed -i "s|^OLLAMA_MODEL=.*|OLLAMA_MODEL=gemma4:e4b|" "${ENV_FILE}"
            echo -e "  ${GREEN}Switched to gemma4:e4b${NC}"
        fi
    fi
    
    # Add LEARNING_ENABLED
    if ! grep -q "^LEARNING_ENABLED=" "${ENV_FILE}"; then
        echo "LEARNING_ENABLED=true" >> "${ENV_FILE}"
        echo "  Added: LEARNING_ENABLED=true"
    fi
    
    # Add FUZZY_MATCH_THRESHOLD
    if ! grep -q "^FUZZY_MATCH_THRESHOLD=" "${ENV_FILE}"; then
        echo "FUZZY_MATCH_THRESHOLD=0.80" >> "${ENV_FILE}"
        echo "  Added: FUZZY_MATCH_THRESHOLD=0.80"
    fi
else
    echo -e "${YELLOW}  Warning: .env not found, skipping config update${NC}"
fi

echo ""
echo -e "${BLUE}Checking for required model...${NC}"

# Get Ollama URL from .env
OLLAMA_URL=$(grep "^OLLAMA_URL=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 || echo "http://localhost:11434")
TARGET_MODEL=$(grep "^OLLAMA_MODEL=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 || echo "gemma4:e4b")

echo "  Ollama URL: ${OLLAMA_URL}"
echo "  Target model: ${TARGET_MODEL}"

# Check if model is available
if curl -s "${OLLAMA_URL}/api/tags" 2>/dev/null | grep -q "${TARGET_MODEL}"; then
    echo -e "  Model status: ${GREEN}AVAILABLE${NC}"
else
    echo -e "  Model status: ${YELLOW}NOT FOUND${NC}"
    echo ""
    echo "  To download the model, run on your Ollama server:"
    echo -e "    ${BLUE}ollama pull ${TARGET_MODEL}${NC}"
    echo ""
fi

echo ""
echo -e "${BLUE}Setting permissions...${NC}"
chmod +x "${SCRIPT_DIR}"/*.py 2>/dev/null || true
chmod 644 "${SCRIPT_DIR}"/*.py 2>/dev/null || true

# Create logs directory for export feature
mkdir -p "${SCRIPT_DIR}/logs"
chmod 755 "${SCRIPT_DIR}/logs"
echo "  Created logs directory"
echo "  Done"

echo ""
echo -e "${BLUE}Restarting service...${NC}"

if systemctl is-active --quiet paperless-classifier 2>/dev/null; then
    systemctl restart paperless-classifier
    echo -e "  Service restarted: ${GREEN}OK${NC}"
elif systemctl is-active --quiet paperless-ai 2>/dev/null; then
    systemctl restart paperless-ai
    echo -e "  Service restarted: ${GREEN}OK${NC}"
else
    echo -e "  ${YELLOW}No systemd service found - restart manually${NC}"
    echo "  Run: python3 classifier_api.py"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Upgrade Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "What's new:"
echo "  • Gemma 4 support - 3-4x faster than Ministral"
echo "  • Learning layer - improves over time from corrections"
echo "  • Dashboard training - teach the system your preferences"
echo "  • Export logs - debug exports for troubleshooting"
echo "  • Re-analyze button - re-process documents with updated prompt"
echo "  • Reset learning - clear examples/mappings for fresh start"
echo ""
echo "Dashboard: http://<your-ip>:8001/dashboard"
echo ""
echo "Dashboard tabs:"
echo "  • Status   - queue, recent activity, re-analyze button"
echo "  • Training - review/correct classifications"
echo "  • Mappings - view/edit learned terms, reset learning"
echo "  • Export   - generate debug logs for troubleshooting"
echo "  • Config   - settings"
echo ""
echo "Backup location: ${BACKUP_DIR}"
echo ""
echo "To rollback: cp ${BACKUP_DIR}/* ${SCRIPT_DIR}/"
echo ""
