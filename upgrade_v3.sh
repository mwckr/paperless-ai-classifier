#!/bin/bash
#
# Paperless AI Classifier - Upgrade to v3
# ========================================
# Changes from v2:
#   - Freeform AI classification (no tag injection into prompt)
#   - Post-processing normalization via fuzzy matching
#   - Configurable sampling parameters (temperature, top_p, top_k)
#   - Queue deduplication & async I/O improvements
#   - Paperless data caching per batch
#   - Removed ministral.py dependency
#
# Handles upgrades from:
#   - v1 (ministral + classifier_api.py only)
#   - v2 (gemma4 + learning + classifier_api_v2.py)
#
# Usage:
#   cd /opt/paperless-classifier
#   wget https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main/upgrade_v3.sh
#   chmod +x upgrade_v3.sh
#   ./upgrade_v3.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD_DIR="/opt/paperless-classifier"

# Determine install directory: use prod dir if it exists, else script dir
if [[ -d "${PROD_DIR}" ]]; then
    INSTALL_DIR="${PROD_DIR}"
else
    INSTALL_DIR="${SCRIPT_DIR}"
fi

# If running from a different dir than prod, inform the user
if [[ "${SCRIPT_DIR}" != "${INSTALL_DIR}" ]]; then
    echo -e "${YELLOW}Note: Running from ${SCRIPT_DIR}${NC}"
    echo -e "${YELLOW}      Production directory detected: ${INSTALL_DIR}${NC}"
    echo ""
fi

BACKUP_DIR="${INSTALL_DIR}/backup_v3_$(date +%Y%m%d_%H%M%S)"
REPO_URL="https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Paperless AI Classifier - Upgrade to v3${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "This upgrade includes:"
echo "  • Freeform AI + smart post-processing (better tag quality)"
echo "  • Configurable sampling parameters (temperature, top_p, top_k)"
echo "  • Queue deduplication & non-blocking async I/O"
echo "  • Paperless data caching (faster batch processing)"
echo "  • Configurable fuzzy match threshold"
echo ""

# Detect current version
CURRENT_VERSION="unknown"
if [[ -f "${INSTALL_DIR}/classifier_api_v2.py" ]] || [[ -f "${INSTALL_DIR}/classifier_api.py" ]]; then
    if [[ -f "${INSTALL_DIR}/learning.py" ]] && [[ -f "${INSTALL_DIR}/gemma4.py" ]]; then
        CURRENT_VERSION="v2"
    elif [[ -f "${INSTALL_DIR}/ministral.py" ]]; then
        CURRENT_VERSION="v1"
    fi
fi

if [[ "${CURRENT_VERSION}" == "unknown" ]]; then
    echo -e "${RED}Error: Could not detect current installation.${NC}"
    echo "Please ensure you're in the paperless-classifier directory."
    echo "  cd /opt/paperless-classifier"
    exit 1
fi

echo -e "Detected current version: ${YELLOW}${CURRENT_VERSION}${NC}"
echo ""

# Check for root
if [[ $EUID -ne 0 ]]; then
    echo -e "${YELLOW}Note: Not running as root. Some operations may fail.${NC}"
fi

# ---- STEP 1: BACKUP ----
echo -e "${YELLOW}[1/7] Creating backup in: ${BACKUP_DIR}${NC}"
mkdir -p "${BACKUP_DIR}"

for file in classifier_api.py classifier_api_v2.py ministral.py gemma4.py learning.py .env classifier_audit.db; do
    if [[ -f "${INSTALL_DIR}/${file}" ]]; then
        cp "${INSTALL_DIR}/${file}" "${BACKUP_DIR}/"
        echo "  Backed up: ${file}"
    fi
done

# ---- STEP 2: DOWNLOAD FILES ----
echo ""
echo -e "${BLUE}[2/7] Downloading new files...${NC}"

download_file() {
    local filename="$1"
    local url="${REPO_URL}/${filename}"

    echo -n "  ${filename}... "
    if curl -fsSL "${url}" -o "${INSTALL_DIR}/${filename}.new" 2>/dev/null; then
        mv "${INSTALL_DIR}/${filename}.new" "${INSTALL_DIR}/${filename}"
        echo -e "${GREEN}OK${NC}"
        return 0
    elif wget -qO "${INSTALL_DIR}/${filename}.new" "${url}" 2>/dev/null; then
        mv "${INSTALL_DIR}/${filename}.new" "${INSTALL_DIR}/${filename}"
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        rm -f "${INSTALL_DIR}/${filename}.new"
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

FILES_TO_DOWNLOAD=(
    "gemma4.py"
    "learning.py"
    "classifier_api_v2.py"
)

DOWNLOAD_FAILED=0
for file in "${FILES_TO_DOWNLOAD[@]}"; do
    if ! download_file "${file}"; then
        DOWNLOAD_FAILED=1
    fi
done

if [[ ${DOWNLOAD_FAILED} -eq 1 ]]; then
    echo ""
    echo -e "${RED}Some downloads failed. Restoring backup...${NC}"
    cp "${BACKUP_DIR}"/* "${INSTALL_DIR}/" 2>/dev/null || true
    echo "Upgrade aborted. Check network connectivity."
    exit 1
fi

# ---- STEP 3: DATABASE MIGRATION ----
echo ""
echo -e "${BLUE}[3/7] Updating database schema...${NC}"

python3 << 'PYEOF'
import sqlite3
from pathlib import Path

db_path = Path("/opt/paperless-classifier/classifier_audit.db")
if not db_path.exists():
    db_path = Path("classifier_audit.db")

conn = sqlite3.connect(db_path)
c = conn.cursor()

# Ensure all tables exist (idempotent)
c.execute('''
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        document_title TEXT,
        timestamp TEXT NOT NULL,
        status TEXT NOT NULL,
        document_type TEXT,
        correspondent TEXT,
        tags TEXT,
        confidence REAL,
        processing_time REAL,
        tokens_used INTEGER,
        auto_approved INTEGER,
        error_message TEXT,
        explanation TEXT,
        ai_raw TEXT
    )
''')

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

# Add columns that may be missing (v1 -> v3)
for col, col_type in [('confidence', 'REAL'), ('ai_raw', 'TEXT'), ('explanation', 'TEXT')]:
    try:
        c.execute(f'ALTER TABLE audit_log ADD COLUMN {col} {col_type}')
        print(f"  Added {col} column to audit_log")
    except:
        pass

# Ensure indexes
c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp DESC)')
c.execute('CREATE INDEX IF NOT EXISTS idx_document_id ON audit_log(document_id)')
c.execute('CREATE INDEX IF NOT EXISTS idx_mappings_lookup ON term_mappings(term_type, ai_term)')
c.execute('CREATE INDEX IF NOT EXISTS idx_examples_verified ON classification_examples(user_verified, confidence DESC)')

conn.commit()
conn.close()
print("  Database schema updated")
PYEOF

# ---- STEP 4: UPDATE .env ----
echo ""
echo -e "${BLUE}[4/7] Updating .env configuration...${NC}"

ENV_FILE="${INSTALL_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
    # Model upgrade prompt
    CURRENT_MODEL=$(grep "^OLLAMA_MODEL=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 || echo "")
    if [[ -n "${CURRENT_MODEL}" ]] && [[ "${CURRENT_MODEL}" != *"gemma4"* ]]; then
        echo "  Current model: ${CURRENT_MODEL}"
        echo -e "${YELLOW}  Recommend switching to gemma4:e4b for best results.${NC}"
        read -p "  Switch to gemma4:e4b? [Y/n]: " switch_model
        if [[ "${switch_model}" != "n" && "${switch_model}" != "N" ]]; then
            sed -i "s|^OLLAMA_MODEL=.*|OLLAMA_MODEL=gemma4:e4b|" "${ENV_FILE}"
            echo -e "  ${GREEN}Switched to gemma4:e4b${NC}"
        fi
    fi

    # Add new v3 config keys
    add_env_key() {
        local key="$1"
        local value="$2"
        if ! grep -q "^${key}=" "${ENV_FILE}"; then
            echo "${key}=${value}" >> "${ENV_FILE}"
            echo "  Added: ${key}=${value}"
        fi
    }

    add_env_key "LEARNING_ENABLED" "true"
    add_env_key "FEW_SHOT_ENABLED" "false"
    add_env_key "INJECT_EXISTING_TYPES" "true"
    add_env_key "FUZZY_MATCH_THRESHOLD" "0.80"
    add_env_key "OLLAMA_TEMPERATURE" "0.7"
    add_env_key "OLLAMA_TOP_P" "0.95"
    add_env_key "OLLAMA_TOP_K" "64"
    add_env_key "GENERATE_EXPLANATIONS" "false"

    # Remove deprecated INJECT_EXISTING_TAGS if present
    if grep -q "^INJECT_EXISTING_TAGS=" "${ENV_FILE}"; then
        sed -i '/^INJECT_EXISTING_TAGS=/d' "${ENV_FILE}"
        echo "  Removed deprecated: INJECT_EXISTING_TAGS"
    fi
else
    echo -e "${YELLOW}  Warning: .env not found, skipping config update${NC}"
fi

# ---- STEP 5: UPDATE SYSTEMD SERVICE ----
echo ""
echo -e "${BLUE}[5/7] Updating systemd service...${NC}"

SERVICE_FILE="/etc/systemd/system/paperless-classifier.service"
if [[ -f "${SERVICE_FILE}" ]]; then
    # Check if it still points to old classifier_api.py
    if grep -q "classifier_api.py" "${SERVICE_FILE}"; then
        sed -i 's|classifier_api.py|classifier_api_v2.py|g' "${SERVICE_FILE}"
        systemctl daemon-reload
        echo -e "  Updated ExecStart to classifier_api_v2.py"
    else
        echo "  Service already points to classifier_api_v2.py"
    fi
else
    echo -e "  ${YELLOW}No systemd service file found — skip${NC}"
fi

# ---- STEP 6: CHECK MODEL ----
echo ""
echo -e "${BLUE}[6/7] Checking model availability...${NC}"

OLLAMA_URL=$(grep "^OLLAMA_URL=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2)
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
TARGET_MODEL=$(grep "^OLLAMA_MODEL=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2)
TARGET_MODEL="${TARGET_MODEL:-gemma4:e4b}"

echo "  Ollama URL: ${OLLAMA_URL}"
echo "  Target model: ${TARGET_MODEL}"

if curl -s "${OLLAMA_URL}/api/tags" 2>/dev/null | grep -q "${TARGET_MODEL}"; then
    echo -e "  Model status: ${GREEN}AVAILABLE${NC}"
else
    echo -e "  Model status: ${YELLOW}NOT FOUND${NC}"
    echo ""
    echo "  To download the model, run on your Ollama server:"
    echo -e "    ${BLUE}ollama pull ${TARGET_MODEL}${NC}"
fi

# ---- STEP 7: RESTART ----
echo ""
echo -e "${BLUE}[7/7] Restarting service...${NC}"

# Create logs directory
mkdir -p "${INSTALL_DIR}/logs"
chmod 755 "${INSTALL_DIR}/logs"

# Set permissions
chmod 644 "${INSTALL_DIR}"/*.py 2>/dev/null || true

if systemctl is-active --quiet paperless-classifier 2>/dev/null; then
    systemctl restart paperless-classifier
    sleep 3
    if systemctl is-active --quiet paperless-classifier; then
        echo -e "  Service restarted: ${GREEN}OK${NC}"
    else
        echo -e "  ${RED}Service failed to start! Check logs:${NC}"
        echo "    journalctl -u paperless-classifier -n 20"
    fi
else
    echo -e "  ${YELLOW}No running service found — start manually:${NC}"
    echo "    python3 ${INSTALL_DIR}/classifier_api_v2.py"
fi

# ---- DONE ----
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Upgrade to v3 Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "What's new in v3:"
echo "  • Freeform AI — no tag injection, better tag quality"
echo "  • Smart post-processing — fuzzy match against existing Paperless data"
echo "  • Configurable sampling — temperature, top_p, top_k via dashboard"
echo "  • Queue dedup — duplicate documents are skipped"
echo "  • Non-blocking I/O — polling and commits no longer block the event loop"
echo "  • Batch caching — Paperless data cached while queue is non-empty"
echo ""
echo "Backup saved to: ${BACKUP_DIR}"
echo "Dashboard: http://<your-ip>:8001/dashboard"
echo ""
echo "If something went wrong:"
echo "  cp ${BACKUP_DIR}/* ${INSTALL_DIR}/"
echo "  systemctl restart paperless-classifier"
