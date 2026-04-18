#!/bin/bash
#
# Paperless AI Classifier — Upgrade to v3
# ========================================
#
# One-command upgrade from v1 (ministral.py + classifier_api.py)
# to v3 (gemma4.py + learning.py + classifier_api_v2.py).
# Also works to update an existing v3 installation.
#
# Run directly:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main/upgrade_v3.sh)"
#
# Or from a local clone:
#   cd /root/paperless-ai-classifier && git pull && bash upgrade_v3.sh
#
# What this script does:
#   1. Backs up current files + database
#   2. Stops the service
#   3. Downloads new files from GitHub (or copies from local clone)
#   4. Migrates database schema (adds tables + columns)
#   5. Adds new .env keys (preserves all existing config)
#   6. Updates systemd ExecStart and restarts
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROD_DIR="/opt/paperless-classifier"
SERVICE_NAME="paperless-classifier"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
REPO_URL="https://raw.githubusercontent.com/mwckr/paperless-ai-classifier/main"
APP_FILES=("classifier_api_v2.py" "gemma4.py" "learning.py")

# Detect if running from a local git clone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo "")"
LOCAL_MODE=false
if [[ -n "${SCRIPT_DIR}" ]]; then
    all_local=true
    for f in "${APP_FILES[@]}"; do
        [[ ! -f "${SCRIPT_DIR}/${f}" ]] && all_local=false && break
    done
    ${all_local} && LOCAL_MODE=true
fi

header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Paperless AI Classifier — Upgrade to v3${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# ---- PRE-FLIGHT ----
header

# Must be root (systemd, /opt writes)
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}This script must be run as root.${NC}"
    echo "  sudo bash upgrade_v3.sh"
    echo "  or: sudo bash -c \"\$(curl -fsSL ${REPO_URL}/upgrade_v3.sh)\""
    exit 1
fi

# Must have production directory
if [[ ! -d "${PROD_DIR}" ]]; then
    echo -e "${RED}Error: ${PROD_DIR} not found.${NC}"
    echo "This script upgrades an existing installation."
    echo "If this is a fresh install, use install.sh or pve-install.sh instead."
    exit 1
fi

# Must have .env
ENV_FILE="${PROD_DIR}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo -e "${RED}Error: ${ENV_FILE} not found.${NC}"
    echo "Cannot upgrade without existing configuration."
    exit 1
fi

# Detect current version
if [[ -f "${PROD_DIR}/classifier_api_v2.py" ]] && [[ -f "${PROD_DIR}/learning.py" ]]; then
    CURRENT="v3 (will update to latest)"
elif [[ -f "${PROD_DIR}/classifier_api.py" ]]; then
    CURRENT="v1 (classifier_api.py + ministral.py)"
else
    echo -e "${RED}Error: Could not detect current version in ${PROD_DIR}.${NC}"
    exit 1
fi

if ${LOCAL_MODE}; then
    echo -e "  Source:     ${CYAN}${SCRIPT_DIR}${NC} (local)"
else
    echo -e "  Source:     ${CYAN}GitHub${NC} (downloading)"
fi
echo -e "  Target:     ${PROD_DIR}"
echo -e "  Detected:   ${YELLOW}${CURRENT}${NC}"
echo ""
echo "This upgrade includes:"
echo "  • Multi-model vision engine (gemma3, ministral, llama, etc.)"
echo "  • Multi-page document analysis with configurable image quality"
echo "  • Learning layer — fuzzy matching normalizes AI output"
echo "  • Review tab — correct and approve classifications"
echo "  • Config editor in dashboard with service restart"
echo "  • Manual document processing by ID"
echo "  • Logs viewer & debug export"
echo "  • Configurable timeout, image size, sampling params"
echo ""

read -p "Continue with upgrade? [Y/n]: " confirm
if [[ "${confirm}" == "n" || "${confirm}" == "N" ]]; then
    echo "Aborted."
    exit 0
fi

# ---- STEP 1: BACKUP ----
BACKUP_DIR="${PROD_DIR}/backup_v3_$(date +%Y%m%d_%H%M%S)"
echo ""
echo -e "${BLUE}[1/6] Backing up current installation...${NC}"
mkdir -p "${BACKUP_DIR}"

for file in classifier_api.py classifier_api_v2.py ministral.py gemma4.py learning.py .env classifier_audit.db; do
    if [[ -f "${PROD_DIR}/${file}" ]]; then
        cp "${PROD_DIR}/${file}" "${BACKUP_DIR}/"
        echo "  → ${file}"
    fi
done
echo -e "  ${GREEN}Saved to: ${BACKUP_DIR}${NC}"

# ---- STEP 2: STOP SERVICE ----
echo ""
echo -e "${BLUE}[2/6] Stopping service...${NC}"
if systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
    systemctl stop "${SERVICE_NAME}"
    echo -e "  ${GREEN}Stopped${NC}"
else
    echo "  Not running — OK"
fi

# ---- STEP 3: INSTALL FILES ----
echo ""
echo -e "${BLUE}[3/6] Installing new files...${NC}"

download_file() {
    local filename="$1"
    local target="${PROD_DIR}/${filename}"
    local url="${REPO_URL}/${filename}"

    echo -n "  ↓ ${filename}... "
    if curl -fsSL "${url}" -o "${target}.tmp" 2>/dev/null; then
        mv "${target}.tmp" "${target}"
        echo -e "${GREEN}OK${NC}"
        return 0
    elif command -v wget &>/dev/null && wget -qO "${target}.tmp" "${url}" 2>/dev/null; then
        mv "${target}.tmp" "${target}"
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        rm -f "${target}.tmp"
        echo -e "${RED}FAILED${NC}"
        return 1
    fi
}

copy_file() {
    local filename="$1"
    cp "${SCRIPT_DIR}/${filename}" "${PROD_DIR}/"
    echo "  → ${filename}"
}

INSTALL_FAILED=0
for f in "${APP_FILES[@]}"; do
    if ${LOCAL_MODE}; then
        copy_file "${f}" || INSTALL_FAILED=1
    else
        download_file "${f}" || INSTALL_FAILED=1
    fi
done

if [[ ${INSTALL_FAILED} -eq 1 ]]; then
    echo ""
    echo -e "${RED}File installation failed. Restoring backup...${NC}"
    cp "${BACKUP_DIR}"/* "${PROD_DIR}/" 2>/dev/null || true
    systemctl start "${SERVICE_NAME}" 2>/dev/null || true
    echo "Upgrade aborted."
    exit 1
fi

# Create logs directory for log viewer / export
mkdir -p "${PROD_DIR}/logs"
# Clear stale Python cache
rm -rf "${PROD_DIR}/__pycache__"
echo "  → Cleared __pycache__"

# ---- STEP 4: DATABASE MIGRATION ----
echo ""
echo -e "${BLUE}[4/6] Migrating database...${NC}"

DB_FILE="${PROD_DIR}/classifier_audit.db"
if [[ ! -f "${DB_FILE}" ]]; then
    echo "  No database found — will be created on first start"
else
    python3 << 'PYEOF'
import sqlite3

db = "/opt/paperless-classifier/classifier_audit.db"
conn = sqlite3.connect(db)
c = conn.cursor()
changes = []

# Check existing audit_log columns
c.execute("PRAGMA table_info(audit_log)")
cols = [row[1] for row in c.fetchall()]

if "ai_raw" not in cols:
    c.execute("ALTER TABLE audit_log ADD COLUMN ai_raw TEXT")
    changes.append("Added ai_raw column to audit_log")

if "explanation" not in cols:
    c.execute("ALTER TABLE audit_log ADD COLUMN explanation TEXT")
    changes.append("Added explanation column to audit_log")

# New table: term_mappings
c.execute("""
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
""")
c.execute("CREATE INDEX IF NOT EXISTS idx_mappings_lookup ON term_mappings(term_type, ai_term)")

# New table: classification_examples
c.execute("""
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
""")
c.execute("CREATE INDEX IF NOT EXISTS idx_examples_verified ON classification_examples(user_verified, confidence DESC)")

c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp DESC)")
c.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON audit_log(document_id)")

conn.commit()
conn.close()

if changes:
    for ch in changes:
        print(f"  → {ch}")
print("  → term_mappings: ready")
print("  → classification_examples: ready")
PYEOF
fi

# ---- STEP 5: UPDATE .env ----
echo ""
echo -e "${BLUE}[5/6] Updating .env configuration...${NC}"
echo "  (existing values are never overwritten)"
echo ""

add_env() {
    local key="$1"
    local value="$2"
    local comment="$3"
    if ! grep -q "^${key}=" "${ENV_FILE}" 2>/dev/null; then
        if [[ -n "${comment}" ]]; then
            echo "" >> "${ENV_FILE}"
            echo "# ${comment}" >> "${ENV_FILE}"
        fi
        echo "${key}=${value}" >> "${ENV_FILE}"
        echo -e "  ${GREEN}+${NC} ${key}=${value}"
    fi
}

add_env "IMAGE_MAX_SIZE" "1024" "Image processing"
add_env "IMAGE_QUALITY" "85"
add_env "OLLAMA_TIMEOUT" "600" "Ollama timeout (seconds)"
add_env "LEARNING_ENABLED" "true" "Learning layer"
add_env "FUZZY_MATCH_THRESHOLD" "0.80"
add_env "FEW_SHOT_ENABLED" "false"
add_env "INJECT_EXISTING_TAGS" "true" "Prompt enhancement"
add_env "INJECT_EXISTING_TYPES" "false"

if ! grep -q "OLLAMA_TEMPERATURE" "${ENV_FILE}" 2>/dev/null; then
    cat >> "${ENV_FILE}" << 'SAMPLING'

# Sampling params — leave commented to use model defaults (recommended)
# OLLAMA_TEMPERATURE=0.7
# OLLAMA_TOP_P=0.95
# OLLAMA_TOP_K=64
SAMPLING
    echo -e "  ${GREEN}+${NC} Sampling params (commented — model defaults)"
fi

# Model recommendation
echo ""
CURRENT_MODEL=$(grep "^OLLAMA_MODEL=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d "'" | tr -d '"')
echo -e "  Current model: ${YELLOW}${CURRENT_MODEL}${NC}"
if [[ "${CURRENT_MODEL}" == "ministral"* ]]; then
    echo -e "  ${YELLOW}gemma3:12b has been tested with significantly better results.${NC}"
    read -p "  Switch OLLAMA_MODEL to gemma3:12b? [Y/n]: " switch_model
    if [[ "${switch_model}" != "n" && "${switch_model}" != "N" ]]; then
        sed -i "s|^OLLAMA_MODEL=.*|OLLAMA_MODEL=gemma3:12b|" "${ENV_FILE}"
        echo -e "  ${GREEN}Model set to gemma3:12b${NC}"
        echo ""
        echo -e "  ${YELLOW}Make sure it's pulled on your Ollama server:${NC}"
        echo -e "    ollama pull gemma3:12b"
    fi
else
    echo "  Model unchanged (editable in dashboard → Config)"
fi

# ---- STEP 6: SYSTEMD & START ----
echo ""
echo -e "${BLUE}[6/6] Updating systemd & starting service...${NC}"

if [[ -f "${SERVICE_FILE}" ]]; then
    if grep -q "classifier_api\.py" "${SERVICE_FILE}" && ! grep -q "classifier_api_v2\.py" "${SERVICE_FILE}"; then
        sed -i 's|classifier_api\.py|classifier_api_v2.py|g' "${SERVICE_FILE}"
        echo "  → ExecStart: classifier_api.py → classifier_api_v2.py"
    fi

    systemctl daemon-reload
    systemctl start "${SERVICE_NAME}"
    sleep 3

    if systemctl is-active --quiet "${SERVICE_NAME}" 2>/dev/null; then
        echo -e "  ${GREEN}Service started successfully${NC}"
    else
        echo -e "  ${RED}Service failed to start!${NC}"
        echo ""
        echo "  Check logs:"
        echo "    journalctl -u ${SERVICE_NAME} -n 30 --no-pager"
        echo ""
        echo "  Rollback:"
        echo "    cp ${BACKUP_DIR}/* ${PROD_DIR}/"
        echo "    systemctl restart ${SERVICE_NAME}"
        exit 1
    fi
else
    echo -e "  ${YELLOW}No systemd service file found.${NC}"
    echo "  Start manually: python3 ${PROD_DIR}/classifier_api_v2.py"
fi

# ---- DONE ----
API_PORT=$(grep "^API_PORT=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d "'")
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Upgrade to v3 complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "What's new:"
echo "  • Multi-model vision engine (gemma3, ministral, llama, etc.)"
echo "  • Multi-page document analysis (configurable quality & size)"
echo "  • Learning layer with fuzzy matching"
echo "  • Review tab — correct classifications, system learns"
echo "  • Data tab — manage term mappings"
echo "  • Config editor with live restart"
echo "  • Manual document processing by ID"
echo "  • Logs viewer & debug export"
echo ""
echo -e "Dashboard:  ${CYAN}http://<your-ip>:${API_PORT:-8001}/dashboard${NC}"
echo "Backup:     ${BACKUP_DIR}"
echo ""
echo "Old files (ministral.py, classifier_api.py) are no longer used."
echo "Safe to delete if desired."
echo ""
echo -e "Rollback:   ${YELLOW}cp ${BACKUP_DIR}/* ${PROD_DIR}/ && systemctl restart ${SERVICE_NAME}${NC}"
