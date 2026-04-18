#!/bin/bash
#
# Paperless AI Classifier — Upgrade to v3
# ========================================
#
# Upgrades from v1 (commit 34fdc13 era: ministral.py + classifier_api.py)
# to v3 (gemma4.py + learning.py + classifier_api_v2.py)
#
# What changes:
#   - New AI engine: gemma4.py (multi-model support, multi-page images)
#   - New learning layer: learning.py (fuzzy matching, term mappings)
#   - New API server: classifier_api_v2.py (dashboard v2 with review/config/logs)
#   - Database: 2 new tables (term_mappings, classification_examples) + ai_raw column
#   - .env: 9+ new configuration keys added (existing keys preserved)
#   - systemd: ExecStart updated to classifier_api_v2.py
#   - Old files (ministral.py, classifier_api.py) left in place but unused
#
# Prerequisites:
#   - Git repo cloned somewhere (e.g. /root/paperless-ai-classifier)
#   - Production install at /opt/paperless-classifier (from original install.sh)
#
# Usage:
#   cd /root/paperless-ai-classifier   # or wherever you cloned the repo
#   git pull
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
SERVICE_NAME="paperless-classifier"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Paperless AI Classifier — Upgrade to v3${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ---- PRE-FLIGHT CHECKS ----

# Must have source files in the git repo
for f in classifier_api_v2.py gemma4.py learning.py; do
    if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
        echo -e "${RED}Error: ${f} not found in ${SCRIPT_DIR}${NC}"
        echo "Make sure you run this from the git repo directory after 'git pull'."
        exit 1
    fi
done

# Must have production directory
if [[ ! -d "${PROD_DIR}" ]]; then
    echo -e "${RED}Error: ${PROD_DIR} not found.${NC}"
    echo "This script upgrades an existing installation. Run install.sh first."
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
    CURRENT="v2/v3 (already upgraded — will update files)"
elif [[ -f "${PROD_DIR}/classifier_api.py" ]]; then
    CURRENT="v1 (classifier_api.py + ministral.py)"
else
    echo -e "${RED}Error: Could not detect current version in ${PROD_DIR}.${NC}"
    exit 1
fi

echo "Detected:    ${CURRENT}"
echo "Source:      ${SCRIPT_DIR}"
echo "Production:  ${PROD_DIR}"
echo ""
echo "This upgrade includes:"
echo "  • Multi-model vision engine (gemma3, ministral, llama, etc.)"
echo "  • Multi-page document analysis with configurable image quality"
echo "  • Learning layer — fuzzy matching normalizes AI output"
echo "  • Review tab — correct classifications, system learns"
echo "  • Config editor in dashboard with service restart"
echo "  • Manual document processing by ID"
echo "  • Logs viewer & debug export"
echo "  • Configurable Ollama timeout, image size, sampling params"
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

# ---- STEP 3: COPY FILES ----
echo ""
echo -e "${BLUE}[3/6] Installing new files...${NC}"

cp "${SCRIPT_DIR}/classifier_api_v2.py" "${PROD_DIR}/"
echo "  → classifier_api_v2.py  (API server + dashboard)"

cp "${SCRIPT_DIR}/gemma4.py" "${PROD_DIR}/"
echo "  → gemma4.py             (vision classification engine)"

cp "${SCRIPT_DIR}/learning.py" "${PROD_DIR}/"
echo "  → learning.py           (fuzzy matching + term mappings)"

# Create logs directory for log viewer / export feature
mkdir -p "${PROD_DIR}/logs"

# Clear Python cache to avoid stale bytecode
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

# Add ai_raw column (new in v3 — stores raw AI response for debugging)
if "ai_raw" not in cols:
    c.execute("ALTER TABLE audit_log ADD COLUMN ai_raw TEXT")
    changes.append("audit_log: added ai_raw column")

# explanation column should exist from v1 but just in case
if "explanation" not in cols:
    c.execute("ALTER TABLE audit_log ADD COLUMN explanation TEXT")
    changes.append("audit_log: added explanation column")

# New table: term_mappings (learning layer — AI term → approved term)
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

# New table: classification_examples (few-shot learning)
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

# Ensure indexes on audit_log
c.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp DESC)")
c.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON audit_log(document_id)")

conn.commit()
conn.close()

if changes:
    for ch in changes:
        print(f"  → {ch}")
else:
    print("  Schema already up to date")
print("  → term_mappings table: ready")
print("  → classification_examples table: ready")
PYEOF
fi

# ---- STEP 5: UPDATE .env ----
echo ""
echo -e "${BLUE}[5/6] Updating .env configuration...${NC}"

# Helper: add key=value to .env if the key doesn't exist yet
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
        echo -e "  + ${key}=${value}"
    fi
}

# Image processing (new in v3)
add_env "IMAGE_MAX_SIZE" "1024" "Image processing — max pixel dimension sent to Ollama"
add_env "IMAGE_QUALITY" "85"

# Ollama timeout
add_env "OLLAMA_TIMEOUT" "600" "Ollama vision request timeout (seconds)"

# Learning layer (new in v3)
add_env "LEARNING_ENABLED" "true" "Learning layer — post-processing normalization"
add_env "FUZZY_MATCH_THRESHOLD" "0.80"
add_env "FEW_SHOT_ENABLED" "false"

# Tag/type injection
add_env "INJECT_EXISTING_TAGS" "true" "Inject existing Paperless tags/types into AI prompt"
add_env "INJECT_EXISTING_TYPES" "false"

# Sampling params — add as comments (model defaults are recommended)
if ! grep -q "OLLAMA_TEMPERATURE" "${ENV_FILE}" 2>/dev/null; then
    echo "" >> "${ENV_FILE}"
    echo "# Sampling params — leave commented to use model defaults (recommended)" >> "${ENV_FILE}"
    echo "# OLLAMA_TEMPERATURE=0.7" >> "${ENV_FILE}"
    echo "# OLLAMA_TOP_P=0.95" >> "${ENV_FILE}"
    echo "# OLLAMA_TOP_K=64" >> "${ENV_FILE}"
    echo -e "  + Sampling params (commented out — uses model defaults)"
fi

# Model recommendation
echo ""
CURRENT_MODEL=$(grep "^OLLAMA_MODEL=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d "'" | tr -d '"')
echo -e "  Current model: ${YELLOW}${CURRENT_MODEL}${NC}"
if [[ "${CURRENT_MODEL}" == "ministral"* ]]; then
    echo -e "  ${YELLOW}Recommendation: gemma3:12b has been tested with significantly better results.${NC}"
    read -p "  Switch OLLAMA_MODEL to gemma3:12b? [Y/n]: " switch_model
    if [[ "${switch_model}" != "n" && "${switch_model}" != "N" ]]; then
        sed -i "s|^OLLAMA_MODEL=.*|OLLAMA_MODEL=gemma3:12b|" "${ENV_FILE}"
        echo -e "  ${GREEN}Model set to gemma3:12b${NC}"
        echo -e "  ${YELLOW}Make sure it's pulled on your Ollama server: ollama pull gemma3:12b${NC}"
    fi
else
    echo "  Keeping current model (can be changed in dashboard Config tab)"
fi

# ---- STEP 6: UPDATE SYSTEMD & START ----
echo ""
echo -e "${BLUE}[6/6] Updating systemd service & starting...${NC}"

if [[ -f "${SERVICE_FILE}" ]]; then
    # Update ExecStart if still pointing to old classifier_api.py
    if grep -q "classifier_api\.py" "${SERVICE_FILE}" && ! grep -q "classifier_api_v2\.py" "${SERVICE_FILE}"; then
        sed -i 's|classifier_api\.py|classifier_api_v2.py|g' "${SERVICE_FILE}"
        echo "  → ExecStart updated: classifier_api.py → classifier_api_v2.py"
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
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Upgrade to v3 complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "What's new:"
echo "  • Multi-model vision engine (gemma3, ministral, llama, etc.)"
echo "  • Multi-page document analysis (configurable DPI & quality)"
echo "  • Learning layer — fuzzy matching normalizes AI output"
echo "  • Review tab — correct classifications, system learns"
echo "  • Data tab — manage term mappings"
echo "  • Config editor with live service restart"
echo "  • Manual document processing by ID"
echo "  • Logs viewer & debug export"
echo ""
API_PORT=$(grep "^API_PORT=" "${ENV_FILE}" 2>/dev/null | cut -d'=' -f2 | tr -d "'")
echo "Dashboard:  http://<your-ip>:${API_PORT:-8001}/dashboard"
echo "Backup:     ${BACKUP_DIR}"
echo ""
echo "Old files (ministral.py, classifier_api.py) are still in ${PROD_DIR}"
echo "but are no longer used. Safe to delete if desired."
echo ""
echo "If something went wrong:"
echo "  cp ${BACKUP_DIR}/* ${PROD_DIR}/"
echo "  systemctl restart ${SERVICE_NAME}"
