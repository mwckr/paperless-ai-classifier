#!/bin/bash
set -e

echo "=============================================="
echo "Paperless Document Classifier Installer"
echo "AI-powered document classification for Paperless-ngx"
echo "=============================================="
echo ""

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo ./install.sh)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/paperless-classifier"

# Check required files exist
if [ ! -f "$SCRIPT_DIR/gemma4.py" ] || [ ! -f "$SCRIPT_DIR/classifier_api_v2.py" ] || [ ! -f "$SCRIPT_DIR/learning.py" ]; then
    echo "Error: gemma4.py, learning.py, and classifier_api_v2.py must be in the same directory as install.sh"
    exit 1
fi

echo "Configuration Setup"
echo "-------------------"
read -p "Paperless URL [http://localhost:8000]: " PAPERLESS_URL
PAPERLESS_URL=${PAPERLESS_URL:-http://localhost:8000}

read -p "Paperless API Token: " PAPERLESS_TOKEN
if [ -z "$PAPERLESS_TOKEN" ]; then
    echo "Error: Paperless token is required"
    exit 1
fi

read -p "Ollama URL [http://localhost:11434]: " OLLAMA_URL
OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}

read -p "Ollama Model [gemma4:e4b]: " OLLAMA_MODEL
OLLAMA_MODEL=${OLLAMA_MODEL:-gemma4:e4b}

read -p "Ollama CPU threads (0 = let Ollama pick) [0]: " OLLAMA_THREADS
OLLAMA_THREADS=${OLLAMA_THREADS:-0}

read -p "Max pages to process [3]: " MAX_PAGES
MAX_PAGES=${MAX_PAGES:-3}

read -p "Auto-commit results to Paperless? [true]: " AUTO_COMMIT
AUTO_COMMIT=${AUTO_COMMIT:-true}

read -p "Poll interval in seconds (0=disabled) [60]: " POLL_INTERVAL
POLL_INTERVAL=${POLL_INTERVAL:-60}

read -p "API port [8001]: " API_PORT
API_PORT=${API_PORT:-8001}

echo ""
echo "Installing to: $INSTALL_DIR"
echo ""

echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip sqlite3 poppler-utils > /dev/null

echo "[2/6] Installing Python packages..."
pip3 install fastapi uvicorn python-dotenv pydantic requests pillow --break-system-packages -q

echo "[3/6] Creating installation directory..."
mkdir -p "$INSTALL_DIR"

echo "[4/6] Copying and configuring application files..."
cp "$SCRIPT_DIR/gemma4.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/learning.py" "$INSTALL_DIR/"
cp "$SCRIPT_DIR/classifier_api_v2.py" "$INSTALL_DIR/"

echo "[5/6] Creating configuration..."
cat > "$INSTALL_DIR/.env" << ENVEOF
# Paperless Configuration
PAPERLESS_URL=$PAPERLESS_URL
PAPERLESS_TOKEN=$PAPERLESS_TOKEN

# Ollama Configuration
OLLAMA_URL=$OLLAMA_URL
OLLAMA_MODEL=$OLLAMA_MODEL
OLLAMA_THREADS=$OLLAMA_THREADS
# Low temperature = consistent names (classification, not creative writing)
OLLAMA_TEMPERATURE=0.1
# Context window and output cap. Raise OLLAMA_NUM_CTX if the log warns that the prompt nearly fills it.
OLLAMA_NUM_CTX=4096
OLLAMA_NUM_PREDICT=512
# Thinking tokens cost minutes on CPU and add nothing for classification
OLLAMA_THINK=false
# Constrain the reply to a JSON schema (needs Ollama >= 0.5)
OLLAMA_STRUCTURED_OUTPUT=true
# How long the model stays in RAM after a request; true = unload as soon as the queue is empty
OLLAMA_KEEP_ALIVE=5m
OLLAMA_UNLOAD_WHEN_IDLE=false
OLLAMA_TIMEOUT=600

# Processing Options
# OCR_TEXT_MODE: hint = send Paperless' OCR text together with page images (recommended)
#                only = text only, no images (fastest, works with non-vision models)
#                off  = images only
OCR_TEXT_MODE=hint
OCR_TEXT_CHARS=2000
MAX_PAGES=$MAX_PAGES
IMAGE_MAX_SIZE=1024
IMAGE_QUALITY=85
IMAGE_GRAYSCALE=false
AUTO_COMMIT=$AUTO_COMMIT
# false = only fill empty document type / correspondent; tags are merged (TAGS_MODE=merge|replace)
OVERWRITE_EXISTING=false
TAGS_MODE=merge
GENERATE_EXPLANATIONS=false

# Learning & Normalization
LEARNING_ENABLED=true
FEW_SHOT_ENABLED=false
INJECT_EXISTING_TYPES=true
INJECT_EXISTING_CORRESPONDENTS=true
INJECT_EXISTING_TAGS=true
PROMPT_TYPE_LIMIT=50
PROMPT_CORRESPONDENT_LIMIT=60
PROMPT_TAG_LIMIT=80
FUZZY_MATCH_THRESHOLD=0.80

# Polling (set to 0 to disable and use webhooks instead)
POLL_INTERVAL=$POLL_INTERVAL

# API Configuration
API_HOST=0.0.0.0
API_PORT=$API_PORT
ENVEOF

echo "[6/6] Creating systemd service..."
cat > /etc/systemd/system/paperless-classifier.service << SVCEOF
[Unit]
Description=Paperless Document Classifier API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=/usr/bin/python3 $INSTALL_DIR/classifier_api_v2.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable paperless-classifier
systemctl start paperless-classifier

sleep 3

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):$API_PORT/dashboard"
echo "Config:    $INSTALL_DIR/.env"
echo "Logs:      journalctl -u paperless-classifier -f"
echo ""

if curl -s "http://localhost:$API_PORT/api/health" > /dev/null 2>&1; then
    echo "[OK] Service is running"
else
    echo "[..] Service may still be starting - check: journalctl -u paperless-classifier -f"
fi
