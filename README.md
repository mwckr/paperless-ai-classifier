# Paperless Document Classifier

AI-powered document classification for Paperless-ngx using Ollama vision models.

## Features

- Automatic document classification using vision AI
- Extracts document type, correspondent, and tags
- Web dashboard for monitoring and configuration
- Automatic polling for new documents
- Audit log of all classifications
- Auto-commit results to Paperless

## Requirements

- Ubuntu/Debian Linux (tested on Ubuntu 24.04)
- Ollama with a vision model installed
- Paperless-ngx instance with API access
- ~10GB RAM recommended for model loading

## Pre-Installation

### 1. Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull a vision model

```bash
ollama pull ministral-3:14b
```

Other compatible models: `llava`, `llava:13b`, `bakllava`

### 3. Get your Paperless API token

Go to Paperless > Settings > API and create/copy your token.

## Installation

```bash
chmod +x install.sh
sudo ./install.sh
```

Follow the prompts to configure your setup.

## Package Contents

```
paperless-classifier/
├── install.sh          # Installer script
├── README.md           # This file
├── ministral.py        # Vision AI classification engine
└── classifier_api.py   # FastAPI service with dashboard
```

## Configuration

After installation, edit `/opt/paperless-classifier/.env`:

| Setting | Description | Default |
|---------|-------------|---------|
| PAPERLESS_URL | Paperless-ngx URL | http://localhost:8000 |
| PAPERLESS_TOKEN | API token | (required) |
| OLLAMA_URL | Ollama API URL | http://localhost:11434 |
| OLLAMA_MODEL | Vision model name | ministral-3:14b |
| OLLAMA_THREADS | CPU threads for Ollama | 10 |
| MAX_PAGES | Max pages to analyze per doc | 3 |
| AUTO_COMMIT | Auto-apply classifications | true |
| POLL_INTERVAL | Seconds between checks (0=off) | 60 |
| API_PORT | Dashboard port | 8001 |

After changing settings:
```bash
sudo systemctl restart paperless-classifier
```

## Usage

### Web Dashboard

Access at `http://your-server:8001/dashboard`

Features:
- Service status and uptime
- Current processing status
- Processing queue
- Recent activity / audit log
- Configuration editor

### Manual Classification

```bash
curl -X POST http://localhost:8001/api/classify \
  -H "Content-Type: application/json" \
  -d '{"document_id": 123}'
```

### Batch Classification

```bash
curl -X POST http://localhost:8001/api/classify/batch \
  -H "Content-Type: application/json" \
  -d '[1, 2, 3, 4, 5]'
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dashboard` | GET | Web dashboard |
| `/api/health` | GET | Health check |
| `/api/queue` | GET | Queue status |
| `/api/stats` | GET | Statistics |
| `/api/audit` | GET | Audit log |
| `/api/classify` | POST | Queue single document |
| `/api/classify/batch` | POST | Queue multiple documents |
| `/api/config` | GET/POST | View/update configuration |
| `/webhook/paperless` | POST | Webhook endpoint |

## Service Management

```bash
# Check status
systemctl status paperless-classifier

# Restart service
systemctl restart paperless-classifier

# View logs
journalctl -u paperless-classifier -f

# Stop service
systemctl stop paperless-classifier

# Start service
systemctl start paperless-classifier
```

## How It Works

1. **Polling**: The service checks Paperless every N seconds for new documents without a document type assigned.

2. **Processing**: For each new document:
   - Fetches the document preview/PDF
   - Converts to optimized JPEG (max 3 pages)
   - Sends to Ollama vision model for analysis
   - Model identifies: document type, correspondent, tags

3. **Commit**: If auto-commit is enabled, the classification is automatically applied to the document in Paperless.

## Troubleshooting

### Service won't start
```bash
journalctl -u paperless-classifier -n 50
```

### Model not found
```bash
ollama list
ollama pull ministral-3:14b
```

### High memory usage
The vision model uses ~9GB RAM. This is normal and expected.
To reduce memory after idle, configure Ollama:
```bash
# In /etc/systemd/system/ollama.service.d/override.conf
Environment="OLLAMA_KEEP_ALIVE=5m"
```

### Documents not being classified
1. Check POLL_INTERVAL is > 0 in .env
2. Verify Paperless token is valid
3. Check Ollama is running: `curl http://localhost:11434/api/tags`
4. Check service logs for errors

### Classification takes too long
- Normal processing time: 2-8 minutes per document on CPU
- Reduce MAX_PAGES to process fewer pages
- Use a smaller model (e.g., `llava:7b`)

## Uninstall

```bash
sudo systemctl stop paperless-classifier
sudo systemctl disable paperless-classifier
sudo rm /etc/systemd/system/paperless-classifier.service
sudo rm -rf /opt/paperless-classifier
sudo systemctl daemon-reload
```

## License

MIT License

## Credits

Built with:
- [Ollama](https://ollama.ai) - Local LLM runtime
- [FastAPI](https://fastapi.tiangolo.com) - Python web framework
- [Paperless-ngx](https://docs.paperless-ngx.com) - Document management system
