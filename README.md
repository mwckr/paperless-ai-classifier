# Paperless Document Classifier

AI-powered document classification for Paperless-ngx using Ollama vision models.

## Features

- Automatic document classification using vision AI
- Extracts document type, correspondent, and tags
- Web dashboard for monitoring
- Polling for new documents
- Audit log of all classifications

## Requirements

- Ubuntu/Debian Linux
- Ollama with a vision model installed
- Paperless-ngx instance with API access
- ~10GB RAM recommended

## Pre-Installation

1. Install Ollama: https://ollama.ai
2. Pull a vision model:
```bash
   ollama pull ministral-3:14b
```
3. Get your Paperless API token from Settings > API

## Installation
```bash
chmod +x install.sh
sudo ./install.sh
```

## Configuration

Edit `/opt/paperless-classifier/.env` after installation.

After changes: `sudo systemctl restart paperless-classifier`

## Service Management
```bash
systemctl status paperless-classifier
systemctl restart paperless-classifier
journalctl -u paperless-classifier -f
```

## Uninstall
```bash
sudo systemctl stop paperless-classifier
sudo systemctl disable paperless-classifier
sudo rm /etc/systemd/system/paperless-classifier.service
sudo rm -rf /opt/paperless-classifier
sudo systemctl daemon-reload
```
