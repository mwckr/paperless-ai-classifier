# CLAUDE.md — Paperless AI Classifier

This file provides guidance for AI assistants working on this codebase.

## Project Overview

**Paperless AI Classifier** is a Python service that automatically classifies documents in [Paperless-ngx](https://docs.paperless-ngx.com/) using a local vision language model (via [Ollama](https://ollama.ai/)). It assigns document types, correspondents, and tags by analyzing document images with the Ministral vision model.

**Architecture summary:**
- `ministral.py` — Vision AI classification engine (fetches images, calls Ollama, parses results, updates Paperless)
- `classifier_api.py` — FastAPI web service with async queue, polling loop, SQLite audit log, and embedded web dashboard

---

## Repository Structure

```
paperless-ai-classifier/
├── classifier_api.py      # FastAPI server, async queue, dashboard, endpoints
├── ministral.py           # Ollama vision classification engine
├── install.sh             # Bash install/deployment script (creates systemd service)
├── classifier_audit.db    # SQLite audit log (gitignored in prod, present in repo)
├── README.md              # User-facing documentation
└── .gitignore             # Ignores .DS_Store
```

No `requirements.txt` exists — dependencies are installed directly by `install.sh`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web framework | FastAPI + Uvicorn (ASGI) |
| HTTP client | `requests` |
| Image processing | Pillow (PIL) + `pdftoppm` (poppler-utils) |
| AI inference | Ollama REST API (local LLM) |
| Document store | Paperless-ngx REST API |
| Audit storage | SQLite 3 (built-in `sqlite3`) |
| Config | `python-dotenv` (.env file) |
| Deployment | systemd service |

---

## Configuration

The service reads configuration from a `.env` file (or environment variables). The `.env` is created by `install.sh` at `/opt/paperless-classifier/.env` for production deployments, or can be placed in the working directory for local runs.

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERLESS_URL` | `http://localhost:8000` | Paperless-ngx server URL |
| `PAPERLESS_TOKEN` | (required) | API token for Paperless authentication |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `ministral-3:14b` | Vision model name |
| `OLLAMA_THREADS` | `10` | CPU threads for Ollama |
| `MAX_PAGES` | `3` | Max PDF pages to analyze |
| `AUTO_COMMIT` | `true` | Auto-apply classifications to Paperless |
| `GENERATE_EXPLANATIONS` | `false` | Generate human-readable explanations |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8001` | API/dashboard port |
| `POLL_INTERVAL` | `60` | Polling interval (seconds) for new documents |

---

## Application Flow

### Startup (`lifespan` in `classifier_api.py`)
1. Initialize SQLite audit database (`init_database()`)
2. Create asyncio `Queue`
3. Launch `process_queue()` as background task
4. Launch `poll_for_new_documents()` as background task

### Polling Loop (`poll_for_new_documents`)
- Queries Paperless API every `POLL_INTERVAL` seconds
- Finds documents with no `document_type` assigned
- Adds document IDs to asyncio Queue

### Processing Loop (`process_queue`)
- Dequeues document IDs
- Calls `ministral.process_document(doc_id, config)`
- Logs result to `audit_log` table
- If `AUTO_COMMIT=true`, applies classification to Paperless

### Classification Engine (`ministral.py`)
1. **Image fetching** (multi-strategy fallback): preview image → PDF-to-JPEG conversion → thumbnail
2. **Vision analysis**: sends images to Ollama, expects JSON response with `document_type`, `correspondent`, `tags`
3. **JSON parsing**: primary JSON parse → regex fallback extraction
4. **Resource management**: `get_or_create_*` helpers for types, correspondents, tags in Paperless
5. **Apply changes**: `update_document_in_paperless()` PATCH call

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` or `/dashboard` | Web dashboard (embedded HTML) |
| GET | `/api/health` | Service health (Paperless + Ollama connectivity) |
| GET | `/api/queue` | Current queue status |
| GET | `/api/stats` | Processing statistics |
| GET | `/api/audit` | Audit log entries |
| GET | `/api/config` | Current configuration |
| POST | `/api/config` | Update configuration |
| POST | `/api/classify` | Queue a single document `{"document_id": N}` |
| POST | `/api/classify/batch` | Queue multiple documents `{"document_ids": [N,...]}` |
| DELETE | `/api/queue/clear` | Clear the queue |
| DELETE | `/api/audit/{entry_id}` | Delete one audit entry |
| DELETE | `/api/audit/status/{status}` | Delete all entries with a given status |
| POST | `/api/audit/cleanup` | Mark stale `processing` entries as `abandoned` |
| POST | `/webhook/paperless[/{doc_id}]` | Webhook triggered by Paperless post-consume script |

---

## Database Schema

**File:** `classifier_audit.db` (SQLite)

**Table: `audit_log`**

```sql
CREATE TABLE audit_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id       INTEGER NOT NULL,
    document_title    TEXT,
    timestamp         TEXT,           -- ISO 8601 format
    status            TEXT,           -- pending | processing | completed | failed | abandoned
    document_type     TEXT,
    correspondent     TEXT,
    tags              TEXT,           -- JSON array string
    confidence        REAL,
    processing_time   REAL,           -- seconds
    tokens_used       INTEGER,        -- estimated
    auto_approved     INTEGER,        -- 0/1 boolean
    error_message     TEXT,
    explanation       TEXT
);
```

---

## Key Code Patterns

### `get_or_create_*` Pattern
Functions like `get_or_create_tag()`, `get_or_create_correspondent()`, and `get_or_create_document_type()` in `ministral.py` search Paperless for an existing resource by name and create it if absent. Always use these rather than directly POSTing to Paperless resource endpoints.

### Configuration Loading
`classifier_api.py` uses `get_config()` which reads from `os.environ` (populated by dotenv). `ministral.py` has its own `load_env()` that searches `.env` in multiple paths. Both approaches must stay in sync — when adding new config keys, add them to both files.

### JSON Parsing with Fallback
`analyze_with_vision()` in `ministral.py` first attempts `json.loads()` on the model response, then falls back to regex extraction (`re.search`) for key fields. Maintain this resilience if modifying the parsing logic.

### Async Queue
The `asyncio.Queue` in `classifier_api.py` holds document IDs (integers). The `process_queue()` task uses `await queue.get()` / `queue.task_done()`. Do not use blocking I/O inside the async queue worker — wrap blocking calls in `asyncio.to_thread()` or use a thread executor if needed.

### Embedded Dashboard
`DASHBOARD_HTML` in `classifier_api.py` is a large multi-line string (~500+ lines of HTML/CSS/JS). The dashboard auto-refreshes every 5 seconds via `setInterval`. When modifying the dashboard, preserve the existing CSS variable structure and JavaScript fetch calls to the `/api/*` endpoints.

---

## Development Workflows

### Running Locally
```bash
# Set environment variables (copy and edit)
cp .env.example .env   # (no .env.example exists; create .env manually)

# Install Python dependencies
pip3 install fastapi uvicorn python-dotenv pydantic requests pillow

# Install system dependency for PDF conversion
apt-get install -y poppler-utils   # Ubuntu/Debian

# Run the API server
python3 classifier_api.py

# Run the classifier CLI (interactive testing)
python3 ministral.py
```

### Testing Classification Manually
`ministral.py` has a `main()` CLI mode (runs when executed directly):
```bash
python3 ministral.py
# Prompts for document ID(s) and whether to generate explanations
# Shows classification results and metrics
# Optionally applies changes to Paperless
```

### Installing as a System Service
```bash
sudo ./install.sh
# Interactive prompts for all configuration values
# Installs to /opt/paperless-classifier/
# Creates and starts systemd service: paperless-classifier
```

### Service Management
```bash
sudo systemctl status paperless-classifier
sudo systemctl restart paperless-classifier
sudo journalctl -u paperless-classifier -f   # Follow logs
```

### Useful API Calls for Development
```bash
# Health check
curl http://localhost:8001/api/health

# Manually classify a document
curl -X POST http://localhost:8001/api/classify -H "Content-Type: application/json" -d '{"document_id": 42}'

# View queue
curl http://localhost:8001/api/queue

# View audit log
curl http://localhost:8001/api/audit
```

---

## Code Conventions

- **Type hints**: Use `Optional[T]`, `Dict[str, Any]`, `List[T]`, `Tuple[T, ...]` throughout
- **Logging**: Use `logging.getLogger(__name__)` with `INFO/DEBUG/WARNING/ERROR` levels; avoid `print()` statements
- **Error handling**: Always use `try/except` with specific exception types; log errors with context before re-raising or returning `None`
- **Config access**: Access config values via the config dict (e.g., `config["PAPERLESS_URL"]`), not direct `os.environ` calls inside functions
- **No test framework**: No pytest or unittest is set up; manual testing via CLI or API calls is the current approach

---

## External Dependencies

| Service | Role | Default URL |
|---------|------|-------------|
| Ollama | Local LLM inference | `http://localhost:11434` |
| Paperless-ngx | Document store and API | `http://localhost:8000` |

Both services must be running and reachable for the classifier to function. The `/api/health` endpoint checks connectivity to both.

---

## Important Notes for AI Assistants

1. **No tests exist** — when adding features, consider whether a manual test via `ministral.py` CLI or the API is sufficient, or suggest adding a test framework.

2. **Single-file architecture** — `classifier_api.py` and `ministral.py` are large, self-contained files. Before extracting code to new files, confirm the user wants to change the deployment model (since `install.sh` copies specific files to `/opt/paperless-classifier/`).

3. **install.sh must stay in sync** — if new Python packages are added as imports, they must be added to the `pip3 install` command in `install.sh`.

4. **Embedded HTML** — the `DASHBOARD_HTML` constant in `classifier_api.py` is intentional (single-file deployment). Do not refactor it to external template files without updating `install.sh`.

5. **Blocking calls in async context** — `requests` calls inside `process_queue()` and `poll_for_new_documents()` are currently blocking. This is a known limitation. If extending these functions significantly, consider wrapping in `asyncio.to_thread()`.

6. **Production path** — the service runs from `/opt/paperless-classifier/` in production, but the repo root is `/home/user/paperless-ai-classifier/`. Path handling in `load_env()` accounts for both.

7. **Document type taxonomy** — `ministral.py` contains a large hardcoded list of document types (~200+). Changes to this list affect classification accuracy and should be made deliberately.
