# Paperless AI Classifier

AI-powered document classification for [Paperless-ngx](https://docs.paperless-ngx.com/) using a local
[Ollama](https://ollama.ai) model. Built for German household paperwork on a home server: no GPU, limited
RAM, CPU inference.

For every new document the service:

1. loads the document from Paperless (OCR text and, optionally, the first pages as images),
2. asks the model for document type, correspondent, tags and a confidence value,
3. normalizes the answer against the names already in Paperless plus corrections you taught it,
4. writes the result back to Paperless (optional) and logs it in an audit database with a web dashboard.

## Features

- Works with vision models (Gemma 4, Gemma 3, Ministral, Qwen-VL, LLaVA) and, in text-only mode,
  with any Ollama model
- Uses the OCR text Paperless already produced, so small print survives even at low image resolution
- Existing document types, correspondents and tags are offered to the model, most used first,
  so names stay consistent
- Learning layer: term mappings and verified examples from your corrections in the dashboard
- Never destroys data by default: existing type and correspondent are kept, tags are merged
- Web dashboard: queue, audit log, correction modal, mappings, config editor, log viewer, debug export
- Polling and webhook triggers

## Requirements

- Debian or Ubuntu Linux (tested on Ubuntu 24.04), `poppler-utils` for PDF rendering
- Ollama 0.5 or newer (structured output) with a model pulled
- Paperless-ngx with an API token
- RAM: depends on the model. `gemma4:e4b` needs roughly 6 to 8 GB while loaded; text-only mode with a 2 to 4B
  model runs in 3 to 4 GB

## Installation

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma4:e4b

git clone <this repo> && cd paperless-ai-classifier
sudo ./install.sh          # interactive, installs to /opt/paperless-classifier and creates a systemd unit
```

Later updates: `git pull && sudo ./deploy.sh` (copies the three application files and restarts the service).
`upgrade_v3.sh` additionally adds any new `.env` keys to an existing installation.

For Proxmox there is `pve-install.sh`, which creates two LXC containers (Ollama and the API).

## Package contents

| File | Role |
|------|------|
| `classifier_api_v2.py` | FastAPI service: queue worker, polling, webhook, SQLite audit log, dashboard |
| `gemma4.py` | Classification engine: fetches documents, builds the prompt, calls Ollama, updates Paperless |
| `learning.py` | Fuzzy matching, term mappings, few-shot examples |
| `install.sh`, `deploy.sh`, `upgrade_v3.sh`, `pve-install.sh` | Deployment scripts |
| `ministral.py`, `classifier_api.py`, `test_gemma4.py`, `upgrade_v2.sh` | Legacy, not deployed by any script |

## Configuration

All settings live in `/opt/paperless-classifier/.env` and can also be changed in the dashboard
(Config tab, followed by a restart).

### Connection

| Setting | Default | Description |
|---------|---------|-------------|
| `PAPERLESS_URL` | `http://localhost:8000` | Paperless-ngx URL |
| `PAPERLESS_TOKEN` | (required) | API token |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama URL |
| `OLLAMA_MODEL` | `gemma4:e4b` | Model name as shown by `ollama list` |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8001` | Dashboard bind address and port |
| `POLL_INTERVAL` | `60` | Seconds between checks for new documents, `0` = webhooks only |

### Ollama runtime

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_THREADS` | `0` | CPU threads. `0` lets Ollama choose (physical cores). More threads than cores slows things down |
| `OLLAMA_TEMPERATURE` | model default | `0.1` recommended. Low values give stable, repeatable names |
| `OLLAMA_TOP_P`, `OLLAMA_TOP_K` | model default | Only sent when set |
| `OLLAMA_NUM_CTX` | `4096` | Context window in tokens. The log warns when a prompt nearly fills it |
| `OLLAMA_NUM_PREDICT` | `512` | Maximum output tokens |
| `OLLAMA_THINK` | `false` | Thinking mode for models that support it. Costs minutes on CPU, adds nothing here |
| `OLLAMA_STRUCTURED_OUTPUT` | `true` | Constrain the reply to a JSON schema. Falls back automatically on servers that reject it |
| `OLLAMA_KEEP_ALIVE` | `5m` | How long the model stays loaded after a request |
| `OLLAMA_UNLOAD_WHEN_IDLE` | `false` | Unload the model as soon as the queue is empty |
| `OLLAMA_TIMEOUT` | `600` | Seconds to wait for one answer |

### Input

| Setting | Default | Description |
|---------|---------|-------------|
| `OCR_TEXT_MODE` | `hint` | `hint`: OCR text plus page images. `only`: OCR text, no images. `off`: images only |
| `OCR_TEXT_CHARS` | `2000` | Characters of OCR text sent (from the start of the document) |
| `MAX_PAGES` | `3` | Pages rendered as images. Every page is one vision encoder pass |
| `IMAGE_MAX_SIZE` | `1024` | Longest side in pixels |
| `IMAGE_QUALITY` | `85` | JPEG quality |
| `IMAGE_GRAYSCALE` | `false` | Grayscale page images (smaller payload) |
| `PDF_RENDER_TIMEOUT` | `120` | Seconds allowed for `pdftoppm` |

### Writing back

| Setting | Default | Description |
|---------|---------|-------------|
| `AUTO_COMMIT` | `true` | Apply results to Paperless |
| `OVERWRITE_EXISTING` | `false` | `false`: only fill empty document type and correspondent. Re-analyze from the dashboard always overwrites |
| `TAGS_MODE` | `merge` | `merge` keeps the document's existing tags (inbox tag, matching rules), `replace` does not |
| `GENERATE_EXPLANATIONS` | `false` | Ask for a one-sentence explanation (shown in the audit log) |

### Prompt and learning

| Setting | Default | Description |
|---------|---------|-------------|
| `INJECT_EXISTING_TYPES` | `true` | Offer existing document types to the model |
| `INJECT_EXISTING_CORRESPONDENTS` | `true` | Offer existing correspondents |
| `INJECT_EXISTING_TAGS` | `true` | Offer existing tags |
| `PROMPT_TYPE_LIMIT` / `PROMPT_CORRESPONDENT_LIMIT` / `PROMPT_TAG_LIMIT` | `50` / `60` / `80` | How many of each list go into the prompt, most used first |
| `LEARNING_ENABLED` | `true` | Fuzzy-match results against Paperless names and learned mappings |
| `FUZZY_MATCH_THRESHOLD` | `0.80` | Similarity needed to map a name onto an existing one |
| `FEW_SHOT_ENABLED` | `false` | Add up to three verified examples to the prompt |

## Tuning for low-powered devices

Per document the model pays for three things: loading itself into RAM, reading the prompt (images are by far
the most expensive part), and writing the answer. The defaults already remove the biggest waste (thinking
tokens, unbounded output, prose around the JSON). Beyond that, in order of effect:

1. **Text-only mode.** `OCR_TEXT_MODE=only` sends no images at all. Paperless has already OCRed the
   document, and for letters, invoices and notices the first 2000 characters carry everything the
   classifier needs. Any model works, including small text models such as `qwen3:4b` or `gemma3:4b`.
   Expect roughly a 5 to 10 times speed-up over three page images on CPU.
2. **Fewer pages.** With `OCR_TEXT_MODE=hint`, `MAX_PAGES=1` keeps the letterhead and logo for the model
   while the OCR text covers the rest.
3. **Smaller images.** `IMAGE_MAX_SIZE=768` is still enough for layout and logos when the OCR text is
   present.
4. **Free RAM between batches.** `OLLAMA_UNLOAD_WHEN_IDLE=true` when Paperless and Ollama share the
   machine. Reloading costs a few seconds per batch.
5. **Threads.** Leave `OLLAMA_THREADS=0`, or set it to the number of physical cores.
6. **Context.** Keep `OLLAMA_NUM_CTX` at 4096 unless the log says the prompt nearly fills it. The KV cache
   grows with the context window.
7. **On the Ollama host**, `OLLAMA_NUM_PARALLEL=1` and `OLLAMA_MAX_LOADED_MODELS=1` avoid a second copy
   of the model in RAM. `OLLAMA_FLASH_ATTENTION=1` with `OLLAMA_KV_CACHE_TYPE=q8_0` roughly halves the
   KV cache on recent Ollama versions.

The log line `Ollama timing: load … | prompt … | output …` after each document shows where the time goes.

## Usage

Dashboard: `http://your-server:8001/`

```bash
# queue one document
curl -X POST http://localhost:8001/api/classify -H "Content-Type: application/json" -d '{"document_id": 123}'
# queue several
curl -X POST http://localhost:8001/api/classify/batch -H "Content-Type: application/json" -d '[1, 2, 3]'
# re-analyze and overwrite existing type/correspondent
curl -X POST http://localhost:8001/api/reanalyze/123
```

Webhook from a Paperless workflow (action "Webhook", URL `http://your-server:8001/webhook/paperless`,
body containing `document_id`) or `POST /webhook/paperless/{id}` from a post-consume script.

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/`, `/dashboard` | GET | Web dashboard |
| `/api/health` | GET | Paperless and Ollama reachability |
| `/api/queue` | GET | Queue status |
| `/api/stats` | GET | Statistics |
| `/api/audit` | GET | Audit log |
| `/api/classify` | POST | Queue one document `{"document_id": N}` |
| `/api/classify/batch` | POST | Queue many `[N, …]` |
| `/api/reanalyze/{id}` | POST | Re-queue with overwrite |
| `/api/config` | GET, POST | Read / update one `.env` key |
| `/api/restart` | POST | Restart the service (systemd restarts it) |
| `/api/mappings`, `/api/learn/*` | various | Learning data |
| `/api/export/debug` | GET | Debug export (config, health, recent runs, log tail) |
| `/api/logs` | GET | Last log lines |
| `/webhook/paperless[/{id}]` | POST | Webhook |

The dashboard and API have no authentication. Keep port 8001 on a trusted network or put it behind a
reverse proxy with auth.

## Service management

```bash
systemctl status paperless-classifier
systemctl restart paperless-classifier
journalctl -u paperless-classifier -f
```

## Troubleshooting

- **Classification fails with `vision_analysis_failed`**: check the log for the Ollama error. Old Ollama
  versions reject `think` or `format`; the service retries without them and remembers that.
- **Prompt nearly fills the context**: the log says so. Raise `OLLAMA_NUM_CTX` or lower `MAX_PAGES`,
  `OCR_TEXT_CHARS` or the `PROMPT_*_LIMIT` values.
- **Too slow**: see the tuning section. The `Ollama timing` log line tells you whether loading,
  prompt processing (images) or generation dominates.
- **Model not found**: `ollama list`, then `ollama pull <model>`.
- **Nothing is picked up**: `POLL_INTERVAL > 0` or a webhook configured, token valid, Ollama reachable.

## Uninstall

```bash
sudo systemctl disable --now paperless-classifier
sudo rm /etc/systemd/system/paperless-classifier.service
sudo rm -rf /opt/paperless-classifier
sudo systemctl daemon-reload
```

## License

MIT
