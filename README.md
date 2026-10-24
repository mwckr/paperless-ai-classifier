# Paperless AI Classifier

Classifies documents in [Paperless-ngx](https://docs.paperless-ngx.com/) with a local
[Ollama](https://ollama.ai) vision model: document type, correspondent and tags. Built for German household
paperwork on a home server or Proxmox container without a GPU and with limited RAM.

For every new document the service renders the first pages as images, asks the model for type, correspondent,
tags and a confidence value, normalizes the answer against the names already in Paperless (plus corrections you
taught it), writes the result back and logs everything in a dashboard.

Page images are the input on purpose. The layout tells the model who the sender is (letterhead, top left), what
the table in the middle lists and that the numbers in the footer are bank details. Flat OCR text loses those
relationships and was found to be worse in testing.

## Features

- Any Ollama vision model (Gemma 4, Gemma 3, Ministral, Qwen-VL, LLaVA)
- Tuned for CPU: thinking off, capped output, explicit context window, JSON-schema constrained answers,
  one render pass per page, model unload when idle
- Existing document types, correspondents and tags are offered to the model, most used first, so names stay
  consistent
- Learning: term mappings and verified examples from your corrections
- Safe write-back: existing type and correspondent are kept, tags are merged, already classified documents are
  skipped
- Dashboard: live status with timing breakdown, activity log with filters and correction modal, learning data,
  config editor with help text (applies live), dry run that shows the exact prompt and raw answer, log viewer,
  one-click update
- Polling and webhook triggers, guided setup, git-based updates

## Requirements

- Debian 12 / Ubuntu 22.04 or newer (Python 3.9+, `python3-venv`, `poppler-utils`, `git`)
- Ollama 0.5 or newer with a model pulled (`ollama pull gemma4:e4b`)
- Paperless-ngx with an API token
- RAM for the model: `gemma4:e4b` needs roughly 6 to 8 GB while loaded

## Install

```bash
sudo git clone https://github.com/mwckr/paperless-ai-classifier.git /opt/paperless-classifier
cd /opt/paperless-classifier
sudo ./install.sh
```

The installer creates a virtual environment, asks for the essential settings one by one (Paperless URL and
token, Ollama URL and model, pages, auto-commit, poll interval, port), tests both connections, writes a
complete `.env` with comments, and installs a systemd service that runs from the checkout.

**Coming from the old copy-based installation** (files in `/opt/paperless-classifier` without `.git`):

```bash
sudo mv /opt/paperless-classifier /opt/paperless-classifier.old
sudo git clone https://github.com/mwckr/paperless-ai-classifier.git /opt/paperless-classifier
cd /opt/paperless-classifier && sudo ./install.sh     # offers to import .env and the audit database
```

**Proxmox:** `pve-install.sh` creates an Ollama container and an API container and runs the same installer.

## Update

```bash
sudo /opt/paperless-classifier/update.sh
```

or the **Update now** button in the dashboard (Tools tab). Both pull, install requirements, run the settings
check and restart. The check is silent when nothing is missing. Settings that a newer version introduced are
appended with their defaults, essential ones are asked, and a failing connection test offers to re-enter the
related values.

## Dashboard

`http://your-server:8001/`

| Tab | What it does |
|-----|--------------|
| Overview | Stats, current document with stage and elapsed time, queue, model in RAM, recent activity |
| Activity | Audit log with status filter, search and paging. Details (timing breakdown, model output, normalization, raw answer), Correct, Redo, Delete |
| Learning | Term mappings and verified examples |
| Config | Every setting with help text. Changes apply immediately, except host and port |
| Tools | Dry run for one document (prompt, raw answer, timing, nothing written), version and update, debug export |
| Logs | Log tail with level filter |

No authentication: keep the port on a trusted network.

## Configuration

Everything lives in `/opt/paperless-classifier/.env` and in the Config tab. The complete list with help text
is in `app/config.py`. The settings that matter most:

| Setting | Default | Notes |
|---------|---------|-------|
| `OLLAMA_MODEL` | `gemma4:e4b` | Any vision model from `ollama list` |
| `OLLAMA_TEMPERATURE` | `0.1` | Low values give stable names |
| `OLLAMA_NUM_CTX` | `4096` | The log warns when a prompt nearly fills it |
| `OLLAMA_NUM_PREDICT` | `512` | Output cap |
| `OLLAMA_THINK` | `false` | Thinking costs minutes on CPU and adds nothing here |
| `OLLAMA_STRUCTURED_OUTPUT` | `true` | JSON schema; falls back automatically on servers that reject it |
| `OLLAMA_KEEP_ALIVE` / `OLLAMA_UNLOAD_WHEN_IDLE` | `5m` / `false` | RAM between batches |
| `OLLAMA_THREADS` | `0` | 0 = let Ollama choose |
| `MAX_PAGES` | `3` | Each page is one vision encoder pass |
| `IMAGE_MAX_SIZE` | `1024` | Longest side |
| `OCR_TEXT_MODE` | `off` | `hint` adds OCR text as a reading aid next to the images, `only` sends no images (last resort) |
| `AUTO_COMMIT` | `true` | Write results to Paperless |
| `OVERWRITE_EXISTING` | `false` | Only fill empty type/correspondent. Re-analyze always overwrites |
| `TAGS_MODE` | `merge` | Keep the document's existing tags |
| `SKIP_CLASSIFIED` | `true` | Documents with type and correspondent are not sent to the model |
| `INJECT_EXISTING_*`, `PROMPT_*_LIMIT` | on, 50/60/80 | Reference lists in the prompt, most used first |
| `LEARNING_ENABLED`, `FUZZY_MATCH_THRESHOLD` | `true`, `0.80` | Normalization |
| `POLL_INTERVAL` | `60` | 0 = webhooks only |

## Tuning for slow hardware

Per document the model pays for loading itself, reading the prompt (the images dominate) and writing the
answer. The `Ollama timing` log line and the Details view show the split. In order of effect:

1. `MAX_PAGES=1` or `2`. The first page carries the letterhead and most of the structure.
2. `IMAGE_MAX_SIZE=768`. Still enough for layout and logos.
3. `OLLAMA_UNLOAD_WHEN_IDLE=true` when Paperless and Ollama share the machine.
4. Leave `OLLAMA_THREADS=0` or set it to the number of physical cores.
5. Keep `OLLAMA_NUM_CTX` at 4096 unless the log complains. The KV cache grows with it.
6. On the Ollama host: `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, and on recent versions
   `OLLAMA_FLASH_ATTENTION=1` with `OLLAMA_KV_CACHE_TYPE=q8_0`.

Use the dry run in the Tools tab to compare settings on one document without touching Paperless.

## Triggers

- Polling: every `POLL_INTERVAL` seconds, all documents newer than the last seen id
- Webhook: Paperless workflow action "Webhook" to `http://your-server:8001/webhook/paperless` with a body
  containing `document_id`, or a post-consume script calling `POST /webhook/paperless/{id}`
- API: `POST /api/classify {"document_id": N}`, `POST /api/classify/batch [N, …]`,
  `POST /api/reanalyze/{id}`

Interactive API docs: `/api/docs`.

## Development

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt pytest
cp .env.example .env            # or run: venv/bin/python -m app.setup
venv/bin/python -m app          # http://localhost:8001
venv/bin/python -m pytest tests/   # unit tests plus an end-to-end run against mock Paperless/Ollama servers
```

`PCLS_ENV`, `PCLS_DB` and `PCLS_LOGS` override the locations of `.env`, the SQLite database and the log directory.

## Service

```bash
systemctl status paperless-classifier
journalctl -u paperless-classifier -f
```

Uninstall: `systemctl disable --now paperless-classifier`, remove `/etc/systemd/system/paperless-classifier.service`
and the checkout.

## License

MIT
