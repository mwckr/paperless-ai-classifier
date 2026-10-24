# CLAUDE.md — Paperless AI Classifier

Guidance for AI assistants working on this codebase.

## Project Overview

**Paperless AI Classifier** is a Python service that classifies documents in
[Paperless-ngx](https://docs.paperless-ngx.com/) with a local [Ollama](https://ollama.ai/) model. It assigns
document type, correspondent and tags. The target deployment is a home server or Proxmox LXC without a GPU
and with limited RAM, so every design decision favours fewer tokens and less resident memory. Prompts and
results are German.

**Deployed files (copied by `install.sh` / `deploy.sh`):**
- `classifier_api_v2.py` — FastAPI service: asyncio queue worker, polling loop, webhook, SQLite audit log, embedded dashboard
- `gemma4.py` — classification engine (any Ollama model, not only Gemma): fetch document, build prompt, call Ollama, normalize, update Paperless
- `learning.py` — fuzzy matching, term mappings, few-shot examples (SQLite tables in the same database)

**Legacy, not deployed by any script:** `ministral.py`, `classifier_api.py`, `test_gemma4.py`, `upgrade_v2.sh`.
Do not extend them. If a change is needed in both, the v2 files are the source of truth.

---

## Repository Structure

```
paperless-ai-classifier/
├── classifier_api_v2.py   # FastAPI server, queue, dashboard, endpoints  (deployed)
├── gemma4.py              # Classification engine                        (deployed)
├── learning.py            # Normalization + learning layer               (deployed)
├── install.sh             # Interactive installer -> /opt/paperless-classifier + systemd unit
├── deploy.sh              # Copy the three deployed files to /opt and restart
├── upgrade_v3.sh          # Upgrade with backup; adds missing .env keys (add_env)
├── pve-install.sh         # Proxmox: creates Ollama LXC + API LXC
├── classifier_audit.db    # SQLite audit/learning database (runtime artifact, empty copy in repo)
├── README.md              # User documentation
└── ministral.py, classifier_api.py, test_gemma4.py, upgrade_v2.sh   # legacy
```

No `requirements.txt`: dependencies are installed by `install.sh`
(`fastapi uvicorn python-dotenv pydantic requests pillow`, plus `poppler-utils` for `pdftoppm`).

---

## Configuration

`.env` next to the script, or `/opt/paperless-classifier/.env`. `get_config()` in `classifier_api_v2.py` is the
single parser (`_env_int` / `_env_float` / `_env_bool` helpers). `gemma4.py` receives the dict via `init_config()`
and never reads the environment itself. **When adding a key:** add it to `get_config()`, to `_ALLOWED_CONFIG_KEYS`,
to the dashboard config form (`items` array and `keys` list in `saveConfig`), to `install.sh`, `pve-install.sh`
and the `add_env` block in `upgrade_v3.sh`, and to the README table.

| Group | Keys |
|-------|------|
| Connection | `PAPERLESS_URL`, `PAPERLESS_TOKEN`, `OLLAMA_URL`, `OLLAMA_MODEL` (default `gemma4:e4b`), `API_HOST`, `API_PORT`, `POLL_INTERVAL` |
| Ollama runtime | `OLLAMA_THREADS` (0 = auto), `OLLAMA_TEMPERATURE` / `OLLAMA_TOP_P` / `OLLAMA_TOP_K` (only sent if set), `OLLAMA_NUM_CTX` (4096), `OLLAMA_NUM_PREDICT` (512), `OLLAMA_THINK` (false), `OLLAMA_STRUCTURED_OUTPUT` (true), `OLLAMA_KEEP_ALIVE` (5m), `OLLAMA_UNLOAD_WHEN_IDLE` (false), `OLLAMA_TIMEOUT` (600) |
| Input | `OCR_TEXT_MODE` (hint / only / off), `OCR_TEXT_CHARS` (2000), `MAX_PAGES` (3), `IMAGE_MAX_SIZE` (1024), `IMAGE_QUALITY` (85), `IMAGE_GRAYSCALE` (false), `PDF_RENDER_TIMEOUT` (120) |
| Write-back | `AUTO_COMMIT` (true), `OVERWRITE_EXISTING` (false), `TAGS_MODE` (merge / replace), `GENERATE_EXPLANATIONS` (false) |
| Prompt / learning | `INJECT_EXISTING_TYPES`, `INJECT_EXISTING_CORRESPONDENTS`, `INJECT_EXISTING_TAGS` (all true), `PROMPT_TYPE_LIMIT` (50), `PROMPT_CORRESPONDENT_LIMIT` (60), `PROMPT_TAG_LIMIT` (80), `LEARNING_ENABLED` (true), `FUZZY_MATCH_THRESHOLD` (0.80), `FEW_SHOT_ENABLED` (false) |

---

## Application Flow

### Startup (`lifespan`)
1. `init_database()` creates `audit_log`, `term_mappings`, `classification_examples`
2. Creates the `asyncio.Queue`, starts `process_queue()` and `poll_for_new_documents()`

### Triggers
- **Polling**: newest-first pages of `/api/documents/` until an already seen id; everything newer is queued.
- **Webhook**: `POST /webhook/paperless` (body with `document_id`, `id`, `pk` or `document`) or `/webhook/paperless/{id}`.
- **API**: `/api/classify`, `/api/classify/batch` (bare JSON list), `/api/reanalyze/{id}` (adds the id to
  `_force_overwrite_ids`, so the result overwrites an existing type/correspondent).

Dedup: `_pending_set` holds an id from enqueue until processing has finished.

### Worker (`process_queue`)
1. `setup_model()` imports `gemma4` and passes a fresh config (config changes need a restart only for the poller)
2. Fetches metadata once (title, OCR `content`, current type/correspondent/tags) and reuses it everywhere
3. `model.process_document(doc_id, learning_enabled, metadata)` in a thread
4. `AUTO_COMMIT`: `model.update_document_in_paperless(doc_id, result, metadata, overwrite)`
5. `log_to_audit(...)` updates the `processing` row to `completed` / `failed`
6. When the queue is empty: `invalidate_cache()` and, with `OLLAMA_UNLOAD_WHEN_IDLE`, `unload_model()`

### Engine (`gemma4.py`)
1. `extract_ocr_excerpt(metadata)` — whitespace-collapsed first `OCR_TEXT_CHARS` characters of Paperless' OCR text
2. `fetch_document_images()` — `pdftoppm -jpeg -scale-to IMAGE_MAX_SIZE -l MAX_PAGES` into a temp dir, no second
   encode pass; image originals go through Pillow; fallbacks preview → thumbnail. Skipped in `OCR_TEXT_MODE=only`
   when OCR text exists.
3. `_build_prompt(ocr_text, num_images)` — context (page count, OCR block, reference lists most-used-first,
   few-shot) → numbered rules → exact JSON shape as the last line
4. `analyze_with_vision()` — `POST /api/chat` with `stream=false`, `think`, `format` (JSON schema from
   `_json_schema()`), `keep_alive`, options `num_ctx` / `num_predict` / sampling / `num_thread`. On HTTP 400 it drops
   `think` and/or `format`, remembers that in `_server_caps`, and retries once. Logs Ollama's real
   `prompt_eval_count` / `eval_count` / durations and warns when the prompt nearly fills `num_ctx`.
5. `_parse_response()` → `_coerce_result()` (types, tag list, confidence in [0, 1], placeholder correspondents → None)
6. `normalize_result()` from `learning.py` (mappings, fuzzy matching, tag filtering)

### Write-back (`update_document_in_paperless`)
- `OVERWRITE_EXISTING=false`: type and correspondent are only set when the document has none
- `TAGS_MODE=merge`: new tag ids are appended to the document's existing tag ids
- Nothing is PATCHed when nothing changes
- `_get_or_create_resource()` checks the cache, then `name__iexact`, then POSTs. Always use the `get_or_create_*`
  helpers instead of POSTing to Paperless resource endpoints directly.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/`, `/dashboard` | Dashboard (embedded HTML) |
| GET | `/api/health` | Paperless + Ollama reachability |
| GET | `/api/queue` | Processing state and pending ids |
| GET | `/api/stats` | Audit statistics |
| GET | `/api/audit?limit=&offset=` | Audit log |
| GET / POST | `/api/config` | Read all (token redacted) / write one key `{"key","value"}` |
| POST | `/api/restart` | Exit; systemd restarts |
| GET | `/api/ollama/models` | Models on the Ollama server |
| POST | `/api/classify` | `{"document_id": N}` |
| POST | `/api/classify/batch` | `[N, ...]` |
| POST | `/api/reanalyze/{id}` | Re-queue with overwrite |
| DELETE | `/api/queue/clear` | Clear queue |
| DELETE | `/api/audit/{id}`, `/api/audit/status/{status}` | Delete audit rows |
| POST | `/api/audit/cleanup` | Mark stale `processing` rows `abandoned` |
| GET / POST / DELETE | `/api/mappings[/{id}]` | Term mappings |
| GET / DELETE | `/api/learn/examples` | Few-shot examples |
| DELETE | `/api/learn/mappings`, `/api/learn/reset` | Clear learning data |
| POST | `/api/learn/correct` | Apply a correction from the dashboard modal |
| GET | `/api/paperless/document/{id}` | Current Paperless values with names |
| GET | `/api/export/debug`, `/api/export/list`, `/api/export/download/{f}` | Debug exports |
| DELETE | `/api/export/{f}` | Delete export |
| GET | `/api/logs?lines=&level=` | Log tail |
| POST | `/webhook/paperless[/{id}]` | Webhook |

No authentication on any endpoint.

---

## Database Schema (`classifier_audit.db`)

```sql
audit_log(id, document_id, document_title, timestamp, status,      -- processing|completed|failed|abandoned
          document_type, correspondent, tags /*JSON*/, confidence, processing_time,
          tokens_used /*prompt_eval_count+eval_count*/, auto_approved, error_message, explanation, ai_raw);
term_mappings(id, term_type /*document_type|correspondent|tag*/, ai_term, approved_term, times_used,
              created_at, last_used, UNIQUE(term_type, ai_term));
classification_examples(id, document_id, document_title, document_type, correspondent, tags /*JSON*/,
                        confidence, user_verified, created_at);
```

---

## Code Conventions

- Type hints (`Optional`, `Dict`, `List`, `Tuple`); `logging.getLogger(__name__)`; no `print()` in deployed files
- `try/except` with context in the log message; return `None` / `False` rather than raising across module boundaries
- Config is read from the config dict (`_config.get(...)`, `_cfg_bool`, `_cfg_int` in `gemma4.py`), never `os.environ` inside functions
- Blocking `requests` calls from the event loop go through `asyncio.to_thread()`
- Dashboard: everything rendered via `innerHTML` from model output or document titles goes through `esc()`;
  attach handlers with `data-*` attributes and `addEventListener`, not hand-built `onclick` strings
- Prompt and model-facing strings are German; code, comments and logs are English
- No test framework in the repo. Changes are verified with mocked Paperless/Ollama servers; keep functions
  pure enough that `_parse_response`, `_coerce_result`, `_build_prompt`, `extract_ocr_excerpt`, `fuzzy_match`
  and `normalize_result` stay testable without network

---

## Notes for AI Assistants

1. **Low-RAM goal.** Anything added to the prompt or to the image pipeline costs CPU time per document.
   Prefer text over images, bound every list, and keep `num_ctx` small.
2. **Single-file deployment.** `install.sh`/`deploy.sh` copy exactly three files. New modules mean updating
   both scripts and `upgrade_v3.sh`.
3. **Embedded dashboard.** `DASHBOARD_HTML` in `classifier_api_v2.py` is intentional. Keep the CSS variable
   structure and the 5-second `refreshAll` interval.
4. **Paperless semantics.** Tag/type/correspondent names are unique case-insensitively; `/download/` returns the
   archived (OCRed) PDF; the document's OCR text is the `content` field of `/api/documents/{id}/`.
5. **Ollama semantics.** Prompts longer than `num_ctx` are truncated silently by the server; `think` and `format`
   may be rejected by old versions (handled); `keep_alive: 0` on `/api/generate` unloads a model.
