# CLAUDE.md — Paperless AI Classifier

Guidance for AI assistants working on this codebase.

## What it is

A Python service that classifies Paperless-ngx documents (type, correspondent, tags) with a local Ollama vision
model. Target hardware: a home server or Proxmox LXC without a GPU and with limited RAM. Prompts and results are
German; code, comments and logs are English.

**Design decisions that are not up for revision without the owner:**

1. **Page images are the input.** Classifying from Paperless' flat OCR text was tested and found worse: the layout
   (sender top left, client table in the middle, bank details in the footer) is what lets the model relate the
   pieces. `OCR_TEXT_MODE` defaults to `off`; `hint` and `only` exist as opt-ins. Do not move accuracy work onto
   the text path. Cost reductions belong in fewer/smaller pages, bounded prompt lists, no thinking, capped output
   and a small context window.
2. **Lean deployment.** A git checkout, a venv, one systemd unit. `install.sh` and `update.sh` are the only
   deployment scripts (plus `pve-install.sh` for Proxmox, which calls `install.sh`). No file copying.
3. **No authentication.** LAN only, by the owner's choice.
4. **Safe write-back by default.** Existing type/correspondent are kept (`OVERWRITE_EXISTING=false`), tags are
   merged (`TAGS_MODE=merge`), already classified documents are skipped (`SKIP_CLASSIFIED=true`). Re-analyze from
   the dashboard overrides all three on purpose.

## Layout

```
app/
  config.py     FIELDS table: every setting with type, default, group, label, help. Drives parsing, validation,
                the dashboard form and the guided setup. Settings.load() / Settings.save(key, value).
  db.py         SQLite (audit_log, term_mappings, classification_examples). init_db() migrates older databases
                by adding missing columns.
  paperless.py  Paperless client: documents, download, cached resource lists (most used first), get_or_create.
  ollama.py     Ollama client: chat with think/format/keep_alive/options, 400-retry that remembers rejected
                fields, real token and timing figures, unload, loaded models.
  images.py     pdftoppm rendering (one pass, straight to JPEG), Pillow for image originals, fallbacks.
  prompt.py     PromptContext -> build_prompt(), json_schema(), parse_response(), coerce_result(), ocr_excerpt().
  learning.py   fuzzy_match(), tag_repeats(), normalize(), few_shot_block(), learn_from_correction().
  classify.py   Classifier.analyze() -> Outcome; write_back(); apply_values() for corrections.
  main.py       FastAPI app: State, worker, poller, all routes, static files.
  setup.py      Guided setup (python -m app.setup [--check] [--non-interactive]).
  static/       index.html, style.css, app.js (vanilla JS, no build step).
tests/          mocks.py (Paperless + Ollama stand-ins), test_units.py, test_e2e.py (real service via uvicorn).
install.sh      venv, guided setup, systemd unit. Idempotent.
update.sh       git pull --ff-only, pip install, settings check, restart. Also run by the dashboard.
pve-install.sh  Proxmox: two LXCs, then install.sh --non-interactive inside the API container.
```

Runtime files next to the checkout, all gitignored: `.env`, `classifier_audit.db`, `logs/`, `venv/`.
`PCLS_ENV`, `PCLS_DB`, `PCLS_LOGS` override their locations (tests use that).

## Flow

1. Trigger: poller (`newest_documents` pages until an already seen id), webhook, `/api/classify`,
   `/api/reanalyze/{id}` (adds the id to `state.force`).
2. `worker()` takes one id, holds `state.inference_lock` (shared with dry runs), runs `process_one()` in a thread:
   `audit_start` -> skip check -> `Classifier.analyze()` -> `write_back()` (if `AUTO_COMMIT`) -> `audit_finish`.
3. `Classifier.analyze()`: document -> images (`fetch_page_images`) -> `PromptContext` (reference lists capped by
   `PROMPT_*_LIMIT`, optional OCR excerpt, few-shot) -> `Ollama.chat()` with schema -> `parse_response` ->
   `coerce_result` -> `learning.normalize()`.
4. When the queue drains: Paperless cache invalidated, model unloaded if `OLLAMA_UNLOAD_WHEN_IDLE`.
5. Settings changes via `/api/config` are validated by the FIELDS table, written to `.env`, and `State.reload()`
   rebuilds the clients. Only `API_HOST`/`API_PORT` need a restart.

## Adding a setting

Add one `Field(...)` to `FIELDS` in `app/config.py`. That is all: parsing, validation, the Config tab, the guided
setup (`ask=True` for essentials) and `update.sh`'s "new setting" handling follow from it. Read it in code as
`settings.KEY`. Mention it in the README table only if users need to know about it.

## Conventions

- Type hints; `logging.getLogger(__name__)`; no `print()` outside `app/setup.py`.
- Blocking HTTP from the event loop goes through `asyncio.to_thread()`; anything that talks to the model runs
  under `state.inference_lock`.
- Dashboard: untrusted strings (model output, titles) go through `esc()` before `innerHTML`; handlers are attached
  with `addEventListener`/`data-*`, never built into `onclick` strings.
- Keep `app/static/*` framework-free and dependency-free.
- Tests: `python -m pytest tests/`. `test_e2e.py` starts the real service against the mocks, so any API change
  must keep it green. Add a unit test for pure functions (prompt, parsing, fuzzy rules).

## Paperless and Ollama facts the code relies on

- Paperless names are unique case-insensitively; `/download/` returns the archived (OCRed) PDF when it exists;
  `document_count` on tags/types/correspondents is used for "most used first".
- Ollama truncates prompts longer than `num_ctx` silently (the client warns at 90%); `think` and `format` may be
  rejected by older servers with HTTP 400 (handled and remembered per client instance); `keep_alive: 0` on
  `/api/generate` unloads a model; `/api/ps` lists loaded models.
