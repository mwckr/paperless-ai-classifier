"""
FastAPI service: queue worker, polling, webhook, dashboard API.
"""
import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, db
from .classify import Classifier, Outcome
from .config import FIELD_MAP, LOG_DIR, ROOT, Settings, schema
from .learning import learn_from_correction
from .ollama import Ollama
from .paperless import Paperless

# ----------------------------------------------------------------------------- logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "classifier.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=3), logging.StreamHandler()],
)
logger = logging.getLogger("app")
STATIC_DIR = Path(__file__).parent / "static"


# ----------------------------------------------------------------------------- state
class State:
    def __init__(self) -> None:
        self.settings: Settings = Settings.load()
        self.paperless = Paperless(self.settings)
        self.ollama = Ollama(self.settings)
        self.queue: "asyncio.Queue[int]" = asyncio.Queue()
        self.pending: Set[int] = set()        # queued or being processed
        self.force: Set[int] = set()          # re-analyze: overwrite + ignore SKIP_CLASSIFIED
        self.cancelled: Set[int] = set()
        self.inference_lock = asyncio.Lock()  # one model request at a time (worker + dry run)
        self.status: Dict[str, Any] = {
            "processing": False, "current_doc": None, "current_title": None, "stage": None,
            "started_at": None, "processed_count": 0, "failed_count": 0, "last_processed": None,
            "service_started": datetime.now().isoformat(timespec="seconds"),
        }
        self.last_seen_doc_id = 0
        self.version = git_version()
        self.apply_log_level()

    def reload(self) -> None:
        self.settings = Settings.load()
        self.paperless = Paperless(self.settings)
        self.ollama = Ollama(self.settings)
        self.apply_log_level()
        logger.info("Settings reloaded")

    def apply_log_level(self) -> None:
        logging.getLogger().setLevel(getattr(logging, str(self.settings.LOG_LEVEL), logging.INFO))

    def classifier(self) -> Classifier:
        def on_stage(stage: str) -> None:
            self.status["stage"] = stage
        return Classifier(self.settings, self.paperless, self.ollama, on_stage=on_stage)


def git_version() -> Dict[str, Any]:
    info: Dict[str, Any] = {"version": __version__, "commit": None, "commit_date": None, "branch": None,
                            "subject": None, "python": platform.python_version()}
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h%n%cs%n%s"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            commit, cdate, subject = (out.stdout.strip().split("\n") + ["", "", ""])[:3]
            info.update(commit=commit, commit_date=cdate, subject=subject)
        branch = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=5)
        if branch.returncode == 0:
            info["branch"] = branch.stdout.strip()
    except Exception:
        pass
    return info


state: State = None  # type: ignore[assignment]  # created in lifespan


# ----------------------------------------------------------------------------- processing
def process_one(doc_id: int, force: bool) -> None:
    """Runs in a worker thread: one document end to end, audit row included."""
    s = state.settings
    clf = state.classifier()
    st = state.status
    st.update(processing=True, current_doc=doc_id, current_title=None, stage="fetching",
              started_at=datetime.now().isoformat(timespec="seconds"))

    doc = state.paperless.get_document(doc_id)
    title = (doc or {}).get("title") or f"Document {doc_id}"
    st["current_title"] = title
    audit_id = db.audit_start(doc_id, title, s.OLLAMA_MODEL)

    if not doc:
        db.audit_finish(audit_id, "failed", error_message="metadata_fetch_failed")
        st["failed_count"] += 1
        return
    if s.SKIP_CLASSIFIED and not force and Classifier.is_classified(doc):
        logger.info(f"Document {doc_id} already has type and correspondent, skipped (SKIP_CLASSIFIED)")
        db.audit_finish(audit_id, "skipped", error_message="already classified")
        return

    out: Outcome = clf.analyze(doc_id, doc)
    if out.ok and s.AUTO_COMMIT:
        clf.write_back(out, doc, overwrite=force or bool(s.OVERWRITE_EXISTING))

    if out.ok:
        r = out.result
        db.audit_finish(audit_id, "completed", document_type=r.get("document_type"),
                        correspondent=r.get("correspondent"), tags=r.get("tags"), confidence=r.get("confidence"),
                        processing_time=out.timing.get("total_s"), tokens_used=out.tokens,
                        auto_approved=out.committed, explanation=r.get("explanation") or None,
                        ai_raw=json.dumps({"model_output": out.model_output, "normalized": r.get("normalized"),
                                           "answer": out.raw_answer[:2000]}, ensure_ascii=False),
                        pages_sent=out.pages_sent, timing=out.timing, changes=out.changes)
        st["processed_count"] += 1
        st["last_processed"] = {"doc_id": doc_id, "title": title, "type": r.get("document_type"),
                                "correspondent": r.get("correspondent"),
                                "time": datetime.now().isoformat(timespec="seconds")}
    else:
        logger.error(f"Document {doc_id} failed: {out.reason}")
        db.audit_finish(audit_id, "failed", error_message=out.reason, processing_time=out.timing.get("total_s"),
                        tokens_used=out.tokens, pages_sent=out.pages_sent, timing=out.timing,
                        ai_raw=out.raw_answer[:2000] or None)
        st["failed_count"] += 1


async def worker() -> None:
    logger.info("Queue worker started")
    while True:
        doc_id = await state.queue.get()
        force = doc_id in state.force
        state.force.discard(doc_id)
        try:
            if doc_id in state.cancelled:
                state.cancelled.discard(doc_id)
                continue
            async with state.inference_lock:
                await asyncio.to_thread(process_one, doc_id, force)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception(f"Worker error on document {doc_id}: {e}")
        finally:
            state.pending.discard(doc_id)
            state.status.update(processing=False, current_doc=None, current_title=None, stage=None, started_at=None)
            state.queue.task_done()
            if state.queue.empty():
                state.paperless.invalidate_cache()
                if state.settings.OLLAMA_UNLOAD_WHEN_IDLE:
                    await asyncio.to_thread(state.ollama.unload)


async def poller() -> None:
    await asyncio.sleep(3)
    try:
        docs, _ = await asyncio.to_thread(state.paperless.newest_documents, 1, 1)
        if docs:
            state.last_seen_doc_id = docs[0]["id"]
            logger.info(f"Polling starts after document id {state.last_seen_doc_id}")
    except Exception as e:
        logger.warning(f"Could not read newest document id: {e}")
    while True:
        interval = int(state.settings.POLL_INTERVAL)
        if interval <= 0:
            await asyncio.sleep(30)
            continue
        await asyncio.sleep(interval)
        try:
            new_docs: List[Dict[str, Any]] = []
            for page in range(1, 11):
                docs, has_next = await asyncio.to_thread(state.paperless.newest_documents, page, 50)
                fresh = [d for d in docs if d["id"] > state.last_seen_doc_id]
                new_docs.extend(fresh)
                if len(fresh) < len(docs) or not has_next:
                    break
            if new_docs:
                logger.info(f"Polling found {len(new_docs)} new document(s)")
                for d in reversed(new_docs):
                    await enqueue(d["id"])
                state.last_seen_doc_id = max(d["id"] for d in new_docs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Polling error: {e}")


async def enqueue(doc_id: int, force: bool = False) -> str:
    if force:
        state.force.add(doc_id)
    if doc_id in state.pending:
        return "already_queued"
    state.pending.add(doc_id)
    state.cancelled.discard(doc_id)
    await state.queue.put(doc_id)
    return "queued"


# ----------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    db.init_db()
    state = State()
    s = state.settings
    logger.info(f"Paperless AI Classifier {__version__} ({state.version.get('commit') or 'no git'}) | "
                f"Paperless {s.PAPERLESS_URL} | Ollama {s.OLLAMA_URL} ({s.OLLAMA_MODEL}) | "
                f"poll {s.POLL_INTERVAL}s | listening on {s.API_HOST}:{s.API_PORT}")
    tasks = [asyncio.create_task(worker()), asyncio.create_task(poller())]
    yield
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Shutdown complete")


app = FastAPI(title="Paperless AI Classifier", version=__version__, lifespan=lifespan, docs_url="/api/docs",
              redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ClassifyRequest(BaseModel):
    document_id: int


class ConfigUpdate(BaseModel):
    key: Optional[str] = None
    value: Optional[str] = None
    values: Optional[Dict[str, str]] = None


class MappingCreate(BaseModel):
    term_type: str
    ai_term: str
    approved_term: str


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    pl, ol = await asyncio.gather(asyncio.to_thread(state.paperless.health), asyncio.to_thread(state.ollama.health))
    return {"api": "healthy", "paperless": pl, "ollama": ol, "version": state.version}


@app.get("/api/status")
async def status():
    loaded = await asyncio.to_thread(state.ollama.loaded)
    return {
        **state.status,
        "queue_size": state.queue.qsize(),
        "pending": sorted(d for d in state.pending if d != state.status["current_doc"]),
        "stats": db.audit_stats(),
        "model": state.settings.OLLAMA_MODEL,
        "models_loaded": loaded,
        "version": state.version,
    }


@app.get("/api/version")
async def version():
    return state.version


# --- queue -------------------------------------------------------------------------
@app.post("/api/classify")
async def classify(req: ClassifyRequest):
    return {"status": await enqueue(req.document_id), "document_id": req.document_id}


@app.post("/api/classify/batch")
async def classify_batch(document_ids: List[int]):
    results = [await enqueue(d) for d in document_ids]
    return {"queued": results.count("queued"), "skipped": results.count("already_queued")}


@app.post("/api/reanalyze/{doc_id}")
async def reanalyze(doc_id: int):
    return {"status": await enqueue(doc_id, force=True), "document_id": doc_id}


@app.delete("/api/queue/clear")
async def queue_clear():
    removed = 0
    while not state.queue.empty():
        try:
            state.queue.get_nowait(); state.queue.task_done(); removed += 1
        except asyncio.QueueEmpty:
            break
    current = state.status["current_doc"]
    state.pending = {current} if current else set()
    state.force.clear()
    return {"removed": removed}


@app.delete("/api/queue/{doc_id}")
async def queue_remove(doc_id: int):
    if doc_id == state.status["current_doc"]:
        raise HTTPException(409, "Document is being processed")
    if doc_id not in state.pending:
        raise HTTPException(404, "Not queued")
    state.cancelled.add(doc_id)
    state.pending.discard(doc_id)
    return {"status": "removed", "document_id": doc_id}


@app.post("/api/dryrun/{doc_id}")
async def dryrun(doc_id: int):
    """Full analysis without writing anything. Returns prompt, raw answer, timing."""
    async with state.inference_lock:
        state.status.update(processing=True, current_doc=doc_id, current_title=f"dry run {doc_id}",
                            started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            out: Outcome = await asyncio.to_thread(state.classifier().analyze, doc_id)
        finally:
            state.status.update(processing=False, current_doc=None, current_title=None, stage=None, started_at=None)
    return out.to_dict()


# --- config --------------------------------------------------------------------------
@app.get("/api/config")
async def config_get():
    return {"values": state.settings.as_dict(redact=True), "schema": schema()}


@app.post("/api/config")
async def config_set(update: ConfigUpdate):
    values = dict(update.values or {})
    if update.key is not None:
        values[update.key] = update.value or ""
    if not values:
        raise HTTPException(400, "Nothing to update")
    restart = False
    changed = []
    for key, value in values.items():
        field = FIELD_MAP.get(key)
        if field is None:
            raise HTTPException(400, f"Unknown setting {key}")
        if field.type == "secret" and value == "***REDACTED***":
            continue
        try:
            Settings.save(key, value)
        except ValueError as e:
            raise HTTPException(400, str(e))
        changed.append(key)
        restart = restart or field.restart
    state.reload()
    return {"updated": changed, "restart_required": restart}


@app.post("/api/restart")
async def restart():
    logger.info("Restart requested from dashboard")
    asyncio.get_running_loop().call_later(0.5, lambda: os._exit(0))
    return {"status": "restarting"}


@app.get("/api/ollama/models")
async def ollama_models():
    return {"models": await asyncio.to_thread(state.ollama.models)}


@app.post("/api/ollama/unload")
async def ollama_unload():
    return {"unloaded": await asyncio.to_thread(state.ollama.unload)}


# --- paperless -----------------------------------------------------------------------
@app.get("/api/paperless/names")
async def paperless_names():
    pl = state.paperless
    return {kind: await asyncio.to_thread(pl.names, kind) for kind in ("document_types", "correspondents", "tags")}


@app.get("/api/paperless/document/{doc_id}")
async def paperless_document(doc_id: int):
    doc = await asyncio.to_thread(state.paperless.document_with_names, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found in Paperless")
    return doc


# --- audit ---------------------------------------------------------------------------
@app.get("/api/audit")
async def audit(limit: int = 50, offset: int = 0, status: Optional[str] = None, q: Optional[str] = None):
    return db.audit_list(min(limit, 500), offset, status or None, (q or "").strip() or None)


@app.get("/api/audit/{audit_id}")
async def audit_entry(audit_id: int):
    row = db.audit_get(audit_id)
    if not row:
        raise HTTPException(404, "Audit entry not found")
    return row


@app.delete("/api/audit/status/{status}")
async def audit_delete_status(status: str):
    if status not in db.AUDIT_STATUSES:
        raise HTTPException(400, f"status must be one of {', '.join(db.AUDIT_STATUSES)}")
    return {"deleted": db.audit_delete_status(status)}


@app.delete("/api/audit/{audit_id}")
async def audit_delete(audit_id: int):
    if not db.audit_delete(audit_id):
        raise HTTPException(404, "Audit entry not found")
    return {"status": "deleted"}


@app.post("/api/audit/cleanup")
async def audit_cleanup():
    return {"updated": db.audit_abandon_stale(1.0)}


# --- learning ----------------------------------------------------------------------------
@app.get("/api/mappings")
async def mappings():
    return {"mappings": db.mapping_list()}


@app.post("/api/mappings")
async def mapping_create(m: MappingCreate):
    if m.term_type not in ("document_type", "correspondent", "tag"):
        raise HTTPException(400, "term_type must be document_type, correspondent or tag")
    if not db.mapping_upsert(m.term_type, m.ai_term, m.approved_term):
        raise HTTPException(400, "Both terms are required and must differ")
    return {"status": "saved"}


@app.delete("/api/mappings/{mapping_id}")
async def mapping_delete(mapping_id: int):
    if not db.mapping_delete(mapping_id):
        raise HTTPException(404, "Mapping not found")
    return {"status": "deleted"}


@app.get("/api/examples")
async def examples():
    return {"examples": db.example_list()}


@app.delete("/api/examples/{example_id}")
async def example_delete(example_id: int):
    if not db.example_delete(example_id):
        raise HTTPException(404, "Example not found")
    return {"status": "deleted"}


@app.delete("/api/learn/mappings")
async def learn_clear_mappings():
    return {"deleted": db.mappings_clear()}


@app.delete("/api/learn/examples")
async def learn_clear_examples():
    return {"deleted": db.examples_clear()}


@app.delete("/api/learn/reset")
async def learn_reset():
    return {"mappings_deleted": db.mappings_clear(), "examples_deleted": db.examples_clear()}


@app.post("/api/learn/correct")
async def learn_correct(request: Request):
    """Body: {"audit_id": N, "corrections": {"document_type": {"ai","user","sync"},
    "correspondent": {...}, "tags": {"ai": [...], "user": [...], "sync"}, "tag_mappings": [{"ai","user"}]}}"""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    audit_row = db.audit_get(int(data.get("audit_id") or 0))
    if not audit_row:
        raise HTTPException(404, "Audit entry not found")
    corrections = data.get("corrections") or {}
    learned = learn_from_correction(audit_row, corrections)

    changed: List[str] = []
    sync_type = (corrections.get("document_type") or {}).get("sync")
    sync_corr = (corrections.get("correspondent") or {}).get("sync")
    sync_tags = (corrections.get("tags") or {}).get("sync")
    if sync_type or sync_corr or sync_tags:
        changed = await asyncio.to_thread(
            state.classifier().apply_values, audit_row["document_id"],
            (corrections.get("document_type") or {}).get("user") if sync_type else None,
            (corrections.get("correspondent") or {}).get("user") if sync_corr else None,
            list((corrections.get("tags") or {}).get("user") or []) if sync_tags else None,
        )
    return {"learned": learned, "paperless_updated": changed}


# --- logs, export, update -----------------------------------------------------------------
@app.get("/api/logs")
async def logs(lines: int = 200, level: Optional[str] = None):
    if not LOG_FILE.exists():
        return {"lines": [], "total": 0}
    content = await asyncio.to_thread(LOG_FILE.read_text, encoding="utf-8", errors="replace")
    all_lines = content.splitlines()
    if level:
        marker = f"| {level.upper()} |"
        all_lines = [ln for ln in all_lines if marker in ln]
    return {"lines": all_lines[-min(max(lines, 1), 2000):], "total": len(all_lines)}


@app.get("/api/export/debug")
async def export_debug():
    pl, ol = await asyncio.gather(asyncio.to_thread(state.paperless.health), asyncio.to_thread(state.ollama.health))
    log_tail = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-500:] if LOG_FILE.exists() else []
    data = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "version": state.version,
        "system": {"platform": platform.platform(), "python": platform.python_version(), "host": platform.node()},
        "config": state.settings.as_dict(redact=True),
        "health": {"paperless": pl, "ollama": ol, "models_loaded": await asyncio.to_thread(state.ollama.loaded)},
        "status": {**state.status, "queue_size": state.queue.qsize()},
        "stats": db.audit_stats(),
        "recent_runs": db.audit_list(50)["logs"],
        "mappings": db.mapping_list(),
        "examples": db.example_list(50),
        "log_tail": log_tail,
    }
    name = f"classifier-debug-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    return JSONResponse(content=json.loads(json.dumps(data, default=str)),
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})


UPDATE_LOG = LOG_DIR / "update.log"


@app.post("/api/update")
async def update_start():
    """Run update.sh (git pull, pip install, restart) outside our own cgroup so the restart does not kill it."""
    script = ROOT / "update.sh"
    if not script.exists():
        raise HTTPException(404, "update.sh not found")
    UPDATE_LOG.write_text(f"{datetime.now().isoformat(timespec='seconds')} starting update\n")
    cmd = ["bash", str(script)]
    env = dict(os.environ, PCLS_UPDATE_LOG=str(UPDATE_LOG))
    try:
        if shutil.which("systemd-run") and os.geteuid() == 0:
            unit = f"pcls-update-{datetime.now().strftime('%H%M%S')}"
            subprocess.Popen(["systemd-run", "--unit", unit, "--collect", "--quiet",
                              f"--setenv=PCLS_UPDATE_LOG={UPDATE_LOG}", *cmd], cwd=str(ROOT))
        else:
            log_fh = open(UPDATE_LOG, "a")
            subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_fh, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
    except Exception as e:
        raise HTTPException(500, f"Could not start update: {e}")
    return {"status": "started"}


@app.get("/api/update/log")
async def update_log():
    if not UPDATE_LOG.exists():
        return {"lines": []}
    return {"lines": UPDATE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]}


# --- webhooks --------------------------------------------------------------------------------
@app.post("/webhook/paperless/{doc_id}")
async def webhook_with_id(doc_id: int):
    logger.info(f"Webhook for document {doc_id}")
    return {"status": await enqueue(doc_id), "document_id": doc_id}


@app.post("/webhook/paperless")
async def webhook(request: Request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    doc_id = None
    for key in ("document_id", "id", "pk", "document_pk"):
        if key in payload:
            doc_id = payload[key]
            break
    if doc_id is None and "document" in payload:
        doc = payload["document"]
        doc_id = doc if isinstance(doc, int) else (doc or {}).get("id") or (doc or {}).get("pk")
    try:
        doc_id = int(doc_id)
    except (TypeError, ValueError):
        logger.warning(f"Webhook without document id: {str(payload)[:200]}")
        return {"status": "ignored", "reason": "no document_id found"}
    logger.info(f"Webhook for document {doc_id}")
    return {"status": await enqueue(doc_id), "document_id": doc_id}


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use: python -m app")
