#!/usr/bin/env python3
"""
Paperless Document Classifier API v2
=====================================
FastAPI service with:
- SQLite audit log and web dashboard
- Gemma 4 / Ministral model support
- Learning layer for continuous improvement
- Dashboard training interface
"""
import os
import asyncio
import sqlite3
import logging
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Load environment variables
ENV_FILE = Path(__file__).parent / ".env"
if not ENV_FILE.exists():
    ENV_FILE = Path("/opt/paperless-classifier/.env")
load_dotenv(ENV_FILE)

# Configuration from .env
def get_config():
    load_dotenv(ENV_FILE, override=True)
    return {
        "PAPERLESS_URL": os.getenv("PAPERLESS_URL", "http://localhost:8000"),
        "PAPERLESS_TOKEN": os.getenv("PAPERLESS_TOKEN", ""),
        "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "gemma4:e4b"),
        "OLLAMA_THREADS": int(os.getenv("OLLAMA_THREADS", "10")),
        "MAX_PAGES": int(os.getenv("MAX_PAGES", "3")),
        "AUTO_COMMIT": os.getenv("AUTO_COMMIT", "true").lower() == "true",
        "GENERATE_EXPLANATIONS": os.getenv("GENERATE_EXPLANATIONS", "false").lower() == "true",
        "LEARNING_ENABLED": os.getenv("LEARNING_ENABLED", "true").lower() == "true",
        "API_HOST": os.getenv("API_HOST", "0.0.0.0"),
        "API_PORT": int(os.getenv("API_PORT", "8001")),
    }

# Database setup
DB_PATH = Path(__file__).parent / "classifier_audit.db"

def init_database():
    """Initialize SQLite database with all tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Audit log table
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            document_title TEXT,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            document_type TEXT,
            correspondent TEXT,
            tags TEXT,
            confidence REAL,
            processing_time REAL,
            tokens_used INTEGER,
            auto_approved INTEGER,
            error_message TEXT,
            explanation TEXT,
            ai_raw TEXT
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp DESC)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_document_id ON audit_log(document_id)')
    
    # Learning: Term mappings
    c.execute('''
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
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_mappings_lookup ON term_mappings(term_type, ai_term)')
    
    # Learning: Classification examples
    c.execute('''
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
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_examples_verified ON classification_examples(user_verified, confidence DESC)')
    
    conn.commit()
    conn.close()

def log_to_audit(
    document_id: int,
    document_title: str = None,
    status: str = "pending",
    document_type: str = None,
    correspondent: str = None,
    tags: List[str] = None,
    confidence: float = None,
    processing_time: float = None,
    tokens_used: int = None,
    auto_approved: bool = False,
    error_message: str = None,
    explanation: str = None,
    ai_raw: str = None
):
    """Log processing result to audit database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO audit_log 
        (document_id, document_title, timestamp, status, document_type, correspondent, 
         tags, confidence, processing_time, tokens_used, auto_approved, error_message, explanation, ai_raw)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        document_id,
        document_title,
        datetime.now().isoformat(),
        status,
        document_type,
        correspondent,
        json.dumps(tags) if tags else None,
        confidence,
        processing_time,
        tokens_used,
        1 if auto_approved else 0,
        error_message,
        explanation,
        ai_raw
    ))
    conn.commit()
    conn.close()

def get_audit_logs(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get audit log entries"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?', (limit, offset))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    for row in rows:
        if row.get('tags'):
            try:
                row['tags'] = json.loads(row['tags'])
            except:
                row['tags'] = []
    return rows

def get_audit_stats() -> Dict:
    """Get audit statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM audit_log WHERE status = "completed"')
    completed = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM audit_log WHERE status = "failed"')
    failed = c.fetchone()[0]
    c.execute('SELECT AVG(processing_time) FROM audit_log WHERE processing_time IS NOT NULL')
    avg_time = c.fetchone()[0] or 0
    c.execute('SELECT AVG(confidence) FROM audit_log WHERE confidence IS NOT NULL')
    avg_confidence = c.fetchone()[0] or 0
    
    # Learning stats
    c.execute('SELECT COUNT(*) FROM term_mappings')
    mappings = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM classification_examples WHERE user_verified = 1')
    verified = c.fetchone()[0]
    
    conn.close()
    return {
        "completed": completed, 
        "failed": failed, 
        "avg_processing_time": round(avg_time, 1),
        "avg_confidence": round(avg_confidence * 100, 1),
        "learned_mappings": mappings,
        "verified_examples": verified
    }

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'classifier_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Queue for document processing
document_queue: asyncio.Queue = None
queue_status: Dict = {
    "processing": False,
    "current_doc": None,
    "current_title": None,
    "started_at": None,
    "queue_size": 0,
    "processed_count": 0,
    "last_processed": None,
    "service_started": None,
    "pending_docs": []
}

# Polling configuration
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))
last_seen_doc_id = 0

# Pydantic models
class ClassifyRequest(BaseModel):
    document_id: int

class ConfigUpdate(BaseModel):
    key: str
    value: str

class CorrectionRequest(BaseModel):
    audit_id: int
    document_type: str = None
    correspondent: str = None
    tags: List[str] = None

class MappingCreate(BaseModel):
    term_type: str
    ai_term: str
    approved_term: str

# Import and configure model module
def setup_model():
    """Import and configure the appropriate model module (Gemma 4 or Ministral)"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    cfg = get_config()
    model_name = cfg["OLLAMA_MODEL"].lower()
    
    # Try Gemma 4 first if model name suggests it
    if "gemma" in model_name:
        try:
            import gemma4 as model_module
            logger.info("Using Gemma 4 model module")
        except ImportError:
            import ministral as model_module
            logger.info("Gemma 4 module not found, using Ministral")
    else:
        import ministral as model_module
        logger.info("Using Ministral model module")
    
    # Configure module
    model_module.PAPERLESS_URL = cfg["PAPERLESS_URL"]
    model_module.PAPERLESS_TOKEN = cfg["PAPERLESS_TOKEN"]
    model_module.OLLAMA_URL = cfg["OLLAMA_URL"]
    model_module.OLLAMA_MODEL = cfg["OLLAMA_MODEL"]
    model_module.NUM_THREADS = cfg["OLLAMA_THREADS"]
    
    return model_module

# Learning functions
def apply_learning_correction(audit_id: int, document_type: str = None, correspondent: str = None, tags: List[str] = None):
    """Apply a user correction and learn from it"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get original AI suggestion
    c.execute('SELECT * FROM audit_log WHERE id = ?', (audit_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    
    original = dict(row)
    original_tags = json.loads(original.get('tags', '[]')) if original.get('tags') else []
    
    # Learn document type mapping
    if document_type and original.get('document_type'):
        ai_type = original['document_type']
        if ai_type.lower() != document_type.lower():
            c.execute('''
                INSERT INTO term_mappings (term_type, ai_term, approved_term)
                VALUES ('document_type', ?, ?)
                ON CONFLICT(term_type, ai_term) DO UPDATE SET 
                    approved_term = excluded.approved_term,
                    times_used = times_used + 1,
                    last_used = strftime('%s', 'now')
            ''', (ai_type, document_type))
            logger.info(f"Learned mapping: document_type '{ai_type}' → '{document_type}'")
    
    # Learn correspondent mapping
    if correspondent and original.get('correspondent'):
        ai_corr = original['correspondent']
        if ai_corr.lower() != correspondent.lower():
            c.execute('''
                INSERT INTO term_mappings (term_type, ai_term, approved_term)
                VALUES ('correspondent', ?, ?)
                ON CONFLICT(term_type, ai_term) DO UPDATE SET 
                    approved_term = excluded.approved_term,
                    times_used = times_used + 1,
                    last_used = strftime('%s', 'now')
            ''', (ai_corr, correspondent))
            logger.info(f"Learned mapping: correspondent '{ai_corr}' → '{correspondent}'")
    
    # Learn tag mappings (fuzzy)
    if tags:
        from difflib import SequenceMatcher
        for ai_tag in original_tags:
            for user_tag in tags:
                ratio = SequenceMatcher(None, ai_tag.lower(), user_tag.lower()).ratio()
                if 0.5 < ratio < 1.0:  # Similar but not exact
                    c.execute('''
                        INSERT INTO term_mappings (term_type, ai_term, approved_term)
                        VALUES ('tag', ?, ?)
                        ON CONFLICT(term_type, ai_term) DO UPDATE SET 
                            approved_term = excluded.approved_term,
                            times_used = times_used + 1,
                            last_used = strftime('%s', 'now')
                    ''', (ai_tag, user_tag))
                    logger.info(f"Learned mapping: tag '{ai_tag}' → '{user_tag}'")
    
    # Add as verified example
    c.execute('''
        INSERT INTO classification_examples 
        (document_id, document_title, document_type, correspondent, tags, confidence, user_verified)
        VALUES (?, ?, ?, ?, ?, ?, 1)
    ''', (
        original['document_id'],
        original.get('document_title', ''),
        document_type or original.get('document_type'),
        correspondent or original.get('correspondent'),
        json.dumps(tags) if tags else original.get('tags'),
        original.get('confidence', 0.9)
    ))
    
    conn.commit()
    conn.close()
    return True


async def poll_for_new_documents():
    """Poll Paperless for new documents"""
    global last_seen_doc_id
    
    if POLL_INTERVAL <= 0:
        logger.info("Polling disabled (POLL_INTERVAL=0)")
        return
    
    logger.info(f"Polling enabled: checking every {POLL_INTERVAL}s")
    await asyncio.sleep(10)
    
    # Get initial max document ID
    try:
        cfg = get_config()
        headers = {"Authorization": f"Token {cfg['PAPERLESS_TOKEN']}"}
        resp = requests.get(
            f"{cfg['PAPERLESS_URL']}/api/documents/",
            headers=headers,
            params={"ordering": "-id", "page_size": 1},
            timeout=10
        )
        if resp.status_code == 200:
            docs = resp.json().get("results", [])
            if docs:
                last_seen_doc_id = docs[0]["id"]
                logger.info(f"Initial max document ID: {last_seen_doc_id}")
    except Exception as e:
        logger.warning(f"Failed to get initial doc ID: {e}")
    
    while True:
        try:
            await asyncio.sleep(POLL_INTERVAL)
            cfg = get_config()
            headers = {"Authorization": f"Token {cfg['PAPERLESS_TOKEN']}"}
            
            resp = requests.get(
                f"{cfg['PAPERLESS_URL']}/api/documents/",
                headers=headers,
                params={"ordering": "-id", "page_size": 10},
                timeout=10
            )
            
            if resp.status_code == 200:
                docs = resp.json().get("results", [])
                new_docs = [d for d in docs if d["id"] > last_seen_doc_id]
                
                if new_docs:
                    logger.info(f"Found {len(new_docs)} new document(s)")
                    for doc in reversed(new_docs):
                        await document_queue.put(doc["id"])
                        queue_status["pending_docs"].append(doc["id"])
                        logger.info(f"Queued new document {doc['id']}: {doc.get('title', 'untitled')}")
                    
                    last_seen_doc_id = max(d["id"] for d in new_docs)
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Polling error: {e}")


async def process_queue():
    """Process documents from queue"""
    await asyncio.sleep(2)
    logger.info("Queue processor started")
    
    while True:
        try:
            doc_id = await document_queue.get()
            cfg = get_config()
            model = setup_model()
            
            queue_status["processing"] = True
            queue_status["current_doc"] = doc_id
            queue_status["started_at"] = datetime.now().isoformat()
            queue_status["queue_size"] = document_queue.qsize()
            
            if doc_id in queue_status["pending_docs"]:
                queue_status["pending_docs"].remove(doc_id)
            
            # Get document title
            try:
                meta = model.get_document_metadata(doc_id)
                doc_title = meta.get("title", f"Document {doc_id}") if meta else f"Document {doc_id}"
                queue_status["current_title"] = doc_title
            except:
                doc_title = f"Document {doc_id}"
                queue_status["current_title"] = doc_title
            
            logger.info(f"Processing document {doc_id}: {doc_title}")
            log_to_audit(doc_id, doc_title, status="processing")
            
            try:
                # Run blocking AI processing in thread pool
                loop = asyncio.get_event_loop()
                
                # Check if using gemma4 or ministral
                if hasattr(model, 'process_document'):
                    result = await loop.run_in_executor(
                        None,
                        lambda: model.process_document(doc_id, apply_learning=cfg["LEARNING_ENABLED"])
                    )
                else:
                    result = await loop.run_in_executor(
                        None,
                        lambda: model.process_document(doc_id, generate_explanation=cfg["GENERATE_EXPLANATIONS"])
                    )
                
                if result["success"]:
                    logger.info(f"Document {doc_id}: {result['document_type']} | {result.get('correspondent', 'n/a')}")
                    
                    committed = False
                    if cfg["AUTO_COMMIT"]:
                        if model.update_document_in_paperless(doc_id, result):
                            logger.info(f"Document {doc_id} committed to Paperless")
                            committed = True
                    
                    log_to_audit(
                        document_id=doc_id,
                        document_title=doc_title,
                        status="completed",
                        document_type=result.get("document_type"),
                        correspondent=result.get("correspondent"),
                        tags=result.get("tags", []),
                        confidence=result.get("confidence"),
                        processing_time=result.get("duration_sec"),
                        tokens_used=result.get("tokens_est"),
                        auto_approved=committed,
                        explanation=result.get("explanation"),
                        ai_raw=json.dumps(result.get("raw")) if result.get("raw") else None
                    )
                    
                    queue_status["processed_count"] += 1
                    queue_status["last_processed"] = {
                        "doc_id": doc_id,
                        "title": doc_title,
                        "type": result["document_type"],
                        "time": datetime.now().isoformat()
                    }
                else:
                    error_msg = result.get("reason", "unknown")
                    logger.error(f"Document {doc_id} failed: {error_msg}")
                    log_to_audit(doc_id, doc_title, status="failed", error_message=error_msg)
                    
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                log_to_audit(doc_id, doc_title, status="failed", error_message=str(e))
            
            finally:
                queue_status["processing"] = False
                queue_status["current_doc"] = None
                queue_status["current_title"] = None
                queue_status["started_at"] = None
                queue_status["queue_size"] = document_queue.qsize()
                document_queue.task_done()
                
        except asyncio.CancelledError:
            logger.info("Queue processor stopped")
            break
        except Exception as e:
            logger.error(f"Queue processor error: {e}")
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global document_queue
    
    init_database()
    document_queue = asyncio.Queue()
    queue_status["service_started"] = datetime.now().isoformat()
    
    worker_task = asyncio.create_task(process_queue())
    poll_task = asyncio.create_task(poll_for_new_documents())
    
    cfg = get_config()
    logger.info("=" * 60)
    logger.info("Paperless Document Classifier API v2")
    logger.info("=" * 60)
    logger.info(f"Paperless: {cfg['PAPERLESS_URL']}")
    logger.info(f"Ollama: {cfg['OLLAMA_URL']} ({cfg['OLLAMA_MODEL']})")
    logger.info(f"Learning: {'enabled' if cfg['LEARNING_ENABLED'] else 'disabled'}")
    logger.info(f"Auto-commit: {cfg['AUTO_COMMIT']}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")
    logger.info(f"Listening on: {cfg['API_HOST']}:{cfg['API_PORT']}")
    logger.info("=" * 60)
    
    yield
    
    worker_task.cancel()
    poll_task.cancel()
    try:
        await worker_task
        await poll_task
    except asyncio.CancelledError:
        pass
    logger.info("API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Paperless Document Classifier",
    description="AI-powered document classification with learning for Paperless-ngx",
    version="2.0.0",
    lifespan=lifespan
)

# Dashboard HTML with training tab
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Paperless Classifier Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }
        h2 { color: #00d4ff; margin: 20px 0 10px; font-size: 1.2em; }
        .tabs { display: flex; gap: 5px; margin-bottom: 20px; }
        .tab { padding: 10px 20px; background: #16213e; border: none; color: #888; cursor: pointer; border-radius: 5px 5px 0 0; font-size: 1em; }
        .tab.active { background: #00d4ff; color: #000; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h3 { color: #00d4ff; margin-bottom: 15px; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-running { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .status-idle { background: #ffaa00; }
        .stat-value { font-size: 2em; font-weight: bold; color: #fff; }
        .stat-label { color: #888; font-size: 0.9em; }
        .learning-stat { font-size: 1.2em; color: #00ff88; }
        .processing-info { background: #1e3a5f; padding: 15px; border-radius: 8px; margin-top: 10px; }
        .processing-info .doc-id { font-size: 1.5em; color: #00d4ff; }
        .processing-info .doc-title { color: #aaa; margin-top: 5px; word-break: break-all; }
        .queue-list { max-height: 150px; overflow-y: auto; }
        .queue-item { padding: 8px; background: #0f1a2e; margin: 5px 0; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 12px 8px; text-align: left; border-bottom: 1px solid #2a2a4a; }
        th { background: #0f1a2e; color: #00d4ff; font-weight: 500; position: sticky; top: 0; }
        tr:hover { background: #1e3a5f; }
        .tag { display: inline-block; background: #00d4ff22; color: #00d4ff; padding: 2px 8px; border-radius: 12px; margin: 2px; font-size: 0.8em; }
        .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; }
        .status-completed { background: #00ff8822; color: #00ff88; }
        .status-failed { background: #ff444422; color: #ff4444; }
        .status-processing { background: #ffaa0022; color: #ffaa00; }
        .status-abandoned { background: #88888822; color: #888888; }
        .config-form { display: grid; gap: 15px; }
        .config-item { display: grid; grid-template-columns: 200px 1fr; gap: 10px; align-items: center; }
        .config-item label { color: #888; }
        .config-item input, .config-item select { background: #0f1a2e; border: 1px solid #2a2a4a; color: #fff; padding: 10px; border-radius: 5px; }
        input, select { background: #0f1a2e; border: 1px solid #2a2a4a; color: #fff; padding: 8px; border-radius: 5px; }
        button { background: #00d4ff; color: #000; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold; }
        button:hover { background: #00b8e6; }
        button.danger { background: #ff4444; color: #fff; }
        button.danger:hover { background: #cc3333; }
        button.warn { background: #ffaa00; color: #000; }
        button.warn:hover { background: #dd9900; }
        button.success { background: #00ff88; color: #000; }
        button.small { padding: 5px 10px; font-size: 0.85em; }
        .delete-btn { background: transparent; border: 1px solid #ff4444; color: #ff4444; width: 28px; height: 28px; border-radius: 4px; cursor: pointer; font-size: 16px; padding: 0; line-height: 1; }
        .delete-btn:hover { background: #ff4444; color: #fff; }
        .edit-btn { background: transparent; border: 1px solid #00d4ff; color: #00d4ff; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .edit-btn:hover { background: #00d4ff; color: #000; }
        .refresh-info { color: #666; font-size: 0.85em; margin-top: 15px; }
        .actions { margin: 20px 0; display: flex; gap: 10px; flex-wrap: wrap; }
        .table-actions { margin-bottom: 15px; display: flex; gap: 10px; flex-wrap: wrap; }
        .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #ffaa00; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px; vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .mapping-type { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; background: #2a2a4a; }
        .mapping-type.tag { background: #00d4ff22; color: #00d4ff; }
        .mapping-type.document_type { background: #00ff8822; color: #00ff88; }
        .mapping-type.correspondent { background: #ffaa0022; color: #ffaa00; }
        .train-form { display: grid; gap: 10px; margin-top: 10px; }
        .train-form input { width: 100%; }
        .train-form .tags-input { display: flex; gap: 5px; flex-wrap: wrap; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; justify-content: center; align-items: center; }
        .modal.active { display: flex; }
        .modal-content { background: #16213e; padding: 30px; border-radius: 10px; max-width: 500px; width: 90%; }
        .modal-content h3 { margin-bottom: 20px; }
        .modal-close { float: right; background: none; border: none; color: #888; font-size: 24px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="#00d4ff"><path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/></svg>
            Paperless Classifier v2
        </h1>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('status')">Status</button>
            <button class="tab" onclick="showTab('training')">Training</button>
            <button class="tab" onclick="showTab('mappings')">Mappings</button>
            <button class="tab" onclick="showTab('config')">Config</button>
        </div>
        
        <!-- STATUS TAB -->
        <div id="tab-status" class="tab-content active">
            <div class="grid">
                <div class="card">
                    <h3>Service Status</h3>
                    <div id="service-status">
                        <span class="status-indicator status-running"></span>
                        <span id="status-text">Running</span>
                    </div>
                    <div style="margin-top: 15px;">
                        <div class="stat-label">Uptime</div>
                        <div id="uptime">--</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Current Processing</h3>
                    <div id="current-processing">
                        <div style="color: #888;">Idle - waiting for documents</div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Statistics</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <div class="stat-value" id="stat-total">0</div>
                            <div class="stat-label">Completed</div>
                        </div>
                        <div>
                            <div class="stat-value" id="stat-failed">0</div>
                            <div class="stat-label">Failed</div>
                        </div>
                        <div>
                            <div class="stat-value" id="stat-avgtime">0s</div>
                            <div class="stat-label">Avg Time</div>
                        </div>
                        <div>
                            <div class="stat-value" id="stat-confidence">0%</div>
                            <div class="stat-label">Avg Confidence</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Learning Progress</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div>
                            <div class="learning-stat" id="stat-mappings">0</div>
                            <div class="stat-label">Learned Mappings</div>
                        </div>
                        <div>
                            <div class="learning-stat" id="stat-verified">0</div>
                            <div class="stat-label">Verified Examples</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>Queue</h3>
                <div id="queue-list" class="queue-list">
                    <div style="color: #888;">Queue is empty</div>
                </div>
            </div>
            
            <div class="actions">
                <button onclick="refreshAll()">Refresh</button>
                <button onclick="clearQueue()" class="danger">Clear Queue</button>
            </div>
            
            <div class="card">
                <h3>Recent Activity</h3>
                <div class="table-actions">
                    <button onclick="cleanupStale()" class="warn">Clean Stale</button>
                    <button onclick="clearByStatus('failed')" class="danger">Clear Failed</button>
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Doc</th>
                                <th>Title</th>
                                <th>Type</th>
                                <th>Correspondent</th>
                                <th>Tags</th>
                                <th>Conf</th>
                                <th>Status</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody id="audit-log"></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- TRAINING TAB -->
        <div id="tab-training" class="tab-content">
            <div class="card">
                <h3>Train the System</h3>
                <p style="color: #888; margin-bottom: 15px;">Review recent classifications and correct any mistakes. Your corrections teach the system to improve.</p>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Document</th>
                                <th>AI Suggested Type</th>
                                <th>AI Suggested Correspondent</th>
                                <th>AI Suggested Tags</th>
                                <th>Confidence</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="training-list"></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- MAPPINGS TAB -->
        <div id="tab-mappings" class="tab-content">
            <div class="card">
                <h3>Learned Term Mappings</h3>
                <p style="color: #888; margin-bottom: 15px;">These mappings are automatically applied to normalize AI suggestions.</p>
                <div class="actions">
                    <button onclick="showAddMappingModal()">Add Mapping</button>
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Type</th>
                                <th>AI Says</th>
                                <th>→</th>
                                <th>Corrected To</th>
                                <th>Times Used</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody id="mappings-list"></tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- CONFIG TAB -->
        <div id="tab-config" class="tab-content">
            <div class="card">
                <h3>Configuration</h3>
                <div class="config-form" id="config-form"></div>
                <div style="margin-top: 20px;">
                    <button onclick="saveConfig()">Save Configuration</button>
                </div>
                <div class="refresh-info">Changes require service restart to take effect.</div>
            </div>
        </div>
        
        <div class="refresh-info">
            Auto-refresh every 5 seconds | Last update: <span id="last-update">--</span>
        </div>
    </div>
    
    <!-- Add Mapping Modal -->
    <div id="add-mapping-modal" class="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <h3>Add Term Mapping</h3>
            <div class="train-form">
                <div>
                    <label>Type</label>
                    <select id="new-mapping-type">
                        <option value="document_type">Document Type</option>
                        <option value="correspondent">Correspondent</option>
                        <option value="tag">Tag</option>
                    </select>
                </div>
                <div>
                    <label>When AI says</label>
                    <input type="text" id="new-mapping-ai" placeholder="AI suggestion to replace">
                </div>
                <div>
                    <label>Correct to</label>
                    <input type="text" id="new-mapping-approved" placeholder="Your preferred term">
                </div>
                <button onclick="addMapping()" class="success">Add Mapping</button>
            </div>
        </div>
    </div>
    
    <!-- Edit Classification Modal -->
    <div id="edit-modal" class="modal">
        <div class="modal-content">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <h3>Correct Classification</h3>
            <input type="hidden" id="edit-audit-id">
            <div class="train-form">
                <div>
                    <label>Document Type</label>
                    <input type="text" id="edit-type" placeholder="Corrected type">
                </div>
                <div>
                    <label>Correspondent</label>
                    <input type="text" id="edit-correspondent" placeholder="Corrected correspondent">
                </div>
                <div>
                    <label>Tags (comma separated)</label>
                    <input type="text" id="edit-tags" placeholder="tag1, tag2, tag3">
                </div>
                <button onclick="submitCorrection()" class="success">Save & Learn</button>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            document.getElementById(`tab-${tabName}`).classList.add('active');
            
            if (tabName === 'training') refreshTraining();
            if (tabName === 'mappings') refreshMappings();
        }
        
        function closeModal() {
            document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
        }
        
        function showAddMappingModal() {
            document.getElementById('add-mapping-modal').classList.add('active');
        }
        
        function showEditModal(auditId, type, correspondent, tags) {
            document.getElementById('edit-audit-id').value = auditId;
            document.getElementById('edit-type').value = type || '';
            document.getElementById('edit-correspondent').value = correspondent || '';
            document.getElementById('edit-tags').value = (tags || []).join(', ');
            document.getElementById('edit-modal').classList.add('active');
        }
        
        async function fetchJson(url) {
            const resp = await fetch(url);
            return resp.json();
        }
        
        async function refreshStatus() {
            try {
                const queue = await fetchJson('/api/queue');
                const stats = await fetchJson('/api/stats');
                
                document.getElementById('stat-total').textContent = stats.stats.completed || 0;
                document.getElementById('stat-failed').textContent = stats.stats.failed || 0;
                document.getElementById('stat-avgtime').textContent = (stats.stats.avg_processing_time || 0) + 's';
                document.getElementById('stat-confidence').textContent = (stats.stats.avg_confidence || 0) + '%';
                document.getElementById('stat-mappings').textContent = stats.stats.learned_mappings || 0;
                document.getElementById('stat-verified').textContent = stats.stats.verified_examples || 0;
                
                const procDiv = document.getElementById('current-processing');
                if (queue.processing) {
                    procDiv.innerHTML = `
                        <div class="processing-info">
                            <div class="doc-id"><span class="spinner"></span>Document #${queue.current_doc}</div>
                            <div class="doc-title">${queue.current_title || ''}</div>
                        </div>
                    `;
                } else {
                    procDiv.innerHTML = '<div style="color: #888;">Idle - waiting for documents</div>';
                }
                
                const queueList = document.getElementById('queue-list');
                if (queue.pending && queue.pending.length > 0) {
                    queueList.innerHTML = queue.pending.map(id => `<div class="queue-item">Document #${id}</div>`).join('');
                } else {
                    queueList.innerHTML = '<div style="color: #888;">Queue is empty</div>';
                }
                
                if (stats.service_started) {
                    const started = new Date(stats.service_started);
                    const diff = Math.floor((new Date() - started) / 1000);
                    document.getElementById('uptime').textContent = `${Math.floor(diff/3600)}h ${Math.floor((diff%3600)/60)}m`;
                }
            } catch (e) { console.error(e); }
        }
        
        async function refreshAuditLog() {
            try {
                const data = await fetchJson('/api/audit?limit=50');
                const tbody = document.getElementById('audit-log');
                
                if (data.logs && data.logs.length > 0) {
                    tbody.innerHTML = data.logs.map(log => {
                        const time = new Date(log.timestamp).toLocaleString();
                        const tags = (log.tags || []).slice(0, 3).map(t => `<span class="tag">${t}</span>`).join('');
                        const statusClass = 'status-' + log.status;
                        const conf = log.confidence ? Math.round(log.confidence * 100) + '%' : '-';
                        
                        return `<tr>
                            <td style="white-space:nowrap">${time}</td>
                            <td>${log.document_id}</td>
                            <td>${(log.document_title || '-').substring(0, 30)}</td>
                            <td>${log.document_type || '-'}</td>
                            <td>${log.correspondent || '-'}</td>
                            <td>${tags || '-'}</td>
                            <td>${conf}</td>
                            <td><span class="status-badge ${statusClass}">${log.status}</span></td>
                            <td><button onclick="deleteEntry(${log.id})" class="delete-btn">×</button></td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="9" style="color: #888;">No activity yet</td></tr>';
                }
            } catch (e) { console.error(e); }
        }
        
        async function refreshTraining() {
            try {
                const data = await fetchJson('/api/audit?limit=20');
                const tbody = document.getElementById('training-list');
                const completedLogs = data.logs.filter(l => l.status === 'completed');
                
                if (completedLogs.length > 0) {
                    tbody.innerHTML = completedLogs.map(log => {
                        const tags = (log.tags || []).map(t => `<span class="tag">${t}</span>`).join('');
                        const conf = log.confidence ? Math.round(log.confidence * 100) + '%' : '-';
                        const tagsStr = (log.tags || []).join(',');
                        
                        return `<tr>
                            <td><strong>#${log.document_id}</strong><br><small>${(log.document_title || '').substring(0, 40)}</small></td>
                            <td>${log.document_type || '-'}</td>
                            <td>${log.correspondent || '-'}</td>
                            <td>${tags || '-'}</td>
                            <td>${conf}</td>
                            <td><button class="edit-btn" onclick="showEditModal(${log.id}, '${log.document_type || ''}', '${log.correspondent || ''}', [${(log.tags||[]).map(t=>"'"+t+"'").join(',')}])">Correct</button></td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="6" style="color: #888;">No completed classifications to review</td></tr>';
                }
            } catch (e) { console.error(e); }
        }
        
        async function refreshMappings() {
            try {
                const data = await fetchJson('/api/mappings');
                const tbody = document.getElementById('mappings-list');
                
                if (data.mappings && data.mappings.length > 0) {
                    tbody.innerHTML = data.mappings.map(m => {
                        return `<tr>
                            <td><span class="mapping-type ${m.term_type}">${m.term_type}</span></td>
                            <td>${m.ai_term}</td>
                            <td>→</td>
                            <td><strong>${m.approved_term}</strong></td>
                            <td>${m.times_used}×</td>
                            <td><button onclick="deleteMapping(${m.id})" class="delete-btn">×</button></td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="6" style="color: #888;">No mappings yet. Correct classifications to start learning!</td></tr>';
                }
            } catch (e) { console.error(e); }
        }
        
        async function refreshConfig() {
            try {
                const data = await fetchJson('/api/config');
                const form = document.getElementById('config-form');
                const items = [
                    { key: 'PAPERLESS_URL', label: 'Paperless URL' },
                    { key: 'PAPERLESS_TOKEN', label: 'Paperless Token', type: 'password' },
                    { key: 'OLLAMA_URL', label: 'Ollama URL' },
                    { key: 'OLLAMA_MODEL', label: 'Model' },
                    { key: 'AUTO_COMMIT', label: 'Auto Commit', type: 'select', options: ['true', 'false'] },
                    { key: 'LEARNING_ENABLED', label: 'Learning Enabled', type: 'select', options: ['true', 'false'] },
                ];
                
                form.innerHTML = items.map(item => {
                    const value = data[item.key] || '';
                    if (item.type === 'select') {
                        return `<div class="config-item"><label>${item.label}</label><select id="config-${item.key}">${item.options.map(o => `<option value="${o}" ${String(value).toLowerCase() === o ? 'selected' : ''}>${o}</option>`).join('')}</select></div>`;
                    }
                    return `<div class="config-item"><label>${item.label}</label><input type="${item.type || 'text'}" id="config-${item.key}" value="${value}"></div>`;
                }).join('');
            } catch (e) { console.error(e); }
        }
        
        async function saveConfig() {
            const keys = ['PAPERLESS_URL', 'PAPERLESS_TOKEN', 'OLLAMA_URL', 'OLLAMA_MODEL', 'AUTO_COMMIT', 'LEARNING_ENABLED'];
            for (const key of keys) {
                const el = document.getElementById(`config-${key}`);
                if (el) await fetch('/api/config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ key, value: el.value }) });
            }
            alert('Saved! Restart service for changes to take effect.');
        }
        
        async function submitCorrection() {
            const auditId = document.getElementById('edit-audit-id').value;
            const type = document.getElementById('edit-type').value;
            const correspondent = document.getElementById('edit-correspondent').value;
            const tagsStr = document.getElementById('edit-tags').value;
            const tags = tagsStr ? tagsStr.split(',').map(t => t.trim()).filter(t => t) : [];
            
            await fetch('/api/learn/correct', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ audit_id: parseInt(auditId), document_type: type, correspondent, tags })
            });
            
            closeModal();
            refreshMappings();
            alert('Correction saved! The system has learned from your input.');
        }
        
        async function addMapping() {
            const type = document.getElementById('new-mapping-type').value;
            const ai = document.getElementById('new-mapping-ai').value;
            const approved = document.getElementById('new-mapping-approved').value;
            
            if (!ai || !approved) { alert('Please fill all fields'); return; }
            
            await fetch('/api/mappings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ term_type: type, ai_term: ai, approved_term: approved })
            });
            
            closeModal();
            refreshMappings();
        }
        
        async function deleteMapping(id) {
            if (confirm('Delete this mapping?')) {
                await fetch('/api/mappings/' + id, { method: 'DELETE' });
                refreshMappings();
            }
        }
        
        async function clearQueue() {
            if (confirm('Clear queue?')) {
                await fetch('/api/queue/clear', { method: 'DELETE' });
                refreshAll();
            }
        }
        
        async function deleteEntry(id) {
            await fetch('/api/audit/' + id, { method: 'DELETE' });
            refreshAll();
        }
        
        async function cleanupStale() {
            const resp = await fetch('/api/audit/cleanup', { method: 'POST' });
            const data = await resp.json();
            alert('Cleaned ' + data.updated + ' stale entries');
            refreshAll();
        }
        
        async function clearByStatus(status) {
            if (confirm('Delete all "' + status + '" entries?')) {
                await fetch('/api/audit/status/' + status, { method: 'DELETE' });
                refreshAll();
            }
        }
        
        function refreshAll() {
            refreshStatus();
            refreshAuditLog();
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }
        
        refreshAll();
        refreshConfig();
        setInterval(refreshAll, 5000);
    </script>
</body>
</html>
'''

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML

@app.get("/api/health")
async def health():
    cfg = get_config()
    status = {"api": "healthy", "paperless": "unknown", "ollama": "unknown"}
    try:
        resp = requests.get(f"{cfg['PAPERLESS_URL']}/api/documents/", timeout=5)
        status["paperless"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        status["paperless"] = "unreachable"
    try:
        resp = requests.get(f"{cfg['OLLAMA_URL']}/api/tags", timeout=5)
        status["ollama"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        status["ollama"] = "unreachable"
    return status

@app.get("/api/queue")
async def get_queue():
    return {
        "processing": queue_status["processing"],
        "current_doc": queue_status["current_doc"],
        "current_title": queue_status["current_title"],
        "started_at": queue_status["started_at"],
        "queue_size": document_queue.qsize() if document_queue else 0,
        "pending": queue_status["pending_docs"]
    }

@app.get("/api/stats")
async def get_stats():
    return {
        "stats": get_audit_stats(),
        "service_started": queue_status["service_started"],
        "processed_count": queue_status["processed_count"],
        "last_processed": queue_status["last_processed"]
    }

@app.get("/api/audit")
async def get_audit(limit: int = 100, offset: int = 0):
    return {"logs": get_audit_logs(limit, offset)}

@app.get("/api/config")
async def get_config_api():
    return get_config()

@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    set_key(str(ENV_FILE), update.key, update.value)
    return {"status": "updated", "key": update.key}

@app.post("/api/classify")
async def classify_document(request: ClassifyRequest):
    await document_queue.put(request.document_id)
    queue_status["pending_docs"].append(request.document_id)
    return {"status": "queued", "document_id": request.document_id}

@app.post("/api/classify/batch")
async def classify_batch(document_ids: List[int]):
    for doc_id in document_ids:
        await document_queue.put(doc_id)
        queue_status["pending_docs"].append(doc_id)
    return {"status": "queued", "count": len(document_ids)}

@app.delete("/api/queue/clear")
async def clear_queue():
    cleared = 0
    while not document_queue.empty():
        try:
            document_queue.get_nowait()
            document_queue.task_done()
            cleared += 1
        except asyncio.QueueEmpty:
            break
    queue_status["pending_docs"] = []
    return {"status": "cleared", "removed": cleared}

@app.delete("/api/audit/{entry_id}")
async def delete_audit_entry(entry_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM audit_log WHERE id = ?", (entry_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return {"status": "deleted" if deleted else "not_found"}

@app.delete("/api/audit/status/{status}")
async def delete_audit_by_status(status: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM audit_log WHERE status = ?", (status,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return {"status": "deleted", "count": deleted}

@app.post("/api/audit/cleanup")
async def cleanup_stale_entries():
    cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE audit_log SET status = 'abandoned' WHERE status = 'processing' AND timestamp < ?", (cutoff,))
    updated = c.rowcount
    conn.commit()
    conn.close()
    return {"status": "cleaned", "updated": updated}

# Learning API endpoints
@app.get("/api/mappings")
async def get_mappings():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM term_mappings ORDER BY times_used DESC')
    mappings = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"mappings": mappings}

@app.post("/api/mappings")
async def create_mapping(mapping: MappingCreate):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO term_mappings (term_type, ai_term, approved_term)
            VALUES (?, ?, ?)
        ''', (mapping.term_type, mapping.ai_term, mapping.approved_term))
        conn.commit()
        return {"status": "created"}
    except sqlite3.IntegrityError:
        return {"status": "exists"}
    finally:
        conn.close()

@app.delete("/api/mappings/{mapping_id}")
async def delete_mapping(mapping_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM term_mappings WHERE id = ?", (mapping_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return {"status": "deleted" if deleted else "not_found"}

@app.post("/api/learn/correct")
async def learn_from_correction(correction: CorrectionRequest):
    success = apply_learning_correction(
        correction.audit_id,
        correction.document_type,
        correction.correspondent,
        correction.tags
    )
    return {"status": "learned" if success else "failed"}

# Webhooks
@app.post("/webhook/paperless/{doc_id}")
async def paperless_webhook_with_id(doc_id: int):
    logger.info(f"Webhook received for document {doc_id}")
    await document_queue.put(doc_id)
    queue_status["pending_docs"].append(doc_id)
    return {"status": "queued", "document_id": doc_id}

@app.post("/webhook/paperless")
async def paperless_webhook(request: Request):
    try:
        payload = await request.json()
    except:
        payload = {}
    
    logger.info(f"Webhook payload: {payload}")
    
    doc_id = None
    for key in ["document_id", "id", "pk", "document_pk"]:
        if key in payload:
            doc_id = payload[key]
            break
    
    if not doc_id and "document" in payload:
        doc = payload["document"]
        doc_id = doc if isinstance(doc, int) else doc.get("id") or doc.get("pk")
    
    if doc_id:
        await document_queue.put(int(doc_id))
        queue_status["pending_docs"].append(int(doc_id))
        logger.info(f"Document {doc_id} queued via webhook")
        return {"status": "queued", "document_id": doc_id}
    
    return {"status": "ignored", "reason": "no document_id found"}

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run("classifier_api:app", host=cfg["API_HOST"], port=cfg["API_PORT"], reload=False)
