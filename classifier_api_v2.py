#!/usr/bin/env python3
"""
Paperless Document Classifier API v2
=====================================
FastAPI service with:
- SQLite audit log and web dashboard
- Gemma 4 / Ministral model support
- Learning layer for continuous improvement
- Dashboard training interface
- Export logs for debugging
"""
import os
import asyncio
import sqlite3
import logging
import json
import requests
import platform
import shutil
import glob
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# Load environment variables
ENV_FILE = Path(__file__).parent / ".env"
if not ENV_FILE.exists():
    ENV_FILE = Path("/opt/paperless-classifier/.env")
load_dotenv(ENV_FILE)

# Logs directory setup
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_RETENTION_DAYS = 7
LOG_MAX_SIZE_MB = 50

# Configuration from .env
def get_config():
    load_dotenv(ENV_FILE, override=True)
    config = {
        "PAPERLESS_URL": os.getenv("PAPERLESS_URL", "http://localhost:8000"),
        "PAPERLESS_TOKEN": os.getenv("PAPERLESS_TOKEN", ""),
        "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "gemma3:12b"),
        "OLLAMA_THREADS": int(os.getenv("OLLAMA_THREADS", "10")),
        "MAX_PAGES": int(os.getenv("MAX_PAGES", "3")),
        "IMAGE_MAX_SIZE": int(os.getenv("IMAGE_MAX_SIZE", "1024")),
        "IMAGE_QUALITY": int(os.getenv("IMAGE_QUALITY", "85")),
        "AUTO_COMMIT": os.getenv("AUTO_COMMIT", "true").lower() == "true",
        "GENERATE_EXPLANATIONS": os.getenv("GENERATE_EXPLANATIONS", "false").lower() == "true",
        "LEARNING_ENABLED": os.getenv("LEARNING_ENABLED", "true").lower() == "true",
        "FEW_SHOT_ENABLED": os.getenv("FEW_SHOT_ENABLED", "false").lower() == "true",
        "INJECT_EXISTING_TYPES": os.getenv("INJECT_EXISTING_TYPES", "false").lower() == "true",
        "INJECT_EXISTING_TAGS": os.getenv("INJECT_EXISTING_TAGS", "true").lower() == "true",
        "FUZZY_MATCH_THRESHOLD": float(os.getenv("FUZZY_MATCH_THRESHOLD", "0.80")),
        "API_HOST": os.getenv("API_HOST", "0.0.0.0"),
        "API_PORT": int(os.getenv("API_PORT", "8001")),
    }
    # Only include sampling params if explicitly set — otherwise let model use its defaults
    if os.getenv("OLLAMA_TEMPERATURE"):
        config["OLLAMA_TEMPERATURE"] = float(os.getenv("OLLAMA_TEMPERATURE"))
    if os.getenv("OLLAMA_TOP_P"):
        config["OLLAMA_TOP_P"] = float(os.getenv("OLLAMA_TOP_P"))
    if os.getenv("OLLAMA_TOP_K"):
        config["OLLAMA_TOP_K"] = int(os.getenv("OLLAMA_TOP_K"))
    return config

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
    """Log processing result to audit database. Updates existing 'processing' entry if completing."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # If we're completing/failing, update the existing 'processing' entry instead of creating new
    if status in ('completed', 'failed', 'abandoned'):
        c.execute('''
            UPDATE audit_log 
            SET status = ?, document_type = ?, correspondent = ?, tags = ?, 
                confidence = ?, processing_time = ?, tokens_used = ?, 
                auto_approved = ?, error_message = ?, explanation = ?, ai_raw = ?,
                timestamp = ?
            WHERE document_id = ? AND status = 'processing'
        ''', (
            status, document_type, correspondent,
            json.dumps(tags) if tags else None,
            confidence, processing_time, tokens_used,
            1 if auto_approved else 0, error_message, explanation, ai_raw,
            datetime.now().isoformat(),
            document_id
        ))
        
        # If we updated an existing row, we're done
        if c.rowcount > 0:
            conn.commit()
            conn.close()
            return
    
    # Otherwise insert new entry (for 'processing' status or if no existing entry found)
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

from logging.handlers import RotatingFileHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        RotatingFileHandler(
            Path(__file__).parent / 'classifier_api.log',
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Queue for document processing
document_queue: asyncio.Queue = None
_pending_set: set = set()  # O(1) dedup lookup
queue_status: Dict = {
    "processing": False,
    "current_doc": None,
    "current_title": None,
    "started_at": None,
    "queue_size": 0,
    "processed_count": 0,
    "last_processed": None,
    "service_started": None,
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
    """Import and configure the vision model module"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    cfg = get_config()
    import gemma4 as model_module
    model_module.init_config(cfg)
    return model_module

# Learning functions
def apply_learning_correction(
    audit_id: int, 
    corrections: Dict,  # {field: {ai_value, user_value, sync_to_paperless}}
    document_id: int = None
):
    """
    Apply user corrections and learn from them.
    
    corrections format:
    {
        'document_type': {'ai': 'Beleg', 'user': 'Rechnung', 'sync': True},
        'correspondent': {'ai': 'Anthropic', 'user': 'Anthropic, PBC', 'sync': True},
        'tags': {'ai': ['a', 'b'], 'user': ['x', 'y'], 'sync': True}
    }
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get the audit entry
    c.execute('SELECT * FROM audit_log WHERE id = ?', (audit_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"success": False, "error": "Audit entry not found"}
    
    original = dict(row)
    doc_id = document_id or original.get('document_id')
    mappings_created = []
    
    # Process document_type correction
    if 'document_type' in corrections:
        corr = corrections['document_type']
        ai_val = corr.get('ai', '').strip()
        user_val = corr.get('user', '').strip()
        
        if ai_val and user_val and ai_val.lower() != user_val.lower():
            c.execute('''
                INSERT INTO term_mappings (term_type, ai_term, approved_term)
                VALUES ('document_type', ?, ?)
                ON CONFLICT(term_type, ai_term) DO UPDATE SET 
                    approved_term = excluded.approved_term,
                    times_used = times_used + 1,
                    last_used = strftime('%s', 'now')
            ''', (ai_val, user_val))
            mappings_created.append(f"document_type: {ai_val} → {user_val}")
            logger.info(f"Learned: document_type '{ai_val}' → '{user_val}'")
    
    # Process correspondent correction
    if 'correspondent' in corrections:
        corr = corrections['correspondent']
        ai_val = corr.get('ai', '').strip()
        user_val = corr.get('user', '').strip()
        
        if ai_val and user_val and ai_val.lower() != user_val.lower():
            c.execute('''
                INSERT INTO term_mappings (term_type, ai_term, approved_term)
                VALUES ('correspondent', ?, ?)
                ON CONFLICT(term_type, ai_term) DO UPDATE SET 
                    approved_term = excluded.approved_term,
                    times_used = times_used + 1,
                    last_used = strftime('%s', 'now')
            ''', (ai_val, user_val))
            mappings_created.append(f"correspondent: {ai_val} → {user_val}")
            logger.info(f"Learned: correspondent '{ai_val}' → '{user_val}'")
    
    # Process tag corrections - ONLY create mappings for explicit 1:1 replacements
    # User must add tag mappings manually via Mappings tab for complex cases
    if 'tag_mappings' in corrections:
        # Explicit tag mappings: [{ai: 'x', user: 'y'}, ...]
        for tm in corrections['tag_mappings']:
            ai_tag = tm.get('ai', '').strip()
            user_tag = tm.get('user', '').strip()
            if ai_tag and user_tag and ai_tag.lower() != user_tag.lower():
                c.execute('''
                    INSERT INTO term_mappings (term_type, ai_term, approved_term)
                    VALUES ('tag', ?, ?)
                    ON CONFLICT(term_type, ai_term) DO UPDATE SET 
                        approved_term = excluded.approved_term,
                        times_used = times_used + 1,
                        last_used = strftime('%s', 'now')
                ''', (ai_tag, user_tag))
                mappings_created.append(f"tag: {ai_tag} → {user_tag}")
                logger.info(f"Learned: tag '{ai_tag}' → '{user_tag}'")
    
    # Add as verified example with user-corrected values
    user_type = corrections.get('document_type', {}).get('user') or original.get('document_type')
    user_corr = corrections.get('correspondent', {}).get('user') or original.get('correspondent')
    user_tags = corrections.get('tags', {}).get('user') or json.loads(original.get('tags', '[]'))
    
    c.execute('''
        INSERT INTO classification_examples 
        (document_id, document_title, document_type, correspondent, tags, confidence, user_verified)
        VALUES (?, ?, ?, ?, ?, ?, 1)
    ''', (
        doc_id,
        original.get('document_title', ''),
        user_type,
        user_corr,
        json.dumps(user_tags) if isinstance(user_tags, list) else user_tags,
        original.get('confidence', 0.9)
    ))
    
    conn.commit()
    conn.close()
    
    # Sync to Paperless if requested
    paperless_updated = False
    if any(c.get('sync') for c in corrections.values() if isinstance(c, dict)):
        paperless_updated = sync_correction_to_paperless(doc_id, corrections)
    
    return {
        "success": True, 
        "mappings_created": mappings_created,
        "paperless_updated": paperless_updated
    }


def sync_correction_to_paperless(doc_id: int, corrections: Dict) -> bool:
    """Sync user corrections to Paperless via API"""
    cfg = get_config()
    headers = {
        "Authorization": f"Token {cfg['PAPERLESS_TOKEN']}",
        "Content-Type": "application/json"
    }
    base_url = cfg['PAPERLESS_URL']
    
    try:
        update_data = {}
        
        # Document type
        if corrections.get('document_type', {}).get('sync'):
            type_name = corrections['document_type'].get('user')
            if type_name:
                # Get or create document type
                resp = requests.get(f"{base_url}/api/document_types/", headers=headers, 
                                   params={"name__iexact": type_name}, timeout=10)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        update_data['document_type'] = results[0]['id']
                    else:
                        # Create new
                        resp = requests.post(f"{base_url}/api/document_types/", headers=headers,
                                           json={"name": type_name}, timeout=10)
                        if resp.status_code == 201:
                            update_data['document_type'] = resp.json()['id']
        
        # Correspondent
        if corrections.get('correspondent', {}).get('sync'):
            corr_name = corrections['correspondent'].get('user')
            if corr_name:
                resp = requests.get(f"{base_url}/api/correspondents/", headers=headers,
                                   params={"name__iexact": corr_name}, timeout=10)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if results:
                        update_data['correspondent'] = results[0]['id']
                    else:
                        resp = requests.post(f"{base_url}/api/correspondents/", headers=headers,
                                           json={"name": corr_name}, timeout=10)
                        if resp.status_code == 201:
                            update_data['correspondent'] = resp.json()['id']
        
        # Tags
        if corrections.get('tags', {}).get('sync'):
            user_tags = corrections['tags'].get('user', [])
            if user_tags:
                tag_ids = []
                for tag_name in user_tags:
                    resp = requests.get(f"{base_url}/api/tags/", headers=headers,
                                       params={"name__iexact": tag_name}, timeout=10)
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        if results:
                            tag_ids.append(results[0]['id'])
                        else:
                            resp = requests.post(f"{base_url}/api/tags/", headers=headers,
                                               json={"name": tag_name}, timeout=10)
                            if resp.status_code == 201:
                                tag_ids.append(resp.json()['id'])
                if tag_ids:
                    update_data['tags'] = tag_ids
        
        # Apply updates to Paperless
        if update_data:
            resp = requests.patch(f"{base_url}/api/documents/{doc_id}/", headers=headers,
                                json=update_data, timeout=15)
            if resp.status_code == 200:
                logger.info(f"Synced corrections to Paperless for doc {doc_id}: {list(update_data.keys())}")
                return True
            else:
                logger.error(f"Failed to sync to Paperless: {resp.status_code} {resp.text}")
                return False
        
        return True  # Nothing to sync
        
    except Exception as e:
        logger.error(f"Error syncing to Paperless: {e}")
        return False


def get_paperless_document(doc_id: int) -> Optional[Dict]:
    """Fetch current document data from Paperless with resolved names"""
    cfg = get_config()
    headers = {"Authorization": f"Token {cfg['PAPERLESS_TOKEN']}"}
    
    try:
        resp = requests.get(f"{cfg['PAPERLESS_URL']}/api/documents/{doc_id}/", 
                          headers=headers, timeout=10)
        if resp.status_code != 200:
            return None
        
        doc = resp.json()
        result = {
            'id': doc['id'],
            'title': doc.get('title', ''),
            'document_type': None,
            'correspondent': None,
            'tags': []
        }
        
        # Resolve names using bulk endpoints (avoid N+1)
        if doc.get('document_type'):
            resp = requests.get(f"{cfg['PAPERLESS_URL']}/api/document_types/{doc['document_type']}/",
                              headers=headers, timeout=10)
            if resp.status_code == 200:
                result['document_type'] = resp.json().get('name')
        
        if doc.get('correspondent'):
            resp = requests.get(f"{cfg['PAPERLESS_URL']}/api/correspondents/{doc['correspondent']}/",
                              headers=headers, timeout=10)
            if resp.status_code == 200:
                result['correspondent'] = resp.json().get('name')
        
        # Fetch all tags in one call and filter
        tag_ids = doc.get('tags', [])
        if tag_ids:
            resp = requests.get(f"{cfg['PAPERLESS_URL']}/api/tags/",
                              headers=headers, params={"page_size": 500}, timeout=10)
            if resp.status_code == 200:
                all_tags = {t['id']: t['name'] for t in resp.json().get('results', [])}
                result['tags'] = [all_tags[tid] for tid in tag_ids if tid in all_tags]
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching from Paperless: {e}")
        return None


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
        resp = await asyncio.to_thread(
            requests.get,
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
            
            resp = await asyncio.to_thread(
                requests.get,
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
                        doc_id = doc["id"]
                        # Dedup: skip if already in queue
                        if doc_id in _pending_set:
                            continue
                        await document_queue.put(doc_id)
                        _pending_set.add(doc_id)
                        logger.info(f"Queued new document {doc_id}: {doc.get('title', 'untitled')}")
                    
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
            
            # Dedup: skip if already processing or recently processed
            if queue_status["current_doc"] == doc_id:
                document_queue.task_done()
                continue
            
            queue_status["processing"] = True
            queue_status["current_doc"] = doc_id
            queue_status["started_at"] = datetime.now().isoformat()
            queue_status["queue_size"] = document_queue.qsize()
            
            _pending_set.discard(doc_id)
            
            # Get document title
            try:
                meta = await asyncio.to_thread(model.get_document_metadata, doc_id)
                doc_title = meta.get("title", f"Document {doc_id}") if meta else f"Document {doc_id}"
                queue_status["current_title"] = doc_title
            except Exception:
                doc_title = f"Document {doc_id}"
                queue_status["current_title"] = doc_title
            
            logger.info(f"Processing document {doc_id}: {doc_title}")
            log_to_audit(doc_id, doc_title, status="processing")
            
            try:
                result = await asyncio.to_thread(
                    model.process_document, doc_id, cfg["LEARNING_ENABLED"]
                )
                
                if result["success"]:
                    logger.info(f"Document {doc_id}: {result['document_type']} | {result.get('correspondent', 'n/a')}")
                    
                    committed = False
                    if cfg["AUTO_COMMIT"]:
                        committed = await asyncio.to_thread(
                            model.update_document_in_paperless, doc_id, result
                        )
                        if committed:
                            logger.info(f"Document {doc_id} committed to Paperless")
                    
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
                
                # Invalidate Paperless cache when queue drains
                if document_queue.empty():
                    model.invalidate_cache()
                    logger.debug("Queue empty — Paperless cache invalidated")
                
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
    logger.info("Paperless Document Classifier API v3")
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
    version="3.0.0",
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
            Paperless Classifier v3
        </h1>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('status')">Status</button>
            <button class="tab" onclick="showTab('training')">Training</button>
            <button class="tab" onclick="showTab('mappings')">Mappings</button>
            <button class="tab" onclick="showTab('export')">Export</button>
            <button class="tab" onclick="showTab('logs')">Logs</button>
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
                                <th>Actions</th>
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
                    <button onclick="resetLearning()" class="danger">Reset All Learning</button>
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
            
            <div class="card">
                <h3>Classification Examples (Few-Shot Learning)</h3>
                <p style="color: #888; margin-bottom: 15px;">Verified examples are shown to the AI as guidance for future classifications.</p>
                <div class="actions">
                    <button onclick="clearExamples()" class="warn">Clear Examples</button>
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Document</th>
                                <th>Type</th>
                                <th>Correspondent</th>
                                <th>Tags</th>
                                <th>Verified</th>
                            </tr>
                        </thead>
                        <tbody id="examples-list"></tbody>
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
        
        <!-- EXPORT TAB -->
        <div id="tab-export" class="tab-content">
            <div class="card">
                <h3>Export Debug Logs</h3>
                <p style="color: #888; margin-bottom: 15px;">
                    Generate comprehensive debug exports for troubleshooting. Includes system info, 
                    recent runs, mappings, and application logs.
                </p>
                <div class="actions">
                    <button onclick="generateExport()" class="success">Generate New Export</button>
                    <button onclick="refreshExports()">Refresh List</button>
                </div>
                <div style="margin-top: 15px; padding: 10px; background: #0f1a2e; border-radius: 5px;">
                    <small style="color: #666;">
                        Logs are stored in /logs folder. Auto-cleanup: files older than 7 days or when total exceeds 50MB.
                    </small>
                </div>
            </div>
            
            <div class="card">
                <h3>Available Exports</h3>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Filename</th>
                                <th>Size</th>
                                <th>Created</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody id="exports-list">
                            <tr><td colspan="4" style="color: #888;">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <!-- LOGS TAB -->
        <div id="tab-logs" class="tab-content">
            <div class="card">
                <h3>Application Logs</h3>
                <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 15px; flex-wrap: wrap;">
                    <select id="log-level-filter" onchange="refreshLogs()" style="background: #0f1a2e; color: #e0e0e0; border: 1px solid #1e3a5f; padding: 6px 10px; border-radius: 4px;">
                        <option value="">All Levels</option>
                        <option value="ERROR">Errors</option>
                        <option value="WARNING">Warnings</option>
                        <option value="INFO">Info</option>
                        <option value="DEBUG">Debug</option>
                    </select>
                    <label style="color: #888; font-size: 13px; display: flex; align-items: center; gap: 5px;">
                        <input type="checkbox" id="log-autoscroll" checked> Auto-scroll
                    </label>
                    <label style="color: #888; font-size: 13px; display: flex; align-items: center; gap: 5px;">
                        <input type="checkbox" id="log-autorefresh" checked> Auto-refresh (3s)
                    </label>
                    <span style="color: #555; font-size: 12px; margin-left: auto;" id="log-line-count"></span>
                </div>
                <div id="log-viewer" style="
                    background: #050a12;
                    border: 1px solid #1e3a5f;
                    border-radius: 6px;
                    padding: 12px;
                    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
                    font-size: 12px;
                    line-height: 1.5;
                    max-height: 600px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                    word-break: break-all;
                    color: #b0b0b0;
                ">Loading logs...</div>
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
        <div class="modal-content" style="max-width: 700px;">
            <button class="modal-close" onclick="closeModal()">&times;</button>
            <h3>Correct Classification</h3>
            <p style="color: #888; margin-bottom: 15px; font-size: 0.9em;">
                Compare AI suggestions with current Paperless data. Check boxes to sync corrections to Paperless.
            </p>
            <input type="hidden" id="edit-audit-id">
            <input type="hidden" id="edit-doc-id">
            
            <div id="edit-loading" style="text-align: center; padding: 20px; color: #888;">
                <span class="spinner"></span> Loading document data...
            </div>
            
            <div id="edit-form" style="display: none;">
                <table style="width: 100%; margin-bottom: 20px;">
                    <thead>
                        <tr>
                            <th style="width: 25%;">Field</th>
                            <th style="width: 30%;">AI Suggested</th>
                            <th style="width: 30%;">Your Correction</th>
                            <th style="width: 15%;">Sync</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Document Type</strong></td>
                            <td><span id="edit-ai-type" class="tag">-</span></td>
                            <td><input type="text" id="edit-type" style="width: 100%;"></td>
                            <td><label><input type="checkbox" id="edit-sync-type" checked> Write</label></td>
                        </tr>
                        <tr>
                            <td><strong>Correspondent</strong></td>
                            <td><span id="edit-ai-corr" class="tag">-</span></td>
                            <td><input type="text" id="edit-correspondent" style="width: 100%;"></td>
                            <td><label><input type="checkbox" id="edit-sync-corr" checked> Write</label></td>
                        </tr>
                        <tr>
                            <td><strong>Tags</strong></td>
                            <td><span id="edit-ai-tags" style="font-size: 0.85em;">-</span></td>
                            <td><input type="text" id="edit-tags" style="width: 100%;" placeholder="comma separated"></td>
                            <td><label><input type="checkbox" id="edit-sync-tags" checked> Write</label></td>
                        </tr>
                    </tbody>
                </table>
                
                <div style="background: #0f1a2e; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <strong style="color: #00d4ff;">Current Paperless Data</strong>
                    <div id="edit-paperless-data" style="margin-top: 10px; font-size: 0.9em; color: #888;">
                        Loading...
                    </div>
                </div>
                
                <div style="display: flex; gap: 10px;">
                    <button onclick="submitCorrection()" class="success">Save & Learn</button>
                    <button onclick="closeModal()">Cancel</button>
                </div>
                <p style="color: #666; font-size: 0.8em; margin-top: 10px;">
                    • Mappings are only created when AI value differs from your correction<br>
                    • "Sync" writes your correction to Paperless, overwriting current data
                </p>
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
            if (tabName === 'export') refreshExports();
            if (tabName === 'logs') refreshLogs();
        }
        
        function closeModal() {
            document.querySelectorAll('.modal').forEach(m => m.classList.remove('active'));
        }
        
        function showAddMappingModal() {
            document.getElementById('add-mapping-modal').classList.add('active');
        }
        
        // Store current edit data
        let currentEditData = { ai: {}, paperless: {} };
        
        async function showEditModal(auditId, docId, aiType, aiCorr, aiTags) {
            // Reset and show loading
            document.getElementById('edit-audit-id').value = auditId;
            document.getElementById('edit-doc-id').value = docId;
            document.getElementById('edit-loading').style.display = 'block';
            document.getElementById('edit-form').style.display = 'none';
            document.getElementById('edit-modal').classList.add('active');
            
            // Store AI data
            currentEditData.ai = {
                document_type: aiType || '',
                correspondent: aiCorr || '',
                tags: aiTags || []
            };
            
            // Display AI data
            document.getElementById('edit-ai-type').textContent = aiType || '-';
            document.getElementById('edit-ai-corr').textContent = aiCorr || '-';
            document.getElementById('edit-ai-tags').innerHTML = (aiTags || []).map(t => `<span class="tag">${t}</span>`).join(' ') || '-';
            
            // Pre-fill correction fields with AI data
            document.getElementById('edit-type').value = aiType || '';
            document.getElementById('edit-correspondent').value = aiCorr || '';
            document.getElementById('edit-tags').value = (aiTags || []).join(', ');
            
            // Fetch current Paperless data
            try {
                const resp = await fetch(`/api/paperless/document/${docId}`);
                const data = await resp.json();
                
                if (data.success && data.document) {
                    currentEditData.paperless = data.document;
                    const pd = data.document;
                    document.getElementById('edit-paperless-data').innerHTML = `
                        <div><strong>Type:</strong> ${pd.document_type || '<em>not set</em>'}</div>
                        <div><strong>Correspondent:</strong> ${pd.correspondent || '<em>not set</em>'}</div>
                        <div><strong>Tags:</strong> ${pd.tags.length ? pd.tags.join(', ') : '<em>none</em>'}</div>
                    `;
                } else {
                    document.getElementById('edit-paperless-data').innerHTML = '<em>Could not fetch Paperless data</em>';
                }
            } catch (e) {
                document.getElementById('edit-paperless-data').innerHTML = '<em>Error fetching data</em>';
            }
            
            // Show form
            document.getElementById('edit-loading').style.display = 'none';
            document.getElementById('edit-form').style.display = 'block';
        }
        
        async function submitCorrection() {
            const auditId = parseInt(document.getElementById('edit-audit-id').value);
            const docId = parseInt(document.getElementById('edit-doc-id').value);
            const userType = document.getElementById('edit-type').value.trim();
            const userCorr = document.getElementById('edit-correspondent').value.trim();
            const userTagsStr = document.getElementById('edit-tags').value;
            const userTags = userTagsStr ? userTagsStr.split(',').map(t => t.trim()).filter(t => t) : [];
            
            const syncType = document.getElementById('edit-sync-type').checked;
            const syncCorr = document.getElementById('edit-sync-corr').checked;
            const syncTags = document.getElementById('edit-sync-tags').checked;
            
            // Build corrections object with explicit AI values
            const corrections = {};
            
            if (userType) {
                corrections.document_type = {
                    ai: currentEditData.ai.document_type,
                    user: userType,
                    sync: syncType
                };
            }
            
            if (userCorr) {
                corrections.correspondent = {
                    ai: currentEditData.ai.correspondent,
                    user: userCorr,
                    sync: syncCorr
                };
            }
            
            if (userTags.length > 0) {
                corrections.tags = {
                    ai: currentEditData.ai.tags,
                    user: userTags,
                    sync: syncTags
                };
            }
            
            // Submit correction
            const resp = await fetch('/api/learn/correct', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ 
                    audit_id: auditId, 
                    document_id: docId,
                    corrections: corrections 
                })
            });
            
            const result = await resp.json();
            closeModal();
            refreshMappings();
            
            let msg = 'Correction saved!';
            if (result.mappings_created && result.mappings_created.length > 0) {
                msg += '\\n\\nLearned mappings:\\n• ' + result.mappings_created.join('\\n• ');
            }
            if (result.paperless_updated) {
                msg += '\\n\\nPaperless updated successfully.';
            }
            alert(msg);
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
                            <td>
                                <button onclick="reanalyze(${log.document_id})" class="edit-btn" title="Re-analyze">↻</button>
                                <button onclick="deleteEntry(${log.id})" class="delete-btn" title="Delete">×</button>
                            </td>
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
                        const tagsArr = JSON.stringify(log.tags || []);
                        
                        return `<tr>
                            <td><strong>#${log.document_id}</strong><br><small>${(log.document_title || '').substring(0, 40)}</small></td>
                            <td>${log.document_type || '-'}</td>
                            <td>${log.correspondent || '-'}</td>
                            <td>${tags || '-'}</td>
                            <td>${conf}</td>
                            <td>
                                <button class="edit-btn" onclick='showEditModal(${log.id}, ${log.document_id}, "${(log.document_type || '').replace(/"/g, '\\"')}", "${(log.correspondent || '').replace(/"/g, '\\"')}", ${tagsArr})'>Correct</button>
                                <button class="edit-btn" onclick="reanalyze(${log.document_id})" title="Re-analyze">↻ Redo</button>
                            </td>
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
                
                // Also refresh examples
                await refreshExamplesList();
            } catch (e) { console.error(e); }
        }
        
        async function refreshExamplesList() {
            try {
                const data = await fetchJson('/api/learn/examples');
                const tbody = document.getElementById('examples-list');
                
                if (data.examples && data.examples.length > 0) {
                    tbody.innerHTML = data.examples.map(ex => {
                        const tags = (ex.tags || []).slice(0, 4).map(t => `<span class="tag">${t}</span>`).join(' ');
                        const verified = ex.user_verified ? '✓' : '-';
                        return `<tr>
                            <td><strong>#${ex.document_id}</strong><br><small>${(ex.document_title || '').substring(0, 30)}</small></td>
                            <td>${ex.document_type || '-'}</td>
                            <td>${ex.correspondent || '-'}</td>
                            <td>${tags || '-'}</td>
                            <td style="color: ${ex.user_verified ? '#00ff88' : '#888'}">${verified}</td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="5" style="color: #888;">No examples yet.</td></tr>';
                }
            } catch (e) { console.error(e); }
        }
        
        async function resetLearning() {
            if (confirm('⚠️ Reset ALL learning data?\\n\\nThis will delete:\\n• All term mappings\\n• All classification examples\\n\\nThe AI will start fresh with no learned patterns.')) {
                const resp = await fetch('/api/learn/reset', { method: 'DELETE' });
                const data = await resp.json();
                alert(`Learning reset!\\n• ${data.examples_deleted} examples deleted\\n• ${data.mappings_deleted} mappings deleted`);
                refreshMappings();
            }
        }
        
        async function clearExamples() {
            if (confirm('Clear all classification examples?\\n\\nMappings will be kept.')) {
                const resp = await fetch('/api/learn/examples', { method: 'DELETE' });
                const data = await resp.json();
                alert(`Cleared ${data.deleted} examples`);
                refreshMappings();
            }
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
                    { key: 'OLLAMA_TEMPERATURE', label: 'Temperature' },
                    { key: 'OLLAMA_TOP_P', label: 'Top P' },
                    { key: 'OLLAMA_TOP_K', label: 'Top K' },
                    { key: 'MAX_PAGES', label: 'Max Pages' },
                    { key: 'IMAGE_MAX_SIZE', label: 'Image Max Size (px)' },
                    { key: 'IMAGE_QUALITY', label: 'Image Quality (75-100)' },
                    { key: 'AUTO_COMMIT', label: 'Auto Commit', type: 'select', options: ['true', 'false'] },
                    { key: 'LEARNING_ENABLED', label: 'Learning Enabled', type: 'select', options: ['true', 'false'] },
                    { key: 'FEW_SHOT_ENABLED', label: 'Few-Shot Examples', type: 'select', options: ['false', 'true'] },
                    { key: 'INJECT_EXISTING_TAGS', label: 'Inject Existing Tags', type: 'select', options: ['true', 'false'] },
                    { key: 'INJECT_EXISTING_TYPES', label: 'Inject Doc Types', type: 'select', options: ['false', 'true'] },
                    { key: 'FUZZY_MATCH_THRESHOLD', label: 'Fuzzy Match Threshold' },
                    { key: 'GENERATE_EXPLANATIONS', label: 'Generate Explanations', type: 'select', options: ['false', 'true'] },
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
            const keys = ['PAPERLESS_URL', 'PAPERLESS_TOKEN', 'OLLAMA_URL', 'OLLAMA_MODEL', 'OLLAMA_TEMPERATURE', 'OLLAMA_TOP_P', 'OLLAMA_TOP_K', 'MAX_PAGES', 'IMAGE_MAX_SIZE', 'IMAGE_QUALITY', 'AUTO_COMMIT', 'LEARNING_ENABLED', 'FEW_SHOT_ENABLED', 'INJECT_EXISTING_TAGS', 'INJECT_EXISTING_TYPES', 'FUZZY_MATCH_THRESHOLD', 'GENERATE_EXPLANATIONS'];
            for (const key of keys) {
                const el = document.getElementById(`config-${key}`);
                if (el) await fetch('/api/config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ key, value: el.value }) });
            }
            alert('Saved! Restart service for changes to take effect.');
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
        
        async function reanalyze(docId) {
            if (confirm('Re-analyze document #' + docId + '? This will queue it for processing.')) {
                await fetch('/api/reanalyze/' + docId, { method: 'POST' });
                alert('Document #' + docId + ' queued for re-analysis');
                refreshAll();
            }
        }
        
        async function generateExport() {
            try {
                const btn = event.target;
                btn.textContent = 'Generating...';
                btn.disabled = true;
                
                const resp = await fetch('/api/export/debug');
                if (resp.ok) {
                    const blob = await resp.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = resp.headers.get('content-disposition')?.split('filename=')[1] || 'debug_export.json';
                    a.click();
                    window.URL.revokeObjectURL(url);
                    refreshExports();
                } else {
                    alert('Export failed: ' + (await resp.text()));
                }
            } catch (e) {
                alert('Export error: ' + e.message);
            } finally {
                event.target.textContent = 'Generate New Export';
                event.target.disabled = false;
            }
        }
        
        async function refreshExports() {
            try {
                const data = await fetchJson('/api/export/list');
                const tbody = document.getElementById('exports-list');
                
                if (data.files && data.files.length > 0) {
                    tbody.innerHTML = data.files.map(f => {
                        const created = new Date(f.created).toLocaleString();
                        return `<tr>
                            <td><code>${f.filename}</code></td>
                            <td>${f.size_kb} KB</td>
                            <td>${created}</td>
                            <td>
                                <button class="edit-btn" onclick="downloadExport('${f.filename}')">Download</button>
                                <button class="delete-btn" onclick="deleteExport('${f.filename}')">×</button>
                            </td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="4" style="color: #888;">No exports yet. Click "Generate New Export" to create one.</td></tr>';
                }
            } catch (e) { console.error(e); }
        }
        
        async function downloadExport(filename) {
            window.location.href = '/api/export/download/' + filename;
        }
        
        async function deleteExport(filename) {
            if (confirm('Delete export ' + filename + '?')) {
                await fetch('/api/export/' + filename, { method: 'DELETE' });
                refreshExports();
            }
        }
        
        function refreshAll() {
            refreshStatus();
            refreshAuditLog();
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }
        
        // Log viewer
        let logRefreshTimer = null;
        
        function colorizeLogLine(line) {
            if (line.includes('| ERROR |')) return `<span style="color:#ff4444">${escapeHtml(line)}</span>`;
            if (line.includes('| WARNING |')) return `<span style="color:#ffaa00">${escapeHtml(line)}</span>`;
            if (line.includes('| DEBUG |')) return `<span style="color:#666">${escapeHtml(line)}</span>`;
            // INFO — highlight key events
            if (line.includes('committed to Paperless') || line.includes('Vision success'))
                return `<span style="color:#00cc88">${escapeHtml(line)}</span>`;
            if (line.includes('Processing document'))
                return `<span style="color:#00d4ff">${escapeHtml(line)}</span>`;
            return escapeHtml(line);
        }
        
        function escapeHtml(text) {
            const d = document.createElement('div');
            d.textContent = text;
            return d.innerHTML;
        }
        
        async function refreshLogs() {
            const viewer = document.getElementById('log-viewer');
            const level = document.getElementById('log-level-filter').value;
            const counter = document.getElementById('log-line-count');
            try {
                let url = '/api/logs?lines=300';
                if (level) url += '&level=' + level;
                const resp = await fetch(url);
                const data = await resp.json();
                
                viewer.innerHTML = data.lines.map(colorizeLogLine).join('\\n');
                counter.textContent = data.lines.length + ' lines' + (data.total > data.lines.length ? ' (of ' + data.total + ' total)' : '');
                
                if (document.getElementById('log-autoscroll').checked) {
                    viewer.scrollTop = viewer.scrollHeight;
                }
            } catch (e) {
                viewer.textContent = 'Failed to load logs: ' + e.message;
            }
            
            // Manage auto-refresh timer
            clearInterval(logRefreshTimer);
            if (document.getElementById('log-autorefresh').checked && 
                document.getElementById('tab-logs').classList.contains('active')) {
                logRefreshTimer = setInterval(refreshLogs, 3000);
            }
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
        resp = await asyncio.to_thread(
            requests.get, f"{cfg['PAPERLESS_URL']}/api/documents/",
            headers={"Authorization": f"Token {cfg['PAPERLESS_TOKEN']}"},
            timeout=5
        )
        status["paperless"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception:
        status["paperless"] = "unreachable"
    try:
        resp = await asyncio.to_thread(
            requests.get, f"{cfg['OLLAMA_URL']}/api/tags", timeout=5
        )
        status["ollama"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except Exception:
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
        "pending": sorted(_pending_set)
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
    cfg = get_config()
    # Redact sensitive values
    safe = dict(cfg)
    if safe.get("PAPERLESS_TOKEN"):
        safe["PAPERLESS_TOKEN"] = "***REDACTED***"
    return safe

_ALLOWED_CONFIG_KEYS = {
    "PAPERLESS_URL", "PAPERLESS_TOKEN", "OLLAMA_URL", "OLLAMA_MODEL",
    "OLLAMA_THREADS", "OLLAMA_TEMPERATURE", "OLLAMA_TOP_P", "OLLAMA_TOP_K",
    "MAX_PAGES", "IMAGE_MAX_SIZE", "IMAGE_QUALITY",
    "AUTO_COMMIT", "GENERATE_EXPLANATIONS", "LEARNING_ENABLED",
    "FEW_SHOT_ENABLED", "INJECT_EXISTING_TAGS", "INJECT_EXISTING_TYPES", "FUZZY_MATCH_THRESHOLD",
    "API_HOST", "API_PORT", "POLL_INTERVAL",
}

@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    if update.key not in _ALLOWED_CONFIG_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown config key: {update.key}")
    # Never write back the redacted placeholder
    if update.key == "PAPERLESS_TOKEN" and update.value == "***REDACTED***":
        return {"status": "skipped", "key": update.key, "reason": "redacted value unchanged"}
    set_key(str(ENV_FILE), update.key, update.value, quote_mode="never")
    return {"status": "updated", "key": update.key}

@app.post("/api/classify")
async def classify_document(request: ClassifyRequest):
    if request.document_id in _pending_set:
        return {"status": "already_queued", "document_id": request.document_id}
    await document_queue.put(request.document_id)
    _pending_set.add(request.document_id)
    return {"status": "queued", "document_id": request.document_id}

@app.post("/api/classify/batch")
async def classify_batch(document_ids: List[int]):
    queued = 0
    for doc_id in document_ids:
        if doc_id not in _pending_set:
            await document_queue.put(doc_id)
            _pending_set.add(doc_id)
            queued += 1
    return {"status": "queued", "count": queued, "skipped": len(document_ids) - queued}

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
    _pending_set.clear()
    return {"status": "cleared", "removed": cleared}

@app.delete("/api/audit/{entry_id}")
async def delete_audit_entry(entry_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM audit_log WHERE id = ?", (entry_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    if not deleted:
        raise HTTPException(status_code=404, detail="Audit entry not found")
    return {"status": "deleted"}

_VALID_STATUSES = {"pending", "processing", "completed", "failed", "abandoned"}

@app.delete("/api/audit/status/{status}")
async def delete_audit_by_status(status: str):
    if status not in _VALID_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {', '.join(_VALID_STATUSES)}")
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
        raise HTTPException(status_code=409, detail="Mapping already exists")
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
    if not deleted:
        raise HTTPException(status_code=404, detail="Mapping not found")
    return {"status": "deleted"}


@app.delete("/api/learn/examples")
async def clear_all_examples():
    """Clear all classification examples (fresh start for learning)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM classification_examples")
    deleted = c.rowcount
    conn.commit()
    conn.close()
    logger.info(f"Cleared {deleted} classification examples")
    return {"status": "cleared", "deleted": deleted}


@app.delete("/api/learn/mappings")
async def clear_all_mappings():
    """Clear all term mappings (fresh start for learning)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM term_mappings")
    deleted = c.rowcount
    conn.commit()
    conn.close()
    logger.info(f"Cleared {deleted} term mappings")
    return {"status": "cleared", "deleted": deleted}


@app.delete("/api/learn/reset")
async def reset_all_learning():
    """Reset ALL learning data (examples + mappings)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM classification_examples")
    examples_deleted = c.rowcount
    c.execute("DELETE FROM term_mappings")
    mappings_deleted = c.rowcount
    conn.commit()
    conn.close()
    logger.info(f"Reset learning: {examples_deleted} examples, {mappings_deleted} mappings cleared")
    return {
        "status": "reset", 
        "examples_deleted": examples_deleted,
        "mappings_deleted": mappings_deleted
    }


@app.get("/api/learn/examples")
async def list_examples():
    """List all classification examples"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM classification_examples ORDER BY created_at DESC")
    rows = [dict(row) for row in c.fetchall()]
    for row in rows:
        if row.get('tags'):
            try:
                row['tags'] = json.loads(row['tags'])
            except:
                pass
    conn.close()
    return {"examples": rows}


@app.post("/api/learn/correct")
async def learn_from_correction(request: Request):
    """
    Apply user corrections with explicit field tracking.
    
    Body format:
    {
        "audit_id": 123,
        "document_id": 74,
        "corrections": {
            "document_type": {"ai": "Beleg", "user": "Rechnung", "sync": true},
            "correspondent": {"ai": "Anthropic", "user": "Anthropic, PBC", "sync": true},
            "tags": {"ai": ["a", "b"], "user": ["x", "y"], "sync": true},
            "tag_mappings": [{"ai": "old-tag", "user": "new-tag"}]
        }
    }
    """
    try:
        data = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    audit_id = data.get('audit_id')
    document_id = data.get('document_id')
    corrections = data.get('corrections', {})
    
    if not audit_id:
        raise HTTPException(status_code=400, detail="audit_id required")
    
    result = apply_learning_correction(audit_id, corrections, document_id)
    return result


@app.get("/api/paperless/document/{doc_id}")
async def get_paperless_doc(doc_id: int):
    """Fetch current document data from Paperless for comparison"""
    doc = get_paperless_document(doc_id)
    if doc:
        return {"success": True, "document": doc}
    return {"success": False, "error": "Could not fetch document"}


# Re-analyze endpoint
@app.post("/api/reanalyze/{doc_id}")
async def reanalyze_document(doc_id: int):
    """Re-queue a document for analysis"""
    await document_queue.put(doc_id)
    _pending_set.add(doc_id)
    logger.info(f"Document {doc_id} queued for re-analysis")
    return {"status": "queued", "document_id": doc_id}


# Export logs functionality
def cleanup_old_logs():
    """Remove logs older than LOG_RETENTION_DAYS"""
    cutoff = datetime.now() - timedelta(days=LOG_RETENTION_DAYS)
    removed = 0
    for log_file in LOGS_DIR.glob("*.json"):
        try:
            # Parse date from filename: debug_YYYYMMDD_HHMMSS.json
            parts = log_file.stem.split("_")
            if len(parts) >= 2:
                date_str = parts[1]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if file_date < cutoff:
                    log_file.unlink()
                    removed += 1
        except:
            pass
    
    # Also check total size
    total_size = sum(f.stat().st_size for f in LOGS_DIR.glob("*.json"))
    if total_size > LOG_MAX_SIZE_MB * 1024 * 1024:
        # Remove oldest files until under limit
        files = sorted(LOGS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime)
        while total_size > LOG_MAX_SIZE_MB * 1024 * 1024 and files:
            oldest = files.pop(0)
            total_size -= oldest.stat().st_size
            oldest.unlink()
            removed += 1
    
    return removed


def generate_debug_export() -> Path:
    """Generate a comprehensive debug export file"""
    cleanup_old_logs()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_file = LOGS_DIR / f"debug_{timestamp}.json"
    
    cfg = get_config()
    
    # System info
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }
    
    # API/Config info (sanitized)
    api_info = {
        "paperless_url": cfg["PAPERLESS_URL"],
        "ollama_url": cfg["OLLAMA_URL"],
        "ollama_model": cfg["OLLAMA_MODEL"],
        "ollama_threads": cfg["OLLAMA_THREADS"],
        "ollama_temperature": cfg.get("OLLAMA_TEMPERATURE", "model default"),
        "ollama_top_p": cfg.get("OLLAMA_TOP_P", "model default"),
        "ollama_top_k": cfg.get("OLLAMA_TOP_K", "model default"),
        "auto_commit": cfg["AUTO_COMMIT"],
        "learning_enabled": cfg["LEARNING_ENABLED"],
        "few_shot_enabled": cfg.get("FEW_SHOT_ENABLED", False),
        "inject_existing_tags": cfg.get("INJECT_EXISTING_TAGS", True),
        "inject_existing_types": cfg.get("INJECT_EXISTING_TYPES", False),
        "fuzzy_match_threshold": cfg.get("FUZZY_MATCH_THRESHOLD", 0.80),
        "generate_explanations": cfg.get("GENERATE_EXPLANATIONS", False),
        "max_pages": cfg["MAX_PAGES"],
    }
    
    # Service status
    service_status = {
        "service_started": queue_status.get("service_started"),
        "processed_count": queue_status.get("processed_count", 0),
        "last_processed": queue_status.get("last_processed"),
        "current_processing": queue_status.get("processing", False),
        "queue_size": document_queue.qsize() if document_queue else 0,
    }
    
    # Health check
    health = {"paperless": "unknown", "ollama": "unknown"}
    try:
        resp = requests.get(f"{cfg['PAPERLESS_URL']}/api/documents/", timeout=5)
        health["paperless"] = "healthy" if resp.status_code == 200 else f"error_{resp.status_code}"
    except Exception as e:
        health["paperless"] = f"unreachable: {str(e)[:50]}"
    try:
        resp = requests.get(f"{cfg['OLLAMA_URL']}/api/tags", timeout=5)
        health["ollama"] = "healthy" if resp.status_code == 200 else f"error_{resp.status_code}"
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            health["ollama_models"] = models
    except Exception as e:
        health["ollama"] = f"unreachable: {str(e)[:50]}"
    
    # Recent audit logs (last 50)
    audit_logs = get_audit_logs(limit=50)
    
    # Statistics
    stats = get_audit_stats()
    
    # Term mappings
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM term_mappings ORDER BY last_used DESC LIMIT 100')
    mappings = [dict(row) for row in c.fetchall()]
    
    # Classification examples
    c.execute('SELECT * FROM classification_examples ORDER BY created_at DESC LIMIT 50')
    examples = [dict(row) for row in c.fetchall()]
    for ex in examples:
        if ex.get('tags'):
            try:
                ex['tags'] = json.loads(ex['tags'])
            except:
                pass
    conn.close()
    
    # Read recent application log
    app_log_content = ""
    app_log_path = Path(__file__).parent / "classifier_api.log"
    if app_log_path.exists():
        try:
            with open(app_log_path, 'r') as f:
                # Last 500 lines
                lines = f.readlines()
                app_log_content = "".join(lines[-500:])
        except:
            app_log_content = "Could not read log file"
    
    # Compile export
    export_data = {
        "export_info": {
            "generated_at": datetime.now().isoformat(),
            "version": "3.0.0",
            "export_type": "debug"
        },
        "system": system_info,
        "api_config": api_info,
        "service_status": service_status,
        "health_check": health,
        "statistics": stats,
        "recent_runs": audit_logs,
        "term_mappings": mappings,
        "classification_examples": examples,
        "application_log": app_log_content
    }
    
    with open(export_file, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    logger.info(f"Debug export generated: {export_file}")
    return export_file


@app.get("/api/export/debug")
async def export_debug_logs():
    """Generate and return a debug export file"""
    try:
        export_file = await asyncio.to_thread(generate_debug_export)
        return FileResponse(
            path=export_file,
            filename=export_file.name,
            media_type="application/json"
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/export/list")
async def list_exports():
    """List available export files"""
    files = []
    for f in sorted(LOGS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        files.append({
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
            "created": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        })
    return {"files": files[:20]}  # Last 20


@app.get("/api/export/download/{filename}")
async def download_export(filename: str):
    """Download a specific export file"""
    file_path = (LOGS_DIR / filename).resolve()
    if not file_path.is_relative_to(LOGS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=file_path.name, media_type="application/json")


@app.delete("/api/export/{filename}")
async def delete_export(filename: str):
    """Delete an export file"""
    file_path = (LOGS_DIR / filename).resolve()
    if not file_path.is_relative_to(LOGS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"status": "deleted", "filename": filename}


# Log viewer
@app.get("/api/logs")
async def get_logs(lines: int = 200, level: str = None):
    """Get recent application log lines. Optional level filter: INFO, WARNING, ERROR, DEBUG"""
    log_path = Path(__file__).parent / "classifier_api.log"
    if not log_path.exists():
        return {"lines": [], "total": 0}
    
    try:
        content = await asyncio.to_thread(log_path.read_text, encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        
        # Filter by level if requested
        if level:
            level_upper = level.upper()
            all_lines = [l for l in all_lines if f"| {level_upper} |" in l]
        
        # Return last N lines
        recent = all_lines[-min(lines, 1000):]
        return {"lines": recent, "total": len(all_lines)}
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to read log file")


# Webhooks
@app.post("/webhook/paperless/{doc_id}")
async def paperless_webhook_with_id(doc_id: int):
    logger.info(f"Webhook received for document {doc_id}")
    if doc_id not in _pending_set:
        await document_queue.put(doc_id)
        _pending_set.add(doc_id)
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
        doc_id = int(doc_id)
        if doc_id not in _pending_set:
            await document_queue.put(doc_id)
            _pending_set.add(doc_id)
        logger.info(f"Document {doc_id} queued via webhook")
        return {"status": "queued", "document_id": doc_id}
    
    return {"status": "ignored", "reason": "no document_id found"}

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run("classifier_api_v2:app", host=cfg["API_HOST"], port=cfg["API_PORT"], reload=False)