#!/usr/bin/env python3
"""
Paperless Document Classifier API
FastAPI service with SQLite audit log and web dashboard.
"""
import os
import asyncio
import sqlite3
import logging
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv, set_key
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import json

# Load environment variables
ENV_FILE = Path("/root/.env")
load_dotenv(ENV_FILE)

# Configuration from .env
def get_config():
    load_dotenv(ENV_FILE, override=True)
    return {
        "PAPERLESS_URL": os.getenv("PAPERLESS_URL", "http://localhost:8000"),
        "PAPERLESS_TOKEN": os.getenv("PAPERLESS_TOKEN", ""),
        "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),
        "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "ministral-3:14b"),
        "OLLAMA_THREADS": int(os.getenv("OLLAMA_THREADS", "10")),
        "MAX_PAGES": int(os.getenv("MAX_PAGES", "3")),
        "AUTO_COMMIT": os.getenv("AUTO_COMMIT", "true").lower() == "true",
        "GENERATE_EXPLANATIONS": os.getenv("GENERATE_EXPLANATIONS", "false").lower() == "true",
        "API_HOST": os.getenv("API_HOST", "0.0.0.0"),
        "API_PORT": int(os.getenv("API_PORT", "8001")),
    }

config = get_config()

# Database setup
DB_PATH = Path("/root/classifier_audit.db")

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
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
            explanation TEXT
        )
    ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp DESC)
    ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_document_id ON audit_log(document_id)
    ''')
    conn.commit()
    conn.close()

def log_to_audit(
    document_id: int,
    document_title: str = None,
    status: str = "pending",
    document_type: str = None,
    correspondent: str = None,
    tags: List[str] = None,
    processing_time: float = None,
    tokens_used: int = None,
    auto_approved: bool = False,
    error_message: str = None,
    explanation: str = None
):
    """Log processing result to audit database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO audit_log 
        (document_id, document_title, timestamp, status, document_type, correspondent, 
         tags, processing_time, tokens_used, auto_approved, error_message, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        document_id,
        document_title,
        datetime.now().isoformat(),
        status,
        document_type,
        correspondent,
        json.dumps(tags) if tags else None,
        processing_time,
        tokens_used,
        1 if auto_approved else 0,
        error_message,
        explanation
    ))
    conn.commit()
    conn.close()

def get_audit_logs(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Get audit log entries"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('''
        SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?
    ''', (limit, offset))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    
    # Parse tags JSON
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
    
    c.execute('SELECT COUNT(*) FROM audit_log')
    total = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM audit_log WHERE status = "completed"')
    completed = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM audit_log WHERE status = "failed"')
    failed = c.fetchone()[0]
    
    c.execute('SELECT AVG(processing_time) FROM audit_log WHERE processing_time IS NOT NULL')
    avg_time = c.fetchone()[0] or 0
    
    c.execute('SELECT document_type, COUNT(*) as cnt FROM audit_log WHERE document_type IS NOT NULL GROUP BY document_type ORDER BY cnt DESC LIMIT 10')
    top_types = [{"type": row[0], "count": row[1]} for row in c.fetchall()]
    
    conn.close()
    
    return {
        "total": total,
        "completed": completed,
        "failed": failed,
        "avg_processing_time": round(avg_time, 1),
        "top_document_types": top_types
    }

# Import ministral module
import sys
sys.path.insert(0, '/root')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('classifier_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Queue for document processing
document_queue: asyncio.Queue = None
processing_lock: asyncio.Lock = None
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


# Pydantic models
class ClassifyRequest(BaseModel):
    document_id: int


class ConfigUpdate(BaseModel):
    key: str
    value: str


# Import and configure ministral
def setup_ministral():
    """Import and configure ministral module"""
    cfg = get_config()
    import ministral
    ministral.PAPERLESS_URL = cfg["PAPERLESS_URL"]
    ministral.PAPERLESS_TOKEN = cfg["PAPERLESS_TOKEN"]
    ministral.OLLAMA_URL = cfg["OLLAMA_URL"]
    ministral.OLLAMA_MODEL = cfg["OLLAMA_MODEL"]
    ministral.NUM_THREADS = cfg["OLLAMA_THREADS"]
    return ministral


# Background worker
async def process_queue():
    """Background worker that processes documents from the queue"""
    global queue_status
    
    logger.info("Queue processor started")
    
    while True:
        try:
            doc_id = await document_queue.get()
            cfg = get_config()
            ministral = setup_ministral()
            
            queue_status["processing"] = True
            queue_status["current_doc"] = doc_id
            queue_status["started_at"] = datetime.now().isoformat()
            queue_status["queue_size"] = document_queue.qsize()
            
            # Update pending list
            if doc_id in queue_status["pending_docs"]:
                queue_status["pending_docs"].remove(doc_id)
            
            # Get document title
            try:
                meta = ministral.get_document_metadata(doc_id)
                doc_title = meta.get("title", f"Document {doc_id}") if meta else f"Document {doc_id}"
                queue_status["current_title"] = doc_title
            except:
                doc_title = f"Document {doc_id}"
                queue_status["current_title"] = doc_title
            
            logger.info(f"Processing document {doc_id}: {doc_title}")
            
            # Log pending status
            log_to_audit(doc_id, doc_title, status="processing")
            
            try:
                result = ministral.process_document(doc_id, generate_explanation=cfg["GENERATE_EXPLANATIONS"])
                
                if result["success"]:
                    logger.info(f"Document {doc_id}: {result['document_type']} | {result.get('correspondent', 'n/a')}")
                    
                    committed = False
                    if cfg["AUTO_COMMIT"]:
                        if ministral.update_document_in_paperless(doc_id, result):
                            logger.info(f"Document {doc_id} committed to Paperless")
                            committed = True
                        else:
                            logger.error(f"Failed to commit document {doc_id}")
                    
                    # Log success
                    log_to_audit(
                        document_id=doc_id,
                        document_title=doc_title,
                        status="completed",
                        document_type=result.get("document_type"),
                        correspondent=result.get("correspondent"),
                        tags=result.get("tags", []),
                        processing_time=result.get("duration_sec"),
                        tokens_used=result.get("tokens_est"),
                        auto_approved=committed,
                        explanation=result.get("explanation")
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
                    log_to_audit(
                        document_id=doc_id,
                        document_title=doc_title,
                        status="failed",
                        error_message=error_msg
                    )
                    
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {e}")
                log_to_audit(
                    document_id=doc_id,
                    document_title=doc_title,
                    status="failed",
                    error_message=str(e)
                )
            
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



# Polling configuration
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
last_seen_doc_id = 0

async def poll_for_new_documents():
    """Poll Paperless for new documents"""
    global last_seen_doc_id
    import requests
    
    if POLL_INTERVAL <= 0:
        logger.info("Polling disabled (POLL_INTERVAL=0)")
        return
    
    logger.info(f"Polling enabled: checking every {POLL_INTERVAL}s")
    await asyncio.sleep(10)  # Wait for startup
    
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
                params={"ordering": "-id", "page_size": 20},
                timeout=10
            )
            
            if resp.status_code == 200:
                docs = resp.json().get("results", [])
                new_docs = [d for d in docs if d["id"] > last_seen_doc_id]
                
                for doc in reversed(new_docs):
                    doc_id = doc["id"]
                    if doc.get("document_type") is not None:
                        logger.debug(f"Document {doc_id} already classified")
                        last_seen_doc_id = max(last_seen_doc_id, doc_id)
                        continue
                    
                    await document_queue.put(doc_id)
                    queue_status["pending_docs"].append(doc_id)
                    logger.info(f"Polled new document {doc_id}: {doc.get('title', 'untitled')}")
                    last_seen_doc_id = max(last_seen_doc_id, doc_id)
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Polling error: {e}")
            await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global document_queue, processing_lock
    
    # Initialize database
    init_database()
    
    # Startup
    document_queue = asyncio.Queue()
    processing_lock = asyncio.Lock()
    queue_status["service_started"] = datetime.now().isoformat()
    
    # Start background worker
    worker_task = asyncio.create_task(process_queue())
    poll_task = asyncio.create_task(poll_for_new_documents())
    
    cfg = get_config()
    logger.info("=" * 60)
    logger.info("Paperless Document Classifier API")
    logger.info("=" * 60)
    logger.info(f"Paperless: {cfg['PAPERLESS_URL']}")
    logger.info(f"Ollama: {cfg['OLLAMA_URL']} ({cfg['OLLAMA_MODEL']})")
    logger.info(f"Auto-commit: {cfg['AUTO_COMMIT']}")
    logger.info(f"Dashboard: http://{cfg['API_HOST']}:{cfg['API_PORT']}/dashboard")
    logger.info("=" * 60)
    
    yield
    
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    logger.info("API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Paperless Document Classifier",
    description="AI-powered document classification for Paperless-ngx",
    version="1.0.0",
    lifespan=lifespan
)


# Dashboard HTML
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
        h1 { color: #00d4ff; margin-bottom: 20px; }
        h2 { color: #00d4ff; margin: 20px 0 10px; font-size: 1.2em; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { background: #16213e; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .card h3 { color: #00d4ff; margin-bottom: 15px; font-size: 1em; text-transform: uppercase; letter-spacing: 1px; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-running { background: #00ff88; box-shadow: 0 0 10px #00ff88; }
        .status-idle { background: #ffaa00; }
        .status-error { background: #ff4444; }
        .stat-value { font-size: 2em; font-weight: bold; color: #fff; }
        .stat-label { color: #888; font-size: 0.9em; }
        .processing-info { background: #1e3a5f; padding: 15px; border-radius: 8px; margin-top: 10px; }
        .processing-info .doc-id { font-size: 1.5em; color: #00d4ff; }
        .processing-info .doc-title { color: #aaa; margin-top: 5px; }
        .queue-list { max-height: 150px; overflow-y: auto; }
        .queue-item { padding: 8px; background: #0f1a2e; margin: 5px 0; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #2a2a4a; }
        th { background: #0f1a2e; color: #00d4ff; font-weight: 500; }
        tr:hover { background: #1e3a5f; }
        .tag { display: inline-block; background: #00d4ff22; color: #00d4ff; padding: 2px 8px; border-radius: 12px; margin: 2px; font-size: 0.85em; }
        .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 0.85em; }
        .status-completed { background: #00ff8822; color: #00ff88; }
        .status-failed { background: #ff444422; color: #ff4444; }
        .status-processing { background: #ffaa0022; color: #ffaa00; }
        .config-form { display: grid; gap: 15px; }
        .config-item { display: grid; grid-template-columns: 200px 1fr; gap: 10px; align-items: center; }
        .config-item label { color: #888; }
        .config-item input, .config-item select { background: #0f1a2e; border: 1px solid #2a2a4a; color: #fff; padding: 10px; border-radius: 5px; }
        .config-item input:focus { border-color: #00d4ff; outline: none; }
        button { background: #00d4ff; color: #000; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-weight: bold; }
        button:hover { background: #00b8e6; }
        button.danger { background: #ff4444; color: #fff; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .refresh-info { color: #666; font-size: 0.85em; margin-top: 10px; }
        .actions { margin: 20px 0; display: flex; gap: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1><svg style="width:24px;height:24px;vertical-align:middle;margin-right:8px" viewBox="0 0 24 24"><path fill="#00d4ff" d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/></svg>Paperless Classifier Dashboard</h1>
        
        <div class="grid">
            <!-- Service Status -->
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
            
            <!-- Current Processing -->
            <div class="card">
                <h3>Current Processing</h3>
                <div id="current-processing">
                    <div style="color: #888;">Idle - waiting for documents</div>
                </div>
            </div>
            
            <!-- Statistics -->
            <div class="card">
                <h3>Statistics</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <div class="stat-value" id="stat-total">0</div>
                        <div class="stat-label">Total Processed</div>
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
                        <div class="stat-value" id="stat-queue">0</div>
                        <div class="stat-label">In Queue</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Queue -->
        <div class="card">
            <h3>Queue</h3>
            <div id="queue-list" class="queue-list">
                <div style="color: #888;">Queue is empty</div>
            </div>
        </div>
        
        <!-- Actions -->
        <div class="actions">
            <button onclick="refreshAll()"><svg style="width:16px;height:16px;vertical-align:middle;margin-right:4px" viewBox="0 0 24 24"><path fill="currentColor" d="M17.65,6.35C16.2,4.9 14.21,4 12,4A8,8 0 0,0 4,12A8,8 0 0,0 12,20C15.73,20 18.84,17.45 19.73,14H17.65C16.83,16.33 14.61,18 12,18A6,6 0 0,1 6,12A6,6 0 0,1 12,6C13.66,6 15.14,6.69 16.22,7.78L13,11H20V4L17.65,6.35Z"/></svg>Refresh</button>
            <button onclick="clearQueue()" class="danger"><svg style="width:16px;height:16px;vertical-align:middle;margin-right:4px" viewBox="0 0 24 24"><path fill="currentColor" d="M19,4H15.5L14.5,3H9.5L8.5,4H5V6H19M6,19A2,2 0 0,0 8,21H16A2,2 0 0,0 18,19V7H6V19Z"/></svg>Clear Queue</button>
        </div>
        
        <!-- Recent Activity -->
        <div class="card">
            <h3>Recent Activity</h3>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Doc ID</th>
                        <th>Title</th>
                        <th>Type</th>
                        <th>Correspondent</th>
                        <th>Tags</th>
                        <th>Status</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody id="audit-log">
                    <tr><td colspan="8" style="color: #888;">Loading...</td></tr>
                </tbody>
            </table>
        </div>
        
        <!-- Configuration -->
        <h2><svg style="width:20px;height:20px;vertical-align:middle;margin-right:8px" viewBox="0 0 24 24"><path fill="#00d4ff" d="M12,15.5A3.5,3.5 0 0,1 8.5,12A3.5,3.5 0 0,1 12,8.5A3.5,3.5 0 0,1 15.5,12A3.5,3.5 0 0,1 12,15.5M19.43,12.97C19.47,12.65 19.5,12.33 19.5,12C19.5,11.67 19.47,11.34 19.43,11L21.54,9.37C21.73,9.22 21.78,8.95 21.66,8.73L19.66,5.27C19.54,5.05 19.27,4.96 19.05,5.05L16.56,6.05C16.04,5.66 15.5,5.32 14.87,5.07L14.5,2.42C14.46,2.18 14.25,2 14,2H10C9.75,2 9.54,2.18 9.5,2.42L9.13,5.07C8.5,5.32 7.96,5.66 7.44,6.05L4.95,5.05C4.73,4.96 4.46,5.05 4.34,5.27L2.34,8.73C2.21,8.95 2.27,9.22 2.46,9.37L4.57,11C4.53,11.34 4.5,11.67 4.5,12C4.5,12.33 4.53,12.65 4.57,12.97L2.46,14.63C2.27,14.78 2.21,15.05 2.34,15.27L4.34,18.73C4.46,18.95 4.73,19.03 4.95,18.95L7.44,17.94C7.96,18.34 8.5,18.68 9.13,18.93L9.5,21.58C9.54,21.82 9.75,22 10,22H14C14.25,22 14.46,21.82 14.5,21.58L14.87,18.93C15.5,18.67 16.04,18.34 16.56,17.94L19.05,18.95C19.27,19.03 19.54,18.95 19.66,18.73L21.66,15.27C21.78,15.05 21.73,14.78 21.54,14.63L19.43,12.97Z"/></svg>Configuration</h2>
        <div class="card">
            <div class="config-form" id="config-form">
                <!-- Filled by JS -->
            </div>
            <div style="margin-top: 20px;">
                <button onclick="saveConfig()"><svg style="width:16px;height:16px;vertical-align:middle;margin-right:4px" viewBox="0 0 24 24"><path fill="currentColor" d="M15,9H5V5H15M12,19A3,3 0 0,1 9,16A3,3 0 0,1 12,13A3,3 0 0,1 15,16A3,3 0 0,1 12,19M17,3H5C3.89,3 3,3.9 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V7L17,3Z"/></svg>Save Configuration</button>
            </div>
            <div class="refresh-info">Changes require service restart to take effect.</div>
        </div>
        
        <div class="refresh-info" style="margin-top: 20px;">
            Auto-refresh every 5 seconds | Last update: <span id="last-update">--</span>
        </div>
    </div>
    
    <script>
        async function fetchJson(url) {
            const resp = await fetch(url);
            return resp.json();
        }
        
        async function refreshStatus() {
            try {
                const queue = await fetchJson('/api/queue');
                const stats = await fetchJson('/api/stats');
                
                // Update stats
                document.getElementById('stat-total').textContent = stats.stats.completed || 0;
                document.getElementById('stat-failed').textContent = stats.stats.failed || 0;
                document.getElementById('stat-avgtime').textContent = (stats.stats.avg_processing_time || 0) + 's';
                document.getElementById('stat-queue').textContent = queue.queue_size || 0;
                
                // Update current processing
                const procDiv = document.getElementById('current-processing');
                if (queue.processing && queue.current_doc) {
                    procDiv.innerHTML = `
                        <div class="processing-info">
                            <div class="doc-id">Document #${queue.current_doc}</div>
                            <div class="doc-title">${queue.current_title || ''}</div>
                            <div style="margin-top: 10px; color: #ffaa00;"><svg style="width:16px;height:16px;vertical-align:middle;margin-right:4px;animation:spin 1s linear infinite" viewBox="0 0 24 24"><path fill="#ffaa00" d="M12,4V2A10,10 0 0,0 2,12H4A8,8 0 0,1 12,4Z"/></svg>Processing...</div>
                        </div>
                    `;
                } else {
                    procDiv.innerHTML = '<div style="color: #888;">Idle - waiting for documents</div>';
                }
                
                // Update queue list
                const queueList = document.getElementById('queue-list');
                if (queue.pending && queue.pending.length > 0) {
                    queueList.innerHTML = queue.pending.map(id => 
                        `<div class="queue-item">Document #${id}</div>`
                    ).join('');
                } else {
                    queueList.innerHTML = '<div style="color: #888;">Queue is empty</div>';
                }
                
                // Update uptime
                if (stats.service_started) {
                    const started = new Date(stats.service_started);
                    const now = new Date();
                    const diff = Math.floor((now - started) / 1000);
                    const hours = Math.floor(diff / 3600);
                    const mins = Math.floor((diff % 3600) / 60);
                    document.getElementById('uptime').textContent = `${hours}h ${mins}m`;
                }
                
            } catch (e) {
                console.error('Status refresh error:', e);
            }
        }
        
        async function refreshAuditLog() {
            try {
                const data = await fetchJson('/api/audit?limit=20');
                const tbody = document.getElementById('audit-log');
                
                if (data.logs && data.logs.length > 0) {
                    tbody.innerHTML = data.logs.map(log => {
                        const time = new Date(log.timestamp).toLocaleString();
                        const tags = (log.tags || []).map(t => `<span class="tag">${t}</span>`).join('');
                        const statusClass = log.status === 'completed' ? 'status-completed' : 
                                           log.status === 'failed' ? 'status-failed' : 'status-processing';
                        const procTime = log.processing_time ? log.processing_time.toFixed(0) + 's' : '-';
                        
                        return `<tr>
                            <td>${time}</td>
                            <td>${log.document_id}</td>
                            <td>${log.document_title || '-'}</td>
                            <td>${log.document_type || '-'}</td>
                            <td>${log.correspondent || '-'}</td>
                            <td>${tags || '-'}</td>
                            <td><span class="status-badge ${statusClass}">${log.status}</span></td>
                            <td>${procTime}</td>
                        </tr>`;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="8" style="color: #888;">No activity yet</td></tr>';
                }
            } catch (e) {
                console.error('Audit log refresh error:', e);
            }
        }
        
        async function refreshConfig() {
            try {
                const data = await fetchJson('/api/config');
                const form = document.getElementById('config-form');
                
                const configItems = [
                    { key: 'PAPERLESS_URL', label: 'Paperless URL', type: 'text' },
                    { key: 'PAPERLESS_TOKEN', label: 'Paperless Token', type: 'password' },
                    { key: 'OLLAMA_URL', label: 'Ollama URL', type: 'text' },
                    { key: 'OLLAMA_MODEL', label: 'Ollama Model', type: 'text' },
                    { key: 'OLLAMA_THREADS', label: 'Ollama Threads', type: 'number' },
                    { key: 'MAX_PAGES', label: 'Max Pages', type: 'number' },
                    { key: 'AUTO_COMMIT', label: 'Auto Commit', type: 'select', options: ['true', 'false'] },
                    { key: 'GENERATE_EXPLANATIONS', label: 'Generate Explanations', type: 'select', options: ['true', 'false'] },
                ];
                
                form.innerHTML = configItems.map(item => {
                    const value = data[item.key] !== undefined ? data[item.key] : '';
                    if (item.type === 'select') {
                        const options = item.options.map(o => 
                            `<option value="${o}" ${String(value).toLowerCase() === o ? 'selected' : ''}>${o}</option>`
                        ).join('');
                        return `<div class="config-item">
                            <label>${item.label}</label>
                            <select id="config-${item.key}">${options}</select>
                        </div>`;
                    }
                    return `<div class="config-item">
                        <label>${item.label}</label>
                        <input type="${item.type}" id="config-${item.key}" value="${value}">
                    </div>`;
                }).join('');
                
            } catch (e) {
                console.error('Config refresh error:', e);
            }
        }
        
        async function saveConfig() {
            const keys = ['PAPERLESS_URL', 'PAPERLESS_TOKEN', 'OLLAMA_URL', 'OLLAMA_MODEL', 
                         'OLLAMA_THREADS', 'MAX_PAGES', 'AUTO_COMMIT', 'GENERATE_EXPLANATIONS'];
            
            for (const key of keys) {
                const el = document.getElementById(`config-${key}`);
                if (el) {
                    await fetch('/api/config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ key, value: el.value })
                    });
                }
            }
            alert('Configuration saved! Restart service for changes to take effect.');
        }
        
        async function clearQueue() {
            if (confirm('Clear all pending documents from queue?')) {
                await fetch('/api/queue/clear', { method: 'DELETE' });
                refreshAll();
            }
        }
        
        function refreshAll() {
            refreshStatus();
            refreshAuditLog();
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        }
        
        // Initial load
        refreshAll();
        refreshConfig();
        
        // Auto-refresh
        setInterval(refreshAll, 5000);
    </script>
</body>
</html>
'''


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard"""
    return DASHBOARD_HTML


@app.get("/api/health")
async def health():
    """Health check"""
    import requests
    cfg = get_config()
    
    health_status = {"api": "healthy", "paperless": "unknown", "ollama": "unknown"}
    
    try:
        resp = requests.get(f"{cfg['PAPERLESS_URL']}/api/", timeout=5)
        health_status["paperless"] = "healthy" if resp.status_code in [200, 401, 403] else "unhealthy"
    except:
        health_status["paperless"] = "unreachable"
    
    try:
        resp = requests.get(f"{cfg['OLLAMA_URL']}/api/tags", timeout=5)
        health_status["ollama"] = "healthy" if resp.status_code == 200 else "unhealthy"
    except:
        health_status["ollama"] = "unreachable"
    
    return health_status


@app.get("/api/queue")
async def get_queue():
    """Get queue status"""
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
    """Get statistics"""
    return {
        "stats": get_audit_stats(),
        "service_started": queue_status["service_started"],
        "processed_count": queue_status["processed_count"],
        "last_processed": queue_status["last_processed"]
    }


@app.get("/api/audit")
async def get_audit(limit: int = 100, offset: int = 0):
    """Get audit log"""
    return {
        "logs": get_audit_logs(limit, offset)
    }


@app.get("/api/config")
async def get_config_api():
    """Get current configuration"""
    return get_config()


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration value"""
    set_key(ENV_FILE, update.key, update.value)
    return {"status": "updated", "key": update.key}


@app.post("/api/classify")
async def classify_document(request: ClassifyRequest):
    """Queue a document for classification"""
    doc_id = request.document_id
    await document_queue.put(doc_id)
    queue_status["pending_docs"].append(doc_id)
    return {"status": "queued", "document_id": doc_id, "queue_position": document_queue.qsize()}


@app.post("/api/classify/batch")
async def classify_batch(document_ids: List[int]):
    """Queue multiple documents"""
    for doc_id in document_ids:
        await document_queue.put(doc_id)
        queue_status["pending_docs"].append(doc_id)
    return {"status": "queued", "count": len(document_ids)}


@app.delete("/api/queue/clear")
async def clear_queue():
    """Clear the queue"""
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




@app.post("/webhook/paperless/{doc_id}")
async def paperless_webhook_with_id(doc_id: int):
    """Webhook with document ID in URL path"""
    logger.info(f"Webhook received for document {doc_id}")
    await document_queue.put(doc_id)
    queue_status["pending_docs"].append(doc_id)
    logger.info(f"Document {doc_id} queued via webhook")
    return {"status": "queued", "document_id": doc_id}

@app.post("/webhook/paperless")
async def paperless_webhook(request: Request):
    """Webhook for Paperless-ngx - accepts any format"""
    # Try to get body as JSON first
    try:
        payload = await request.json()
        logger.info(f"Webhook JSON payload: {payload}")
    except:
        # Try form data
        try:
            form = await request.form()
            payload = dict(form)
            logger.info(f"Webhook form payload: {payload}")
        except:
            # Try raw body
            body = await request.body()
            logger.info(f"Webhook raw body: {body}")
            payload = {}
    
    # Extract document ID from various possible formats
    doc_id = None
    
    # Direct fields
    if "document_id" in payload:
        doc_id = payload["document_id"]
    elif "id" in payload:
        doc_id = payload["id"]
    elif "pk" in payload:
        doc_id = payload["pk"]
    
    # Nested document object
    if not doc_id and "document" in payload:
        doc = payload["document"]
        if isinstance(doc, int):
            doc_id = doc
        elif isinstance(doc, str) and doc.isdigit():
            doc_id = int(doc)
        elif isinstance(doc, dict):
            doc_id = doc.get("id") or doc.get("pk")
    
    # Convert to int if string
    if isinstance(doc_id, str) and doc_id.isdigit():
        doc_id = int(doc_id)
    
    if doc_id:
        await document_queue.put(doc_id)
        queue_status["pending_docs"].append(doc_id)
        logger.info(f"Document {doc_id} queued via webhook")
        return {"status": "queued", "document_id": doc_id}
    
    logger.warning(f"Could not extract document_id from webhook payload: {payload}")
    return {"status": "ignored", "reason": "no document_id found", "received": str(payload)[:200]}


if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run("classifier_api:app", host=cfg["API_HOST"], port=cfg["API_PORT"], reload=False)

@app.post("/webhook/debug")
async def debug_webhook(request: Request):
    """Debug endpoint to see what Paperless sends"""
    headers = dict(request.headers)
    body = await request.body()
    logger.info(f"DEBUG Headers: {headers}")
    logger.info(f"DEBUG Body: {body[:500]}")
    try:
        json_body = await request.json()
        logger.info(f"DEBUG JSON: {json_body}")
    except:
        pass
    return {"status": "received"}

@app.post("/webhook/debug")
async def debug_webhook(request: Request):
    """Debug endpoint to see what Paperless sends"""
    headers = dict(request.headers)
    body = await request.body()
    logger.info(f"DEBUG Headers: {headers}")
    logger.info(f"DEBUG Body: {body[:500]}")
    try:
        json_body = await request.json()
        logger.info(f"DEBUG JSON: {json_body}")
    except:
        pass
    return {"status": "received"}
