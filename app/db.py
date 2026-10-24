"""
SQLite storage: audit log, learned term mappings, few-shot examples.
Opens the existing classifier_audit.db of older installations and adds any
missing columns, so nothing is lost on upgrade.
"""
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional

from .config import DB_FILE

logger = logging.getLogger(__name__)

AUDIT_STATUSES = ("processing", "completed", "failed", "skipped", "abandoned")

_AUDIT_COLUMNS = {
    # name: type (used for ALTER TABLE on older databases)
    "document_title": "TEXT", "timestamp": "TEXT", "status": "TEXT", "document_type": "TEXT",
    "correspondent": "TEXT", "tags": "TEXT", "confidence": "REAL", "processing_time": "REAL",
    "tokens_used": "INTEGER", "auto_approved": "INTEGER", "error_message": "TEXT", "explanation": "TEXT",
    "ai_raw": "TEXT", "model": "TEXT", "pages_sent": "INTEGER", "timing": "TEXT", "changes": "TEXT",
}


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute('''
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
                ai_raw TEXT,
                model TEXT,
                pages_sent INTEGER,
                timing TEXT,
                changes TEXT
            )''')
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(audit_log)")}
        for name, ctype in _AUDIT_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE audit_log ADD COLUMN {name} {ctype}")
                logger.info(f"Migrated audit_log: added column {name}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON audit_log(document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON audit_log(status)")

        conn.execute('''
            CREATE TABLE IF NOT EXISTS term_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term_type TEXT NOT NULL,
                ai_term TEXT NOT NULL,
                approved_term TEXT NOT NULL,
                times_used INTEGER DEFAULT 1,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                last_used INTEGER DEFAULT (strftime('%s', 'now')),
                UNIQUE(term_type, ai_term)
            )''')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mappings_lookup ON term_mappings(term_type, ai_term)")

        conn.execute('''
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
            )''')
        conn.execute("CREATE INDEX IF NOT EXISTS idx_examples_verified "
                     "ON classification_examples(user_verified, confidence DESC)")


def _row(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    for key in ("tags", "timing", "changes"):
        if d.get(key):
            try:
                d[key] = json.loads(d[key])
            except (TypeError, ValueError):
                d[key] = [] if key != "timing" else {}
    return d


# ----------------------------------------------------------------------------- audit log

def audit_start(document_id: int, title: str, model: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO audit_log (document_id, document_title, timestamp, status, model) VALUES (?, ?, ?, 'processing', ?)",
            (document_id, title, datetime.now().isoformat(timespec="seconds"), model))
        return cur.lastrowid


def audit_finish(audit_id: int, status: str, **fields: Any) -> None:
    """Complete a 'processing' row. Accepted fields: any audit_log column."""
    cols = {k: v for k, v in fields.items() if k in _AUDIT_COLUMNS}
    for key in ("tags", "timing", "changes"):
        if key in cols and cols[key] is not None and not isinstance(cols[key], str):
            cols[key] = json.dumps(cols[key], ensure_ascii=False)
    if "auto_approved" in cols:
        cols["auto_approved"] = 1 if cols["auto_approved"] else 0
    cols["status"] = status
    cols["timestamp"] = datetime.now().isoformat(timespec="seconds")
    assignments = ", ".join(f"{k} = ?" for k in cols)
    with connect() as conn:
        conn.execute(f"UPDATE audit_log SET {assignments} WHERE id = ?", (*cols.values(), audit_id))


def audit_list(limit: int = 50, offset: int = 0, status: Optional[str] = None,
               query: Optional[str] = None) -> Dict[str, Any]:
    where, params = [], []
    if status:
        where.append("status = ?"); params.append(status)
    if query:
        like = f"%{query}%"
        where.append("(document_title LIKE ? OR document_type LIKE ? OR correspondent LIKE ? OR tags LIKE ? "
                     "OR CAST(document_id AS TEXT) = ?)")
        params += [like, like, like, like, query]
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    with connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM audit_log{clause}", params).fetchone()[0]
        rows = conn.execute(f"SELECT * FROM audit_log{clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                            (*params, limit, offset)).fetchall()
    return {"total": total, "logs": [_row(r) for r in rows]}


def audit_get(audit_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM audit_log WHERE id = ?", (audit_id,)).fetchone()
    return _row(row) if row else None


def audit_stats() -> Dict[str, Any]:
    with connect() as conn:
        counts = {r["status"]: r["n"] for r in conn.execute("SELECT status, COUNT(*) AS n FROM audit_log GROUP BY status")}
        avg = conn.execute("SELECT AVG(processing_time), AVG(confidence) FROM audit_log WHERE status = 'completed'").fetchone()
        today = datetime.now().strftime("%Y-%m-%d")
        done_today = conn.execute("SELECT COUNT(*) FROM audit_log WHERE status = 'completed' AND timestamp LIKE ?",
                                  (today + "%",)).fetchone()[0]
        mappings = conn.execute("SELECT COUNT(*) FROM term_mappings").fetchone()[0]
        verified = conn.execute("SELECT COUNT(*) FROM classification_examples WHERE user_verified = 1").fetchone()[0]
    return {
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "skipped": counts.get("skipped", 0),
        "abandoned": counts.get("abandoned", 0),
        "completed_today": done_today,
        "avg_processing_time": round(avg[0] or 0, 1),
        "avg_confidence": round((avg[1] or 0) * 100, 1),
        "learned_mappings": mappings,
        "verified_examples": verified,
    }


def audit_delete(audit_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM audit_log WHERE id = ?", (audit_id,)).rowcount > 0


def audit_delete_status(status: str) -> int:
    with connect() as conn:
        return conn.execute("DELETE FROM audit_log WHERE status = ?", (status,)).rowcount


def audit_abandon_stale(older_than_hours: float = 1.0) -> int:
    cutoff = (datetime.now() - timedelta(hours=older_than_hours)).isoformat(timespec="seconds")
    with connect() as conn:
        return conn.execute("UPDATE audit_log SET status = 'abandoned' WHERE status = 'processing' AND timestamp < ?",
                            (cutoff,)).rowcount


# ----------------------------------------------------------------------------- term mappings

def mapping_lookup(term_type: str, ai_term: str) -> Optional[str]:
    if not ai_term:
        return None
    with connect() as conn:
        row = conn.execute("SELECT approved_term FROM term_mappings WHERE term_type = ? AND LOWER(ai_term) = LOWER(?)",
                           (term_type, ai_term.strip())).fetchone()
    return row["approved_term"] if row else None


def mapping_upsert(term_type: str, ai_term: str, approved_term: str) -> bool:
    ai_term, approved_term = (ai_term or "").strip(), (approved_term or "").strip()
    if not ai_term or not approved_term or ai_term.lower() == approved_term.lower():
        return False
    with connect() as conn:
        conn.execute('''
            INSERT INTO term_mappings (term_type, ai_term, approved_term) VALUES (?, ?, ?)
            ON CONFLICT(term_type, ai_term) DO UPDATE SET
                approved_term = excluded.approved_term,
                times_used = times_used + 1,
                last_used = strftime('%s', 'now')''', (term_type, ai_term, approved_term))
    logger.info(f"Learned {term_type}: '{ai_term}' -> '{approved_term}'")
    return True


def mapping_list() -> List[Dict[str, Any]]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM term_mappings ORDER BY term_type, times_used DESC, ai_term")]


def mapping_delete(mapping_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM term_mappings WHERE id = ?", (mapping_id,)).rowcount > 0


def mappings_clear() -> int:
    with connect() as conn:
        return conn.execute("DELETE FROM term_mappings").rowcount


# ----------------------------------------------------------------------------- examples

def example_add(document_id: Optional[int], title: str, document_type: str, correspondent: Optional[str],
                tags: List[str], confidence: float, verified: bool) -> None:
    with connect() as conn:
        conn.execute('''
            INSERT INTO classification_examples
            (document_id, document_title, document_type, correspondent, tags, confidence, user_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (document_id, title or "", document_type, correspondent, json.dumps(tags or [], ensure_ascii=False),
             confidence or 0.0, 1 if verified else 0))


def example_list(limit: int = 200, verified_only: bool = False) -> List[Dict[str, Any]]:
    clause = " WHERE user_verified = 1" if verified_only else ""
    with connect() as conn:
        rows = conn.execute(f"SELECT * FROM classification_examples{clause} "
                            "ORDER BY user_verified DESC, confidence DESC, created_at DESC LIMIT ?", (limit,))
        return [_row(r) for r in rows]


def example_delete(example_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM classification_examples WHERE id = ?", (example_id,)).rowcount > 0


def examples_clear() -> int:
    with connect() as conn:
        return conn.execute("DELETE FROM classification_examples").rowcount
