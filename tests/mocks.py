"""
Minimal Paperless-ngx and Ollama stand-ins for tests. No external dependencies.

    from tests.mocks import MockServers
    with MockServers() as m:
        m.paperless_url, m.ollama_url, m.state
"""
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from PIL import Image

TOKEN = "testtoken"


def _pdf(pages: int = 3) -> bytes:
    buf = io.BytesIO()
    imgs = [Image.new("RGB", (600, 800), "white") for _ in range(pages)]
    imgs[0].save(buf, "PDF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def default_state() -> Dict[str, Any]:
    return {
        "docs": {
            101: {"id": 101, "title": "Telekom Rechnung", "document_type": None, "correspondent": None,
                  "tags": [9], "content": "Telekom Deutschland GmbH\nRechnung Nr. 12345 für Ihren DSL Anschluss\n" + "Text " * 600},
            102: {"id": 102, "title": "Mietvertrag", "document_type": 1, "correspondent": 2, "tags": [],
                  "content": "Mietvertrag über Wohnraum"},
            103: {"id": 103, "title": "Unklar", "document_type": None, "correspondent": None, "tags": [], "content": ""},
        },
        "tags": [{"id": 9, "name": "Inbox", "document_count": 40}, {"id": 1, "name": "dsl", "document_count": 5},
                 {"id": 2, "name": "strom", "document_count": 3}],
        "document_types": [{"id": 1, "name": "Vertrag", "document_count": 3}, {"id": 2, "name": "Rechnung", "document_count": 9}],
        "correspondents": [{"id": 2, "name": "Hausverwaltung Müller", "document_count": 3},
                           {"id": 3, "name": "Deutsche Telekom", "document_count": 9}],
        "next_id": 100,
        "patches": [],
        "created": [],
        "ollama_requests": [],
        "reject_think_once": False,
        "answers": {},          # prompt substring -> dict answer; default answer otherwise
        "next_answers": [],     # consumed in order before the lookup above
        "pdf": _pdf(),
    }


class _Handler(BaseHTTPRequestHandler):
    state: Dict[str, Any] = {}
    lock = threading.Lock()

    def log_message(self, *a):  # silence
        pass

    def _json(self, code: int, obj: Any) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}


class PaperlessHandler(_Handler):
    def do_GET(self):
        st = self.state
        u = urlparse(self.path); q = parse_qs(u.query); parts = [p for p in u.path.split("/") if p]
        if self.headers.get("Authorization") != f"Token {TOKEN}":
            return self._json(401, {"detail": "Invalid token."})
        if parts[:2] == ["api", "documents"]:
            if len(parts) == 2:
                docs = sorted(st["docs"].values(), key=lambda d: -d["id"])
                size = int(q.get("page_size", ["25"])[0]); page = int(q.get("page", ["1"])[0])
                chunk = docs[(page - 1) * size: page * size]
                nxt = f"{self.path}&page={page + 1}" if page * size < len(docs) else None
                return self._json(200, {"count": len(docs), "next": nxt, "results": chunk})
            doc_id = int(parts[2])
            if doc_id not in st["docs"]:
                return self._json(404, {"detail": "Not found."})
            if len(parts) == 3:
                return self._json(200, st["docs"][doc_id])
            if parts[3] == "download":
                pdf = st["pdf"]
                self.send_response(200); self.send_header("Content-Type", "application/pdf")
                self.send_header("Content-Length", str(len(pdf))); self.end_headers(); self.wfile.write(pdf); return
            return self._json(404, {})
        if parts[:2] in (["api", "tags"], ["api", "document_types"], ["api", "correspondents"]):
            items = st[parts[1]]
            if "name__iexact" in q:
                items = [i for i in items if i["name"].lower() == q["name__iexact"][0].lower()]
            return self._json(200, {"count": len(items), "next": None, "results": items})
        return self._json(404, {})

    def do_POST(self):
        st = self.state
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        body = self._body()
        if parts[:2] in (["api", "tags"], ["api", "document_types"], ["api", "correspondents"]):
            with self.lock:
                st["next_id"] += 1
                item = {"id": st["next_id"], "name": body["name"], "document_count": 0}
                st[parts[1]].append(item)
                st["created"].append((parts[1], body["name"]))
            return self._json(201, item)
        return self._json(404, {})

    def do_PATCH(self):
        st = self.state
        parts = [p for p in urlparse(self.path).path.split("/") if p]
        body = self._body()
        doc_id = int(parts[2])
        with self.lock:
            st["patches"].append((doc_id, body))
            st["docs"][doc_id].update(body)
        return self._json(200, st["docs"][doc_id])


class OllamaHandler(_Handler):
    def do_GET(self):
        if self.path.startswith("/api/ps"):
            return self._json(200, {"models": [{"name": "gemma4:e4b", "size": 6_000_000_000, "expires_at": "2030-01-01T00:00:00Z"}]})
        return self._json(200, {"models": [{"name": "gemma4:e4b"}, {"name": "llava:7b"}]})

    def do_POST(self):
        st = self.state
        body = self._body()
        if self.path == "/api/generate":
            with self.lock:
                st["ollama_requests"].append({"path": "generate", "body": body})
            return self._json(200, {"done": True})
        with self.lock:
            st["ollama_requests"].append({"path": "chat", "body": body})
            if st["reject_think_once"] and "think" in body:
                st["reject_think_once"] = False
                return self._json(400, {"error": "gemma4:e4b does not support thinking"})
        prompt = body["messages"][0]["content"]
        answer = None
        with self.lock:
            if st["next_answers"]:
                answer = st["next_answers"].pop(0)
        for needle, ans in ([] if answer is not None else st["answers"].items()):
            if needle in prompt:
                answer = ans
                break
        if answer is None:
            answer = {"dokumenttyp": "Rechnungen", "absender": "telekom",
                      "tags": ["DSL", "festnetz", "Rechnung", "Telekom"], "konfidenz": 0.93}
        content = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return self._json(200, {
            "model": body["model"], "message": {"role": "assistant", "content": content, "thinking": ""},
            "done": True, "prompt_eval_count": 1200, "eval_count": 60, "load_duration": 2_000_000_000,
            "prompt_eval_duration": 30_000_000_000, "eval_duration": 5_000_000_000,
        })


class MockServers:
    def __init__(self, state: Dict[str, Any] = None):
        self.state = state or default_state()

    def __enter__(self):
        PaperlessHandler.state = self.state
        OllamaHandler.state = self.state
        self.paperless = ThreadingHTTPServer(("127.0.0.1", 0), PaperlessHandler)
        self.ollama = ThreadingHTTPServer(("127.0.0.1", 0), OllamaHandler)
        for srv in (self.paperless, self.ollama):
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.paperless_url = f"http://127.0.0.1:{self.paperless.server_address[1]}"
        self.ollama_url = f"http://127.0.0.1:{self.ollama.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self.paperless.shutdown(); self.ollama.shutdown()

    def chats(self):
        return [r for r in self.state["ollama_requests"] if r["path"] == "chat"]
