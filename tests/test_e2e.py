"""End-to-end: the real service under uvicorn against mock Paperless and Ollama servers.
Run:  python -m pytest tests/test_e2e.py -s   or   python tests/test_e2e.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.mocks import TOKEN, MockServers  # noqa: E402


class Service:
    def __init__(self, mocks: MockServers, extra_env: str = ""):
        self.work = tempfile.mkdtemp(prefix="pcls-e2e-")
        self.port = 18000 + os.getpid() % 1000
        self.base = f"http://127.0.0.1:{self.port}"
        Path(self.work, ".env").write_text(
            f"PAPERLESS_URL={mocks.paperless_url}\nPAPERLESS_TOKEN={TOKEN}\nOLLAMA_URL={mocks.ollama_url}\n"
            f"OLLAMA_MODEL=gemma4:e4b\nMAX_PAGES=2\nPOLL_INTERVAL=0\nAPI_HOST=127.0.0.1\nAPI_PORT={self.port}\n"
            f"OLLAMA_UNLOAD_WHEN_IDLE=true\n{extra_env}\n")
        self.log = open(Path(self.work, "service.log"), "w")

    def __enter__(self):
        env = dict(os.environ, PCLS_ENV=str(Path(self.work, ".env")), PCLS_DB=str(Path(self.work, "audit.db")),
                   PCLS_LOGS=str(Path(self.work, "logs")))
        self.proc = subprocess.Popen([sys.executable, "-m", "app"], cwd=str(ROOT), env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(80):
            try:
                if requests.get(f"{self.base}/api/health", timeout=2).status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(0.25)
        raise RuntimeError("service did not start:\n" + Path(self.work, "service.log").read_text())

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()

    def get(self, path, **kw):
        r = requests.get(self.base + path, timeout=30, **kw); r.raise_for_status(); return r.json()

    def post(self, path, body=None):
        r = requests.post(self.base + path, json=body, timeout=120); r.raise_for_status(); return r.json()

    def delete(self, path):
        r = requests.delete(self.base + path, timeout=30); r.raise_for_status(); return r.json()

    def wait_idle(self, timeout=60):
        deadline = time.time() + timeout
        time.sleep(0.8)
        while time.time() < deadline:
            st = self.get("/api/status")
            if not st["processing"] and st["queue_size"] == 0 and not st["pending"]:
                time.sleep(0.3)
                return
            time.sleep(0.3)
        raise AssertionError("queue never went idle")

    def tail(self, n=40):
        return "\n".join(Path(self.work, "service.log").read_text().splitlines()[-n:])


def test_full_flow():
    with MockServers() as m, Service(m) as svc:
        st = m.state
        h = svc.get("/api/health")
        assert h["paperless"] == "healthy" and h["ollama"] == "healthy"

        # --- webhook, dedup, request shape, write-back with merge ---
        assert svc.post("/webhook/paperless", {"document_id": 101})["status"] == "queued"
        assert svc.post("/webhook/paperless/101")["status"] == "already_queued"
        svc.wait_idle()
        chats = m.chats()
        assert len(chats) == 1, svc.tail()
        body = chats[0]["body"]
        assert body["think"] is False and isinstance(body["format"], dict) and body["keep_alive"] == "5m"
        opts = body["options"]
        assert opts["num_ctx"] == 4096 and opts["num_predict"] == 512 and opts["temperature"] == 0.1 and "num_thread" not in opts
        assert len(body["messages"][0]["images"]) == 2
        prompt = body["messages"][0]["content"]
        assert "OCR" not in prompt, "OCR text must be off by default"
        assert "Deutsche Telekom" in prompt and "Rechnung" in prompt and prompt.index("Inbox") < prompt.index("dsl")
        gens = [r for r in st["ollama_requests"] if r["path"] == "generate"]
        assert gens and gens[-1]["body"]["keep_alive"] == 0, "model should be unloaded when idle"
        patches = [p for p in st["patches"] if p[0] == 101]
        assert len(patches) == 1
        p = patches[0][1]
        assert p["document_type"] == 2 and p["correspondent"] == 3
        assert 9 in p["tags"] and 1 in p["tags"], "inbox tag kept, dsl reused"
        assert [n for k, n in st["created"] if k == "tags"] == ["festnetz"]
        rows = svc.get("/api/audit?status=completed")["logs"]
        assert rows and rows[0]["document_id"] == 101 and rows[0]["tokens_used"] == 1260
        assert rows[0]["timing"]["prompt_tokens"] == 1200 and rows[0]["changes"] == ["document_type", "correspondent", "tags"]
        details = svc.get(f"/api/audit/{rows[0]['id']}")
        assert json.loads(details["ai_raw"])["model_output"]["document_type"] == "rechnungen"

        # --- already classified document is skipped, forced re-analyze is not ---
        svc.post("/api/classify", {"document_id": 102}); svc.wait_idle()
        assert len(m.chats()) == 1
        assert svc.get("/api/audit?status=skipped")["total"] == 1
        st["reject_think_once"] = True
        st["next_answers"].append({"dokumenttyp": "Vertrag", "absender": "Hausverwaltung Mueller", "tags": ["wohnung", "miete"], "konfidenz": 0.8})
        svc.post("/api/reanalyze/102"); svc.wait_idle()
        chats = m.chats()
        assert len(chats) == 3 and "think" in chats[1]["body"] and "think" not in chats[2]["body"], "retry without think"
        p102 = [p for p in st["patches"] if p[0] == 102]
        assert len(p102) == 1 and "document_type" not in p102[0][1] and p102[0][1].get("tags"), p102

        # --- dry run writes nothing ---
        n_patches = len(st["patches"])
        dry = svc.post("/api/dryrun/101")
        assert dry["ok"] and dry["prompt"] and dry["raw_answer"] and dry["timing"]["total_s"] >= 0
        assert len(st["patches"]) == n_patches

        # --- config: live reload, validation, redaction ---
        cfg = svc.get("/api/config")
        assert cfg["values"]["PAPERLESS_TOKEN"] == "***REDACTED***" and any(f["key"] == "OCR_TEXT_MODE" for f in cfg["schema"])
        r = svc.post("/api/config", {"values": {"OCR_TEXT_MODE": "hint", "MAX_PAGES": "1"}})
        assert r["updated"] == ["OCR_TEXT_MODE", "MAX_PAGES"] and r["restart_required"] is False
        bad = requests.post(svc.base + "/api/config", json={"key": "MAX_PAGES", "value": "x"})
        assert bad.status_code == 400
        svc.post("/api/classify", {"document_id": 103}); svc.wait_idle()  # no OCR content -> images only
        last = m.chats()[-1]["body"]
        assert len(last["messages"][0]["images"]) == 1
        svc.post("/api/reanalyze/101"); svc.wait_idle()
        last = m.chats()[-1]["body"]
        assert "Lesehilfe" in last["messages"][0]["content"] and len(last["messages"][0]["images"]) == 1
        svc.post("/api/config", {"key": "OCR_TEXT_MODE", "value": "only"})
        svc.post("/api/reanalyze/101"); svc.wait_idle()
        assert "images" not in m.chats()[-1]["body"]["messages"][0]

        # --- correction: mapping learned and written to Paperless ---
        row = svc.get("/api/audit?status=completed&q=101")["logs"][0]
        r = svc.post("/api/learn/correct", {"audit_id": row["id"], "corrections": {
            "document_type": {"ai": row["document_type"], "user": "Telefonrechnung", "sync": True},
            "correspondent": {"ai": row["correspondent"], "user": row["correspondent"], "sync": False},
            "tags": {"ai": row["tags"], "user": ["dsl", "festnetz"], "sync": True}}})
        assert any("document_type" in x for x in r["learned"]) and "document_type" in r["paperless_updated"]
        assert any(mp["approved_term"] == "Telefonrechnung" for mp in svc.get("/api/mappings")["mappings"])
        assert svc.get("/api/examples")["examples"]
        names = svc.get("/api/paperless/names")
        assert "Telefonrechnung" in names["document_types"]

        # --- misc endpoints ---
        assert svc.get("/api/paperless/document/101")["document_type"] == "Telefonrechnung"
        assert "gemma4:e4b" in svc.get("/api/ollama/models")["models"]
        assert svc.get("/api/status")["models_loaded"]
        assert svc.get("/api/logs?lines=50")["lines"]
        dbg = requests.get(svc.base + "/api/export/debug", timeout=30)
        assert dbg.status_code == 200 and dbg.json()["health"]["paperless"] == "healthy"
        html = requests.get(svc.base + "/", timeout=10).text
        assert "<title>Paperless AI Classifier" in html and requests.get(svc.base + "/static/app.js", timeout=10).status_code == 200
        assert svc.delete("/api/learn/reset")["mappings_deleted"] == 1
        assert svc.get("/api/status")["stats"]["failed"] == 0, svc.tail()


def test_polling_pages_through_bulk_import():
    with MockServers() as m:
        for i in range(200, 260):
            m.state["docs"][i] = {"id": i, "title": f"Bulk {i}", "document_type": None, "correspondent": None, "tags": [], "content": ""}
        with Service(m, extra_env="POLL_INTERVAL=1\nSKIP_CLASSIFIED=false\n") as svc:
            time.sleep(5)   # poller reads the newest id 3 s after start
            for i in range(300, 375):   # 75 new docs after startup, more than one page of 50
                m.state["docs"][i] = {"id": i, "title": f"New {i}", "document_type": None, "correspondent": None, "tags": [], "content": ""}
            deadline = time.time() + 40
            while time.time() < deadline:
                st = svc.get("/api/status")
                if st["stats"]["completed"] >= 75:
                    break
                time.sleep(1)
            ids = {r["document_id"] for r in svc.get("/api/audit?limit=200")["logs"]}
            assert ids >= set(range(300, 375)), (len(ids), svc.tail())
            assert not any(i < 300 for i in ids), "documents that existed at startup must not be re-queued"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failures += 1; print("FAIL", name, "::", repr(e))
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)
