"""Unit tests for the pure parts: prompt, parsing, coercion, fuzzy matching, images, config.
Run:  python -m pytest tests/   or   python tests/test_units.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp(prefix="pcls-unit-")
os.environ.setdefault("PCLS_ENV", os.path.join(_tmp, ".env"))
os.environ.setdefault("PCLS_DB", os.path.join(_tmp, "audit.db"))
os.environ.setdefault("PCLS_LOGS", os.path.join(_tmp, "logs"))

from PIL import Image  # noqa: E402

from app import db, learning  # noqa: E402
from app.config import Settings, validate  # noqa: E402
from app.images import encode_image, render_pdf_pages  # noqa: E402
from app.prompt import PromptContext, build_prompt, coerce_result, json_schema, ocr_excerpt, parse_response  # noqa: E402

db.init_db()


# --------------------------------------------------------------------------- parsing
def test_parse_code_block_and_trailing_comma():
    r = parse_response('```json\n{"dokumenttyp": "Rechnung", "absender": "Telekom", "tags": ["dsl"], "konfidenz": 0.9,}\n```')
    assert r["document_type"] == "Rechnung" and r["confidence"] == 0.9


def test_parse_raw_newline_in_string():
    r = parse_response('Ergebnis: {"dokumenttyp": "vertrag", "absender": "A", "tags": ["x"], "konfidenz": 1, "erklaerung": "Zeile 1\nZeile 2"}')
    assert r["explanation"] == "Zeile 1 Zeile 2"


def test_parse_no_json():
    assert parse_response("kein json") is None
    assert parse_response("[1, 2]") is None


def test_coerce_types():
    c = coerce_result({"document_type": " Rechnung ", "correspondent": "Keine", "tags": "dsl, festnetz; mobilfunk", "confidence": "85%"})
    assert c["tags"] == ["dsl", "festnetz", "mobilfunk"]
    assert abs(c["confidence"] - 0.85) < 1e-9
    assert c["correspondent"] is None and c["document_type"] == "rechnung"
    c = coerce_result({"document_type": ["bescheid"], "tags": None, "confidence": None})
    assert c["document_type"] == "bescheid" and c["tags"] == [] and c["confidence"] == 0.0
    c = coerce_result({"document_type": "x", "tags": [1, "#steuer", "", "none"], "confidence": 90})
    assert c["tags"] == ["1", "steuer"] and c["confidence"] == 0.9


# --------------------------------------------------------------------------- prompt
def test_prompt_layout():
    ctx = PromptContext(num_images=2, types=["rechnung"], correspondents=["Deutsche Telekom"], tags=["dsl", "strom"])
    p = build_prompt(ctx)
    assert p.startswith("Die 2 Bilder")
    assert "Dokumenttypen: rechnung" in p and "Absender: Deutsche Telekom" in p and "Tags: dsl, strom" in p
    assert "OCR" not in p
    assert p.rstrip().endswith('"konfidenz": 0.8}')
    assert "erklaerung" not in p


def test_prompt_ocr_hint_marks_images_authoritative():
    p = build_prompt(PromptContext(num_images=1, ocr_text="Rechnung Nr 1"))
    assert "Lesehilfe" in p and "Maßgeblich sind die Bilder" in p and "<<<\nRechnung Nr 1\n>>>" in p
    p = build_prompt(PromptContext(num_images=0, ocr_text="Rechnung Nr 1"))
    assert "Layout ging verloren" in p and "Bild" not in p.split("OCR-Text")[0]


def test_prompt_explanations_and_schema():
    p = build_prompt(PromptContext(num_images=1, explanations=True))
    assert "5. erklaerung" in p and '"erklaerung": "..."' in p
    assert "erklaerung" in json_schema(True)["required"] and "erklaerung" not in json_schema(False)["required"]


def test_ocr_excerpt():
    t = ocr_excerpt("Zeile  eins\r\n\r\n\r\nZeile   zwei\t\tdrei " + "wort " * 200, 300)
    assert t.startswith("Zeile eins\nZeile zwei drei") and t.endswith("…") and len(t) <= 303
    assert ocr_excerpt(None, 100) == "" and ocr_excerpt("", 100) == ""


# --------------------------------------------------------------------------- fuzzy / normalize
def test_fuzzy_rules():
    fm = learning.fuzzy_match
    assert fm("Rechnung", ["rechnung", "vertrag"], 0.8) == "rechnung"
    assert fm("rechnungen", ["rechnung", "vertrag"], 0.8) == "rechnung"
    assert fm("amt", ["finanzamt"], 0.8) is None
    assert fm("bahn", ["autobahn"], 0.8) is None
    assert fm("telekom", ["Deutsche Telekom", "Vodafone"], 0.8, partial_names=True) == "Deutsche Telekom"
    assert fm("telekom", ["Deutsche Telekom", "Vodafone"], 0.8) is None
    assert fm("stadtwerke", ["Stadtwerke Bonn", "Stadtwerke Köln"], 0.8, partial_names=True) is None
    assert fm("miete", ["mietvertrag", "nebenkosten"], 0.8) is None


def test_tag_repeats():
    assert learning.tag_repeats("Telekom", "Deutsche Telekom")
    assert learning.tag_repeats("rechnungen", "rechnung")
    assert not learning.tag_repeats("dsl", "Deutsche Telekom")


def test_normalize():
    n = learning.normalize(
        {"document_type": "Rechnungen", "correspondent": "telekom", "tags": ["DSL", "dsl", "Rechnung", "Telekom", "ab", "festnetz-anschluss"], "confidence": 0.9},
        types=["Rechnung"], correspondents=["Deutsche Telekom"], tags=["dsl", "festnetz"], threshold=0.8)
    assert n["document_type"] == "Rechnung" and n["correspondent"] == "Deutsche Telekom"
    assert n["tags"] == ["dsl", "festnetz-anschluss"]
    assert any("repeats correspondent" in x for x in n["normalized"])


def test_mapping_applied():
    db.mapping_upsert("document_type", "beleg", "Rechnung")
    n = learning.normalize({"document_type": "Beleg", "correspondent": None, "tags": [], "confidence": 0.5},
                           types=[], correspondents=[], tags=[], threshold=0.8)
    assert n["document_type"] == "Rechnung" and n["correspondent"] is None
    db.mappings_clear()


# --------------------------------------------------------------------------- images
def _pdf(pages=4):
    buf = io.BytesIO()
    imgs = [Image.new("RGB", (1240, 1754), "white") for _ in range(pages)]
    imgs[0].save(buf, "PDF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def test_render_pdf_pages():
    out = render_pdf_pages(_pdf(), max_pages=2, max_size=800, quality=70, grayscale=True, timeout=60)
    assert len(out) == 2
    im = Image.open(io.BytesIO(out[0]))
    assert im.format == "JPEG" and max(im.size) == 800 and im.mode == "L"
    out = render_pdf_pages(_pdf(1), max_pages=3, max_size=600, quality=70, grayscale=False, timeout=60)
    assert len(out) == 1 and Image.open(io.BytesIO(out[0])).mode == "RGB"


def test_render_bad_pdf():
    try:
        render_pdf_pages(b"not a pdf", 1, 800, 70, False, 30)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "pdftoppm" in str(e)


def test_encode_image():
    buf = io.BytesIO(); Image.new("RGB", (3000, 2000), "white").save(buf, "PNG")
    im = Image.open(io.BytesIO(encode_image(buf.getvalue(), 1024, 85, False)))
    assert max(im.size) == 1024 and im.format == "JPEG"


# --------------------------------------------------------------------------- config
def test_config_defaults_and_validation():
    s = Settings.load()
    assert s.OLLAMA_MODEL == "gemma4:e4b" and s.OCR_TEXT_MODE == "off" and s.OLLAMA_TOP_P is None and s.OLLAMA_THREADS == 0
    assert validate("OLLAMA_TOP_P", "") == "" and validate("AUTO_COMMIT", "TRUE") == "true"
    for key, bad in (("MAX_PAGES", "x"), ("OCR_TEXT_MODE", "maybe"), ("API_PORT", "1\n2")):
        try:
            validate(key, bad); assert False, key
        except ValueError:
            pass
    Settings.save("MAX_PAGES", "2")
    Settings.save("OLLAMA_TOP_P", "")
    s = Settings.load()
    assert s.MAX_PAGES == 2 and s.OLLAMA_TOP_P is None


def test_audit_roundtrip():
    aid = db.audit_start(7, "Doc 7", "m")
    db.audit_finish(aid, "completed", document_type="rechnung", tags=["a"], confidence=0.7, timing={"total_s": 1.5}, changes=["tags"])
    row = db.audit_get(aid)
    assert row["status"] == "completed" and row["tags"] == ["a"] and row["timing"]["total_s"] == 1.5 and row["changes"] == ["tags"]
    assert db.audit_list(status="completed")["total"] >= 1
    assert db.audit_stats()["completed"] >= 1
    db.audit_delete(aid)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failures += 1; print("FAIL", name, "::", repr(e))
    print("FAILURES:", failures)
    sys.exit(1 if failures else 0)
