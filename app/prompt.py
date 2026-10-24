"""
Prompt, JSON schema and answer parsing.

Prompt layout (small models follow this best): context first (page count,
reference lists from Paperless, optional OCR excerpt, few-shot examples), then
short numbered rules, then the exact JSON shape as the very last line.

Page images are the primary input on purpose: the layout (letterhead top left,
client table in the middle, bank details in the footer) is what lets the model
tell sender from recipient and subject from boilerplate. Flat OCR text loses
those relationships and was found to hurt results.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .paperless import NO_NAME

_RE_CODE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_RE_TRAILING_COMMA_OBJ = re.compile(r",\s*}")
_RE_TRAILING_COMMA_ARR = re.compile(r",\s*]")
_RE_SPACES = re.compile(r"[ \t\f\v]+")
_RE_BLANK_LINES = re.compile(r"\n\s*\n+")

FIELD_MAP = {
    "dokumenttyp": "document_type",
    "absender": "correspondent",
    "konfidenz": "confidence",
    "zusammenfassung": "summary",
    "erklärung": "explanation",
    "erklaerung": "explanation",
}


@dataclass
class PromptContext:
    num_images: int = 0
    ocr_text: str = ""
    types: List[str] = field(default_factory=list)
    correspondents: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    few_shot: str = ""
    explanations: bool = False


def ocr_excerpt(content: Optional[str], limit: int) -> str:
    """Whitespace-collapsed start of Paperless' OCR text."""
    if not content or not isinstance(content, str):
        return ""
    text = _RE_SPACES.sub(" ", content.replace("\r", "\n"))
    text = _RE_BLANK_LINES.sub("\n", text).strip()
    limit = max(200, int(limit))
    if len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        text = text[: cut if cut > limit * 0.6 else limit].rstrip() + " …"
    return text


def build_prompt(ctx: PromptContext) -> str:
    parts: List[str] = []

    if ctx.num_images == 1:
        parts.append("Das Bild zeigt die erste Seite eines Dokuments.")
    elif ctx.num_images > 1:
        parts.append(f"Die {ctx.num_images} Bilder sind die ersten {ctx.num_images} Seiten EINES Dokuments, in Reihenfolge.")

    refs = []
    if ctx.types:
        refs.append("- Dokumenttypen: " + ", ".join(ctx.types))
    if ctx.correspondents:
        refs.append("- Absender: " + ", ".join(ctx.correspondents))
    if ctx.tags:
        refs.append("- Tags: " + ", ".join(ctx.tags))
    if refs:
        parts.append("Bereits in Paperless vorhanden (wenn passend, exakt so wiederverwenden):\n" + "\n".join(refs))

    if ctx.ocr_text and ctx.num_images > 0:
        parts.append("Zusätzlich der Anfang des OCR-Texts, nur als Lesehilfe für kleine Schrift. "
                     "Maßgeblich sind die Bilder: Layout, Briefkopf, Logo und Position der Angaben entscheiden, "
                     f"wer Absender ist und was der Inhalt ist.\n<<<\n{ctx.ocr_text}\n>>>")
    elif ctx.ocr_text:
        parts.append("OCR-Text des Dokuments (Anfang, kann Erkennungsfehler enthalten; das Layout ging verloren):"
                     f"\n<<<\n{ctx.ocr_text}\n>>>")

    if ctx.few_shot:
        parts.append(ctx.few_shot.strip())

    rules = [
        "1. dokumenttyp: Art des Dokuments. Kurz, Singular, kleingeschrieben "
        "(z.B. rechnung, vertrag, bescheid, kündigung, mahnung, lohnabrechnung, kontoauszug). "
        "Passt ein vorhandener Dokumenttyp, nimm genau diesen.",
        "2. absender: Firma, Behörde oder Person, die das Dokument verschickt hat "
        "(Briefkopf, Logo, Unterschrift). Nicht der Empfänger. Kurzer, gebräuchlicher Name ohne Adresse. "
        "Passt ein vorhandener Absender, übernimm ihn exakt. Nicht erkennbar: \"keine\".",
        "3. tags: 3 bis 5 Schlagwörter zum konkreten Inhalt: Welches Produkt? Welche Dienstleistung? "
        "Welcher Anlass? Welches Thema? Kleingeschrieben, Substantive. Vorhandene Tags bevorzugen, wenn sie passen. "
        "KEINE Tags für: den Dokumenttyp, den Absender, generische Wörter (rechnung, dokument, zahlung, betrag, "
        "mwst, brief, information), Zahlen, Daten, Beträge, Namen von Privatpersonen.",
        "4. konfidenz: Zahl von 0 bis 1, wie sicher du dir bei dieser Einordnung bist.",
    ]
    shape = '{"dokumenttyp": "...", "absender": "...", "tags": ["...", "...", "..."], "konfidenz": 0.8'
    if ctx.explanations:
        rules.append("5. erklaerung: Ein Satz, woran du Dokumenttyp und Absender erkannt hast.")
        shape += ', "erklaerung": "..."'
    shape += "}"

    parts.append("Du klassifizierst dieses Dokument für Paperless-ngx. Bestimme:\n" + "\n".join(rules)
                 + "\n\nSprache: Deutsch. Englisch nur für eingedeutschte Begriffe (streaming, cloud, app)."
                 "\nAntworte ausschließlich mit einem JSON-Objekt in genau dieser Form, ohne weiteren Text:\n" + shape)
    return "\n\n".join(parts)


def json_schema(explanations: bool) -> Dict[str, Any]:
    props: Dict[str, Any] = {
        "dokumenttyp": {"type": "string"},
        "absender": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
        "konfidenz": {"type": "number"},
    }
    required = ["dokumenttyp", "absender", "tags", "konfidenz"]
    if explanations:
        props["erklaerung"] = {"type": "string"}
        required.append("erklaerung")
    return {"type": "object", "properties": props, "required": required}


def parse_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON object from a model answer. Returns None when there is
    none; raises json.JSONDecodeError when it is there but broken."""
    if not raw:
        return None
    if "```" in raw:
        m = _RE_CODE_BLOCK.search(raw)
        if m:
            raw = m.group(1).strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    text = _RE_TRAILING_COMMA_ARR.sub("]", _RE_TRAILING_COMMA_OBJ.sub("}", raw[start:end]))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # bare control characters inside strings (newlines in explanations)
        clean, in_str, esc = [], False, False
        for ch in text:
            if esc:
                clean.append(ch); esc = False; continue
            if ch == "\\":
                esc = True; clean.append(ch); continue
            if ch == '"':
                in_str = not in_str
            clean.append(" " if in_str and ord(ch) < 32 else ch)
        parsed = json.loads("".join(clean))
    if not isinstance(parsed, dict):
        return None
    for de, en in FIELD_MAP.items():
        if de in parsed:
            parsed[en] = parsed.pop(de)
    return parsed


def coerce_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Type-safe view of the model output: strings, a list of tag strings,
    confidence as float in [0, 1], placeholder correspondents as None."""
    def text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value).strip().strip('"').strip()

    out = dict(parsed)
    out["document_type"] = text(out.get("document_type")).lower()
    corr = text(out.get("correspondent"))
    out["correspondent"] = None if corr.lower() in NO_NAME else corr

    tags = out.get("tags")
    if isinstance(tags, str):
        tags = re.split(r"[,;\n]+", tags)
    elif isinstance(tags, dict):
        tags = list(tags.values())
    elif not isinstance(tags, (list, tuple)):
        tags = []
    out["tags"] = [t for t in (text(x).lstrip("#") for x in tags) if t and t.lower() not in NO_NAME]

    conf = out.get("confidence")
    try:
        conf = float(str(conf).strip().rstrip("%")) if conf is not None else 0.0
    except ValueError:
        conf = 0.0
    if conf > 1.0:
        conf /= 100.0
    out["confidence"] = max(0.0, min(conf, 1.0))
    out["explanation"] = text(out.get("explanation"))
    out["summary"] = text(out.get("summary"))
    return out
