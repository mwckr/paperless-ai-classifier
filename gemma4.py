#!/usr/bin/env python3
"""
Vision Model Integration for Paperless AI Classifier
=====================================================
Supports any Ollama model (Gemma 3/4, Ministral, Qwen-VL, LLaVA, or a plain
text model when OCR_TEXT_MODE=only). Freeform classification with
post-processing normalization.

Config is passed in from classifier_api_v2.py - no direct .env reading.

Design notes for low-powered / low-RAM hosts
--------------------------------------------
* Paperless already ran OCR. The OCR text is injected into the prompt
  (OCR_TEXT_MODE=hint) or used instead of images (OCR_TEXT_MODE=only). Text
  tokens are far cheaper than vision encoder passes and the model can read
  small print that a downscaled page image loses.
* Page images are rendered once by pdftoppm directly at the target size and
  quality. No second decode/encode pass through Pillow.
* Thinking is disabled (OLLAMA_THINK=false), generation is capped
  (OLLAMA_NUM_PREDICT), the context window is explicit (OLLAMA_NUM_CTX) and the
  reply is constrained to a JSON schema (OLLAMA_STRUCTURED_OUTPUT). Together
  these remove most of the wasted CPU time per document.
* keep_alive is sent with every request and the model can be unloaded when the
  queue drains (OLLAMA_UNLOAD_WHEN_IDLE) so RAM is free between batches.
"""
import os
import requests
import json
import base64
import time
import logging
import re
import tempfile
import subprocess
import glob
from typing import Any, Optional, Tuple, Dict, List
from PIL import Image
import io

from learning import build_few_shot_prompt, normalize_result

logger = logging.getLogger(__name__)

# Module-level config cache (set by init_config)
_config: Dict[str, Any] = {}

# Paperless data cache — reused while queue is non-empty, with TTL.
# Each entry is a list of {"id", "name", "document_count"} dicts sorted by
# document_count (most used first).
_paperless_cache: Dict[str, Any] = {
    "tags": None,
    "document_types": None,
    "correspondents": None,
    "fetched_at": {},
}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Remembered capabilities of the Ollama server / model. None = unknown.
# Set to False after the server rejected the parameter once, so later
# requests skip the failing option without another round trip.
_server_caps: Dict[str, Optional[bool]] = {"think": None, "format": None}

# Pre-compiled regexes for response parsing
_RE_CODE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_RE_TRAILING_COMMA_OBJ = re.compile(r",\s*}")
_RE_TRAILING_COMMA_ARR = re.compile(r",\s*]")
_RE_WHITESPACE = re.compile(r"[ \t\f\v]+")
_RE_BLANK_LINES = re.compile(r"\n\s*\n+")

# Placeholder answers that mean "no correspondent"
_NO_CORRESPONDENT = {"", "keine", "keiner", "kein", "none", "null", "unbekannt", "unknown", "n/a", "-"}


def init_config(config: Dict):
    """Initialize module with config from classifier_api"""
    global _config
    _config = config
    logger.debug(f"Vision config initialized: model={config.get('OLLAMA_MODEL')}")


def _cfg_bool(key: str, default: bool) -> bool:
    val = _config.get(key, default)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_config.get(key, default))
    except (TypeError, ValueError):
        return default


def invalidate_cache():
    """Invalidate cached Paperless data (called when queue drains)"""
    _paperless_cache["tags"] = None
    _paperless_cache["document_types"] = None
    _paperless_cache["correspondents"] = None
    _paperless_cache["fetched_at"] = {}


def _cache_valid(kind: str) -> bool:
    if _paperless_cache.get(kind) is None:
        return False
    fetched = _paperless_cache["fetched_at"].get(kind, 0.0)
    return (time.time() - fetched) <= _CACHE_TTL_SECONDS


def get_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Token {_config.get('PAPERLESS_TOKEN', '')}",
        "Content-Type": "application/json"
    }


def _paperless_url(path: str) -> str:
    return f"{str(_config.get('PAPERLESS_URL', '')).rstrip('/')}{path}"


# =============================================================================
# PAPERLESS API
# =============================================================================

def _fetch_all_pages(endpoint: str, page_size: int = 500, max_pages: int = 20) -> List[Dict]:
    """Fetch every page of a paginated Paperless list endpoint."""
    results: List[Dict] = []
    url = _paperless_url(f"/api/{endpoint}/")
    params: Optional[Dict] = {"page_size": page_size}
    for _ in range(max_pages):
        response = requests.get(url, headers=get_headers(), params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        results.extend(data.get("results", []))
        url = data.get("next")
        params = None  # "next" already carries the query string
        if not url:
            break
    return results


def _get_resources(kind: str) -> List[Dict]:
    """Cached list of Paperless resources (tags/document_types/correspondents),
    most used first."""
    if _cache_valid(kind):
        return _paperless_cache[kind]
    try:
        raw = _fetch_all_pages(kind)
        items = [
            {
                "id": r.get("id"),
                "name": r.get("name", ""),
                "document_count": r.get("document_count") or 0,
            }
            for r in raw if r.get("name")
        ]
        items.sort(key=lambda r: (-r["document_count"], r["name"].lower()))
        _paperless_cache[kind] = items
        _paperless_cache["fetched_at"][kind] = time.time()
        logger.info(f"Loaded {len(items)} existing {kind.replace('_', ' ')}")
        return items
    except Exception as e:
        logger.warning(f"Could not fetch {kind}: {e}")
        return []


def get_existing_tags() -> List[str]:
    """All existing tag names, most used first (cached with TTL)."""
    return [r["name"] for r in _get_resources("tags")]


def get_existing_document_types() -> List[str]:
    """All existing document type names, most used first (cached with TTL)."""
    return [r["name"] for r in _get_resources("document_types")]


def get_existing_correspondents() -> List[str]:
    """All existing correspondent names, most used first (cached with TTL)."""
    return [r["name"] for r in _get_resources("correspondents")]


def list_documents(limit: int = 20) -> List[int]:
    """List available document IDs"""
    try:
        response = requests.get(
            _paperless_url("/api/documents/"),
            headers=get_headers(),
            timeout=20,
            params={"page_size": limit}
        )
        response.raise_for_status()
        data = response.json()
        doc_ids = [doc['id'] for doc in data.get("results", [])]
        logger.info(f"Found {data.get('count', len(doc_ids))} documents, showing {len(doc_ids)}")
        return doc_ids
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []


def get_document_metadata(doc_id: int) -> Optional[Dict]:
    """Fetch document metadata from Paperless (includes the OCR text in 'content')"""
    try:
        response = requests.get(
            _paperless_url(f"/api/documents/{doc_id}/"),
            headers=get_headers(),
            timeout=20
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {doc_id}: {e}")
        return None


# =============================================================================
# IMAGES
# =============================================================================

def _encode_pil_image(img: Image.Image, max_size: int, quality: int, grayscale: bool) -> bytes:
    """Downscale a PIL image in place and return JPEG bytes."""
    mode = "L" if grayscale else "RGB"
    try:
        # JPEG decoder can skip work when we only need a smaller image
        img.draft(mode, (max_size, max_size))
    except Exception:
        pass
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    if img.mode != mode:
        img = img.convert(mode)
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _render_pdf_pages(pdf_bytes: bytes, max_pages: int, max_size: int,
                      quality: int, grayscale: bool) -> List[bytes]:
    """Render the first pages of a PDF straight to JPEG with pdftoppm.
    pdftoppm scales to the target size itself, so no second encode is needed."""
    with tempfile.TemporaryDirectory(prefix="pcls-") as tmpdir:
        pdf_path = os.path.join(tmpdir, "doc.pdf")
        with open(pdf_path, "wb") as fh:
            fh.write(pdf_bytes)
        base_path = os.path.join(tmpdir, "page")
        # Note: poppler ignores -gray for JPEG output, so grayscale is applied
        # below with Pillow (opt-in, costs one extra encode per page).
        cmd = [
            "pdftoppm", "-jpeg",
            "-jpegopt", f"quality={quality},optimize=y",
            "-scale-to", str(max_size),
            "-f", "1", "-l", str(max(1, max_pages)),
            pdf_path, base_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=_cfg_int("PDF_RENDER_TIMEOUT", 120))
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", "replace").strip()[:300]
            raise RuntimeError(f"pdftoppm exit {result.returncode}: {err}")
        page_files = sorted(glob.glob(f"{base_path}-*.jpg"))
        if not page_files:
            raise RuntimeError("pdftoppm produced no pages")
        pages = []
        for path in page_files[:max_pages]:
            if grayscale:
                with Image.open(path) as img:
                    buffer = io.BytesIO()
                    img.convert("L").save(buffer, "JPEG", quality=quality, optimize=True)
                    pages.append(buffer.getvalue())
            else:
                with open(path, "rb") as fh:
                    pages.append(fh.read())
        return pages


def fetch_document_images(doc_id: int) -> Optional[List[Tuple[bytes, float]]]:
    """Fetch document as list of page images from Paperless.
    Returns list of (jpeg_bytes, size_kb) tuples — one per page.
    Resolution and quality controlled by IMAGE_MAX_SIZE, IMAGE_QUALITY and IMAGE_GRAYSCALE."""
    headers = {"Authorization": f"Token {_config.get('PAPERLESS_TOKEN')}"}
    max_pages = max(1, _cfg_int("MAX_PAGES", 3))
    max_size = max(256, _cfg_int("IMAGE_MAX_SIZE", 1024))
    quality = min(100, max(30, _cfg_int("IMAGE_QUALITY", 85)))
    grayscale = _cfg_bool("IMAGE_GRAYSCALE", False)

    # Strategy 1: download the (archived, OCR-processed) document and render pages
    try:
        response = requests.get(
            _paperless_url(f"/api/documents/{doc_id}/download/"),
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")

        if content_type.startswith("image/"):
            img = Image.open(io.BytesIO(response.content))
            img_bytes = _encode_pil_image(img, max_size, quality, grayscale)
            img.close()
            size_kb = len(img_bytes) / 1024
            logger.info(f"Document is image: {size_kb:.1f} KB")
            return [(img_bytes, size_kb)]

        pages = _render_pdf_pages(response.content, max_pages, max_size, quality, grayscale)
        page_images = [(p, len(p) / 1024) for p in pages]
        total_kb = sum(s for _, s in page_images)
        logger.info(f"PDF → {len(page_images)} page image(s), {max_size}px, total {total_kb:.1f} KB")
        return page_images
    except Exception as e:
        logger.warning(f"PDF conversion error for {doc_id}: {e}")

    # Fallback: preview image, then thumbnail. Both are re-encoded to the target size.
    for path, label in ((f"/api/documents/{doc_id}/preview/", "preview"),
                        (f"/api/documents/{doc_id}/thumb/", "thumbnail")):
        try:
            response = requests.get(_paperless_url(path), headers=headers, timeout=30)
            if response.status_code != 200:
                continue
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                continue
            img = Image.open(io.BytesIO(response.content))
            img_bytes = _encode_pil_image(img, max_size, quality, grayscale)
            img.close()
            size_kb = len(img_bytes) / 1024
            logger.info(f"Using {label} image: {size_kb:.1f} KB")
            return [(img_bytes, size_kb)]
        except Exception as e:
            logger.debug(f"{label} fetch failed: {e}")

    logger.error(f"Failed to fetch any image for document {doc_id}")
    return None


# =============================================================================
# OCR TEXT
# =============================================================================

def extract_ocr_excerpt(metadata: Optional[Dict]) -> str:
    """Return a whitespace-collapsed excerpt of Paperless' OCR text.
    The start of a document (letterhead, subject line) carries most of the
    signal, so the excerpt is simply the first OCR_TEXT_CHARS characters."""
    if not metadata:
        return ""
    content = metadata.get("content") or ""
    if not isinstance(content, str):
        return ""
    text = content.replace("\r", "\n")
    text = _RE_WHITESPACE.sub(" ", text)
    text = _RE_BLANK_LINES.sub("\n", text).strip()
    limit = max(200, _cfg_int("OCR_TEXT_CHARS", 2000))
    if len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        text = text[: cut if cut > limit * 0.6 else limit].rstrip() + " …"
    return text


# =============================================================================
# VISION ANALYSIS
# =============================================================================

def _format_hint_list(names: List[str], limit: int) -> str:
    names = [n.strip() for n in names if n and n.strip()]
    if limit > 0:
        names = names[:limit]
    return ", ".join(names)


def _build_prompt(ocr_text: str = "", num_images: int = 0) -> str:
    """Build the classification prompt.

    Layout (small models follow this best): context first (page count, OCR
    excerpt, reference lists from Paperless), then short numbered rules, then
    the exact output shape as the very last thing the model reads."""
    want_explanation = _cfg_bool("GENERATE_EXPLANATIONS", False)

    # --- Context -----------------------------------------------------------
    context_lines = []
    if num_images > 0:
        if num_images == 1:
            context_lines.append("Das Bild zeigt die erste Seite eines Dokuments.")
        else:
            context_lines.append(
                f"Die {num_images} Bilder sind die ersten {num_images} Seiten EINES Dokuments, in Reihenfolge."
            )
    if ocr_text:
        context_lines.append(
            "OCR-Text des Dokuments (Anfang, kann Erkennungsfehler enthalten):\n"
            f"<<<\n{ocr_text}\n>>>"
        )

    # --- Reference lists from Paperless -------------------------------------
    ref_lines = []
    if _cfg_bool("INJECT_EXISTING_TYPES", True):
        types_str = _format_hint_list(get_existing_document_types(), _cfg_int("PROMPT_TYPE_LIMIT", 50))
        if types_str:
            ref_lines.append(f"- Dokumenttypen: {types_str}")
    if _cfg_bool("INJECT_EXISTING_CORRESPONDENTS", True):
        corr_str = _format_hint_list(get_existing_correspondents(), _cfg_int("PROMPT_CORRESPONDENT_LIMIT", 60))
        if corr_str:
            ref_lines.append(f"- Absender: {corr_str}")
    if _cfg_bool("INJECT_EXISTING_TAGS", True):
        tags_str = _format_hint_list(get_existing_tags(), _cfg_int("PROMPT_TAG_LIMIT", 80))
        if tags_str:
            ref_lines.append(f"- Tags: {tags_str}")
    if ref_lines:
        context_lines.append(
            "Bereits in Paperless vorhanden (wenn passend, exakt so wiederverwenden):\n" + "\n".join(ref_lines)
        )

    # Few-shot examples from learning layer
    if _cfg_bool("FEW_SHOT_ENABLED", False):
        few_shot = build_few_shot_prompt(limit=3)
        if few_shot:
            context_lines.append(few_shot.strip())

    context = ("\n\n".join(context_lines) + "\n\n") if context_lines else ""

    # --- Rules ---------------------------------------------------------------
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
    if want_explanation:
        rules.append("5. erklaerung: Ein Satz, woran du Dokumenttyp und Absender erkannt hast.")

    if want_explanation:
        json_shape = ('{"dokumenttyp": "...", "absender": "...", "tags": ["...", "...", "..."], '
                      '"konfidenz": 0.8, "erklaerung": "..."}')
    else:
        json_shape = '{"dokumenttyp": "...", "absender": "...", "tags": ["...", "...", "..."], "konfidenz": 0.8}'

    prompt = (
        f"{context}"
        "Du klassifizierst dieses Dokument für Paperless-ngx. Bestimme:\n"
        + "\n".join(rules) +
        "\n\nSprache: Deutsch. Englisch nur für eingedeutschte Begriffe (streaming, cloud, app)."
        "\nAntworte ausschließlich mit einem JSON-Objekt in genau dieser Form, ohne weiteren Text:\n"
        f"{json_shape}"
    )
    return prompt


def _json_schema() -> Dict:
    """JSON schema used for Ollama structured output. Keeps the model from
    producing prose, markdown fences or invalid JSON."""
    props = {
        "dokumenttyp": {"type": "string"},
        "absender": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 6},
        "konfidenz": {"type": "number"},
    }
    required = ["dokumenttyp", "absender", "tags", "konfidenz"]
    if _cfg_bool("GENERATE_EXPLANATIONS", False):
        props["erklaerung"] = {"type": "string"}
        required.append("erklaerung")
    return {"type": "object", "properties": props, "required": required}


def _parse_response(raw: str) -> Optional[Dict]:
    """Parse JSON from model response with fallback handling.
    Returns a dict with English field names, or None if no JSON object was found.
    Raises json.JSONDecodeError if a JSON object was found but is unparseable."""
    if not raw:
        return None

    # Handle markdown code blocks
    if "```" in raw:
        match = _RE_CODE_BLOCK.search(raw)
        if match:
            raw = match.group(1).strip()

    # Find JSON object
    json_start = raw.find("{")
    json_end = raw.rfind("}") + 1
    if json_start < 0 or json_end <= json_start:
        return None

    json_str = raw[json_start:json_end]
    json_str = _RE_TRAILING_COMMA_OBJ.sub("}", json_str)
    json_str = _RE_TRAILING_COMMA_ARR.sub("]", json_str)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Models put raw newlines/tabs inside string values. Replace bare
        # control characters inside strings with spaces, keep valid escapes.
        clean = []
        in_string = False
        escape = False
        for ch in json_str:
            if escape:
                clean.append(ch)
                escape = False
                continue
            if ch == '\\':
                escape = True
                clean.append(ch)
                continue
            if ch == '"':
                in_string = not in_string
            if in_string and ord(ch) < 32:
                clean.append(' ')
                continue
            clean.append(ch)
        parsed = json.loads(''.join(clean))

    if not isinstance(parsed, dict):
        return None

    # Normalize German field names to English
    field_map = {
        'dokumenttyp': 'document_type',
        'absender': 'correspondent',
        'konfidenz': 'confidence',
        'zusammenfassung': 'summary',
        'erklärung': 'explanation',
        'erklaerung': 'explanation',
    }
    for de_key, en_key in field_map.items():
        if de_key in parsed:
            parsed[en_key] = parsed.pop(de_key)

    return parsed


def _coerce_result(parsed: Dict) -> Dict:
    """Make the parsed model output type-safe: strings where strings are
    expected, a list of strings for tags, a float in [0, 1] for confidence."""
    out = dict(parsed)

    def as_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return str(value).strip().strip('"').strip()

    out["document_type"] = as_text(out.get("document_type")).lower()

    corr = as_text(out.get("correspondent"))
    out["correspondent"] = None if corr.lower() in _NO_CORRESPONDENT else corr

    tags = out.get("tags")
    if isinstance(tags, str):
        tags = re.split(r"[,;\n]+", tags)
    elif isinstance(tags, dict):
        tags = list(tags.values())
    elif not isinstance(tags, (list, tuple)):
        tags = []
    clean_tags: List[str] = []
    for tag in tags:
        text = as_text(tag).lstrip("#")
        if text and text.lower() not in _NO_CORRESPONDENT:
            clean_tags.append(text)
    out["tags"] = clean_tags

    conf = out.get("confidence")
    try:
        if isinstance(conf, str):
            conf = conf.strip().rstrip("%")
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 0.0
    if conf > 1.0:
        conf = conf / 100.0  # model answered in percent
    out["confidence"] = max(0.0, min(conf, 1.0))

    out["explanation"] = as_text(out.get("explanation"))
    out["summary"] = as_text(out.get("summary"))
    return out


def _build_options() -> Dict:
    """Ollama runtime options. Sampling params are only sent when configured
    so the model's own defaults apply otherwise."""
    options: Dict[str, Any] = {}
    if 'OLLAMA_TEMPERATURE' in _config:
        options['temperature'] = _config['OLLAMA_TEMPERATURE']
    if 'OLLAMA_TOP_P' in _config:
        options['top_p'] = _config['OLLAMA_TOP_P']
    if 'OLLAMA_TOP_K' in _config:
        options['top_k'] = _config['OLLAMA_TOP_K']
    threads = _cfg_int("OLLAMA_THREADS", 0)
    if threads > 0:
        options['num_thread'] = threads
    num_ctx = _cfg_int("OLLAMA_NUM_CTX", 4096)
    if num_ctx > 0:
        options['num_ctx'] = num_ctx
    num_predict = _cfg_int("OLLAMA_NUM_PREDICT", 512)
    if num_predict > 0:
        options['num_predict'] = num_predict
    return options


def _post_chat(payload: Dict, timeout: int) -> requests.Response:
    return requests.post(f"{str(_config.get('OLLAMA_URL', '')).rstrip('/')}/api/chat",
                         json=payload, timeout=(10, timeout))


def analyze_with_vision(page_images: List[bytes], ocr_text: str = "") -> Tuple[bool, Optional[Dict], float, int]:
    """
    Analyze document with the configured Ollama model.
    page_images: list of JPEG page images (may be empty when OCR_TEXT_MODE=only).
    ocr_text: OCR excerpt to include in the prompt (may be empty).
    Returns: (success, result_dict, elapsed_seconds, tokens_used)
    tokens_used is prompt_eval_count + eval_count as reported by Ollama.
    """
    images_b64 = [base64.b64encode(img).decode('utf-8') for img in page_images]
    prompt = _build_prompt(ocr_text=ocr_text, num_images=len(images_b64))
    logger.debug(f"Prompt ({len(prompt)} chars):\n{prompt}")

    options = _build_options()
    message: Dict[str, Any] = {"role": "user", "content": prompt}
    if images_b64:
        message["images"] = images_b64

    payload: Dict[str, Any] = {
        "model": _config.get('OLLAMA_MODEL', 'gemma4:e4b'),
        "messages": [message],
        "stream": False,
        "keep_alive": _config.get('OLLAMA_KEEP_ALIVE', '5m'),
    }
    if options:
        payload["options"] = options
    # Thinking tokens are pure cost for a classification task on CPU.
    if _server_caps["think"] is not False:
        payload["think"] = _cfg_bool("OLLAMA_THINK", False)
    if _cfg_bool("OLLAMA_STRUCTURED_OUTPUT", True) and _server_caps["format"] is not False:
        payload["format"] = _json_schema()

    vision_timeout = _cfg_int("OLLAMA_TIMEOUT", 600)
    start = time.time()
    try:
        logger.debug(f"Vision request: {len(images_b64)} images, {len(ocr_text)} chars OCR, "
                     f"model {payload['model']}, options={options or 'model defaults'}, timeout={vision_timeout}s")
        response = _post_chat(payload, vision_timeout)

        # Older Ollama versions (or non-thinking models) reject some fields with
        # HTTP 400. Drop the field, remember that, and retry once.
        if response.status_code == 400:
            body = response.text.lower()
            retried = False
            if "think" in payload and ("think" in body or "does not support" in body):
                logger.warning(f"Ollama rejected 'think' ({response.text[:120]}); retrying without it")
                _server_caps["think"] = False
                payload.pop("think", None)
                retried = True
            if "format" in payload and ("format" in body or "schema" in body or "grammar" in body or not retried):
                logger.warning(f"Ollama rejected 'format' ({response.text[:120]}); retrying without structured output")
                _server_caps["format"] = False
                payload.pop("format", None)
                retried = True
            if retried:
                response = _post_chat(payload, vision_timeout)

        elapsed = time.time() - start
        if response.status_code != 200:
            logger.error(f"Vision API error {response.status_code}: {response.text[:200]}")
            return False, None, elapsed, 0

        result = response.json()
        message = result.get("message", {}) or {}
        content = (message.get("content") or "").strip()
        thinking = (message.get("thinking") or "").strip()

        prompt_tokens = int(result.get("prompt_eval_count") or 0)
        eval_tokens = int(result.get("eval_count") or 0)
        tokens_used = prompt_tokens + eval_tokens
        load_s = (result.get("load_duration") or 0) / 1e9
        prompt_s = (result.get("prompt_eval_duration") or 0) / 1e9
        eval_s = (result.get("eval_duration") or 0) / 1e9
        logger.info(
            f"Ollama timing: load {load_s:.1f}s | prompt {prompt_tokens} tok in {prompt_s:.1f}s "
            f"| output {eval_tokens} tok in {eval_s:.1f}s | total {elapsed:.1f}s"
        )

        num_ctx = options.get("num_ctx")
        if num_ctx and prompt_tokens >= 0.9 * num_ctx:
            logger.warning(
                f"Prompt used {prompt_tokens} of {num_ctx} context tokens. Ollama truncates silently when the "
                "prompt does not fit. Raise OLLAMA_NUM_CTX or lower MAX_PAGES / OCR_TEXT_CHARS / PROMPT_*_LIMIT."
            )
        num_predict = options.get("num_predict")
        if num_predict and eval_tokens >= num_predict and not content:
            logger.warning(f"Model hit OLLAMA_NUM_PREDICT={num_predict} before answering. "
                           "Disable thinking (OLLAMA_THINK=false) or raise OLLAMA_NUM_PREDICT.")

        # Use content primarily, fall back to thinking (for thinking-mode models)
        raw = content if content else thinking
        if not raw:
            logger.error("Empty response from vision model")
            return False, None, elapsed, tokens_used

        try:
            parsed = _parse_response(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.warning(f"Raw: {raw[:300]}")
            return False, None, elapsed, tokens_used
        if not parsed:
            logger.warning(f"No JSON found in response: {raw[:200]}")
            return False, None, elapsed, tokens_used

        parsed = _coerce_result(parsed)
        if not parsed["document_type"]:
            logger.warning(f"Model returned no document type: {raw[:200]}")
            return False, None, elapsed, tokens_used

        logger.info(f"Vision success: {parsed['document_type']} | {parsed.get('correspondent')} | "
                    f"{elapsed:.1f}s | {tokens_used} tokens")
        return True, parsed, elapsed, tokens_used

    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        logger.error(f"Vision timeout after {elapsed:.1f}s (OLLAMA_TIMEOUT={vision_timeout})")
        return False, None, elapsed, 0
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Vision error: {e}")
        return False, None, elapsed, 0


def unload_model() -> bool:
    """Ask Ollama to unload the model now (frees RAM between batches)."""
    model = _config.get('OLLAMA_MODEL')
    if not model:
        return False
    try:
        resp = requests.post(
            f"{str(_config.get('OLLAMA_URL', '')).rstrip('/')}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=30,
        )
        ok = resp.status_code == 200
        logger.info(f"Unloaded model {model}" if ok else f"Unload of {model} failed: {resp.status_code}")
        return ok
    except Exception as e:
        logger.warning(f"Could not unload model {model}: {e}")
        return False


# =============================================================================
# PAPERLESS UPDATE
# =============================================================================

def _get_or_create_resource(endpoint: str, name: str) -> Optional[int]:
    """Generic get-or-create for Paperless resources (tags, document_types, correspondents)"""
    name = (name or "").strip()
    if not name:
        return None
    try:
        # Try the cache first (exact, case-insensitive)
        for item in _paperless_cache.get(endpoint) or []:
            if item["name"].lower() == name.lower() and item.get("id"):
                return item["id"]

        response = requests.get(
            _paperless_url(f"/api/{endpoint}/"),
            headers=get_headers(),
            params={"name__iexact": name},
            timeout=10
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]["id"]

        response = requests.post(
            _paperless_url(f"/api/{endpoint}/"),
            headers=get_headers(),
            json={"name": name},
            timeout=10
        )
        if response.status_code == 201:
            new_id = response.json()["id"]
            logger.info(f"Created new {endpoint[:-1] if endpoint.endswith('s') else endpoint}: '{name}'")
            cached = _paperless_cache.get(endpoint)
            if cached is not None:
                cached.append({"id": new_id, "name": name, "document_count": 0})
            return new_id
        logger.warning(f"Could not create {endpoint} '{name}': {response.status_code} {response.text[:120]}")
    except Exception as e:
        logger.error(f"{endpoint} error for '{name}': {e}")
    return None


def get_or_create_tag(tag_name: str) -> Optional[int]:
    return _get_or_create_resource("tags", tag_name)


def get_or_create_document_type(type_name: str) -> Optional[int]:
    return _get_or_create_resource("document_types", type_name)


def get_or_create_correspondent(name: str) -> Optional[int]:
    if not name or name.strip().lower() in _NO_CORRESPONDENT:
        return None
    return _get_or_create_resource("correspondents", name)


def update_document_in_paperless(doc_id: int, result: Dict, metadata: Optional[Dict] = None,
                                 overwrite: Optional[bool] = None) -> bool:
    """Update document in Paperless with classification results.

    overwrite=False (default, OVERWRITE_EXISTING): only fill document type and
    correspondent when the document has none yet. Tags are merged with the
    document's existing tags unless TAGS_MODE=replace, so inbox tags and tags
    set by Paperless' own matching survive."""
    if overwrite is None:
        overwrite = _cfg_bool("OVERWRITE_EXISTING", False)
    merge_tags = str(_config.get("TAGS_MODE", "merge")).lower() != "replace"
    try:
        if metadata is None:
            metadata = get_document_metadata(doc_id) or {}

        update_data: Dict[str, Any] = {}
        skipped = []

        doc_type = result.get('document_type')
        if doc_type:
            if metadata.get('document_type') and not overwrite:
                skipped.append("document_type")
            else:
                type_id = get_or_create_document_type(doc_type)
                if type_id and type_id != metadata.get('document_type'):
                    update_data['document_type'] = type_id

        correspondent = result.get('correspondent')
        if correspondent:
            if metadata.get('correspondent') and not overwrite:
                skipped.append("correspondent")
            else:
                corr_id = get_or_create_correspondent(correspondent)
                if corr_id and corr_id != metadata.get('correspondent'):
                    update_data['correspondent'] = corr_id

        tags = [t for t in (result.get('tags') or []) if isinstance(t, str) and t.strip()]
        if tags:
            tag_ids: List[int] = []
            for tag in tags[:5]:  # Limit to 5 tags
                tag_id = get_or_create_tag(tag)
                if tag_id and tag_id not in tag_ids:
                    tag_ids.append(tag_id)
            existing_ids = list(metadata.get('tags') or []) if merge_tags else []
            merged = existing_ids + [t for t in tag_ids if t not in existing_ids]
            if merged and set(merged) != set(metadata.get('tags') or []):
                update_data['tags'] = merged

        if skipped:
            logger.info(f"Document {doc_id}: kept existing {', '.join(skipped)} (OVERWRITE_EXISTING=false)")
        if not update_data:
            logger.info(f"Document {doc_id}: nothing to update in Paperless")
            return False

        response = requests.patch(
            _paperless_url(f"/api/documents/{doc_id}/"),
            headers=get_headers(),
            json=update_data,
            timeout=15
        )
        response.raise_for_status()
        logger.info(f"Updated document {doc_id}: {', '.join(update_data.keys())} "
                    f"(type={doc_type}, correspondent={correspondent}, tags={len(tags)})")
        return True
    except Exception as e:
        logger.error(f"Failed to update document {doc_id}: {e}")
        return False


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_document(doc_id: int, apply_learning: bool = True, metadata: Optional[Dict] = None) -> Dict:
    """Process a single document through model analysis.
    metadata may be passed in by the caller to save a Paperless round trip."""
    logger.info(f"Processing document {doc_id} with {_config.get('OLLAMA_MODEL', 'vision model')}")

    if metadata is None:
        metadata = get_document_metadata(doc_id)
    if not metadata:
        return {"doc_id": doc_id, "success": False, "reason": "metadata_fetch_failed"}

    ocr_mode = str(_config.get("OCR_TEXT_MODE", "hint")).lower()
    ocr_text = extract_ocr_excerpt(metadata) if ocr_mode in ("hint", "only") else ""

    page_images: List[Tuple[bytes, float]] = []
    if ocr_mode == "only" and ocr_text:
        logger.info(f"OCR_TEXT_MODE=only: classifying from {len(ocr_text)} chars of OCR text, no images")
    else:
        if ocr_mode == "only":
            logger.info(f"Document {doc_id} has no OCR text, falling back to page images")
        page_images = fetch_document_images(doc_id) or []
        if not page_images and not ocr_text:
            return {"doc_id": doc_id, "title": metadata.get("title", ""),
                    "success": False, "reason": "image_fetch_failed"}
        if not page_images:
            logger.warning(f"Document {doc_id}: no images available, using OCR text only")

    image_bytes_list = [img for img, _ in page_images]
    total_size_kb = sum(s for _, s in page_images)
    success, vision_result, elapsed, tokens = analyze_with_vision(image_bytes_list, ocr_text=ocr_text)

    if not success or not vision_result:
        return {
            "doc_id": doc_id,
            "title": metadata.get("title", ""),
            "success": False,
            "reason": "vision_analysis_failed",
            "duration_sec": elapsed,
            "image_size_kb": total_size_kb
        }

    # Apply learning normalization: fuzzy-match tags/types/correspondents
    # against existing Paperless data + learned term mappings
    if apply_learning:
        threshold = float(_config.get('FUZZY_MATCH_THRESHOLD', 0.80))
        vision_result = normalize_result(
            vision_result,
            existing_tags=get_existing_tags(),
            existing_types=get_existing_document_types(),
            existing_correspondents=get_existing_correspondents(),
            threshold=threshold
        )

    return {
        "doc_id": doc_id,
        "title": metadata.get("title", ""),
        "success": True,
        "duration_sec": elapsed,
        "tokens_est": tokens,
        "image_size_kb": total_size_kb,
        "pages_sent": len(image_bytes_list),
        "ocr_chars": len(ocr_text),
        "document_type": vision_result.get('document_type'),
        "correspondent": vision_result.get('correspondent'),
        "tags": vision_result.get('tags', []),
        "confidence": vision_result.get('confidence', 0.0),
        "summary": vision_result.get('summary', ''),
        "explanation": vision_result.get('explanation', ''),
        "raw": vision_result
    }


# For backward compatibility with classifier_api.py
def analyze_document(doc_id: int) -> Dict:
    """Alias for process_document"""
    return process_document(doc_id)
