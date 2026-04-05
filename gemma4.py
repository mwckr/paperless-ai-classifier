#!/usr/bin/env python3
"""
Vision Model Integration for Paperless AI Classifier
=====================================================
Optimized for Google's Gemma 4 with thinking mode support.
Supports freeform classification with post-processing normalization.

Config is passed in from classifier_api.py - no direct .env reading.
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
from typing import Optional, Tuple, Dict, List
from pathlib import Path
from PIL import Image
import io

from learning import build_few_shot_prompt, normalize_result

logger = logging.getLogger(__name__)

# Module-level config cache (set by init_config)
_config = {}

# Paperless data cache — reused while queue is non-empty, with TTL
_paperless_cache = {
    "tags": None,
    "types": None,
    "correspondents": None,
    "fetched_at": 0.0,
}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Pre-compiled regexes for response parsing
_RE_CODE_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_RE_TRAILING_COMMA_OBJ = re.compile(r",\s*}")
_RE_TRAILING_COMMA_ARR = re.compile(r",\s*]")


def init_config(config: Dict):
    """Initialize module with config from classifier_api"""
    global _config
    _config = config
    logger.debug(f"Vision config initialized: model={config.get('OLLAMA_MODEL')}")


def invalidate_cache():
    """Invalidate cached Paperless data (called when queue drains)"""
    _paperless_cache["tags"] = None
    _paperless_cache["types"] = None
    _paperless_cache["correspondents"] = None
    _paperless_cache["fetched_at"] = 0.0


def _cache_expired() -> bool:
    """Check if cache has exceeded TTL"""
    return (time.time() - _paperless_cache["fetched_at"]) > _CACHE_TTL_SECONDS


def get_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Token {_config.get('PAPERLESS_TOKEN', '')}",
        "Content-Type": "application/json"
    }


# =============================================================================
# PAPERLESS API
# =============================================================================

def get_existing_tags() -> List[str]:
    """Fetch all existing tags from Paperless (cached per batch with TTL)"""
    if _paperless_cache["tags"] is not None and not _cache_expired():
        return _paperless_cache["tags"]
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/tags/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": 500}
        )
        response.raise_for_status()
        tags = [tag['name'] for tag in response.json().get("results", [])]
        logger.info(f"Loaded {len(tags)} existing tags")
        _paperless_cache["tags"] = tags
        _paperless_cache["fetched_at"] = time.time()
        return tags
    except Exception as e:
        logger.warning(f"Could not fetch existing tags: {e}")
        return []


def get_existing_document_types() -> List[str]:
    """Fetch all existing document types from Paperless (cached per batch with TTL)"""
    if _paperless_cache["types"] is not None and not _cache_expired():
        return _paperless_cache["types"]
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/document_types/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": 500}
        )
        response.raise_for_status()
        types = [t['name'] for t in response.json().get("results", [])]
        logger.info(f"Loaded {len(types)} existing document types")
        _paperless_cache["types"] = types
        return types
    except Exception as e:
        logger.warning(f"Could not fetch document types: {e}")
        return []


def get_existing_correspondents() -> List[str]:
    """Fetch all existing correspondents from Paperless (cached per batch with TTL)"""
    if _paperless_cache["correspondents"] is not None and not _cache_expired():
        return _paperless_cache["correspondents"]
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/correspondents/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": 500}
        )
        response.raise_for_status()
        correspondents = [c['name'] for c in response.json().get("results", [])]
        logger.info(f"Loaded {len(correspondents)} existing correspondents")
        _paperless_cache["correspondents"] = correspondents
        return correspondents
    except Exception as e:
        logger.warning(f"Could not fetch correspondents: {e}")
        return []


def list_documents(limit: int = 20) -> List[int]:
    """List available document IDs"""
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/documents/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": limit}
        )
        response.raise_for_status()
        data = response.json()
        doc_ids = [doc['id'] for doc in data.get("results", [])]
        logger.info(f"Found {data['count']} documents, showing {len(doc_ids)}")
        return doc_ids
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        return []


def get_document_metadata(doc_id: int) -> Optional[Dict]:
    """Fetch document metadata from Paperless"""
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/documents/{doc_id}/",
            headers=get_headers(),
            timeout=20
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {doc_id}: {e}")
        return None


def fetch_document_image(doc_id: int) -> Optional[Tuple[bytes, str, float]]:
    """Fetch document image from Paperless with multi-strategy fallback"""
    headers = {"Authorization": f"Token {_config.get('PAPERLESS_TOKEN')}"}
    max_pages = _config.get('MAX_PAGES', 3)
    
    # Try preview first
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/documents/{doc_id}/preview/",
            headers=headers,
            timeout=30
        )
        if response.status_code == 200:
            content_type = response.headers.get('content-type', '')
            if content_type.startswith('image/'):
                size_kb = len(response.content) / 1024
                logger.info(f"Got preview image: {size_kb:.1f} KB")
                return response.content, content_type, size_kb
    except Exception as e:
        logger.debug(f"Preview fetch failed: {e}")
    
    # Try PDF conversion
    pdf_path = None
    base_path = None
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/documents/{doc_id}/download/",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        pdf_bytes = response.content
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
            pdf_file.write(pdf_bytes)
            pdf_path = pdf_file.name
        
        base_path = tempfile.mktemp()
        result = subprocess.run(
            ['pdftoppm', '-jpeg', '-r', '72', '-scale-to', '1024',
             '-l', str(max_pages), pdf_path, base_path],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0:
            page_files = sorted(glob.glob(f"{base_path}-*.jpg"))
            if page_files:
                if len(page_files) == 1:
                    img = Image.open(page_files[0])
                    max_size = 1024
                    if max(img.size) > max_size:
                        ratio = max_size / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    buffer = io.BytesIO()
                    img.convert('RGB').save(buffer, 'JPEG', quality=85)
                    img_bytes = buffer.getvalue()
                    img.close()
                else:
                    images = [Image.open(f) for f in page_files]
                    total_height = sum(im.height for im in images)
                    max_width = max(im.width for im in images)
                    combined = Image.new('RGB', (max_width, total_height))
                    y_offset = 0
                    for im in images:
                        combined.paste(im, (0, y_offset))
                        y_offset += im.height
                    max_size = 1024
                    if max(combined.size) > max_size:
                        ratio = max_size / max(combined.size)
                        new_size = (int(combined.size[0] * ratio), int(combined.size[1] * ratio))
                        combined = combined.resize(new_size, Image.LANCZOS)
                    buffer = io.BytesIO()
                    combined.save(buffer, 'JPEG', quality=85)
                    img_bytes = buffer.getvalue()
                    for im in images:
                        im.close()
                
                # Cleanup page files
                for f in page_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                
                size_kb = len(img_bytes) / 1024
                logger.info(f"Converted PDF to image: {size_kb:.1f} KB ({len(page_files)} pages)")
                return img_bytes, 'image/jpeg', size_kb
    except Exception as e:
        logger.warning(f"PDF conversion error: {e}")
    finally:
        # Always clean up the PDF temp file
        if pdf_path:
            try:
                os.remove(pdf_path)
            except OSError:
                pass
    
    # Fallback to thumbnail
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/documents/{doc_id}/thumb/",
            headers=headers,
            timeout=25
        )
        response.raise_for_status()
        size_kb = len(response.content) / 1024
        logger.info(f"Using thumbnail: {size_kb:.1f} KB")
        return response.content, 'image/webp', size_kb
    except Exception as e:
        logger.error(f"Failed to fetch image: {e}")
        return None


# =============================================================================
# VISION ANALYSIS
# =============================================================================

def _build_prompt() -> str:
    """Build the classification prompt. No tag injection — tags are normalized post-hoc."""
    explanation_request = ""
    if _config.get('GENERATE_EXPLANATIONS', False):
        explanation_request = '\n4. erklärung - Kurze Begründung deiner Einordnung'

    # Optionally inject existing document types (small list, low hallucination risk)
    types_hint = ""
    if _config.get('INJECT_EXISTING_TYPES', True):
        existing_types = get_existing_document_types()
        if existing_types:
            types_list = ", ".join(existing_types[:50])
            types_hint = f"\nExistierende Dokumenttypen in Paperless (bei Übereinstimmung bevorzugt verwenden): {types_list}\n"

    # Few-shot examples from learning layer
    few_shot = ""
    if _config.get('FEW_SHOT_ENABLED', False):
        few_shot = build_few_shot_prompt(limit=3)
        if few_shot:
            few_shot = "\n" + few_shot + "\n"

    if _config.get('GENERATE_EXPLANATIONS', False):
        json_format = '{{"dokumenttyp": "...", "absender": "...", "tags": ["...", "...", "..."], "zusammenfassung": "Ein Satz", "erklärung": "..."}}'
    else:
        json_format = '{{"dokumenttyp": "...", "absender": "...", "tags": ["...", "...", "..."], "zusammenfassung": "Ein Satz"}}'

    prompt = f"""Analysiere dieses Dokument für die Archivierung in Paperless-ngx.

Bestimme:
1. dokumenttyp - Was für ein Dokument ist das? (kleingeschrieben)
2. absender - Wer hat dieses Dokument erstellt/versendet?
3. tags - 3 bis 5 präzise Suchbegriffe (jeweils mindestens 3 Zeichen), mit denen man genau dieses Dokument in einem Archiv wiederfinden würde. Qualität vor Quantität — lieber 3 gute als 5 mittelmäßige.

Regeln für tags:
- NIEMALS den Dokumenttyp, Absender oder deren Varianten als Tag verwenden
- KEINE generischen Begriffe wie "Rechnung", "Dokument", "Zahlung", "MwSt", "Betrag"
- KONKRET und SPEZIFISCH: Was unterscheidet dieses Dokument von anderen des gleichen Typs?
- Gute Tags: Produktnamen, Dienstleistungen, Zeiträume (z.B. "März 2026"), Vertragsnummern
- Deutsch bevorzugen, englische Begriffe nur wenn im Deutschen üblich (z.B. "Streaming", "Cloud"){explanation_request}
{types_hint}{few_shot}
Antworte nur mit JSON:
{json_format}"""

    return prompt


def _parse_response(raw: str) -> Optional[Dict]:
    """Parse JSON from model response with fallback handling."""
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
    # Clean up common JSON issues
    json_str = _RE_TRAILING_COMMA_OBJ.sub("}", json_str)
    json_str = _RE_TRAILING_COMMA_ARR.sub("]", json_str)

    parsed = json.loads(json_str)

    # Normalize German field names to English
    field_map = {
        'dokumenttyp': 'document_type',
        'absender': 'correspondent',
        'konfidenz': 'confidence',
        'zusammenfassung': 'summary',
        'erklärung': 'explanation',
    }
    for de_key, en_key in field_map.items():
        if de_key in parsed:
            parsed[en_key] = parsed.pop(de_key)

    return parsed


def analyze_with_vision(image_bytes: bytes, content_type: str) -> Tuple[bool, Optional[Dict], float, int]:
    """
    Analyze document with vision model.
    Returns: (success, result_dict, elapsed_seconds, estimated_tokens)
    """
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    prompt = _build_prompt()
    
    # Sampling parameters — configurable via .env, with Gemma4-tuned defaults
    temperature = _config.get('OLLAMA_TEMPERATURE', 0.7)
    top_p = _config.get('OLLAMA_TOP_P', 0.95)
    top_k = _config.get('OLLAMA_TOP_K', 64)
    
    payload = {
        "model": _config.get('OLLAMA_MODEL', 'gemma4:e4b'),
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_ctx": 8192,
            "num_thread": _config.get('OLLAMA_THREADS', 10)
        }
    }
    
    start = time.time()
    try:
        logger.debug(f"Vision request: image {len(image_b64)} chars, model {_config.get('OLLAMA_MODEL')}")
        response = requests.post(f"{_config.get('OLLAMA_URL')}/api/chat", json=payload, timeout=600)
        elapsed = time.time() - start
        
        if response.status_code != 200:
            logger.error(f"Vision API error {response.status_code}: {response.text[:200]}")
            return False, None, elapsed, 0
        
        result = response.json()
        message = result.get("message", {})
        
        # Gemma 4 separates thinking and content
        content = message.get("content", "").strip()
        thinking = message.get("thinking", "")
        
        estimated_tokens = len(prompt.split()) + len(content.split()) + len(thinking.split())
        
        # Use content primarily, fall back to thinking
        raw = content if content else thinking
        
        if not raw:
            logger.error("Empty response from vision model")
            return False, None, elapsed, estimated_tokens
        
        try:
            parsed = _parse_response(raw)
            if parsed:
                logger.info(f"Vision success: {parsed.get('document_type')} | {elapsed:.1f}s | ~{estimated_tokens} tokens")
                return True, parsed, elapsed, estimated_tokens
            else:
                logger.warning(f"No JSON found in response: {raw[:200]}")
                return False, None, elapsed, estimated_tokens
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            logger.warning(f"Raw: {raw[:300]}")
            return False, None, elapsed, estimated_tokens
            
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        logger.error(f"Vision timeout after {elapsed:.1f}s")
        return False, None, elapsed, 0
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Vision error: {e}")
        return False, None, elapsed, 0


# =============================================================================
# PAPERLESS UPDATE
# =============================================================================

def _get_or_create_resource(endpoint: str, name: str) -> Optional[int]:
    """Generic get-or-create for Paperless resources (tags, document_types, correspondents)"""
    try:
        response = requests.get(
            f"{_config.get('PAPERLESS_URL')}/api/{endpoint}/",
            headers=get_headers(),
            params={"name__iexact": name},
            timeout=10
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]["id"]
        
        response = requests.post(
            f"{_config.get('PAPERLESS_URL')}/api/{endpoint}/",
            headers=get_headers(),
            json={"name": name},
            timeout=10
        )
        if response.status_code == 201:
            return response.json()["id"]
    except Exception as e:
        logger.error(f"{endpoint} error for '{name}': {e}")
    return None


def get_or_create_tag(tag_name: str) -> Optional[int]:
    return _get_or_create_resource("tags", tag_name)


def get_or_create_document_type(type_name: str) -> Optional[int]:
    return _get_or_create_resource("document_types", type_name)


def get_or_create_correspondent(name: str) -> Optional[int]:
    if not name or name.lower() in ('keine', 'none', 'null', ''):
        return None
    return _get_or_create_resource("correspondents", name)


def update_document_in_paperless(doc_id: int, result: Dict) -> bool:
    """Update document in Paperless with classification results"""
    try:
        update_data = {}
        
        # Document type
        doc_type = result.get('document_type')
        if doc_type:
            type_id = get_or_create_document_type(doc_type)
            if type_id:
                update_data['document_type'] = type_id
        
        # Correspondent
        correspondent = result.get('correspondent')
        if correspondent:
            corr_id = get_or_create_correspondent(correspondent)
            if corr_id:
                update_data['correspondent'] = corr_id
        
        # Tags
        tags = result.get('tags', [])
        if tags:
            tag_ids = []
            for tag in tags[:5]:  # Limit to 5 tags
                tag_id = get_or_create_tag(tag)
                if tag_id:
                    tag_ids.append(tag_id)
            if tag_ids:
                update_data['tags'] = tag_ids
        
        if not update_data:
            logger.warning(f"No valid data to update for {doc_id}")
            return False
        
        response = requests.patch(
            f"{_config.get('PAPERLESS_URL')}/api/documents/{doc_id}/",
            headers=get_headers(),
            json=update_data,
            timeout=15
        )
        response.raise_for_status()
        logger.info(f"Updated document {doc_id}: type={doc_type}, correspondent={correspondent}, tags={len(tags)}")
        return True
    except Exception as e:
        logger.error(f"Failed to update document {doc_id}: {e}")
        return False


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_document(doc_id: int, apply_learning: bool = True) -> Dict:
    """Process a single document through Gemma 4 vision analysis"""
    logger.info(f"Processing document {doc_id} with Gemma 4")
    
    metadata = get_document_metadata(doc_id)
    if not metadata:
        return {"doc_id": doc_id, "success": False, "reason": "metadata_fetch_failed"}
    
    image_data = fetch_document_image(doc_id)
    if not image_data:
        return {"doc_id": doc_id, "success": False, "reason": "image_fetch_failed"}
    
    image_bytes, content_type, size_kb = image_data
    success, vision_result, elapsed, tokens = analyze_with_vision(image_bytes, content_type)
    
    if not success or not vision_result:
        return {
            "doc_id": doc_id,
            "title": metadata.get("title", ""),
            "success": False,
            "reason": "vision_analysis_failed",
            "duration_sec": elapsed,
            "image_size_kb": size_kb
        }
    
    # Apply learning normalization: fuzzy-match tags/types/correspondents
    # against existing Paperless data + learned term mappings
    if apply_learning:
        existing_tags = get_existing_tags()
        existing_types = get_existing_document_types()
        existing_correspondents = get_existing_correspondents()
        
        threshold = _config.get('FUZZY_MATCH_THRESHOLD', 0.80)
        vision_result = normalize_result(
            vision_result,
            existing_tags=existing_tags,
            existing_types=existing_types,
            existing_correspondents=existing_correspondents,
            threshold=threshold
        )
    
    result = {
        "doc_id": doc_id,
        "title": metadata.get("title", ""),
        "success": True,
        "duration_sec": elapsed,
        "tokens_est": tokens,
        "image_size_kb": size_kb,
        "document_type": vision_result.get('document_type'),
        "correspondent": vision_result.get('correspondent'),
        "tags": vision_result.get('tags', []),
        "confidence": vision_result.get('confidence', 0.0),
        "summary": vision_result.get('summary', ''),
        "explanation": vision_result.get('explanation', ''),
        "raw": vision_result
    }
    
    return result


# For backward compatibility with classifier_api.py
def analyze_document(doc_id: int) -> Dict:
    """Alias for process_document"""
    return process_document(doc_id)