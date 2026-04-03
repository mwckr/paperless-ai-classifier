#!/usr/bin/env python3
"""
Gemma 4 Vision Integration for Paperless AI Classifier
=======================================================
Optimized for Google's Gemma 4 model with thinking mode support.

Key differences from Ministral:
- Handles thinking/content response split
- Uses Gemma 4 recommended sampling parameters
- German output for Paperless fields
- No token limits (let model finish naturally)
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

# Import learning module
try:
    from learning import build_few_shot_prompt, normalize_result
except ImportError:
    def build_few_shot_prompt(limit=3): return ""
    def normalize_result(result, **kwargs): return result

logger = logging.getLogger(__name__)

# Configuration from environment
PAPERLESS_URL = os.getenv("PAPERLESS_URL", "http://localhost:8000")
PAPERLESS_TOKEN = os.getenv("PAPERLESS_TOKEN", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
NUM_THREADS = int(os.getenv("OLLAMA_THREADS", "10"))


def get_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Token {PAPERLESS_TOKEN}",
        "Content-Type": "application/json"
    }


# =============================================================================
# PAPERLESS API
# =============================================================================

def get_existing_tags() -> List[str]:
    """Fetch all existing tags from Paperless"""
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/tags/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": 500}
        )
        response.raise_for_status()
        tags = [tag['name'] for tag in response.json().get("results", [])]
        logger.info(f"Loaded {len(tags)} existing tags")
        return tags
    except Exception as e:
        logger.warning(f"Could not fetch existing tags: {e}")
        return []


def get_existing_document_types() -> List[str]:
    """Fetch all existing document types from Paperless"""
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/document_types/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": 500}
        )
        response.raise_for_status()
        types = [t['name'] for t in response.json().get("results", [])]
        logger.info(f"Loaded {len(types)} existing document types")
        return types
    except Exception as e:
        logger.warning(f"Could not fetch document types: {e}")
        return []


def get_existing_correspondents() -> List[str]:
    """Fetch all existing correspondents from Paperless"""
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/correspondents/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": 500}
        )
        response.raise_for_status()
        correspondents = [c['name'] for c in response.json().get("results", [])]
        logger.info(f"Loaded {len(correspondents)} existing correspondents")
        return correspondents
    except Exception as e:
        logger.warning(f"Could not fetch correspondents: {e}")
        return []


def list_documents(limit: int = 20) -> List[int]:
    """List available document IDs"""
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/",
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
            f"{PAPERLESS_URL}/api/documents/{doc_id}/",
            headers=get_headers(),
            timeout=20
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch metadata for {doc_id}: {e}")
        return None


def fetch_document_image(doc_id: int) -> Optional[Tuple[bytes, str, float]]:
    """Fetch document image from Paperless"""
    headers = {"Authorization": f"Token {PAPERLESS_TOKEN}"}
    
    # Try preview first
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/preview/",
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
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
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
            ['pdftoppm', '-jpeg', '-r', '100', '-scale-to', '1024', '-l', '3', pdf_path, base_path],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode == 0:
            page_files = sorted(glob.glob(f"{base_path}-*.jpg"))
            if page_files:
                if len(page_files) == 1:
                    with open(page_files[0], 'rb') as f:
                        img_bytes = f.read()
                else:
                    # Combine pages vertically
                    images = [Image.open(f) for f in page_files]
                    total_height = sum(img.height for img in images)
                    max_width = max(img.width for img in images)
                    combined = Image.new('RGB', (max_width, total_height))
                    y_offset = 0
                    for img in images:
                        combined.paste(img, (0, y_offset))
                        y_offset += img.height
                    buffer = io.BytesIO()
                    combined.save(buffer, 'JPEG', quality=85)
                    img_bytes = buffer.getvalue()
                    for img in images:
                        img.close()
                
                # Cleanup
                for f in page_files:
                    try:
                        os.remove(f)
                    except:
                        pass
                try:
                    os.remove(pdf_path)
                except:
                    pass
                
                size_kb = len(img_bytes) / 1024
                logger.info(f"Converted PDF to image: {size_kb:.1f} KB ({len(page_files)} pages)")
                return img_bytes, 'image/jpeg', size_kb
    except Exception as e:
        logger.warning(f"PDF conversion error: {e}")
    
    # Fallback to thumbnail
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/thumb/",
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
# GEMMA 4 VISION ANALYSIS
# =============================================================================

def analyze_with_vision(image_bytes: bytes, content_type: str) -> Tuple[bool, Optional[Dict], float, int]:
    """
    Analyze document with Gemma 4 vision model.
    Returns: (success, result_dict, elapsed_seconds, estimated_tokens)
    """
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Get few-shot examples from learning system
    few_shot = build_few_shot_prompt(limit=3)
    
    # Get existing Paperless data for context
    existing_tags = get_existing_tags()
    existing_tags_str = ", ".join(existing_tags[:50]) if existing_tags else "keine vorhanden"
    
    # Build prompt - German output for Paperless fields
    prompt = f"""Analysiere dieses Dokument sorgfältig.

{few_shot}
Bestimme:
1. Dokumenttyp (z.B. "Rechnung", "Vertrag", "Bescheid", "Kontoauszug", etc.)
2. Absender/Firma (Name wie er auf dem Dokument erscheint)
3. 3-5 relevante Tags zur Organisation
4. Kurze Zusammenfassung (1 Satz)

WICHTIG: 
- dokumenttyp und tags MÜSSEN auf Deutsch sein
- Wenn die Überschrift den Typ nennt (z.B. "Rechnung"), diesen DIREKT übernehmen
- Tags sollten allgemein + spezifisch sein

EXISTIERENDE TAGS (wenn passend, exakt wiederverwenden):
{existing_tags_str}

Antworte NUR mit gültigem JSON:
{{"dokumenttyp": "Typ auf Deutsch", "absender": "Firmenname", "tags": ["tag1", "tag2", "tag3"], "zusammenfassung": "Kurze Beschreibung", "konfidenz": 0.95}}"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [image_b64]
        }],
        "stream": False,
        "options": {
            # Gemma 4 recommended settings
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "num_thread": NUM_THREADS
            # No num_predict - let model finish naturally
        }
    }
    
    start = time.time()
    try:
        logger.debug(f"Gemma 4 request: image {len(image_b64)} chars")
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        elapsed = time.time() - start
        
        if response.status_code != 200:
            logger.error(f"Gemma 4 API error {response.status_code}: {response.text[:200]}")
            return False, None, elapsed, 0
        
        result = response.json()
        message = result.get("message", {})
        
        # Gemma 4 separates thinking and content
        content = message.get("content", "").strip()
        thinking = message.get("thinking", "")
        
        # Estimate tokens
        estimated_tokens = len(prompt.split()) + len(content.split()) + len(thinking.split())
        
        # Use content primarily, fall back to thinking
        raw = content if content else thinking
        
        if not raw:
            logger.error("Empty response from Gemma 4")
            return False, None, elapsed, estimated_tokens
        
        # Parse JSON
        try:
            # Handle markdown code blocks
            if "```" in raw:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                if match:
                    raw = match.group(1).strip()
            
            # Find JSON object
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = raw[json_start:json_end]
                # Clean up common issues
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                
                parsed = json.loads(json_str)
                
                # Normalize field names for compatibility
                if 'dokumenttyp' in parsed:
                    parsed['document_type'] = parsed.pop('dokumenttyp')
                if 'absender' in parsed:
                    parsed['correspondent'] = parsed.pop('absender')
                if 'konfidenz' in parsed:
                    parsed['confidence'] = parsed.pop('konfidenz')
                if 'zusammenfassung' in parsed:
                    parsed['summary'] = parsed.pop('zusammenfassung')
                
                logger.info(f"Gemma 4 success: {parsed.get('document_type')} | {elapsed:.1f}s | ~{estimated_tokens} tokens")
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
        logger.error(f"Gemma 4 timeout after {elapsed:.1f}s")
        return False, None, elapsed, 0
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Gemma 4 error: {e}")
        return False, None, elapsed, 0


# =============================================================================
# PAPERLESS UPDATE
# =============================================================================

def get_or_create_tag(tag_name: str) -> Optional[int]:
    """Get existing tag ID or create new tag"""
    try:
        # Search for existing
        response = requests.get(
            f"{PAPERLESS_URL}/api/tags/",
            headers=get_headers(),
            params={"name__iexact": tag_name},
            timeout=10
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]["id"]
        
        # Create new
        response = requests.post(
            f"{PAPERLESS_URL}/api/tags/",
            headers=get_headers(),
            json={"name": tag_name},
            timeout=10
        )
        if response.status_code == 201:
            return response.json()["id"]
    except Exception as e:
        logger.error(f"Tag error for '{tag_name}': {e}")
    return None


def get_or_create_document_type(type_name: str) -> Optional[int]:
    """Get existing document type ID or create new"""
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/document_types/",
            headers=get_headers(),
            params={"name__iexact": type_name},
            timeout=10
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]["id"]
        
        response = requests.post(
            f"{PAPERLESS_URL}/api/document_types/",
            headers=get_headers(),
            json={"name": type_name},
            timeout=10
        )
        if response.status_code == 201:
            return response.json()["id"]
    except Exception as e:
        logger.error(f"Document type error for '{type_name}': {e}")
    return None


def get_or_create_correspondent(name: str) -> Optional[int]:
    """Get existing correspondent ID or create new"""
    if not name or name.lower() in ['keine', 'none', 'null', '']:
        return None
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/correspondents/",
            headers=get_headers(),
            params={"name__iexact": name},
            timeout=10
        )
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                return results[0]["id"]
        
        response = requests.post(
            f"{PAPERLESS_URL}/api/correspondents/",
            headers=get_headers(),
            json={"name": name},
            timeout=10
        )
        if response.status_code == 201:
            return response.json()["id"]
    except Exception as e:
        logger.error(f"Correspondent error for '{name}': {e}")
    return None


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
            f"{PAPERLESS_URL}/api/documents/{doc_id}/",
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
    
    # Apply learning normalization
    if apply_learning:
        existing_tags = get_existing_tags()
        existing_types = get_existing_document_types()
        existing_correspondents = get_existing_correspondents()
        
        vision_result = normalize_result(
            vision_result,
            existing_tags=existing_tags,
            existing_types=existing_types,
            existing_correspondents=existing_correspondents
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
        "raw": vision_result
    }
    
    return result


# For backward compatibility with classifier_api.py
def analyze_document(doc_id: int) -> Dict:
    """Alias for process_document"""
    return process_document(doc_id)
