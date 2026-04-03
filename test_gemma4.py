#!/usr/bin/env python3
"""
Gemma 4 Test Script - Freeform Classification
==============================================
Tests Gemma 4's ability to classify documents without guidance.
Dry run only - no changes to Paperless.

Run from /opt/paperless-classifier/
"""
import requests
import json
import base64
import sys
import time
import os
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv()

# ============================================================================
# CONFIGURATION (from .env)
# ============================================================================

PAPERLESS_URL = os.getenv("PAPERLESS_URL", "").rstrip("/")
PAPERLESS_TOKEN = os.getenv("PAPERLESS_TOKEN", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")

# Override model for this test
OLLAMA_MODEL = "gemma4:e4b"

NUM_THREADS = int(os.getenv("OLLAMA_THREADS", "10"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "3"))
IMAGE_DPI = 150

print(f"Loaded config:")
print(f"  PAPERLESS_URL: {PAPERLESS_URL}")
print(f"  OLLAMA_URL:    {OLLAMA_URL}")
print(f"  MODEL:         {OLLAMA_MODEL}")
print()

# ============================================================================
# PAPERLESS API
# ============================================================================

def get_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Token {PAPERLESS_TOKEN}",
        "Content-Type": "application/json"
    }


def fetch_existing_tags() -> Dict[int, str]:
    """Fetch all existing tags from Paperless"""
    try:
        resp = requests.get(
            f"{PAPERLESS_URL}/api/tags/",
            headers=get_headers(),
            params={"page_size": 500},
            timeout=20
        )
        resp.raise_for_status()
        tags = {t["id"]: t["name"] for t in resp.json().get("results", [])}
        print(f"Loaded {len(tags)} existing tags from Paperless")
        return tags
    except Exception as e:
        print(f"Warning: Could not fetch tags: {e}")
        return {}


def fetch_existing_doc_types() -> Dict[int, str]:
    """Fetch all existing document types from Paperless"""
    try:
        resp = requests.get(
            f"{PAPERLESS_URL}/api/document_types/",
            headers=get_headers(),
            params={"page_size": 500},
            timeout=20
        )
        resp.raise_for_status()
        types = {t["id"]: t["name"] for t in resp.json().get("results", [])}
        print(f"Loaded {len(types)} existing document types from Paperless")
        return types
    except Exception as e:
        print(f"Warning: Could not fetch document types: {e}")
        return {}


def fetch_existing_correspondents() -> Dict[int, str]:
    """Fetch all existing correspondents from Paperless"""
    try:
        resp = requests.get(
            f"{PAPERLESS_URL}/api/correspondents/",
            headers=get_headers(),
            params={"page_size": 500},
            timeout=20
        )
        resp.raise_for_status()
        corrs = {c["id"]: c["name"] for c in resp.json().get("results", [])}
        print(f"Loaded {len(corrs)} existing correspondents from Paperless")
        return corrs
    except Exception as e:
        print(f"Warning: Could not fetch correspondents: {e}")
        return {}


def list_documents(limit: int = 30) -> List[Tuple[int, str]]:
    """List documents with IDs and titles"""
    try:
        resp = requests.get(
            f"{PAPERLESS_URL}/api/documents/",
            headers=get_headers(),
            params={"page_size": limit},
            timeout=20
        )
        resp.raise_for_status()
        docs = [(d["id"], d.get("title", "Untitled")) for d in resp.json().get("results", [])]
        return docs
    except Exception as e:
        print(f"Error listing documents: {e}")
        return []


def fetch_document_image(doc_id: int) -> Optional[Tuple[bytes, str]]:
    """Fetch document preview image"""
    headers = {"Authorization": f"Token {PAPERLESS_TOKEN}"}
    
    # Try preview
    try:
        resp = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/preview/",
            headers=headers, timeout=30
        )
        if resp.status_code == 200:
            ct = resp.headers.get("content-type", "")
            if ct.startswith("image/"):
                return resp.content, ct
    except:
        pass
    
    # Try PDF conversion
    try:
        resp = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
            headers=headers, timeout=30
        )
        if resp.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(resp.content)
                pdf_path = f.name
            
            img_base = tempfile.mktemp()
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(IMAGE_DPI), "-singlefile", 
                 "-f", "1", "-l", "1", pdf_path, img_base],
                capture_output=True, timeout=30
            )
            
            img_path = img_base + ".png"
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_data = f.read()
                os.unlink(img_path)
                os.unlink(pdf_path)
                return img_data, "image/png"
    except:
        pass
    
    return None


# ============================================================================
# GEMMA 4 ANALYSIS - Freeform (no guidance)
# ============================================================================

def analyze_with_gemma4(image_bytes: bytes, content_type: str) -> Optional[Dict]:
    """
    Analyze document with Gemma 4 - letting it decide freely.
    No predefined categories or tag lists.
    """
    
    # Encode image
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Strict prompt focused on actual content identification
    prompt = """Du bist ein Dokumentenklassifizierer für Paperless-ngx. Analysiere dieses Dokument.

AUFGABE: Bestimme Dokumenttyp, Absender und 5 beschreibende Tags.

═══════════════════════════════════════════════════════════════════
DOKUMENTTYP (immer kleingeschrieben):
═══════════════════════════════════════════════════════════════════
Wähle den passendsten:
- rechnung (für Kaufbelege, Rechnungen)
- vertrag (für Verträge, Vereinbarungen)
- erstattung (für Rückzahlungen, Gutschriften, Auszahlungen)
- fahrkarte (für Tickets, Bordkarten, Buchungsbestätigungen)
- bescheid (für behördliche Mitteilungen)
- kontoauszug (für Bankauszüge)
- versicherung (für Policen, Beitragsrechnungen)
- arztbrief (für medizinische Dokumente)
- kostenvoranschlag (für Angebote, Schätzungen)

═══════════════════════════════════════════════════════════════════
TAGS - DIE WICHTIGSTE AUFGABE:
═══════════════════════════════════════════════════════════════════
Beantworte diese Fragen für die Tags:
1. Was ist das HAUPTPRODUKT oder die HAUPTSACHE? (z.B. "steelcase-bürostuhl", "zigbee-adapter", "ice-ticket")
2. Welche KATEGORIE? (z.B. "büromöbel", "smarthome", "bahnreise")
3. Welcher KONTEXT? (z.B. "arbeit", "privat", "haushalt")
4. Welche DETAILS sind relevant? (z.B. "verspätung", "erstattung", "abo")
5. OPTIONAL: Zeitraum oder Route (z.B. "märz-2026", "berlin-köln")

VERBOTENE WÖRTER (NIEMALS als Tag verwenden):
❌ dienstleistung, beleg, zahlung, kosten, kunde, kauf, produkt
❌ rechnung, vertrag, dokument (= Dokumenttyp, nicht Tag!)
❌ Der Firmenname des Absenders
❌ Zufällige Wörter aus dem Dokument die nicht das Hauptthema sind

BEISPIELE:

Steelcase Bürostuhl-Rechnung von LEiK GmbH:
→ Typ: rechnung | Tags: ["steelcase-please", "bürostuhl", "büromöbel", "arbeit", "vorkasse"]

Home Assistant Adapter von BerryBase:
→ Typ: rechnung | Tags: ["home-assistant-zbt2", "zigbee-adapter", "smarthome", "haustechnik", "kreditkarte"]

DB Verspätungs-Erstattung:
→ Typ: erstattung | Tags: ["fahrgastrechte", "verspätung", "bahnreise", "berlin-köln", "rückerstattung"]

FlixTrain Bordkarte Berlin-Köln:
→ Typ: fahrkarte | Tags: ["flixtrain", "berlin-köln", "zugticket", "fernreise", "sitzplatz"]

Claude Pro Abo-Rechnung:
→ Typ: rechnung | Tags: ["claude-pro", "ki-assistent", "software-abo", "arbeit", "monatlich"]

═══════════════════════════════════════════════════════════════════

Antworte NUR mit validem JSON:
{
    "document_type": "typ_kleingeschrieben",
    "correspondent": "Firmenname",
    "tags": ["hauptprodukt", "kategorie", "kontext", "detail", "optional"],
    "summary": "Ein Satz Zusammenfassung",
    "confidence": 0.95
}"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [b64]
        }],
        "stream": False,
        "options": {
            # Gemma 4 recommended settings from docs
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            # No num_predict limit - let model finish naturally
            "num_thread": NUM_THREADS
        }
    }
    
    try:
        start = time.time()
        print(f"  Sending request to {OLLAMA_URL}/api/chat ...")
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
        elapsed = time.time() - start
        
        if resp.status_code != 200:
            print(f"  API error: {resp.status_code} - {resp.text[:200]}")
            return None
        
        response_json = resp.json()
        message = response_json.get("message", {})
        
        # Gemma 4 separates thinking and content
        content = message.get("content", "").strip()
        thinking = message.get("thinking", "")
        
        print(f"  Response received in {elapsed:.1f}s")
        print(f"  Content length: {len(content)} chars")
        print(f"  Thinking length: {len(thinking)} chars")
        
        # Primary: use content field (final answer)
        raw = content
        
        # Fallback: if content empty, try to find JSON in thinking
        if not raw and thinking:
            print(f"  Content empty - checking thinking for JSON...")
            print(f"  Thinking preview: {thinking[:300]}...")
            raw = thinking
        
        if not raw:
            print(f"  Both content and thinking are empty")
            print(f"  Full API response: {resp.text[:500]}")
            return None
        
        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```" in raw:
                import re
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
                if match:
                    json_str = match.group(1).strip()
                    print(f"  Found JSON in code block")
                    parsed = json.loads(json_str)
                    parsed["_elapsed"] = elapsed
                    return parsed
            
            # Try to find JSON object in response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = raw[json_start:json_end]
                print(f"  Extracting JSON from position {json_start} to {json_end}")
                parsed = json.loads(json_str)
                parsed["_elapsed"] = elapsed
                return parsed
            else:
                print(f"  No JSON object found in response")
                print(f"  Response content:\n{raw[:500]}")
                return None
                
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            print(f"  Attempted to parse: {raw[json_start:json_start+300] if json_start >= 0 else raw[:300]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"  Request timeout (300s)")
        return None
    except Exception as e:
        print(f"  Request error: {type(e).__name__}: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 70)
    print("GEMMA 4 TEST - Freeform Classification")
    print("=" * 70)
    print(f"Model:  {OLLAMA_MODEL}")
    print(f"Mode:   DRY RUN (no changes to Paperless)")
    print(f"Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if model is available
    print("Checking model availability...")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"Available models: {', '.join(models)}")
            if not any(OLLAMA_MODEL in m or m in OLLAMA_MODEL for m in models):
                print(f"⚠️  Warning: {OLLAMA_MODEL} not found in model list!")
                print(f"   You may need to run: ollama pull {OLLAMA_MODEL}")
        else:
            print(f"Could not fetch model list: {resp.status_code}")
    except Exception as e:
        print(f"Could not connect to Ollama: {e}")
        sys.exit(1)
    print()
    
    # Test configuration
    print("-" * 70)
    print("TEST CONFIGURATION")
    print("-" * 70)
    print("Choose what existing Paperless data to load for comparison.")
    print("Loading data does NOT guide the model - it's only used to check")
    print("if Gemma's suggestions match existing entries.")
    print()
    
    def ask_yes_no(prompt: str, default: bool = True) -> bool:
        default_str = "Y/n" if default else "y/N"
        answer = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not answer:
            return default
        return answer in ("y", "yes", "ja", "j")
    
    load_tags = ask_yes_no("Load existing tags?", default=True)
    load_correspondents = ask_yes_no("Load existing correspondents?", default=True)
    load_doc_types = ask_yes_no("Load existing document types?", default=True)
    
    print()
    print("Configuration:")
    print(f"  • Load tags:           {'Yes' if load_tags else 'No'}")
    print(f"  • Load correspondents: {'Yes' if load_correspondents else 'No'}")
    print(f"  • Load document types: {'Yes' if load_doc_types else 'No'}")
    print()
    
    # Load existing metadata based on configuration
    existing_tags = fetch_existing_tags() if load_tags else {}
    existing_correspondents = fetch_existing_correspondents() if load_correspondents else {}
    existing_types = fetch_existing_doc_types() if load_doc_types else {}
    print()
    
    # List documents
    docs = list_documents(limit=30)
    if not docs:
        print("No documents found!")
        sys.exit(1)
    
    print(f"Available documents ({len(docs)}):")
    for i, (did, title) in enumerate(docs[:15]):
        print(f"  {did:4d}: {title[:55]}")
    if len(docs) > 15:
        print(f"  ... and {len(docs)-15} more")
    print()
    
    # Select documents
    user = input("Enter ID(s) comma-separated, 'all' for first 5, or Enter for first: ").strip()
    
    if user.lower() == "all":
        selected = [d[0] for d in docs[:5]]
    elif "," in user:
        selected = [int(x.strip()) for x in user.split(",") if x.strip().isdigit()]
    elif user.isdigit():
        selected = [int(user)]
    else:
        selected = [docs[0][0]] if docs else []
    
    print(f"\nProcessing {len(selected)} document(s)...")
    print("-" * 70)
    
    # Process documents
    results = []
    
    for doc_id in selected:
        # Get title
        title = next((t for d, t in docs if d == doc_id), "Unknown")
        print(f"\n📄 Document {doc_id}: {title[:50]}")
        
        # Fetch image
        print("  Fetching image...")
        img_data = fetch_document_image(doc_id)
        
        if not img_data:
            print("  ❌ Could not fetch image")
            results.append({"doc_id": doc_id, "title": title, "success": False, "error": "No image"})
            continue
        
        img_bytes, ct = img_data
        print(f"  Image: {len(img_bytes)/1024:.0f}KB ({ct})")
        
        # Analyze with Gemma 4
        print("  Analyzing with Gemma 4...")
        result = analyze_with_gemma4(img_bytes, ct)
        
        if result:
            print(f"  ✅ Analysis complete ({result.get('_elapsed', 0):.1f}s)")
            results.append({
                "doc_id": doc_id,
                "title": title,
                "success": True,
                **result
            })
        else:
            print("  ❌ Analysis failed")
            results.append({"doc_id": doc_id, "title": title, "success": False, "error": "Analysis failed"})
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    
    print()
    print("=" * 70)
    print("RESULTS - DRY RUN (nothing committed)")
    print("=" * 70)
    
    success_count = sum(1 for r in results if r.get("success"))
    print(f"\nProcessed: {len(results)} | Success: {success_count} | Failed: {len(results) - success_count}")
    
    for r in results:
        print()
        print("-" * 70)
        print(f"📄 Document {r['doc_id']}: {r['title']}")
        
        if not r.get("success"):
            print(f"   ❌ Error: {r.get('error', 'Unknown')}")
            continue
        
        print(f"   Document Type:  {r.get('document_type', 'N/A')}")
        print(f"   Correspondent:  {r.get('correspondent', 'N/A')}")
        print(f"   Tags:           {', '.join(r.get('tags', []))}")
        print(f"   Summary:        {r.get('summary', 'N/A')}")
        print(f"   Confidence:     {r.get('confidence', 0):.0%}")
        print(f"   Language:       {r.get('language', 'N/A')}")
        print(f"   Time:           {r.get('_elapsed', 0):.1f}s")
        
        # Compare with existing metadata (only if loaded)
        doc_type = r.get("document_type", "").lower()
        correspondent = r.get("correspondent", "").lower() if r.get("correspondent") else ""
        tags = [t.lower() for t in r.get("tags", [])]
        
        # Check if document type exists
        if existing_types:
            existing_type_names = [n.lower() for n in existing_types.values()]
            type_match = "✓ EXISTS" if any(doc_type in et or et in doc_type for et in existing_type_names) else "○ NEW"
            print(f"   Type Status:    {type_match}")
        
        # Check if correspondent exists
        if existing_correspondents and correspondent:
            existing_corr_names = [n.lower() for n in existing_correspondents.values()]
            corr_match = "✓ EXISTS" if any(correspondent in ec or ec in correspondent for ec in existing_corr_names) else "○ NEW"
            print(f"   Corr Status:    {corr_match}")
        
        # Check tags
        if existing_tags:
            existing_tag_names = [n.lower() for n in existing_tags.values()]
            for tag in tags:
                tag_match = "✓" if any(tag in et or et in tag for et in existing_tag_names) else "○"
                print(f"   Tag '{tag}': {tag_match}")
    
    print()
    print("=" * 70)
    print("DRY RUN COMPLETE - No changes made to Paperless")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
