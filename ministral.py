#!/usr/bin/env python3
"""
Ministral 3 14B Vision Integration Test
Document classification using Ministral 3's dedicated vision encoder
- Optimized prompt formatting for Mistral's chat template
- Full v5 document types (200+ types across 15+ categories)
- Vision + OCR analysis with explanation generation
- Monitors latency and token usage
- Integrates Paperless image fetching
"""
import requests
import json
import base64
import sys
import time
import logging
import re
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, List
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('ministral_vision.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PAPERLESS_URL = "http://192.168.1.201:8000"
PAPERLESS_TOKEN = "ceb3ef3a18218fa5df247416978aa58b7be63d18"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "ministral-3:14b"  # Optimized for vision
NUM_THREADS = 10

# ============================================================================
# FULL DOCUMENT TYPES FROM v5 (200+ types, 15+ categories)
# ============================================================================

DOCUMENT_TYPES = {
    "behoerden": [
        "bescheid", "bewilligungsbescheid", "ablehnungsbescheid", "änderungsbescheid",
        "aufhebungsbescheid", "festsetzungsbescheid", "leistungsbescheid", "gebührenbescheid",
        "kostenbescheid", "anhörung", "vorladung", "ladung", "amtliche-mitteilung",
        "behördenbrief", "verwaltungsakt", "widerspruchsbescheid", "einspruchsentscheidung",
        "genehmigung", "erlaubnis", "zulassung", "anordnung", "verfügung", "auflage",
        "aufforderung", "erinnerung", "zwangsgeldandrohung", "vollstreckungsankündigung"
    ],
    "finanzamt": [
        "steuerbescheid", "einkommensteuerbescheid", "umsatzsteuerbescheid",
        "gewerbesteuerbescheid", "grundsteuerbescheid", "erbschaftsteuerbescheid",
        "schenkungsteuerbescheid", "steuererklärung", "steuervorauszahlung",
        "umsatzsteuervoranmeldung", "steuerbescheinigung", "lohnsteuerbescheinigung",
        "freistellungsbescheinigung", "nichtveranlagungsbescheinigung",
        "steuernummer-mitteilung", "steuer-id-mitteilung", "mahnung-finanzamt",
        "vollstreckung-finanzamt", "stundungsbescheid", "erlass", "zinsbescheid",
        "verspätungszuschlag", "säumniszuschlag"
    ],
    "soziales": [
        "kindergeldbescheid", "elterngeldbescheid", "wohngeldbescheid",
        "arbeitslosengeld-bescheid", "bürgergeld-bescheid", "sozialhilfebescheid",
        "grundsicherungsbescheid", "pflegegeldbescheid", "krankengeld-bescheid",
        "mutterschaftsgeld", "unterhaltsbescheid", "unterhaltsvorschuss",
        "bafög-bescheid", "bildungsgutschein", "eingliederungszuschuss",
        "gründungszuschuss", "kurzarbeitergeld", "insolvenzgeld", "kinderzuschlag",
        "betreuungsgeld"
    ],
    "rente": [
        "renteninformation", "rentenbescheid", "rentenauskunft", "versicherungsverlauf",
        "kontenklärung", "nachversicherung", "beitragsbescheid-rv",
        "erwerbsminderungsrente", "altersrente", "witwenrente", "waisenrente",
        "riester-bescheid", "betriebsrente", "versorgungsauskunft"
    ],
    "strafen": [
        "bußgeldbescheid", "verwarnungsgeld", "strafzettel", "knöllchen",
        "blitzer-bescheid", "ordnungswidrigkeitenbescheid", "fahrverbot",
        "punktemitteilung", "strafbefehl", "geldstrafe", "gerichtskostenbescheid"
    ],
    "meldewesen": [
        "meldebescheinigung", "anmeldebestätigung", "abmeldebestätigung",
        "ummeldebescheinigung", "aufenthaltstitel", "aufenthaltserlaubnis",
        "niederlassungserlaubnis", "einbürgerungsurkunde", "geburtsurkunde",
        "heiratsurkunde", "sterbeurkunde", "scheidungsurkunde",
        "namensänderungsurkunde", "abstammungsurkunde", "familienbuchauszug",
        "eheurkunde", "lebenspartnerschaftsurkunde"
    ],
    "ausweise": [
        "personalausweis", "reisepass", "kinderreisepass", "führerschein",
        "fahrerlaubnis", "internationaler-führerschein", "schwerbehindertenausweis",
        "waffenschein", "jagdschein", "angelschein", "fischereierlaubnis"
    ],
    "immobilien": [
        "baugenehmigung", "bauantrag", "bauvoranfrage", "nutzungsänderung",
        "abrissgenehmigung", "grundbuchauszug", "flurkarte", "liegenschaftskataster",
        "teilungserklärung", "wohnflächenberechnung", "energieausweis",
        "grundsteuererklärung", "erschließungsbescheid", "anliegerbescheid"
    ],
    "kfz": [
        "fahrzeugschein", "fahrzeugbrief", "zulassungsbescheinigung",
        "abmeldebescheinigung-kfz", "ummeldebescheinigung-kfz", "kurzzeitkennzeichen",
        "ausfuhrkennzeichen", "feinstaubplakette", "tüv-bericht", "hauptuntersuchung",
        "abgasuntersuchung", "unfallbericht", "gutachten-kfz", "werkstattrechnung",
        "kfz-versicherung", "schadensfreiheitsrabatt"
    ],
    "arbeit": [
        "lebenslauf", "bewerbung", "bewerbungsschreiben", "anschreiben",
        "arbeitsvertrag", "änderungsvertrag", "aufhebungsvertrag", "kündigung",
        "kündigungsschreiben", "kündigungsbestätigung", "abmahnung", "arbeitszeugnis",
        "zwischenzeugnis", "praktikumszeugnis", "ausbildungszeugnis", "lohnabrechnung",
        "gehaltsabrechnung", "entgeltabrechnung", "arbeitsbescheinigung",
        "arbeitszeitnachweis", "urlaubsantrag", "urlaubsbescheinigung",
        "reisekostenabrechnung", "spesenabrechnung", "dienstvertrag",
        "honorarvertrag", "werkvertrag", "freelancer-vertrag", "minijob-vertrag",
        "tarifvertrag", "betriebsvereinbarung"
    ],
    "bildung": [
        "schulzeugnis", "halbjahreszeugnis", "jahreszeugnis", "abschlusszeugnis",
        "abitur", "fachabitur", "realschulabschluss", "hauptschulabschluss",
        "bachelor-zeugnis", "master-zeugnis", "diplom", "doktorurkunde", "promotion",
        "immatrikulationsbescheinigung", "exmatrikulationsbescheinigung",
        "studienbescheinigung", "transcript", "zertifikat", "fortbildungszertifikat",
        "sprachzertifikat", "teilnahmebescheinigung", "ausbildungsvertrag",
        "praktikumsvertrag", "praktikumsbescheinigung", "berufsschulzeugnis",
        "gesellenbrief", "meisterbrief"
    ],
    "versicherung": [
        "versicherungspolice", "versicherungsschein", "versicherungsvertrag",
        "nachtrag", "beitragsrechnung", "beitragsbescheid", "leistungsabrechnung",
        "schadensmeldung", "schadensregulierung", "deckungszusage",
        "ablehnung-versicherung", "kündigung-versicherung",
        "kündigungsbestätigung-vers", "krankenversicherungskarte",
        "mitgliedsbescheinigung", "familienversicherung", "zusatzversicherung",
        "beitragssatz", "bonusheft"
    ],
    "bank": [
        "kontoauszug", "jahreskontoauszug", "depotauszug", "wertpapierabrechnung",
        "kontovertrag", "kontoeröffnung", "kontokündigung", "kreditvertrag",
        "darlehensvertrag", "tilgungsplan", "zins-und-tilgungsplan", "bürgschaft",
        "sicherungsübereignung", "grundschuldbestellung", "freistellungsauftrag",
        "jahressteuerbescheinigung", "verlustbescheinigung", "bankbestätigung",
        "finanzierungszusage", "kreditablehnung", "inkasso", "pfändung",
        "kontopfändung", "p-konto-bescheinigung"
    ],
    "wohnen": [
        "mietvertrag", "untermietvertrag", "staffelmietvertrag", "indexmietvertrag",
        "mieterhöhung", "mietminderung", "nebenkostenabrechnung",
        "betriebskostenabrechnung", "heizkostenabrechnung", "kaution",
        "kautionsabrechnung", "wohnungsübergabeprotokoll", "übergabeprotokoll",
        "abnahmeprotokoll", "mängelanzeige", "instandsetzungsaufforderung",
        "modernisierungsankündigung", "eigenbedarfskündigung", "räumungsklage",
        "hausordnung", "eigentümerversammlung", "hausgeldabrechnung", "wirtschaftsplan"
    ],
    "vertraege": [
        "vertrag", "kaufvertrag", "notarvertrag", "schenkungsvertrag",
        "tauschvertrag", "dienstleistungsvertrag", "wartungsvertrag",
        "servicevertrag", "lizenzvertrag", "nutzungsvertrag", "pachtvertrag",
        "leasingvertrag", "abo-vertrag", "rahmenvertrag", "zusatzvereinbarung",
        "vollmacht", "generalvollmacht", "vorsorgevollmacht", "patientenverfügung",
        "testament", "erbvertrag"
    ],
    "rechnungen": [
        "rechnung", "sammelrechnung", "abschlagsrechnung", "schlussrechnung",
        "teilrechnung", "stornorechnung", "gutschrift", "proformarechnung",
        "stromrechnung", "gasrechnung", "wasserrechnung", "fernwärmerechnung",
        "telefonrechnung", "mobilfunkrechnung", "internetrechnung", "kabelrechnung",
        "rundfunkbeitrag", "müllgebühren", "abwassergebühren", "schornsteinfeger",
        "quittung", "kassenbon", "zahlungsbeleg"
    ],
    "mahnungen": [
        "zahlungserinnerung", "mahnung", "erste-mahnung", "zweite-mahnung",
        "letzte-mahnung", "mahnbescheid", "vollstreckungsbescheid",
        "inkassoschreiben", "forderungsaufstellung", "ratenzahlungsvereinbarung",
        "vergleich", "schuldanerkenntnis"
    ],
    "korrespondenz": [
        "brief", "geschäftsbrief", "privatbrief", "einschreiben",
        "einwurf-einschreiben", "rückschein", "bestätigung", "zusage", "absage",
        "einladung", "mitteilung", "ankündigung", "information", "rundschreiben",
        "newsletter", "werbung", "antwort", "stellungnahme", "widerspruch",
        "einspruch", "beschwerde", "reklamation", "anfrage", "angebot",
        "kostenvoranschlag", "auftragsbestätigung", "lieferschein", "bestellung"
    ],
    "medizin": [
        "arztbrief", "überweisungsschein", "einweisung", "entlassungsbrief",
        "befund", "laborbericht", "blutbild", "röntgenbefund", "mrt-befund",
        "ct-befund", "ultraschallbefund", "pathologiebefund", "histologie",
        "op-bericht", "behandlungsplan", "therapieplan", "attest",
        "krankschreibung", "arbeitsunfähigkeitsbescheinigung", "rezept",
        "privatrezept", "kassenrezept", "heilmittelverordnung",
        "hilfsmittelverordnung", "impfausweis", "impfbescheinigung", "mutterpass",
        "kinderuntersuchungsheft", "vorsorgeuntersuchung", "gutachten-medizinisch",
        "pflegegutachten", "pflegegrad", "medikamentenplan", "organspendeausweis"
    ],
    "recht": [
        "anwaltsschreiben", "mandatsvertrag", "vollmacht-anwalt", "klage",
        "klageschrift", "klageerwiderung", "urteil", "beschluss",
        "gerichtsbeschluss", "vergleich-gericht", "vollstreckungstitel",
        "pfändungsbeschluss", "eidesstattliche-versicherung", "zeugnis-gericht",
        "sachverständigengutachten", "rechtsmittelbelehrung", "rechtskraftvermerk",
        "scheidungsurteil", "sorgerechtsurteil", "umgangsregelung",
        "unterhaltsurteil", "erbschein", "testamentseröffnung", "nachlassverzeichnis"
    ],
    "mitgliedschaft": [
        "mitgliedsausweis", "mitgliedsantrag", "mitgliedsbestätigung",
        "beitragsrechnung-verein", "vereinssatzung", "protokoll-versammlung",
        "abonnement", "abo-bestätigung", "kündigung-abo", "fitnessstudio-vertrag",
        "vereinsbeitritt", "vereinsaustritt"
    ],
    "einkauf": [
        "kaufbeleg", "garantieschein", "garantiekarte", "gewährleistung",
        "reparaturauftrag", "reparaturrechnung", "rücksendebeleg",
        "umtauschbeleg", "widerrufsbelehrung", "widerruf"
    ],
    "reise": [
        "buchungsbestätigung", "reisebestätigung", "flugticket", "e-ticket",
        "boardingpass", "bahnticket", "bahncard", "hotelrechnung", "hotelbuchung",
        "mietwagenvertrag", "reiseversicherung", "reiserücktritt", "visum",
        "visumsantrag"
    ],
    "sonstiges": [
        "protokoll", "bericht", "gutachten", "expertise", "stellungnahme",
        "dokumentation", "anleitung", "bedienungsanleitung", "datenblatt",
        "technische-dokumentation", "foto", "scan", "kopie", "notiz", "liste",
        "tabelle", "sonstiges"
    ]
}

# Flatten for easy access
ALL_DOCUMENT_TYPES = []
for category, types_list in DOCUMENT_TYPES.items():
    ALL_DOCUMENT_TYPES.extend(types_list)

# Document types without correspondents
NO_CORRESPONDENT_TYPES = {
    "lebenslauf", "bewerbung", "personalausweis", "reisepass", "führerschein",
    "geburtsurkunde", "heiratsurkunde", "sterbeurkunde", "testament",
    "impfausweis", "foto", "scan", "kopie", "notiz"
}


def get_headers():
    return {
        "Authorization": f"Token {PAPERLESS_TOKEN}",
        "Content-Type": "application/json"
    }



def get_or_create_tag(tag_name: str) -> Optional[int]:
    """Get existing tag ID or create new tag, returns tag ID"""
    tag_name = tag_name.lower().strip()
    if not tag_name:
        return None
    try:
        # Search for existing tag
        response = requests.get(
            f"{PAPERLESS_URL}/api/tags/",
            headers=get_headers(),
            params={"name__iexact": tag_name},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            return results[0]["id"]
        
        # Create new tag
        response = requests.post(
            f"{PAPERLESS_URL}/api/tags/",
            headers=get_headers(),
            json={"name": tag_name},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Created new tag: {tag_name}")
        return response.json()["id"]
    except Exception as e:
        logger.warning(f"Failed to get/create tag '{tag_name}': {e}")
        return None


def get_or_create_correspondent(name: str) -> Optional[int]:
    """Get existing correspondent ID or create new one"""
    if not name or name.lower() == "keine":
        return None
    try:
        # Search for existing
        response = requests.get(
            f"{PAPERLESS_URL}/api/correspondents/",
            headers=get_headers(),
            params={"name__icontains": name[:50]},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            return results[0]["id"]
        
        # Create new
        response = requests.post(
            f"{PAPERLESS_URL}/api/correspondents/",
            headers=get_headers(),
            json={"name": name},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Created new correspondent: {name}")
        return response.json()["id"]
    except Exception as e:
        logger.warning(f"Failed to get/create correspondent '{name}': {e}")
        return None


def get_or_create_document_type(type_name: str) -> Optional[int]:
    """Get existing document type ID or create new one"""
    if not type_name:
        return None
    try:
        # Search for existing
        response = requests.get(
            f"{PAPERLESS_URL}/api/document_types/",
            headers=get_headers(),
            params={"name__iexact": type_name},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if results:
            return results[0]["id"]
        
        # Create new
        response = requests.post(
            f"{PAPERLESS_URL}/api/document_types/",
            headers=get_headers(),
            json={"name": type_name},
            timeout=10
        )
        response.raise_for_status()
        logger.info(f"Created new document type: {type_name}")
        return response.json()["id"]
    except Exception as e:
        logger.warning(f"Failed to get/create document type '{type_name}': {e}")
        return None


def update_document_in_paperless(doc_id: int, result: Dict) -> bool:
    """Update document in Paperless with classification results"""
    try:
        update_data = {}
        
        # Document type
        if result.get("document_type"):
            type_id = get_or_create_document_type(result["document_type"])
            if type_id:
                update_data["document_type"] = type_id
        
        # Correspondent
        if result.get("correspondent") and result["correspondent"].lower() != "keine":
            corr_id = get_or_create_correspondent(result["correspondent"])
            if corr_id:
                update_data["correspondent"] = corr_id
        
        # Tags
        if result.get("tags"):
            tag_ids = []
            for tag in result["tags"]:
                tag_id = get_or_create_tag(tag)
                if tag_id:
                    tag_ids.append(tag_id)
            if tag_ids:
                update_data["tags"] = tag_ids
        
        if not update_data:
            logger.warning(f"No data to update for document {doc_id}")
            return False
        
        # Update document
        response = requests.patch(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/",
            headers=get_headers(),
            json=update_data,
            timeout=15
        )
        response.raise_for_status()
        logger.info(f"Updated document {doc_id}: type={result.get('document_type')}, correspondent={result.get('correspondent')}, tags={len(result.get('tags', []))}")
        return True
    except Exception as e:
        logger.error(f"Failed to update document {doc_id}: {e}")
        return False




def get_existing_tags() -> List[str]:
    """Fetch all existing tags from Paperless for reuse"""
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/tags/",
            headers=get_headers(),
            timeout=20,
            params={"page_size": 500}
        )
        response.raise_for_status()
        data = response.json()
        tags = [tag['name'].lower() for tag in data.get("results", [])]
        logger.info(f"Loaded {len(tags)} existing tags for reuse")
        return tags
    except Exception as e:
        logger.warning(f"Could not fetch existing tags: {e}")
        return []


def display_existing_tags():
    """Display all existing tags at startup"""
    tags = get_existing_tags()
    print(f"Existing tags in Paperless ({len(tags)}):")
    if tags:
        # Display in columns
        cols = 4
        for i in range(0, len(tags), cols):
            row = tags[i:i+cols]
            print("  " + "  |  ".join(f"{t:<25}" for t in row))
    else:
        print("  (keine)")
    print()


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


def fetch_document_image(doc_id: int) -> Optional[Tuple[bytes, str, int]]:
    """Fetch document image from Paperless"""
    headers = {"Authorization": f"Token {PAPERLESS_TOKEN}"}
    
    try:
        # Try preview first
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/preview/",
            headers=headers,
            timeout=25
        )
        
        if response.status_code == 200:
            content_type = response.headers.get('content-type', 'application/octet-stream')
            
            if content_type.startswith('image/'):
                size_kb = len(response.content) / 1024
                logger.info(f"Got preview image: {size_kb:.1f} KB ({content_type})")
                return response.content, content_type, size_kb
            
            logger.info("Preview is PDF, attempting conversion")
    except Exception as e:
        logger.debug(f"Preview fetch failed: {e}")
    
    # Fetch original PDF
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/download/",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_size_kb = len(pdf_bytes) / 1024
        logger.info(f"Got original PDF: {pdf_size_kb:.1f} KB")
        
        # Convert PDF to image using pdftoppm
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_file:
                pdf_file.write(pdf_bytes)
                pdf_path = pdf_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
                img_path = img_file.name
            
            # Use pdftoppm to convert ALL pages to JPEG (optimized size)
            import glob
            from PIL import Image
            base_path = img_path.replace('.png', '').replace('.jpg', '')
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
                        # Combine all pages vertically
                        images = [Image.open(f) for f in page_files]
                        total_height = sum(img.height for img in images)
                        max_width = max(img.width for img in images)
                        combined = Image.new('RGB', (max_width, total_height))
                        y_offset = 0
                        for img in images:
                            combined.paste(img, (0, y_offset))
                            y_offset += img.height
                        import io
                        buffer = io.BytesIO()
                        combined.save(buffer, 'JPEG', quality=85)
                        img_bytes = buffer.getvalue()
                        for img in images:
                            img.close()
                    # Cleanup temp files
                    for f in page_files:
                        try:
                            os.remove(f)
                        except:
                            pass
                    img_size_kb = len(img_bytes) / 1024
                    logger.info(f"Converted PDF to image: {img_size_kb:.1f} KB (JPEG, {len(page_files)} pages)")
                
                # Cleanup temp files
                import os
                try:
                    os.remove(pdf_path)
                    os.remove(img_path)
                except:
                    pass
                
                return img_bytes, 'image/png', img_size_kb
            else:
                logger.warning(f"PDF conversion failed: {result.stderr.decode()}")
        except FileNotFoundError:
            logger.warning("pdftoppm not found, falling back to thumbnail")
        except Exception as e:
            logger.warning(f"PDF conversion error: {e}")
    
    except Exception as e:
        logger.debug(f"PDF fetch failed: {e}")
    
    # Fallback to thumbnail
    try:
        response = requests.get(
            f"{PAPERLESS_URL}/api/documents/{doc_id}/thumb/",
            headers=headers,
            timeout=25
        )
        response.raise_for_status()
        content_type = response.headers.get('content-type', 'image/webp')
        size_kb = len(response.content) / 1024
        logger.info(f"Using thumbnail fallback: {size_kb:.1f} KB ({content_type})")
        return response.content, content_type, size_kb
    except Exception as e:
        logger.error(f"Failed to fetch image: {e}")
        return None


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
        logger.error(f"Failed to fetch metadata: {e}")
        return None


def analyze_with_vision(image_bytes: bytes, content_type: str) -> Tuple[bool, Optional[Dict], float, int]:
    """
    Send image to Ministral 3 vision model.
    Optimized for Mistral's chat format.
    Returns (success, result_dict, elapsed_time, estimated_tokens)
    """
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    # Build comprehensive type list grouped by category
    type_suggestions = []
    for category, types_list in DOCUMENT_TYPES.items():
        type_suggestions.append(f"{category}: {', '.join(types_list[:6])}")
    type_list_str = "\n".join(type_suggestions)
    
    # Get existing tags for reuse
    existing_tags = get_existing_tags()
    existing_tags_str = ", ".join(existing_tags[:100]) if existing_tags else "keine vorhanden"
    
    # REASONING-FIRST PROMPT: Describe before classifying
    prompt = f"""Analysiere dieses Dokument in dieser Reihenfolge:

SCHRITT 1 - BESCHREIBUNG (2-3 Sätze):
Beschreibe was du auf der ERSTEN SEITE siehst: Überschrift, Absender, Hauptzweck.
Ignoriere Anlagen, AGB oder Zertifikate auf späteren Seiten.

WICHTIG: Wenn die Überschrift den Dokumenttyp EXPLIZIT nennt (z.B. "Rechnung", "Checkliste", "Vertrag", "Bescheid"), übernimm diesen Namen DIREKT als Dokumenttyp. Nicht interpretieren, nicht umformulieren.

SCHRITT 2 - KLASSIFIKATION:
Basierend auf deiner Beschreibung, fülle das JSON aus.

WICHTIG für dokumenttyp:
1. ERST prüfen: Nennt die Überschrift den Typ explizit? (Rechnung, Checkliste, Vertrag, Bescheid, etc.)
   → Wenn ja: Diesen Namen als dokumenttyp verwenden!
2. NUR wenn kein expliziter Typ genannt wird: Aus dem Inhalt ableiten
3. NIEMALS Kategorienamen verwenden (soziales, behoerden, rechnungen = FALSCH)

DOKUMENTTYPEN:
{type_list_str}

EXISTIERENDE TAGS (wenn passend, exakt wiederverwenden):
{existing_tags_str}

TAG-REGELN:
- 2-3 allgemeine + 2-3 spezifische Tags
- Bei RECHNUNGEN: Mindestens 1 Tag mit dem konkreten PRODUKT/ARTIKEL (z.B. "ubiquiti-switch", "macbook-pro", "waschmaschine")
- Abkürzungen ausschreiben (MBP=macbook-pro, UDM=unifi-dream-machine)
- KEINE Daten, Orte oder Beträge

ANTWORT FORMAT:
Beschreibung: [deine 2-3 Sätze hier]
JSON: {{"dokumenttyp": "...", "absender": "...", "tags": ["...", "...", "...", "...", "..."]}}"""

    # MISTRAL CHAT FORMAT: Cleaner than Gemma's approach
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.05,  # Lower temperature for Mistral = more consistent JSON
            "num_predict": 400,   # Room for description + JSON
            "num_thread": NUM_THREADS,
            "num_ctx": 8192
        }
    }
    
    start = time.time()
    try:
        logger.debug(f"Vision request: image {len(image_b64)} chars, prompt {len(prompt)} chars")
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=1500)
        elapsed = time.time() - start
        
        if response.status_code != 200:
            logger.error(f"Vision API error {response.status_code}: {response.text[:200]}")
            return False, None, elapsed, 0
        
        result = response.json()
        llm_response = result["message"]["content"].strip()
        
        # Estimate tokens
        input_tokens = len(prompt.split()) + len(image_b64) // 4
        output_tokens = len(llm_response.split())
        estimated_tokens = input_tokens + output_tokens
        
        # Parse JSON with robust handling
        try:
            # Strip markdown code blocks
            clean_response = llm_response.replace('```json', '').replace('```', '').strip()
            
            start_idx = clean_response.find('{')
            end_idx = clean_response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = clean_response[start_idx:end_idx]
                json_str = re.sub(r",\s*}", "}", json_str)
                json_str = re.sub(r",\s*]", "]", json_str)
                
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    # Manual extraction fallback
                    dtype_m = re.search(r'"dokumenttyp"\s*:\s*"([^"]+)"', clean_response)
                    sender_m = re.search(r'"absender"\s*:\s*"([^"]+)"', clean_response)
                    tags_m = re.search(r'"tags"\s*:\s*\[([^\]]+)\]', clean_response)
                    if dtype_m:
                        tags = re.findall(r'"([^"]+)"', tags_m.group(1)) if tags_m else []
                        parsed = {"dokumenttyp": dtype_m.group(1), "absender": sender_m.group(1) if sender_m else "keine", "tags": tags[:5]}
                    else:
                        raise ValueError("No dokumenttyp found")
                
                # Tag-based correction for CV/Lebenslauf
                dtype = parsed.get("dokumenttyp", "")
                tags = [t.lower() for t in parsed.get("tags", [])]
                if any(t in tags for t in ["lebenslauf", "cv", "berufserfahrung", "bewerbung"]):
                    if dtype not in ["lebenslauf", "bewerbung", "bewerbungsschreiben", "anschreiben"]:
                        logger.info(f"Correcting type from {dtype} to lebenslauf based on tags")
                        parsed["dokumenttyp"] = "lebenslauf"
                
                # Log the inline analysis if present
                if parsed.get('analyse'):
                    logger.debug(f"Model analysis: {parsed.get('analyse')}")
                logger.info(f"Vision success: {parsed.get('dokumenttyp')} | {elapsed:.1f}s | ~{estimated_tokens} tokens")
                return True, parsed, elapsed, estimated_tokens
            else:
                logger.warning("No JSON found in response")
                logger.warning(f"Raw response: {llm_response[:500]}")
                return False, None, elapsed, estimated_tokens
        except Exception as e:
            logger.warning(f"JSON parse error: {e}")
            logger.warning(f"Raw response: {llm_response[:500]}")
            return False, None, elapsed, estimated_tokens
    
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        logger.error(f"Vision timeout after {elapsed:.1f}s")
        return False, None, elapsed, 0
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Vision error: {e}")
        return False, None, elapsed, 0


def explain_decision(image_bytes: bytes, result: Dict) -> Optional[str]:
    """
    Ask Ministral 3 to explain WHY it classified the document this way.
    Optimized for Mistral's chat template.
    """
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

    explanation_prompt = f"""Dokument-Klassifikation Analyse:
Typ: {result.get('dokumenttyp')}
Absender: {result.get('absender', 'keine')}

Erkläre in 2-3 Sätzen, WARUM diese Klassifikation richtig ist.
Nenne konkrete visuelle Merkmale (Layout, Beträge, Überschriften, Struktur).
Antworte kurz und sachlich auf Deutsch."""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": explanation_prompt,
                "images": [image_b64]
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 500,
            "num_thread": NUM_THREADS,
            "num_ctx": 8192
        }
    }

    try:
        response = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
        if response.status_code != 200:
            logger.error(f"Explain API error {response.status_code}")
            return None
        data = response.json()
        explanation = data["message"]["content"].strip()
        logger.info(f"Explanation: {explanation}")
        return explanation
    except Exception as e:
        logger.error(f"Explain error: {e}")
        return None


def normalize_type(doc_type: str) -> str:
    """Normalize document type - preserve original if reasonable"""
    doc_type_lower = doc_type.lower().strip()
    
    # Exact match
    if doc_type_lower in ALL_DOCUMENT_TYPES:
        return doc_type_lower
    
    # If the type looks valid (not a category), keep it as-is
    # This allows new types like "checkliste" to be created
    category_names = list(DOCUMENT_TYPES.keys())
    if doc_type_lower not in category_names and len(doc_type_lower) > 2:
        return doc_type_lower
    
    # Only fuzzy match as last resort for very short/ambiguous types
    for known_type in ALL_DOCUMENT_TYPES:
        if known_type == doc_type_lower or doc_type_lower == known_type:
            return known_type
    
    return doc_type_lower


def process_document(doc_id: int, generate_explanation: bool = False) -> Dict:
    """Process a single document through vision analysis"""
    logger.info(f"Processing document {doc_id}")
    
    metadata = get_document_metadata(doc_id)
    if not metadata:
        logger.error(f"Could not fetch metadata for {doc_id}")
        return {"doc_id": doc_id, "success": False, "reason": "metadata_fetch_failed"}
    
    image_data = fetch_document_image(doc_id)
    if not image_data:
        logger.error(f"Could not fetch image for {doc_id}")
        return {"doc_id": doc_id, "success": False, "reason": "image_fetch_failed"}
    
    image_bytes, content_type, size_kb = image_data
    success, vision_result, elapsed, tokens = analyze_with_vision(image_bytes, content_type)
    
    if not success or not vision_result:
        logger.error(f"Vision analysis failed for {doc_id}")
        return {
            "doc_id": doc_id,
            "title": metadata.get("title", ""),
            "success": False,
            "reason": "vision_analysis_failed",
            "duration_sec": elapsed,
            "image_size_kb": size_kb
        }
    
    # Normalize output
    doc_type = normalize_type(vision_result.get("dokumenttyp", "unknown"))
    
    result = {
        "doc_id": doc_id,
        "title": metadata.get("title", ""),
        "success": True,
        "duration_sec": elapsed,
        "tokens_est": tokens,
        "image_size_kb": size_kb,
        "document_type": doc_type,
        "correspondent": vision_result.get("absender"),
        "tags": vision_result.get("tags", []),
        "raw": vision_result
    }
    
    # Generate explanation if requested
    if generate_explanation:
        explanation = explain_decision(image_bytes, vision_result)
        if explanation:
            result["explanation"] = explanation
    
    return result


def main():
    print("\n" + "=" * 80)
    print("Ministral 3 14B Vision Integration Test")
    print("Optimized for Mistral's chat template and vision capabilities")
    print("=" * 80)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Known doc types: {len(ALL_DOCUMENT_TYPES)} across {len(DOCUMENT_TYPES)} categories")
    print(f"VRAM: ~9GB (optimized for edge deployment)")
    
    # Display existing tags
    display_existing_tags()
    
    # Get documents
    doc_ids = list_documents(limit=20)
    if not doc_ids:
        logger.error("No documents found")
        sys.exit(1)
    
    print(f"Available: {len(doc_ids)} documents")
    print(f"IDs: {', '.join(map(str, doc_ids[:10]))}")
    print()
    
    # Select document(s)
    user_input = input(f"Enter doc ID or range (e.g. '82' or '82,83,84'): ").strip()
    
    if not user_input:
        selected = doc_ids[:1]
    elif ',' in user_input:
        selected = [int(x.strip()) for x in user_input.split(',') if x.strip().isdigit()]
    else:
        try:
            selected = [int(user_input)]
        except ValueError:
            print("Invalid input")
            sys.exit(1)
    
    # Ask if user wants explanations
    explain_mode = input("Generate explanations for classifications? (y/N): ").strip().lower() == 'y'
    
    print()
    print("-" * 80)
    
    # Process documents
    results = []
    metrics = {
        "total_docs": len(selected),
        "successful": 0,
        "failed": 0,
        "total_time": 0.0,
        "total_tokens": 0,
        "total_image_size_kb": 0.0,
    }
    
    for doc_id in selected:
        result = process_document(doc_id, generate_explanation=explain_mode)
        results.append(result)
        
        if result["success"]:
            metrics["successful"] += 1
            metrics["total_time"] += result.get("duration_sec", 0)
            metrics["total_tokens"] += result.get("tokens_est", 0)
            metrics["total_image_size_kb"] += result.get("image_size_kb", 0)
            
            print(f"ID {doc_id:4d} | {result['document_type']:<20} | {result['duration_sec']:6.1f}s | {result['tokens_est']:5d} tok | {result.get('title', '')[:35]}")
        else:
            metrics["failed"] += 1
            print(f"ID {doc_id:4d} | FAILED | {result.get('reason', 'unknown')}")
    
    print("-" * 80)
    print()
    
    # Summary
    print("METRICS (Ministral 3 14B @ 9GB VRAM)")
    print("-" * 80)
    print(f"Documents processed: {metrics['successful']}/{metrics['total_docs']}")
    
    if metrics['successful'] > 0:
        avg_time = metrics['total_time'] / metrics['successful']
        print(f"Avg latency: {avg_time:.1f}s per document")
        print(f"Total time: {metrics['total_time']:.1f}s")
        print(f"Total tokens: {metrics['total_tokens']}")
        print(f"Avg tokens: {metrics['total_tokens'] / metrics['successful']:.0f}")
        print(f"Total image size: {metrics['total_image_size_kb']:.1f} KB")
        print(f"Avg image size: {metrics['total_image_size_kb'] / metrics['successful']:.1f} KB")
    
    print()
    
    # Detailed results
    successful_results = [r for r in results if r["success"]]
    if len(results) > 0:
        print("RESULTS")
        print("-" * 80)
        for result in results:
            if result["success"]:
                print(f"\nID {result['doc_id']}: {result['title']}")
                print(f"  Type: {result['document_type']}")
                if result.get('correspondent'):
                    print(f"  From: {result['correspondent']}")
                print(f"  Tags: {', '.join(result.get('tags', []))}")
                if result.get('raw', {}).get('analyse'):
                    print(f"  Analysis: {result['raw']['analyse']}")
                print(f"  Duration: {result['duration_sec']:.1f}s | Tokens: {result['tokens_est']}")
                if result.get('explanation'):
                    print(f"  Reason: {result['explanation']}")
    
    # Ask to apply changes
    if successful_results:
        print()
        print("=" * 80)
        apply = input(f"Apply changes to Paperless for {len(successful_results)} document(s)? (y/N): ").strip().lower()
        if apply == 'y':
            print("\nApplying changes...")
            applied = 0
            for result in successful_results:
                if update_document_in_paperless(result['doc_id'], result):
                    print(f"  ✓ Document {result['doc_id']} updated")
                    applied += 1
                else:
                    print(f"  ✗ Document {result['doc_id']} failed")
            print(f"\nApplied changes to {applied}/{len(successful_results)} documents")
        else:
            print("\nDry run - no changes applied")
    
    print()
    print(f"Log: ministral_vision.log")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    main()
