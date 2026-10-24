"""
Learning layer: normalize model answers against the names that already exist
in Paperless and against term mappings learned from corrections. No fine-tuning.
"""
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from . import db
from .paperless import NO_NAME

logger = logging.getLogger(__name__)

_WORD_BOUNDARY = r"(?<![\wäöüß]){}(?![\wäöüß])"


def _containment(a: str, b: str, partial_names: bool) -> float:
    """Score when the shorter string is a whole word (or words) of the longer one.

    partial_names=False (tags, document types): the shorter must also cover
    nearly all of the longer, so 'amt' never maps to 'finanzamt' and the generic
    'versicherung' is not mapped to 'kfz versicherung'.
    partial_names=True (correspondents): a short name inside a long official
    one is fine ('telekom' -> 'Deutsche Telekom')."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < 4 or not re.search(_WORD_BOUNDARY.format(re.escape(short)), long_):
        return 0.0
    if partial_names:
        return 0.9
    return 0.95 if len(short) / len(long_) >= 0.8 else 0.0


def fuzzy_match(query: str, candidates: List[str], threshold: float, partial_names: bool = False) -> Optional[str]:
    """Best candidate for query, or None when nothing reaches the threshold or
    two candidates tie ('stadtwerke' vs 'Stadtwerke Bonn' / 'Stadtwerke Köln')."""
    if not query or not candidates:
        return None
    q = query.lower().strip()
    best, best_score, ties = None, 0.0, 0
    for cand in candidates:
        c = cand.lower().strip()
        if q == c:
            return cand
        score = _containment(q, c, partial_names) or SequenceMatcher(None, q, c).ratio()
        if score < threshold:
            continue
        if score > best_score:
            best, best_score, ties = cand, score, 0
        elif score == best_score:
            ties += 1
    if best and ties:
        logger.debug(f"Fuzzy match for '{query}' is ambiguous ({ties + 1} candidates), keeping original")
        return None
    if best:
        logger.debug(f"Fuzzy match: '{query}' -> '{best}' ({best_score:.0%})")
    return best


def tag_repeats(tag: str, reference: str) -> bool:
    """True when a tag just repeats the document type or correspondent: same text,
    an inflection ('rechnungen' ~ 'rechnung') or one word of a multi-word name."""
    tag, reference = tag.lower().strip(), reference.lower().strip()
    if not tag or not reference:
        return False
    if tag == reference:
        return True
    if len(tag) >= 4 and re.search(_WORD_BOUNDARY.format(re.escape(tag)), reference):
        return True
    return SequenceMatcher(None, tag, reference).ratio() >= 0.85


def normalize(result: Dict[str, Any], types: List[str], correspondents: List[str], tags: List[str],
              threshold: float) -> Dict[str, Any]:
    """Apply learned mappings and fuzzy matching. Returns a new dict and records
    what changed under 'normalized' for the audit log / dry run."""
    out = dict(result)
    notes: List[str] = []

    # document type
    raw_type = (result.get("document_type") or "").strip()
    if raw_type:
        mapped = db.mapping_lookup("document_type", raw_type)
        if mapped:
            out["document_type"] = mapped
            notes.append(f"type: '{raw_type}' -> '{mapped}' (mapping)")
        else:
            match = fuzzy_match(raw_type, types, threshold)
            if match and match.lower() != raw_type.lower():
                notes.append(f"type: '{raw_type}' -> '{match}' (fuzzy)")
            out["document_type"] = match or raw_type.lower()

    # correspondent
    raw_corr = (result.get("correspondent") or "").strip()
    if raw_corr and raw_corr.lower() not in NO_NAME:
        mapped = db.mapping_lookup("correspondent", raw_corr)
        if mapped:
            out["correspondent"] = mapped
            notes.append(f"correspondent: '{raw_corr}' -> '{mapped}' (mapping)")
        else:
            match = fuzzy_match(raw_corr, correspondents, threshold, partial_names=True)
            if match and match != raw_corr:
                notes.append(f"correspondent: '{raw_corr}' -> '{match}' (fuzzy)")
            out["correspondent"] = match or raw_corr
    else:
        out["correspondent"] = None

    # tags
    type_refs = {raw_type, out.get("document_type") or ""}
    corr_refs = {raw_corr, out.get("correspondent") or ""}
    seen, clean = set(), []
    for tag in result.get("tags") or []:
        if not isinstance(tag, str) or not tag.strip():
            continue
        t = tag.strip()
        if len(t) < 3:
            notes.append(f"tag dropped (too short): '{t}'"); continue
        if any(tag_repeats(t, ref) for ref in type_refs if ref):
            notes.append(f"tag dropped (repeats type): '{t}'"); continue
        if any(tag_repeats(t, ref) for ref in corr_refs if ref):
            notes.append(f"tag dropped (repeats correspondent): '{t}'"); continue
        mapped = db.mapping_lookup("tag", t)
        if mapped:
            notes.append(f"tag: '{t}' -> '{mapped}' (mapping)")
            t = mapped
        else:
            match = fuzzy_match(t, tags, threshold)
            if match and match != t:
                notes.append(f"tag: '{t}' -> '{match}' (fuzzy)")
                t = match
        if t.lower() not in seen:
            seen.add(t.lower())
            clean.append(t)
    out["tags"] = clean
    out["normalized"] = notes
    return out


def few_shot_block(limit: int = 3) -> str:
    examples = db.example_list(limit=limit, verified_only=True)
    if not examples:
        return ""
    lines = ["Beispiele korrekter Klassifikationen:"]
    for ex in examples:
        tags = ", ".join((ex.get("tags") or [])[:4])
        lines.append(f'Dokument "{ex.get("document_title") or ""}" -> Typ: {ex.get("document_type")}, '
                     f'Absender: {ex.get("correspondent") or "keine"}, Tags: [{tags}]')
    return "\n".join(lines)


def learn_from_correction(audit: Dict[str, Any], corrections: Dict[str, Any]) -> List[str]:
    """Create term mappings from a dashboard correction and store a verified example.

    corrections: {"document_type": {"ai": ..., "user": ...}, "correspondent": {...},
                  "tags": {"user": [...]}, "tag_mappings": [{"ai": ..., "user": ...}]}"""
    learned: List[str] = []
    for kind in ("document_type", "correspondent"):
        corr = corrections.get(kind) or {}
        ai, user = (corr.get("ai") or "").strip(), (corr.get("user") or "").strip()
        if ai and user and db.mapping_upsert(kind, ai, user):
            learned.append(f"{kind}: {ai} -> {user}")
    for tm in corrections.get("tag_mappings") or []:
        ai, user = (tm.get("ai") or "").strip(), (tm.get("user") or "").strip()
        if ai and user and db.mapping_upsert("tag", ai, user):
            learned.append(f"tag: {ai} -> {user}")

    user_type = (corrections.get("document_type") or {}).get("user") or audit.get("document_type")
    user_corr = (corrections.get("correspondent") or {}).get("user") or audit.get("correspondent")
    user_tags = (corrections.get("tags") or {}).get("user")
    if user_tags is None:
        user_tags = audit.get("tags") or []
    if user_type:
        db.example_add(audit.get("document_id"), audit.get("document_title") or "", user_type, user_corr,
                       list(user_tags), float(audit.get("confidence") or 0.9), verified=True)
    return learned
