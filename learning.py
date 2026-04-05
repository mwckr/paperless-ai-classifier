#!/usr/bin/env python3
"""
Learning Layer for Paperless AI Classifier
==========================================
Implements term mappings and few-shot examples without model fine-tuning.

Features:
- Term normalization (AI term → approved term)
- Few-shot example injection for better accuracy
- Fuzzy matching for correspondents
- Confidence-based auto-apply of mappings
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "classifier_audit.db"

# Minimum similarity for fuzzy matching (0.0 - 1.0)
FUZZY_MATCH_THRESHOLD = 0.80

# Minimum times_used to auto-apply a mapping
AUTO_APPLY_MIN_CONFIDENCE = 3


def init_learning_tables():
    """Initialize learning tables in the database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Term mappings: AI term → approved term
    c.execute('''
        CREATE TABLE IF NOT EXISTS term_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term_type TEXT NOT NULL,
            ai_term TEXT NOT NULL,
            approved_term TEXT NOT NULL,
            times_used INTEGER DEFAULT 1,
            created_at INTEGER DEFAULT (strftime('%s', 'now')),
            last_used INTEGER DEFAULT (strftime('%s', 'now')),
            UNIQUE(term_type, ai_term)
        )
    ''')
    
    # Classification examples for few-shot learning
    c.execute('''
        CREATE TABLE IF NOT EXISTS classification_examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            document_title TEXT,
            document_type TEXT,
            correspondent TEXT,
            tags TEXT,
            confidence REAL,
            user_verified INTEGER DEFAULT 0,
            created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
    ''')
    
    # Indexes for fast lookups
    c.execute('CREATE INDEX IF NOT EXISTS idx_mappings_lookup ON term_mappings(term_type, ai_term)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_mappings_approved ON term_mappings(term_type, approved_term)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_examples_verified ON classification_examples(user_verified, confidence DESC)')
    
    conn.commit()
    conn.close()
    logger.info("Learning tables initialized")


# =============================================================================
# TERM MAPPINGS
# =============================================================================

def get_mapping(term_type: str, ai_term: str) -> Optional[str]:
    """
    Look up an approved term for an AI-suggested term.
    Returns None if no mapping exists.
    """
    if not ai_term:
        return None
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT approved_term, times_used 
        FROM term_mappings 
        WHERE term_type = ? AND LOWER(ai_term) = LOWER(?)
    ''', (term_type, ai_term))
    row = c.fetchone()
    conn.close()
    
    if row:
        approved_term, times_used = row
        logger.debug(f"Mapping found: {ai_term} → {approved_term} (used {times_used}x)")
        return approved_term
    
    return None


def add_or_update_mapping(term_type: str, ai_term: str, approved_term: str):
    """
    Add a new mapping or increment times_used if it exists.
    Called when user accepts/corrects a classification.
    """
    if not ai_term or not approved_term:
        return
    
    # Don't create identity mappings
    if ai_term.lower().strip() == approved_term.lower().strip():
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Try to update existing
    c.execute('''
        UPDATE term_mappings 
        SET times_used = times_used + 1, 
            last_used = strftime('%s', 'now'),
            approved_term = ?
        WHERE term_type = ? AND LOWER(ai_term) = LOWER(?)
    ''', (approved_term, term_type, ai_term))
    
    if c.rowcount == 0:
        # Insert new mapping
        c.execute('''
            INSERT INTO term_mappings (term_type, ai_term, approved_term)
            VALUES (?, ?, ?)
        ''', (term_type, ai_term, approved_term))
        logger.info(f"New mapping: {term_type} '{ai_term}' → '{approved_term}'")
    else:
        logger.debug(f"Updated mapping: {term_type} '{ai_term}' → '{approved_term}'")
    
    conn.commit()
    conn.close()


def get_all_mappings(term_type: Optional[str] = None) -> List[Dict]:
    """Get all mappings, optionally filtered by type"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    if term_type:
        c.execute('''
            SELECT * FROM term_mappings 
            WHERE term_type = ? 
            ORDER BY times_used DESC
        ''', (term_type,))
    else:
        c.execute('SELECT * FROM term_mappings ORDER BY term_type, times_used DESC')
    
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def delete_mapping(mapping_id: int) -> bool:
    """Delete a mapping by ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM term_mappings WHERE id = ?', (mapping_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# =============================================================================
# FUZZY MATCHING
# =============================================================================

def fuzzy_match(query: str, candidates: List[str], threshold: float = FUZZY_MATCH_THRESHOLD) -> Optional[str]:
    """
    Find the best fuzzy match for a query string among candidates.
    Returns None if no match meets the threshold.
    """
    if not query or not candidates:
        return None
    
    query_lower = query.lower().strip()
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        
        # Exact match
        if query_lower == candidate_lower:
            return candidate
        
        # Substring match (one contains the other)
        if query_lower in candidate_lower or candidate_lower in query_lower:
            score = 0.95
        else:
            # Sequence matcher for similarity
            score = SequenceMatcher(None, query_lower, candidate_lower).ratio()
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = candidate
    
    if best_match:
        logger.debug(f"Fuzzy match: '{query}' → '{best_match}' ({best_score:.0%})")
    
    return best_match


# =============================================================================
# CLASSIFICATION EXAMPLES (Few-Shot)
# =============================================================================

def add_example(
    document_id: int,
    document_title: str,
    document_type: str,
    correspondent: Optional[str],
    tags: List[str],
    confidence: float = 0.0,
    user_verified: bool = False
):
    """Add a classification example for few-shot learning"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO classification_examples 
        (document_id, document_title, document_type, correspondent, tags, confidence, user_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        document_id,
        document_title,
        document_type,
        correspondent,
        json.dumps(tags) if tags else '[]',
        confidence,
        1 if user_verified else 0
    ))
    
    conn.commit()
    conn.close()
    logger.debug(f"Added example: {document_title} ({document_type})")


def get_few_shot_examples(limit: int = 5, verified_only: bool = True) -> List[Dict]:
    """
    Get high-quality examples for few-shot prompting.
    Prioritizes user-verified examples with high confidence.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    if verified_only:
        c.execute('''
            SELECT * FROM classification_examples 
            WHERE user_verified = 1 
            ORDER BY confidence DESC, created_at DESC 
            LIMIT ?
        ''', (limit,))
    else:
        c.execute('''
            SELECT * FROM classification_examples 
            ORDER BY user_verified DESC, confidence DESC, created_at DESC 
            LIMIT ?
        ''', (limit,))
    
    rows = []
    for row in c.fetchall():
        r = dict(row)
        if r.get('tags'):
            try:
                r['tags'] = json.loads(r['tags'])
            except:
                r['tags'] = []
        rows.append(r)
    
    conn.close()
    return rows


def mark_example_verified(example_id: int) -> bool:
    """Mark an example as user-verified"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE classification_examples SET user_verified = 1 WHERE id = ?', (example_id,))
    updated = c.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def delete_example(example_id: int) -> bool:
    """Delete an example by ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM classification_examples WHERE id = ?', (example_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# =============================================================================
# NORMALIZATION PIPELINE
# =============================================================================

def normalize_result(
    result: Dict,
    existing_tags: List[str] = None,
    existing_types: List[str] = None,
    existing_correspondents: List[str] = None,
    threshold: float = None
) -> Dict:
    """
    Apply learned mappings and fuzzy matching to normalize AI results.
    This is called AFTER AI analysis, BEFORE committing to Paperless.
    
    Does NOT modify the original dict - returns a new one.
    """
    if threshold is None:
        threshold = FUZZY_MATCH_THRESHOLD
    normalized = result.copy()
    
    # 1. Document type normalization
    doc_type = result.get('document_type') or result.get('dokumenttyp')
    if doc_type:
        # Normalize to lowercase first
        doc_type_lower = doc_type.lower().strip()
        
        # Check for learned mapping first
        mapped = get_mapping('document_type', doc_type_lower)
        if mapped:
            normalized['document_type'] = mapped.lower()
            normalized['_type_mapped'] = True
        elif existing_types:
            # Fuzzy match against existing types (case-insensitive)
            existing_lower = [t.lower() for t in existing_types]
            matched = fuzzy_match(doc_type_lower, existing_lower, threshold=threshold)
            if matched:
                # Find original case from existing_types
                for orig in existing_types:
                    if orig.lower() == matched.lower():
                        normalized['document_type'] = orig
                        break
                else:
                    normalized['document_type'] = matched
                normalized['_type_fuzzy'] = True
            else:
                normalized['document_type'] = doc_type_lower
        else:
            normalized['document_type'] = doc_type_lower
    
    # 2. Correspondent normalization
    correspondent = result.get('correspondent') or result.get('absender')
    if correspondent and correspondent.lower() not in ['keine', 'none', 'null', '']:
        # Check for learned mapping
        mapped = get_mapping('correspondent', correspondent)
        if mapped:
            normalized['correspondent'] = mapped
            normalized['_corr_mapped'] = True
        elif existing_correspondents:
            # Fuzzy match
            matched = fuzzy_match(correspondent, existing_correspondents, threshold=threshold)
            if matched:
                normalized['correspondent'] = matched
                normalized['_corr_fuzzy'] = True
            else:
                normalized['correspondent'] = correspondent
        else:
            normalized['correspondent'] = correspondent
    else:
        normalized['correspondent'] = None
    
    # 3. Tag normalization - simple, no hardcoded filters
    tags = result.get('tags', [])
    normalized_tags = []
    
    # Get doc type and correspondent for comparison
    doc_type_lower = (normalized.get('document_type') or '').lower().strip()
    correspondent_lower = (normalized.get('correspondent') or '').lower().strip()
    
    for tag in tags:
        if not tag or not isinstance(tag, str):
            continue
            
        tag_clean = tag.strip()
        tag_lower = tag_clean.lower()
        
        # Only skip if tag exactly matches document type or correspondent
        if tag_lower == doc_type_lower:
            logger.debug(f"Skipped tag matching doc type: {tag}")
            continue
        if tag_lower == correspondent_lower:
            logger.debug(f"Skipped tag matching correspondent: {tag}")
            continue
        
        # Check for learned mapping
        mapped = get_mapping('tag', tag_clean)
        if mapped:
            normalized_tags.append(mapped)
        elif existing_tags:
            # Fuzzy match against existing tags
            matched = fuzzy_match(tag_clean, existing_tags, threshold=threshold)
            normalized_tags.append(matched if matched else tag_clean)
        else:
            normalized_tags.append(tag_clean)
    
    # Deduplicate while preserving order
    seen = set()
    normalized['tags'] = []
    for tag in normalized_tags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            normalized['tags'].append(tag)
    
    return normalized


def learn_from_correction(
    ai_result: Dict,
    user_result: Dict,
    document_id: int = None,
    document_title: str = None
):
    """
    Learn from a user correction by creating mappings.
    Called when user modifies AI suggestion before committing.
    
    ai_result: What the AI originally suggested
    user_result: What the user accepted/corrected
    """
    # Document type
    ai_type = ai_result.get('document_type') or ai_result.get('dokumenttyp')
    user_type = user_result.get('document_type') or user_result.get('dokumenttyp')
    if ai_type and user_type and ai_type.lower() != user_type.lower():
        add_or_update_mapping('document_type', ai_type, user_type)
    
    # Correspondent
    ai_corr = ai_result.get('correspondent') or ai_result.get('absender')
    user_corr = user_result.get('correspondent') or user_result.get('absender')
    if ai_corr and user_corr and ai_corr.lower() != user_corr.lower():
        add_or_update_mapping('correspondent', ai_corr, user_corr)
    
    # Tags (learn each difference)
    ai_tags = set(t.lower() for t in (ai_result.get('tags') or []))
    user_tags = user_result.get('tags') or []
    
    for user_tag in user_tags:
        user_tag_lower = user_tag.lower()
        # Find potential AI tag this replaces
        for ai_tag in ai_tags:
            # If similar but not identical, it's a correction
            ratio = SequenceMatcher(None, ai_tag, user_tag_lower).ratio()
            if 0.5 < ratio < 1.0:  # Similar but not exact
                add_or_update_mapping('tag', ai_tag, user_tag)
                break
    
    # Add as verified example
    if document_id and user_type:
        add_example(
            document_id=document_id,
            document_title=document_title or '',
            document_type=user_type,
            correspondent=user_corr,
            tags=user_tags,
            confidence=user_result.get('confidence', 0.9),
            user_verified=True
        )


def build_few_shot_prompt(limit: int = 3) -> str:
    """
    Build a few-shot examples string to inject into the prompt.
    Returns empty string if no verified examples exist.
    """
    examples = get_few_shot_examples(limit=limit, verified_only=True)
    
    if not examples:
        return ""
    
    lines = ["Hier sind Beispiele korrekter Klassifikationen:\n"]
    
    for ex in examples:
        tags_str = ", ".join(ex.get('tags', [])[:4])
        lines.append(f"Dokument: \"{ex['document_title']}\"")
        lines.append(f"→ Typ: {ex['document_type']}, Absender: {ex['correspondent'] or 'keine'}, Tags: [{tags_str}]\n")
    
    return "\n".join(lines)


def get_learning_stats() -> Dict:
    """Get statistics about the learning system"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM term_mappings')
    total_mappings = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM term_mappings WHERE times_used >= ?', (AUTO_APPLY_MIN_CONFIDENCE,))
    confident_mappings = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM classification_examples')
    total_examples = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM classification_examples WHERE user_verified = 1')
    verified_examples = c.fetchone()[0]
    
    c.execute('SELECT term_type, COUNT(*) FROM term_mappings GROUP BY term_type')
    by_type = dict(c.fetchall())
    
    conn.close()
    
    return {
        'total_mappings': total_mappings,
        'confident_mappings': confident_mappings,
        'total_examples': total_examples,
        'verified_examples': verified_examples,
        'mappings_by_type': by_type
    }


# Initialize tables on import
init_learning_tables()
