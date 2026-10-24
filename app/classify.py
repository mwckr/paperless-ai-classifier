"""
The classification pipeline for one document:
fetch -> page images -> prompt -> Ollama -> parse -> normalize -> write back.
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import learning
from .config import Settings
from .images import fetch_page_images
from .ollama import Ollama
from .paperless import Paperless
from .prompt import PromptContext, build_prompt, coerce_result, json_schema, ocr_excerpt, parse_response

logger = logging.getLogger(__name__)


@dataclass
class Outcome:
    doc_id: int
    title: str = ""
    ok: bool = False
    reason: str = ""
    model: str = ""
    result: Dict[str, Any] = field(default_factory=dict)        # normalized answer
    model_output: Dict[str, Any] = field(default_factory=dict)  # answer before normalization
    prompt: str = ""
    raw_answer: str = ""
    thinking: str = ""
    timing: Dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    pages_sent: int = 0
    image_kb: float = 0.0
    ocr_chars: int = 0
    committed: bool = False
    changes: List[str] = field(default_factory=list)   # fields written to Paperless
    kept: List[str] = field(default_factory=list)      # fields left alone because they were set

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Classifier:
    def __init__(self, settings: Settings, paperless: Paperless, ollama: Ollama,
                 on_stage: Optional[Callable[[str], None]] = None):
        self.s = settings
        self.paperless = paperless
        self.ollama = ollama
        self._on_stage = on_stage or (lambda stage: None)

    def _stage(self, name: str) -> None:
        try:
            self._on_stage(name)
        except Exception:
            pass

    @staticmethod
    def is_classified(doc: Dict[str, Any]) -> bool:
        return bool(doc.get("document_type")) and bool(doc.get("correspondent"))

    # ------------------------------------------------------------------ analyze
    def analyze(self, doc_id: int, doc: Optional[Dict[str, Any]] = None) -> Outcome:
        s = self.s
        out = Outcome(doc_id=doc_id, model=s.OLLAMA_MODEL)
        self._stage("fetching")
        if doc is None:
            doc = self.paperless.get_document(doc_id)
        if not doc:
            out.reason = "metadata_fetch_failed"
            return out
        out.title = doc.get("title") or f"Document {doc_id}"

        ocr_mode = s.OCR_TEXT_MODE
        ocr_text = ocr_excerpt(doc.get("content"), s.OCR_TEXT_CHARS) if ocr_mode in ("hint", "only") else ""
        images: List[bytes] = []
        if ocr_mode == "only" and ocr_text:
            logger.info(f"Document {doc_id}: OCR text only ({len(ocr_text)} chars), no images")
        else:
            self._stage("rendering")
            images = fetch_page_images(self.paperless, doc_id, s)
            if not images and not ocr_text:
                out.reason = "image_fetch_failed"
                return out
        out.pages_sent = len(images)
        out.image_kb = round(sum(len(i) for i in images) / 1024, 1)
        out.ocr_chars = len(ocr_text)

        ctx = PromptContext(
            num_images=len(images),
            ocr_text=ocr_text,
            types=self.paperless.names("document_types")[: s.PROMPT_TYPE_LIMIT] if s.INJECT_EXISTING_TYPES else [],
            correspondents=self.paperless.names("correspondents")[: s.PROMPT_CORRESPONDENT_LIMIT]
            if s.INJECT_EXISTING_CORRESPONDENTS else [],
            tags=self.paperless.names("tags")[: s.PROMPT_TAG_LIMIT] if s.INJECT_EXISTING_TAGS else [],
            few_shot=learning.few_shot_block(3) if s.FEW_SHOT_ENABLED else "",
            explanations=bool(s.GENERATE_EXPLANATIONS),
        )
        out.prompt = build_prompt(ctx)
        logger.debug(f"Prompt ({len(out.prompt)} chars):\n{out.prompt}")

        self._stage("inference")
        chat = self.ollama.chat(out.prompt, images, json_schema(ctx.explanations))
        out.timing = chat.timing()
        out.tokens = chat.tokens
        out.raw_answer = chat.content
        out.thinking = chat.thinking
        if not chat.ok:
            out.reason = f"ollama: {chat.error}"
            return out
        raw = chat.content or chat.thinking
        if not raw:
            out.reason = "empty_response"
            return out
        try:
            parsed = parse_response(raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Document {doc_id}: JSON parse error {e}; answer: {raw[:300]}")
            out.reason = "invalid_json"
            return out
        if not parsed:
            logger.warning(f"Document {doc_id}: no JSON in answer: {raw[:300]}")
            out.reason = "no_json"
            return out
        out.model_output = coerce_result(parsed)
        if not out.model_output["document_type"]:
            out.reason = "no_document_type"
            return out

        self._stage("normalizing")
        if s.LEARNING_ENABLED:
            out.result = learning.normalize(
                out.model_output,
                types=self.paperless.names("document_types"),
                correspondents=self.paperless.names("correspondents"),
                tags=self.paperless.names("tags"),
                threshold=float(s.FUZZY_MATCH_THRESHOLD),
            )
        else:
            out.result = dict(out.model_output, normalized=[])
        out.ok = True
        logger.info(f"Document {doc_id}: {out.result['document_type']} | {out.result.get('correspondent')} | "
                    f"{out.result['tags']} | conf {out.result['confidence']:.2f} | {out.timing.get('total_s')}s")
        return out

    # ------------------------------------------------------------------ write back
    def write_back(self, out: Outcome, doc: Dict[str, Any], overwrite: bool) -> None:
        """Apply out.result to the document. Fills in out.committed / changes / kept."""
        self._stage("writing")
        changed, kept = self._apply(doc, out.result.get("document_type"), out.result.get("correspondent"),
                                    out.result.get("tags") or [], overwrite=overwrite,
                                    replace_tags=(self.s.TAGS_MODE == "replace"))
        out.changes, out.kept = changed, kept
        out.committed = bool(changed)

    def apply_values(self, doc_id: int, document_type: Optional[str], correspondent: Optional[str],
                     tags: Optional[List[str]]) -> List[str]:
        """Write user-chosen values (dashboard correction). Always overwrites; a tag list replaces."""
        doc = self.paperless.get_document(doc_id)
        if not doc:
            return []
        changed, _ = self._apply(doc, document_type, correspondent, tags, overwrite=True, replace_tags=True,
                                 skip_tags=tags is None)
        return changed

    def _apply(self, doc: Dict[str, Any], document_type: Optional[str], correspondent: Optional[str],
               tags: List[str], overwrite: bool, replace_tags: bool, skip_tags: bool = False):
        doc_id = doc["id"]
        data: Dict[str, Any] = {}
        changed: List[str] = []
        kept: List[str] = []

        if document_type:
            if doc.get("document_type") and not overwrite:
                kept.append("document_type")
            else:
                type_id = self.paperless.get_or_create("document_types", document_type)
                if type_id and type_id != doc.get("document_type"):
                    data["document_type"] = type_id
        if correspondent:
            if doc.get("correspondent") and not overwrite:
                kept.append("correspondent")
            else:
                corr_id = self.paperless.get_or_create("correspondents", correspondent)
                if corr_id and corr_id != doc.get("correspondent"):
                    data["correspondent"] = corr_id
        if not skip_tags:
            new_ids: List[int] = []
            for tag in tags[:5]:
                tag_id = self.paperless.get_or_create("tags", tag)
                if tag_id and tag_id not in new_ids:
                    new_ids.append(tag_id)
            current = list(doc.get("tags") or [])
            merged = new_ids if replace_tags else current + [t for t in new_ids if t not in current]
            if (new_ids or replace_tags) and set(merged) != set(current):
                data["tags"] = merged

        if kept:
            logger.info(f"Document {doc_id}: kept existing {', '.join(kept)} (OVERWRITE_EXISTING=false)")
        if not data:
            logger.info(f"Document {doc_id}: nothing to change in Paperless")
            return changed, kept
        if self.paperless.patch_document(doc_id, data):
            changed = list(data.keys())
            logger.info(f"Document {doc_id}: updated {', '.join(changed)}")
        return changed, kept
