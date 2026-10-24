"""
Configuration
=============
One table (FIELDS) describes every setting: key, type, default, group, label and
help text. The same table drives parsing of the .env file, validation of
dashboard edits and the config form itself, so a new setting is added in exactly
one place.

Precedence: .env file > process environment > default.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import dotenv_values, set_key

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(os.environ.get("PCLS_ENV", ROOT / ".env"))
DB_FILE = Path(os.environ.get("PCLS_DB", ROOT / "classifier_audit.db"))
LOG_DIR = Path(os.environ.get("PCLS_LOGS", ROOT / "logs"))

_TRUE = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Field:
    key: str
    type: str                 # str | secret | int | float | float_opt | int_opt | bool | choice
    default: Any
    group: str
    label: str
    help: str = ""
    options: Tuple[str, ...] = ()
    restart: bool = False     # change needs a service restart
    ask: bool = False         # asked one by one in the guided setup (install / first run)


FIELDS: List[Field] = [
    # --- Connection -----------------------------------------------------------
    Field("PAPERLESS_URL", "str", "http://localhost:8000", "Connection", "Paperless URL", ask=True),
    Field("PAPERLESS_TOKEN", "secret", "", "Connection", "Paperless API token",
          "Paperless > Settings > API tokens (or the user's profile page).", ask=True),
    Field("OLLAMA_URL", "str", "http://localhost:11434", "Connection", "Ollama URL", ask=True),
    Field("OLLAMA_MODEL", "str", "gemma4:e4b", "Connection", "Model",
          "Any vision model from `ollama list`. Text-only models work with OCR text mode 'only'.", ask=True),

    # --- Ollama runtime ---------------------------------------------------------
    Field("OLLAMA_THREADS", "int", 0, "Ollama", "CPU threads",
          "0 = let Ollama choose (physical cores). More threads than cores slows inference down."),
    Field("OLLAMA_TEMPERATURE", "float_opt", 0.1, "Ollama", "Temperature",
          "Low values give stable, repeatable names. Empty = model default."),
    Field("OLLAMA_TOP_P", "float_opt", None, "Ollama", "Top P", "Empty = model default."),
    Field("OLLAMA_TOP_K", "int_opt", None, "Ollama", "Top K", "Empty = model default."),
    Field("OLLAMA_NUM_CTX", "int", 4096, "Ollama", "Context window (tokens)",
          "Ollama truncates silently when a prompt does not fit. The log warns when a prompt nearly fills it. "
          "A bigger window costs RAM (KV cache)."),
    Field("OLLAMA_NUM_PREDICT", "int", 512, "Ollama", "Max output tokens"),
    Field("OLLAMA_THINK", "bool", False, "Ollama", "Thinking mode",
          "Thinking tokens cost minutes on CPU and add nothing for classification."),
    Field("OLLAMA_STRUCTURED_OUTPUT", "bool", True, "Ollama", "JSON schema output",
          "Constrains the answer to the expected JSON. Falls back automatically on servers that reject it."),
    Field("OLLAMA_KEEP_ALIVE", "str", "5m", "Ollama", "Keep model loaded",
          "How long the model stays in RAM after a request (e.g. 5m, 1h, -1 = forever)."),
    Field("OLLAMA_UNLOAD_WHEN_IDLE", "bool", False, "Ollama", "Unload when queue is empty",
          "Frees the model's RAM between batches. Reloading costs a few seconds."),
    Field("OLLAMA_TIMEOUT", "int", 600, "Ollama", "Request timeout (s)"),

    # --- Input ------------------------------------------------------------------
    Field("MAX_PAGES", "int", 3, "Input", "Pages per document",
          "Every page is one vision encoder pass. 1 or 2 is often enough: the first page carries the letterhead.",
          ask=True),
    Field("IMAGE_MAX_SIZE", "int", 1024, "Input", "Image size (longest side, px)"),
    Field("IMAGE_QUALITY", "int", 85, "Input", "JPEG quality"),
    Field("IMAGE_GRAYSCALE", "bool", False, "Input", "Grayscale images", "Smaller payload, same token count."),
    Field("PDF_RENDER_TIMEOUT", "int", 120, "Input", "pdftoppm timeout (s)"),
    Field("OCR_TEXT_MODE", "choice", "off", "Input", "Paperless OCR text",
          "off = page images only (recommended; the layout tells the model who the sender is). "
          "hint = images plus OCR text as a reading aid. only = text without images, last resort for hardware "
          "that cannot run a vision model.", options=("off", "hint", "only")),
    Field("OCR_TEXT_CHARS", "int", 2000, "Input", "OCR text characters", "Used by hint/only mode."),

    # --- Write-back --------------------------------------------------------------
    Field("AUTO_COMMIT", "bool", True, "Write-back", "Write results to Paperless",
          "false = classify and log only, nothing is changed in Paperless.", ask=True),
    Field("OVERWRITE_EXISTING", "bool", False, "Write-back", "Overwrite existing type/correspondent",
          "false = only fill empty fields. Re-analyze from the dashboard always overwrites."),
    Field("TAGS_MODE", "choice", "merge", "Write-back", "Tags",
          "merge keeps the document's existing tags (inbox tag, matching rules).", options=("merge", "replace")),
    Field("SKIP_CLASSIFIED", "bool", True, "Write-back", "Skip already classified documents",
          "Documents that already have a type and a correspondent are not sent to the model "
          "(polling and webhook). Re-analyze ignores this."),
    Field("GENERATE_EXPLANATIONS", "bool", False, "Write-back", "Ask for an explanation",
          "One sentence per document, shown in the audit log. Costs output tokens."),

    # --- Prompt --------------------------------------------------------------------
    Field("INJECT_EXISTING_TYPES", "bool", True, "Prompt", "Offer existing document types"),
    Field("INJECT_EXISTING_CORRESPONDENTS", "bool", True, "Prompt", "Offer existing correspondents"),
    Field("INJECT_EXISTING_TAGS", "bool", True, "Prompt", "Offer existing tags"),
    Field("PROMPT_TYPE_LIMIT", "int", 50, "Prompt", "Max document types in prompt", "Most used first."),
    Field("PROMPT_CORRESPONDENT_LIMIT", "int", 60, "Prompt", "Max correspondents in prompt", "Most used first."),
    Field("PROMPT_TAG_LIMIT", "int", 80, "Prompt", "Max tags in prompt", "Most used first."),
    Field("FEW_SHOT_ENABLED", "bool", False, "Prompt", "Few-shot examples",
          "Adds up to three verified examples from your corrections."),

    # --- Learning --------------------------------------------------------------------
    Field("LEARNING_ENABLED", "bool", True, "Learning", "Normalize against Paperless names",
          "Fuzzy-match the answer against existing names and learned term mappings."),
    Field("FUZZY_MATCH_THRESHOLD", "float", 0.80, "Learning", "Fuzzy match threshold", "0.0 to 1.0"),

    # --- Service ---------------------------------------------------------------------
    Field("POLL_INTERVAL", "int", 60, "Service", "Poll interval (s)", "0 = webhooks only.", ask=True),
    Field("API_HOST", "str", "0.0.0.0", "Service", "Bind address", restart=True),
    Field("API_PORT", "int", 8001, "Service", "Dashboard port", restart=True, ask=True),
    Field("LOG_LEVEL", "choice", "INFO", "Service", "Log level", options=("DEBUG", "INFO", "WARNING")),
]
FIELD_MAP: Dict[str, Field] = {f.key: f for f in FIELDS}
GROUPS: List[str] = []
for _f in FIELDS:
    if _f.group not in GROUPS:
        GROUPS.append(_f.group)


def _parse(field: Field, raw: Optional[str]) -> Any:
    """Parse one raw string according to the field type; invalid input falls back to the default."""
    if raw is None:
        return field.default
    raw = raw.strip()
    t = field.type
    if t in ("str", "secret"):
        return raw if raw else field.default
    if t == "bool":
        return raw.lower() in _TRUE if raw else field.default
    if t == "choice":
        return raw.lower() if raw.lower() in field.options else field.default
    if t in ("float_opt", "int_opt") and raw == "":
        return None
    try:
        if t in ("int", "int_opt"):
            return int(float(raw))
        if t in ("float", "float_opt"):
            return float(raw)
    except ValueError:
        logger.warning(f"Invalid value for {field.key}={raw!r}, using default {field.default!r}")
        return field.default
    return raw


def validate(key: str, value: str) -> str:
    """Validate a value coming from the dashboard. Returns the string to store or raises ValueError."""
    field = FIELD_MAP.get(key)
    if field is None:
        raise ValueError(f"Unknown setting {key}")
    value = (value or "").strip()
    if "\n" in value or "\r" in value:
        raise ValueError("Values must be a single line")
    t = field.type
    if t in ("float_opt", "int_opt") and value == "":
        return ""
    if t in ("int", "int_opt"):
        try:
            int(float(value))
        except ValueError:
            raise ValueError(f"{field.label} must be a whole number")
    elif t in ("float", "float_opt"):
        try:
            float(value)
        except ValueError:
            raise ValueError(f"{field.label} must be a number")
    elif t == "bool":
        if value.lower() not in ("true", "false"):
            raise ValueError(f"{field.label} must be true or false")
        value = value.lower()
    elif t == "choice":
        if value.lower() not in field.options:
            raise ValueError(f"{field.label} must be one of {', '.join(field.options)}")
        value = value.lower()
    return value


class Settings:
    """Immutable snapshot of all settings. Attribute access by key: settings.OLLAMA_MODEL"""

    def __init__(self, values: Dict[str, Any]):
        self._values = values

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name)

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def as_dict(self, redact: bool = True) -> Dict[str, Any]:
        out = dict(self._values)
        if redact:
            for f in FIELDS:
                if f.type == "secret" and out.get(f.key):
                    out[f.key] = "***REDACTED***"
        return out

    @classmethod
    def load(cls) -> "Settings":
        file_values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
        values: Dict[str, Any] = {}
        for f in FIELDS:
            raw = file_values.get(f.key)
            if raw is None:
                raw = os.environ.get(f.key)
            values[f.key] = _parse(f, raw)
        return cls(values)

    @staticmethod
    def save(key: str, value: str) -> None:
        """Write one validated value to the .env file (creating it if needed)."""
        value = validate(key, value)
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENV_FILE.touch(exist_ok=True)
        set_key(str(ENV_FILE), key, value, quote_mode="never")


def schema() -> List[Dict[str, Any]]:
    """Config form description for the dashboard."""
    return [
        {
            "key": f.key, "type": f.type, "group": f.group, "label": f.label, "help": f.help,
            "options": list(f.options), "default": f.default, "restart": f.restart,
        }
        for f in FIELDS
    ]
