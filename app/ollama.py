"""
Ollama client tuned for CPU hosts: explicit context window and output cap,
thinking off, JSON-schema constrained output, keep_alive per request, and
real token/timing figures from the server response.
"""
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    ok: bool
    content: str = ""
    thinking: str = ""
    error: str = ""
    prompt_tokens: int = 0
    eval_tokens: int = 0
    load_s: float = 0.0
    prompt_s: float = 0.0
    eval_s: float = 0.0
    total_s: float = 0.0
    dropped: List[str] = field(default_factory=list)   # request fields the server rejected

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.eval_tokens

    def timing(self) -> Dict[str, float]:
        return {"load_s": round(self.load_s, 1), "prompt_s": round(self.prompt_s, 1),
                "eval_s": round(self.eval_s, 1), "total_s": round(self.total_s, 1),
                "prompt_tokens": self.prompt_tokens, "eval_tokens": self.eval_tokens}


class Ollama:
    def __init__(self, settings: Settings):
        self.s = settings
        self.base = str(settings.OLLAMA_URL).rstrip("/")
        self.model = settings.OLLAMA_MODEL
        # Remembered server rejections so later requests skip the field right away
        self.unsupported: set = set()

    # ------------------------------------------------------------------ status
    def health(self) -> str:
        try:
            resp = requests.get(f"{self.base}/api/tags", timeout=5)
            return "healthy" if resp.status_code == 200 else f"error {resp.status_code}"
        except requests.RequestException:
            return "unreachable"

    def models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.base}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except requests.RequestException as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []

    def loaded(self) -> List[Dict[str, Any]]:
        """Models currently in RAM (Ollama /api/ps)."""
        try:
            resp = requests.get(f"{self.base}/api/ps", timeout=5)
            resp.raise_for_status()
            return [{"name": m.get("name"), "size_mb": round((m.get("size") or 0) / 1e6),
                     "expires_at": m.get("expires_at")} for m in resp.json().get("models", [])]
        except requests.RequestException:
            return []

    def unload(self) -> bool:
        try:
            resp = requests.post(f"{self.base}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
            ok = resp.status_code == 200
            logger.info(f"Unloaded {self.model}" if ok else f"Unload failed: {resp.status_code}")
            return ok
        except requests.RequestException as e:
            logger.warning(f"Could not unload {self.model}: {e}")
            return False

    # ------------------------------------------------------------------ chat
    def options(self) -> Dict[str, Any]:
        s = self.s
        opts: Dict[str, Any] = {}
        if s.OLLAMA_TEMPERATURE is not None:
            opts["temperature"] = s.OLLAMA_TEMPERATURE
        if s.OLLAMA_TOP_P is not None:
            opts["top_p"] = s.OLLAMA_TOP_P
        if s.OLLAMA_TOP_K is not None:
            opts["top_k"] = s.OLLAMA_TOP_K
        if s.OLLAMA_THREADS and s.OLLAMA_THREADS > 0:
            opts["num_thread"] = s.OLLAMA_THREADS
        if s.OLLAMA_NUM_CTX > 0:
            opts["num_ctx"] = s.OLLAMA_NUM_CTX
        if s.OLLAMA_NUM_PREDICT > 0:
            opts["num_predict"] = s.OLLAMA_NUM_PREDICT
        return opts

    def build_payload(self, prompt: str, images: List[bytes], schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        message: Dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = [base64.b64encode(img).decode("ascii") for img in images]
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message],
            "stream": False,
            "keep_alive": self.s.OLLAMA_KEEP_ALIVE,
            "options": self.options(),
        }
        if "think" not in self.unsupported:
            payload["think"] = bool(self.s.OLLAMA_THINK)
        if schema and self.s.OLLAMA_STRUCTURED_OUTPUT and "format" not in self.unsupported:
            payload["format"] = schema
        return payload

    def chat(self, prompt: str, images: List[bytes], schema: Optional[Dict[str, Any]] = None) -> ChatResult:
        payload = self.build_payload(prompt, images, schema)
        timeout = (10, int(self.s.OLLAMA_TIMEOUT))
        result = ChatResult(ok=False)
        start = time.time()
        try:
            logger.debug(f"Ollama request: {len(images)} image(s), {len(prompt)} chars, options={payload['options']}")
            resp = requests.post(f"{self.base}/api/chat", json=payload, timeout=timeout)

            # Older servers reject newer fields with HTTP 400. Drop, remember, retry once.
            if resp.status_code == 400:
                body = resp.text.lower()
                for key in ("think", "format"):
                    if key in payload and (key in body or "does not support" in body or "unknown" in body
                                           or key == "format"):
                        logger.warning(f"Ollama rejected '{key}' ({resp.text[:100]}); retrying without it")
                        payload.pop(key)
                        self.unsupported.add(key)
                        result.dropped.append(key)
                if result.dropped:
                    resp = requests.post(f"{self.base}/api/chat", json=payload, timeout=timeout)

            result.total_s = time.time() - start
            if resp.status_code != 200:
                result.error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.error(f"Ollama error {result.error}")
                return result

            data = resp.json()
            message = data.get("message") or {}
            result.content = (message.get("content") or "").strip()
            result.thinking = (message.get("thinking") or "").strip()
            result.prompt_tokens = int(data.get("prompt_eval_count") or 0)
            result.eval_tokens = int(data.get("eval_count") or 0)
            result.load_s = (data.get("load_duration") or 0) / 1e9
            result.prompt_s = (data.get("prompt_eval_duration") or 0) / 1e9
            result.eval_s = (data.get("eval_duration") or 0) / 1e9
            logger.info(f"Ollama timing: load {result.load_s:.1f}s | prompt {result.prompt_tokens} tok in "
                        f"{result.prompt_s:.1f}s | output {result.eval_tokens} tok in {result.eval_s:.1f}s "
                        f"| total {result.total_s:.1f}s")

            num_ctx = payload["options"].get("num_ctx")
            if num_ctx and result.prompt_tokens >= 0.9 * num_ctx:
                logger.warning(f"Prompt used {result.prompt_tokens} of {num_ctx} context tokens. Ollama truncates "
                               "silently when it does not fit: raise OLLAMA_NUM_CTX or lower MAX_PAGES / PROMPT_*_LIMIT.")
            num_predict = payload["options"].get("num_predict")
            if num_predict and result.eval_tokens >= num_predict and not result.content:
                logger.warning(f"Model hit OLLAMA_NUM_PREDICT={num_predict} before answering "
                               "(thinking enabled?). Set OLLAMA_THINK=false or raise OLLAMA_NUM_PREDICT.")
            result.ok = True
            return result
        except requests.exceptions.Timeout:
            result.total_s = time.time() - start
            result.error = f"timeout after {result.total_s:.0f}s (OLLAMA_TIMEOUT={self.s.OLLAMA_TIMEOUT})"
            logger.error(f"Ollama {result.error}")
            return result
        except requests.RequestException as e:
            result.total_s = time.time() - start
            result.error = str(e)
            logger.error(f"Ollama request failed: {e}")
            return result
