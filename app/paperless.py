"""
Paperless-ngx REST client.
Resource lists (tags, document types, correspondents) are cached per client
instance, most used first, with a TTL. A client is rebuilt whenever settings
change, which also drops the cache.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import Settings

logger = logging.getLogger(__name__)

RESOURCE_KINDS = ("tags", "document_types", "correspondents")
CACHE_TTL_SECONDS = 300

# Answers that mean "no correspondent"
NO_NAME = {"", "keine", "keiner", "kein", "none", "null", "unbekannt", "unknown", "n/a", "-"}


class Paperless:
    def __init__(self, settings: Settings):
        self.base = str(settings.PAPERLESS_URL).rstrip("/")
        self.token = settings.PAPERLESS_TOKEN
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {self.token}"})
        self._cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

    # ------------------------------------------------------------------ http
    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _get_json(self, path: str, timeout: int = 20, **params: Any) -> Any:
        resp = self.session.get(self._url(path), params=params or None, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> str:
        try:
            resp = self.session.get(self._url("/api/documents/"), params={"page_size": 1}, timeout=5)
            return "healthy" if resp.status_code == 200 else f"error {resp.status_code}"
        except requests.RequestException:
            return "unreachable"

    # ------------------------------------------------------------------ documents
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        try:
            return self._get_json(f"/api/documents/{doc_id}/")
        except requests.RequestException as e:
            logger.error(f"Failed to fetch document {doc_id}: {e}")
            return None

    def download(self, doc_id: int, timeout: int = 60) -> Tuple[str, bytes]:
        """Archived (OCRed) PDF when available, otherwise the original. Returns (content_type, bytes)."""
        resp = self.session.get(self._url(f"/api/documents/{doc_id}/download/"), timeout=timeout)
        resp.raise_for_status()
        return resp.headers.get("content-type", ""), resp.content

    def fetch_image(self, doc_id: int, kind: str) -> Optional[bytes]:
        """kind: 'preview' or 'thumb'. Returns bytes only when the server sends an image."""
        try:
            resp = self.session.get(self._url(f"/api/documents/{doc_id}/{kind}/"), timeout=30)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                return resp.content
        except requests.RequestException as e:
            logger.debug(f"{kind} fetch failed for {doc_id}: {e}")
        return None

    def newest_documents(self, page: int = 1, page_size: int = 50) -> Tuple[List[Dict[str, Any]], bool]:
        """Documents ordered by id descending. Returns (results, has_next)."""
        data = self._get_json("/api/documents/", timeout=10, ordering="-id", page_size=page_size, page=page,
                              fields="id,title,document_type,correspondent")
        return data.get("results", []), bool(data.get("next"))

    def patch_document(self, doc_id: int, data: Dict[str, Any]) -> bool:
        try:
            resp = self.session.patch(self._url(f"/api/documents/{doc_id}/"), json=data, timeout=15)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False

    # ------------------------------------------------------------------ resources
    def _fetch_all(self, kind: str, page_size: int = 500, max_pages: int = 20) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        url = self._url(f"/api/{kind}/")
        params: Optional[Dict[str, Any]] = {"page_size": page_size}
        for _ in range(max_pages):
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            url = data.get("next")
            params = None
            if not url:
                break
        return results

    def resources(self, kind: str) -> List[Dict[str, Any]]:
        """Cached [{id, name, document_count}] for a kind, most used first."""
        cached = self._cache.get(kind)
        if cached and time.time() - cached[0] <= CACHE_TTL_SECONDS:
            return cached[1]
        try:
            items = [{"id": r.get("id"), "name": r.get("name", ""), "document_count": r.get("document_count") or 0}
                     for r in self._fetch_all(kind) if r.get("name")]
            items.sort(key=lambda r: (-r["document_count"], r["name"].lower()))
            self._cache[kind] = (time.time(), items)
            logger.info(f"Loaded {len(items)} {kind.replace('_', ' ')} from Paperless")
            return items
        except requests.RequestException as e:
            logger.warning(f"Could not fetch {kind}: {e}")
            return cached[1] if cached else []

    def names(self, kind: str) -> List[str]:
        return [r["name"] for r in self.resources(kind)]

    def invalidate_cache(self) -> None:
        self._cache.clear()

    def get_or_create(self, kind: str, name: str) -> Optional[int]:
        """Find a tag/type/correspondent by name (case-insensitive) or create it."""
        name = (name or "").strip()
        if not name or name.lower() in NO_NAME:
            return None
        for item in self.resources(kind):
            if item["name"].lower() == name.lower() and item.get("id"):
                return item["id"]
        try:
            data = self._get_json(f"/api/{kind}/", timeout=10, name__iexact=name)
            results = data.get("results", [])
            if results:
                return results[0]["id"]
            resp = self.session.post(self._url(f"/api/{kind}/"), json={"name": name}, timeout=10)
            if resp.status_code == 201:
                new_id = resp.json()["id"]
                logger.info(f"Created {kind[:-1]} '{name}' (id {new_id})")
                cached = self._cache.get(kind)
                if cached:
                    cached[1].append({"id": new_id, "name": name, "document_count": 0})
                return new_id
            logger.warning(f"Could not create {kind} '{name}': {resp.status_code} {resp.text[:120]}")
        except requests.RequestException as e:
            logger.error(f"{kind} lookup/create failed for '{name}': {e}")
        return None

    def document_with_names(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Current Paperless values of a document with ids resolved to names (for the dashboard)."""
        doc = self.get_document(doc_id)
        if not doc:
            return None
        types = {r["id"]: r["name"] for r in self.resources("document_types")}
        corrs = {r["id"]: r["name"] for r in self.resources("correspondents")}
        tags = {r["id"]: r["name"] for r in self.resources("tags")}
        return {
            "id": doc["id"],
            "title": doc.get("title", ""),
            "document_type": types.get(doc.get("document_type")),
            "correspondent": corrs.get(doc.get("correspondent")),
            "tags": [tags[t] for t in doc.get("tags", []) if t in tags],
        }
