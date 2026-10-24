"""
Page images for the vision model.
PDFs are rendered once by pdftoppm straight to JPEG at the target size, so no
second decode/encode pass is needed. Image originals go through Pillow.
"""
import glob
import io
import logging
import os
import subprocess
import tempfile
from typing import List, Optional

from PIL import Image

from .config import Settings
from .paperless import Paperless

logger = logging.getLogger(__name__)


def encode_image(data: bytes, max_size: int, quality: int, grayscale: bool) -> bytes:
    """Downscale an image file (any Pillow format) and return JPEG bytes."""
    mode = "L" if grayscale else "RGB"
    with Image.open(io.BytesIO(data)) as img:
        try:
            img.draft(mode, (max_size, max_size))  # JPEG decoder shortcut
        except Exception:
            pass
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        if img.mode != mode:
            img = img.convert(mode)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=quality, optimize=True)
        return out.getvalue()


def render_pdf_pages(pdf_bytes: bytes, max_pages: int, max_size: int, quality: int,
                     grayscale: bool, timeout: int) -> List[bytes]:
    """First `max_pages` pages of a PDF as JPEG bytes. Raises RuntimeError on failure."""
    with tempfile.TemporaryDirectory(prefix="pcls-") as tmp:
        pdf_path = os.path.join(tmp, "doc.pdf")
        with open(pdf_path, "wb") as fh:
            fh.write(pdf_bytes)
        base = os.path.join(tmp, "page")
        # poppler ignores -gray for JPEG output; grayscale is applied below with Pillow
        cmd = ["pdftoppm", "-jpeg", "-jpegopt", f"quality={quality},optimize=y",
               "-scale-to", str(max_size), "-f", "1", "-l", str(max(1, max_pages)), pdf_path, base]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except FileNotFoundError:
            raise RuntimeError("pdftoppm not found: install poppler-utils")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"pdftoppm timed out after {timeout}s")
        if proc.returncode != 0:
            raise RuntimeError(f"pdftoppm exit {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:300]}")
        files = sorted(glob.glob(f"{base}-*.jpg"))[:max_pages]
        if not files:
            raise RuntimeError("pdftoppm produced no pages")
        pages = []
        for path in files:
            with open(path, "rb") as fh:
                data = fh.read()
            pages.append(encode_image(data, max_size, quality, True) if grayscale else data)
        return pages


def fetch_page_images(paperless: Paperless, doc_id: int, s: Settings) -> List[bytes]:
    """Page images for a document: archived PDF first, then preview, then thumbnail."""
    max_pages = max(1, int(s.MAX_PAGES))
    max_size = max(256, int(s.IMAGE_MAX_SIZE))
    quality = min(100, max(30, int(s.IMAGE_QUALITY)))
    gray = bool(s.IMAGE_GRAYSCALE)
    try:
        content_type, data = paperless.download(doc_id)
        if content_type.startswith("image/"):
            img = encode_image(data, max_size, quality, gray)
            logger.info(f"Document {doc_id} is an image: {len(img) / 1024:.0f} KB")
            return [img]
        pages = render_pdf_pages(data, max_pages, max_size, quality, gray, int(s.PDF_RENDER_TIMEOUT))
        logger.info(f"Document {doc_id}: {len(pages)} page image(s) at {max_size}px, "
                    f"{sum(len(p) for p in pages) / 1024:.0f} KB")
        return pages
    except Exception as e:
        logger.warning(f"Document {doc_id}: PDF rendering failed ({e}), trying preview/thumbnail")

    for kind in ("preview", "thumb"):
        data: Optional[bytes] = paperless.fetch_image(doc_id, kind)
        if data:
            try:
                img = encode_image(data, max_size, quality, gray)
                logger.info(f"Document {doc_id}: using {kind} image, {len(img) / 1024:.0f} KB")
                return [img]
            except Exception as e:
                logger.debug(f"{kind} unusable: {e}")
    logger.error(f"Document {doc_id}: no image could be obtained")
    return []
