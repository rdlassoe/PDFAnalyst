"""Extracción de texto por página (PyMuPDF), con OCR opcional (Tesseract) y respaldo con pypdf."""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (AGPL-3.0, ver docs/LICENSES.md)

from ..config import get_settings

log = logging.getLogger(__name__)

_NUL_AND_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTISPACE = re.compile(r"[ \t ]+")


class PdfError(Exception):
    """PDF inválido, cifrado o ilegible."""


@dataclass
class PageText:
    number: int  # base 1
    text: str
    ocr_used: bool = False


@dataclass
class ExtractedPdf:
    pages: list[PageText]
    page_count: int
    title: str | None = None
    author: str | None = None
    ocr_pages: int = 0
    warnings: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Normaliza texto: PostgreSQL rechaza NUL, y los guiones de fin de línea rompen palabras."""
    text = _NUL_AND_CTRL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_MULTISPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()


def validate_pdf_bytes(data: bytes, max_bytes: int) -> int:
    """Valida firma, tamaño y que se pueda abrir. Devuelve el número de páginas."""
    if not data:
        raise PdfError("El archivo está vacío")
    if len(data) > max_bytes:
        raise PdfError(f"El archivo supera el máximo de {max_bytes // (1024 * 1024)} MB")
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise PdfError("El archivo no es un PDF válido (firma %PDF- ausente)")
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise PdfError("El PDF está protegido con contraseña")
            if doc.page_count < 1:
                raise PdfError("El PDF no contiene páginas")
            return doc.page_count
    except PdfError:
        raise
    except Exception as exc:  # PyMuPDF lanza varias excepciones propias
        raise PdfError(f"No se pudo abrir el PDF: {exc}") from exc


def _page_paragraphs(page: "fitz.Page") -> str:
    """Texto de la página como párrafos (bloques) en orden de lectura, separados por línea en blanco."""
    blocks = page.get_text("blocks", sort=True)
    paras: list[str] = []
    for b in blocks:
        if b[6] != 0:  # 0 = texto, 1 = imagen
            continue
        cleaned = clean_text(b[4])
        if cleaned:
            # Dentro de un bloque, las líneas son saltos de renglón, no de párrafo.
            paras.append(re.sub(r"\s*\n\s*", " ", cleaned))
    return "\n\n".join(paras)


def _ocr_page(page: "fitz.Page") -> str:
    import pytesseract
    from PIL import Image

    s = get_settings()
    pix = page.get_pixmap(dpi=s.ocr_dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    raw = pytesseract.image_to_string(img, lang=s.ocr_langs)
    # Tesseract separa párrafos con línea en blanco
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", clean_text(raw))]
    return "\n\n".join(p for p in paras if p)


def ocr_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_pdf(path: Path) -> ExtractedPdf:
    s = get_settings()
    try:
        doc = fitz.open(path)
    except Exception as exc:
        log.warning("PyMuPDF no pudo abrir %s (%s); intentando con pypdf", path.name, exc)
        return _extract_with_pypdf(path)

    result = ExtractedPdf(pages=[], page_count=doc.page_count)
    meta = doc.metadata or {}
    result.title = clean_text(meta.get("title") or "") or None
    result.author = clean_text(meta.get("author") or "") or None

    ocr_ok = s.ocr_enabled and ocr_available()
    if s.ocr_enabled and not ocr_ok:
        result.warnings.append("OCR habilitado pero Tesseract no está disponible")

    with doc:
        for index in range(doc.page_count):
            page = doc[index]
            text = _page_paragraphs(page)
            used_ocr = False
            if len(text) < s.ocr_min_chars and ocr_ok:
                try:
                    ocr_text = _ocr_page(page)
                    if len(ocr_text) > len(text):
                        text, used_ocr = ocr_text, True
                        result.ocr_pages += 1
                except Exception as exc:
                    result.warnings.append(f"OCR falló en la página {index + 1}: {exc}")
            result.pages.append(PageText(number=index + 1, text=text, ocr_used=used_ocr))
    return result


def _extract_with_pypdf(path: Path) -> ExtractedPdf:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise PdfError("El PDF está cifrado")
        pages = [
            PageText(number=i + 1, text=clean_text(p.extract_text() or ""))
            for i, p in enumerate(reader.pages)
        ]
    except PdfError:
        raise
    except Exception as exc:
        raise PdfError(f"No se pudo leer el PDF: {exc}") from exc
    meta = reader.metadata
    return ExtractedPdf(
        pages=pages,
        page_count=len(pages),
        title=clean_text(getattr(meta, "title", None) or "") or None,
        author=clean_text(getattr(meta, "author", None) or "") or None,
        warnings=["Extraído con pypdf (respaldo)"],
    )
