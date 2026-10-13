"""Generadores de PDFs de prueba (PyMuPDF). Los PDFs reales de evaluación viven en eval/corpus."""

from __future__ import annotations

import pymupdf as fitz


def make_pdf(pages: list[str], fontsize: int = 11) -> bytes:
    """Cada elemento de `pages` es una página; los párrafos se separan con línea en blanco."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        y = 72
        for para in text.split("\n\n"):
            rect = fitz.Rect(72, y, 523, y + 200)
            rc = page.insert_textbox(rect, para, fontsize=fontsize, fontname="helv")
            used = 200 - rc if rc > 0 else 200
            y += used + 12
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf(text: str) -> bytes:
    """PDF 'escaneado': el texto solo existe como imagen (requiere OCR para extraerse)."""
    src = fitz.open()
    p = src.new_page()
    p.insert_textbox(fitz.Rect(72, 72, 523, 500), text, fontsize=20, fontname="helv")
    pix = p.get_pixmap(dpi=200)
    src.close()
    out = fitz.open()
    page = out.new_page()
    page.insert_image(page.rect, pixmap=pix)
    data = out.tobytes()
    out.close()
    return data
