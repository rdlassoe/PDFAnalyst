"""OCR real con Tesseract (se omite si no está instalado, p. ej. en el host sin Tesseract)."""

import pytest

from app.config import get_settings
from app.services.ingestion import register_upload
from app.services.pdf_extract import extract_pdf, ocr_available
from app.worker import process_one

from .pdfs import make_scanned_pdf

pytestmark = [pytest.mark.ocr, pytest.mark.skipif(not ocr_available(), reason="Tesseract no instalado")]

TEXT = "La garantia del producto es de veinticuatro meses desde la fecha de compra. Codigo de servicio TX-9087."


@pytest.fixture
def ocr_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "ocr_enabled", True)


def test_scanned_page_is_read_with_ocr(ocr_on, tmp_path):
    p = tmp_path / "scan.pdf"
    p.write_bytes(make_scanned_pdf(TEXT))
    ex = extract_pdf(p)
    assert ex.ocr_pages == 1 and ex.pages[0].ocr_used
    t = ex.pages[0].text.lower()
    assert "garantia" in t and "veinticuatro" in t and "9087" in t


def test_scanned_pdf_full_pipeline_is_searchable(ocr_on, session, embedder):
    from app.services.retrieval import retrieve

    doc = register_upload(session, "escaneado.pdf", make_scanned_pdf(TEXT)).document
    while process_one(session):
        pass
    session.expire_all()
    assert session.get(type(doc), doc.id).status == "ready"
    r = retrieve(session, "¿Cuánto dura la garantía?")
    assert r.hits and r.hits[0].filename == "escaneado.pdf"


def test_scanned_pdf_without_ocr_fails_with_clear_message(monkeypatch, session):
    monkeypatch.setattr(get_settings(), "ocr_enabled", False)
    monkeypatch.setattr(get_settings(), "pdf_figures_enabled", False)  # sin figuras ni OCR no hay nada que leer
    doc = register_upload(session, "escaneado.pdf", make_scanned_pdf(TEXT)).document
    while process_one(session):
        pass
    session.expire_all()
    d = session.get(type(doc), doc.id)
    assert d.status == "failed" and "OCR" in d.error


def test_scanned_pdf_without_ocr_is_indexed_through_its_page_image_with_a_warning(monkeypatch, session):
    """Con las figuras activadas, la página escaneada es una figura: se vectoriza (y se puede describir con un modelo de visión) en lugar de fallar."""
    monkeypatch.setattr(get_settings(), "ocr_enabled", False)
    doc = register_upload(session, "escaneado.pdf", make_scanned_pdf(TEXT)).document
    assert process_one(session)
    session.expire_all()
    d = session.get(type(doc), doc.id)
    assert d.status == "processing" and d.meta["figures"]["count"] == 1
    assert any("No se extrajo texto del PDF" in w and "solo se indexaron sus figuras" in w for w in d.meta["warnings"])
