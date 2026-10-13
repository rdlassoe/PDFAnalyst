"""Fase 2/3 — ingesta, deduplicación, idempotencia y embeddings reales."""

import pytest
from sqlalchemy import func, select, text

from app.models import ChunkEmbedding, Document, DocumentChunk, DocumentPage, ProcessingJob
from app.services import storage
from app.services.embeddings import EmbeddingError, Embedder, get_or_create_model_row
from app.services.ingestion import delete_document, register_upload, run_embed, run_ingest
from app.services.pdf_extract import PdfError, clean_text, extract_pdf, validate_pdf_bytes
from app.worker import claim_job, process_one

from .pdfs import make_pdf

PAGES = [
    "Introducción\n\nEste informe describe la política de seguridad de la información de la empresa Acme. "
    "Incluye controles de acceso, gestión de contraseñas y respuesta a incidentes.",
    "Capítulo 2\n\nLa contraseña debe tener al menos 14 caracteres y cambiarse cada 90 días. "
    "El código de la norma interna es SEC-2024-017.",
    "Capítulo 3\n\nLos incidentes graves deben notificarse al comité en menos de 24 horas.",
]


def _run_all_jobs(session):
    while process_one(session):
        pass


# ------------------------------ validación y seguridad -----------------------------------------
def test_validate_rejects_non_pdf_empty_oversized_and_corrupt():
    with pytest.raises(PdfError, match="vacío"):
        validate_pdf_bytes(b"", 10)
    with pytest.raises(PdfError, match="no es un PDF"):
        validate_pdf_bytes(b"MZ\x90\x00 esto es un exe", 1000)
    with pytest.raises(PdfError, match="supera"):
        validate_pdf_bytes(make_pdf(["hola"]), 10)
    with pytest.raises(PdfError, match="No se pudo abrir"):
        validate_pdf_bytes(b"%PDF-1.7\nbasura que no es un pdf", 10_000)


def test_validate_returns_page_count():
    assert validate_pdf_bytes(make_pdf(PAGES), 10**7) == 3


def test_storage_blocks_path_traversal(tmp_path):
    for bad in ("../../etc/passwd", "..\\x.pdf", "abc.pdf", "/abs/path.pdf", "a" * 32 + ".exe"):
        with pytest.raises(storage.StorageError):
            storage.resolve_storage_path(bad)
    ok = storage.new_storage_name()
    assert storage.resolve_storage_path(ok).parent == storage.storage_root()


def test_display_name_sanitized():
    assert storage.sanitize_display_name("../../etc/pass\x00wd.pdf") == "passwd.pdf"
    assert storage.sanitize_display_name("C:\\Users\\x\\informe.pdf") == "informe.pdf"
    assert storage.sanitize_display_name("") == "documento.pdf"


def test_clean_text_strips_nul_and_dehyphenates():
    assert clean_text("hola\x00 mun-\ndo") == "hola mundo"


# ------------------------------ extracción -----------------------------------------------------
def test_extract_pages_keep_page_numbers(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(make_pdf(PAGES))
    ex = extract_pdf(p)
    assert ex.page_count == 3
    assert [pg.number for pg in ex.pages] == [1, 2, 3]
    assert "SEC-2024-017" in ex.pages[1].text
    assert "24 horas" in ex.pages[2].text


# ------------------------------ ingesta con BD -------------------------------------------------
def test_upload_creates_document_and_job_and_dedups(session):
    data = make_pdf(PAGES)
    r1 = register_upload(session, "../informe.pdf", data)
    assert r1.created and r1.document.filename == "informe.pdf" and r1.document.page_count == 3
    assert storage.resolve_storage_path(r1.document.storage_name).exists()

    r2 = register_upload(session, "otro-nombre.pdf", data)
    assert not r2.created and r2.document.id == r1.document.id
    assert session.scalar(select(func.count()).select_from(Document)) == 1
    assert session.scalar(select(func.count()).select_from(ProcessingJob)) == 1


def test_invalid_upload_leaves_no_trace(session):
    with pytest.raises(PdfError):
        register_upload(session, "x.pdf", b"no soy pdf")
    assert session.scalar(select(func.count()).select_from(Document)) == 0
    assert list(storage.storage_root().glob("*.pdf")) == []


def test_full_pipeline_pages_chunks_embeddings(session, embedder):
    doc = register_upload(session, "informe.pdf", make_pdf(PAGES)).document
    _run_all_jobs(session)
    session.expire_all()

    doc = session.get(Document, doc.id)
    assert doc.status == "ready", doc.error
    assert session.scalar(select(func.count()).select_from(DocumentPage)) == 3
    n_chunks = session.scalar(select(func.count()).select_from(DocumentChunk))
    assert n_chunks >= 1
    model = get_or_create_model_row(session, embedder)
    assert model.dimension == 384
    assert session.scalar(select(func.count()).select_from(ChunkEmbedding)) == n_chunks
    dims = set(session.scalars(text("SELECT vector_dims(embedding) FROM chunk_embeddings")))
    assert dims == {384}
    jobs = session.scalars(select(ProcessingJob.status)).all()
    assert set(jobs) == {"done"}


def test_reprocessing_is_idempotent(session, embedder):
    doc = register_upload(session, "informe.pdf", make_pdf(PAGES)).document
    _run_all_jobs(session)
    before = (
        session.scalar(select(func.count()).select_from(DocumentChunk)),
        session.scalar(select(func.count()).select_from(ChunkEmbedding)),
    )
    # reprocesar desde cero: mismos conteos, sin duplicados
    from app.services.ingestion import enqueue_job

    enqueue_job(session, doc.id, "ingest")
    session.commit()
    _run_all_jobs(session)
    after = (
        session.scalar(select(func.count()).select_from(DocumentChunk)),
        session.scalar(select(func.count()).select_from(ChunkEmbedding)),
    )
    assert before == after
    assert session.get(Document, doc.id).status == "ready"


def test_embed_job_is_resumable_and_no_duplicates(session, embedder):
    doc = register_upload(session, "informe.pdf", make_pdf(PAGES)).document
    _run_all_jobs(session)
    from app.services.ingestion import enqueue_job

    # borrar parte de los vectores y reejecutar solo `embed`: se completan los que faltan
    session.execute(text("DELETE FROM chunk_embeddings WHERE chunk_id = (SELECT min(id) FROM document_chunks)"))
    session.commit()
    enqueue_job(session, doc.id, "embed")
    session.commit()
    _run_all_jobs(session)
    n_chunks = session.scalar(select(func.count()).select_from(DocumentChunk))
    assert session.scalar(select(func.count()).select_from(ChunkEmbedding)) == n_chunks


def test_pdf_without_text_fails_cleanly_and_is_retried_on_reupload(session):
    # PDF válido pero sin texto y sin OCR → fallo permanente con mensaje claro
    import pymupdf as fitz

    d = fitz.open()
    d.new_page()
    blank = d.tobytes()
    doc = register_upload(session, "vacio.pdf", blank).document
    _run_all_jobs(session)
    session.expire_all()
    doc = session.get(Document, doc.id)
    assert doc.status == "failed" and "texto" in doc.error
    job = session.scalar(select(ProcessingJob))
    assert job.status == "failed" and job.attempts == 1  # error permanente: no se reintenta en bucle
    # volver a subir lo reencola
    r = register_upload(session, "vacio.pdf", blank)
    assert not r.created and r.document.status == "uploaded"


def test_delete_document_removes_file_and_rows(session, embedder):
    doc = register_upload(session, "informe.pdf", make_pdf(PAGES)).document
    _run_all_jobs(session)
    path = storage.resolve_storage_path(doc.storage_name)
    assert path.exists()
    delete_document(session, session.get(Document, doc.id))
    assert not path.exists()
    for t in ("documents", "document_chunks", "chunk_embeddings", "processing_jobs"):
        assert session.scalar(text(f"SELECT count(*) FROM {t}")) == 0


# ------------------------------ embeddings: validación ------------------------------------------
class _BadDimEmbedder(Embedder):
    provider, name, query_prefix, passage_prefix = "test", "bad", "", ""

    def __init__(self, dim_reported, dim_real):
        self._r, self._real = dim_reported, dim_real

    @property
    def dimension(self):
        return self._r

    def _encode(self, texts):
        import numpy as np

        return np.ones((len(texts), self._real), dtype="float32")


def test_embedder_rejects_wrong_shape_and_nan():
    import numpy as np

    with pytest.raises(EmbeddingError, match="Forma"):
        _BadDimEmbedder(4, 3).embed_passages(["a"])

    class Nan(_BadDimEmbedder):
        def _encode(self, texts):
            return np.full((len(texts), 4), np.nan, dtype="float32")

    with pytest.raises(EmbeddingError, match="no finitos"):
        Nan(4, 4).embed_query("a")


def test_registered_model_dimension_conflict_is_blocked(session):
    get_or_create_model_row(session, _BadDimEmbedder(4, 4))
    session.commit()
    with pytest.raises(EmbeddingError, match="dimensión"):
        get_or_create_model_row(session, _BadDimEmbedder(8, 8))


def test_real_embeddings_are_normalized_and_semantic(embedder):
    import numpy as np

    v = embedder.embed_passages(["El gato duerme en el sofá.", "Un felino descansa en el sillón.", "Factura de electricidad de marzo."])
    assert v.shape == (3, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    sim_close, sim_far = float(v[0] @ v[1]), float(v[0] @ v[2])
    assert sim_close > sim_far
