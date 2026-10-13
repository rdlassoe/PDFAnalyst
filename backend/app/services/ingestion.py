"""Ingesta: validación → registro → (job ingest) extracción+chunking → (job embed) embeddings.

Idempotencia (ver docs/ARCHITECTURE.md §4):
- Subir el mismo PDF (mismo SHA-256) no crea un documento nuevo.
- `run_ingest` reemplaza páginas y chunks en UNA transacción: un fallo no deja datos a medias.
- `run_embed` solo calcula los embeddings que faltan para el modelo (reanudable, sin duplicados).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import (
    ChunkEmbedding,
    Document,
    DocumentChunk,
    DocumentPage,
    EmbeddingModel,
    ProcessingJob,
)
from . import storage
from .chunking import chunk_pages
from .embeddings import Embedder, get_embedder, get_or_create_model_row
from .pdf_extract import PdfError, extract_pdf, validate_pdf_bytes

log = logging.getLogger(__name__)


@dataclass
class UploadResult:
    document: Document
    created: bool  # False si ya existía (duplicado por hash)


def enqueue_job(session: Session, document_id: uuid.UUID, kind: str, payload: dict | None = None) -> ProcessingJob:
    """Encola un job salvo que ya haya uno activo del mismo tipo para el documento."""
    existing = session.scalar(
        select(ProcessingJob).where(
            ProcessingJob.document_id == document_id,
            ProcessingJob.kind == kind,
            ProcessingJob.status.in_(("queued", "running")),
        )
    )
    if existing:
        return existing
    job = ProcessingJob(document_id=document_id, kind=kind, payload=payload or {})
    session.add(job)
    session.flush()
    return job


def register_upload(session: Session, filename: str, data: bytes) -> UploadResult:
    """Valida el PDF, evita duplicados por hash, lo guarda y encola su procesamiento."""
    s = get_settings()
    page_count = validate_pdf_bytes(data, s.max_upload_bytes)  # lanza PdfError
    digest = storage.sha256_of(data)

    existing = session.scalar(select(Document).where(Document.sha256 == digest))
    if existing:
        if existing.status == "failed":  # reintento natural al volver a subirlo
            existing.status, existing.error = "uploaded", None
            enqueue_job(session, existing.id, "ingest")
            session.commit()
        return UploadResult(existing, created=False)

    storage_name = storage.new_storage_name()
    storage.write_pdf(storage_name, data)
    doc = Document(
        sha256=digest,
        filename=storage.sanitize_display_name(filename),
        storage_name=storage_name,
        size_bytes=len(data),
        page_count=page_count,
        status="uploaded",
    )
    try:
        session.add(doc)
        session.flush()
        enqueue_job(session, doc.id, "ingest")
        session.commit()
    except Exception:
        session.rollback()
        storage.delete_pdf(storage_name)  # no dejar huérfanos en disco
        raise
    return UploadResult(doc, created=True)


def delete_document(session: Session, doc: Document) -> None:
    storage_name = doc.storage_name
    session.delete(doc)
    session.commit()
    storage.delete_pdf(storage_name)


# ----------------------------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------------------------
def run_ingest(session: Session, job: ProcessingJob) -> None:
    doc = session.get(Document, job.document_id)
    if doc is None:
        raise PdfError("Documento inexistente")
    s = get_settings()

    doc.status, doc.error = "processing", None
    session.commit()

    extracted = extract_pdf(storage.resolve_storage_path(doc.storage_name))
    job.progress = 0.4
    session.commit()

    chunks = chunk_pages(
        ((p.number, p.text) for p in extracted.pages),
        target=s.chunk_target_chars,
        overlap=s.chunk_overlap_chars,
    )
    if not chunks:
        raise PdfError(
            "No se extrajo texto utilizable del PDF"
            + ("" if s.ocr_enabled else " (el OCR está deshabilitado)")
        )

    # Reemplazo atómico de páginas y chunks (los embeddings se eliminan por cascada).
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
    session.execute(delete(DocumentPage).where(DocumentPage.document_id == doc.id))
    session.execute(
        insert(DocumentPage),
        [
            {"document_id": doc.id, "page_number": p.number, "text": p.text, "ocr_used": p.ocr_used}
            for p in extracted.pages
        ],
    )
    session.execute(
        insert(DocumentChunk),
        [
            {
                "document_id": doc.id,
                "chunk_index": c.index,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "text": c.text,
            }
            for c in chunks
        ],
    )
    doc.page_count = extracted.page_count
    doc.title = extracted.title or doc.title
    doc.author = extracted.author or doc.author
    doc.meta = {
        **(doc.meta or {}),
        "ocr_pages": extracted.ocr_pages,
        "chunk_count": len(chunks),
        "warnings": extracted.warnings,
    }
    enqueue_job(session, doc.id, "embed")
    session.commit()  # todo o nada
    log.info("Ingest %s: %d páginas, %d chunks, %d con OCR", doc.id, len(extracted.pages), len(chunks), extracted.ocr_pages)


def run_embed(session: Session, job: ProcessingJob, embedder: Embedder | None = None) -> None:
    embedder = embedder or get_embedder()
    s = get_settings()
    doc = session.get(Document, job.document_id)
    if doc is None:
        raise PdfError("Documento inexistente")

    model = get_or_create_model_row(session, embedder)
    session.commit()

    missing = list(
        session.execute(
            select(DocumentChunk.id, DocumentChunk.text)
            .where(
                DocumentChunk.document_id == doc.id,
                ~select(ChunkEmbedding.chunk_id)
                .where(ChunkEmbedding.chunk_id == DocumentChunk.id, ChunkEmbedding.model_id == model.id)
                .exists(),
            )
            .order_by(DocumentChunk.chunk_index)
        )
    )
    total = max(len(missing), 1)
    batch = s.embedding_batch_size
    for start in range(0, len(missing), batch):
        part = missing[start : start + batch]
        vectors = embedder.embed_passages([text for _, text in part])
        assert vectors.shape[1] == model.dimension  # defensa adicional (ya validado en _check)
        stmt = pg_insert(ChunkEmbedding).values(
            [
                {"chunk_id": cid, "model_id": model.id, "embedding": vec.tolist()}
                for (cid, _), vec in zip(part, vectors)
            ]
        )
        session.execute(stmt.on_conflict_do_nothing(index_elements=["chunk_id", "model_id"]))
        job.progress = 0.4 + 0.6 * min(start + batch, len(missing)) / total
        session.commit()

    if model.id == active_model_id(session, embedder):
        doc.status, doc.error = "ready", None
    session.commit()
    log.info("Embed %s: %d vectores nuevos (modelo %s)", doc.id, len(missing), model.name)


def active_model_id(session: Session, embedder: Embedder | None = None) -> int:
    embedder = embedder or get_embedder()
    return get_or_create_model_row(session, embedder).id


def requeue_all_for_active_model(session: Session) -> int:
    """Tras cambiar de modelo de embeddings: encola `embed` para todos los documentos con chunks."""
    model_id = active_model_id(session)
    doc_ids = session.scalars(
        select(DocumentChunk.document_id).group_by(DocumentChunk.document_id)
    ).all()
    for did in doc_ids:
        session.execute(
            Document.__table__.update().where(Document.id == did).values(status="processing")
        )
        enqueue_job(session, did, "embed", {"model_id": model_id})
    session.commit()
    return len(doc_ids)


def embedding_counts(session: Session) -> list[dict]:
    rows = session.execute(
        select(EmbeddingModel.id, EmbeddingModel.provider, EmbeddingModel.name, EmbeddingModel.dimension,
               func.count(ChunkEmbedding.chunk_id))
        .join(ChunkEmbedding, ChunkEmbedding.model_id == EmbeddingModel.id, isouter=True)
        .group_by(EmbeddingModel.id)
        .order_by(EmbeddingModel.id)
    )
    return [
        {"id": i, "provider": p, "name": n, "dimension": d, "vectors": c} for i, p, n, d, c in rows
    ]
