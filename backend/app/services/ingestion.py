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
from . import media_ingest, storage
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
    """Valida el archivo (PDF o imagen, por su firma), evita duplicados por hash, lo guarda y encola su procesamiento."""
    s = get_settings()
    if data and not data.startswith(storage.PDF_MAGIC):
        if not storage.detect_image_type(data):
            raise PdfError("Formato no admitido: suba un PDF o una imagen PNG, JPEG o WebP")
        doc, created = media_ingest.register_image(session, filename, data, enqueue_job)
        return UploadResult(doc, created)
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
    storage_name, doc_id = doc.storage_name, doc.id
    session.delete(doc)
    session.commit()
    storage.delete_blob(storage_name)
    storage.delete_thumb(doc_id)
    storage.delete_figures(doc_id)


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
    if doc.media_type == "image":
        media_ingest.run_ingest_image(session, job, doc, enqueue_job)
        return

    extracted = extract_pdf(storage.resolve_storage_path(doc.storage_name))
    job.progress = 0.4
    session.commit()

    chunks = chunk_pages(
        ((p.number, p.text) for p in extracted.pages),
        target=s.chunk_target_chars,
        overlap=s.chunk_overlap_chars,
    )
    # Figuras (imágenes incrustadas y gráficos): OCR, descripción con el modelo de visión y vector. Antes de la transacción de reemplazo: es lo lento.
    fig_rows, fig_info, fig_warnings = media_ingest.process_pdf_figures(session, job, doc, storage.resolve_storage_path(doc.storage_name))
    if not chunks and not fig_rows:
        raise PdfError(
            "No se extrajo texto utilizable del PDF"
            + ("" if s.ocr_enabled else " (el OCR está deshabilitado)")
        )
    if not chunks:  # solo hay figuras: las páginas son imágenes y el OCR no leyó texto
        fig_warnings.append(
            "No se extrajo texto del PDF (sus páginas son imágenes y el OCR está desactivado o no leyó nada): solo se indexaron sus figuras. "
            "Sin un modelo de visión activo (Ajustes) o con el OCR habilitado no se podrá responder con su contenido."
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
        ]
        + [{"document_id": doc.id, "chunk_index": len(chunks) + i, **r} for i, r in enumerate(fig_rows)],
    )
    doc.page_count = extracted.page_count
    doc.title = extracted.title or doc.title
    doc.author = extracted.author or doc.author
    doc.meta = {
        **(doc.meta or {}),
        "ocr_pages": extracted.ocr_pages,
        "chunk_count": len(chunks) + len(fig_rows),
        "warnings": extracted.warnings + fig_warnings,
        "figures": fig_info,
    }
    enqueue_job(session, doc.id, "embed")
    session.commit()  # todo o nada
    log.info("Ingest %s: %d páginas, %d chunks, %d con OCR", doc.id, len(extracted.pages), len(chunks), extracted.ocr_pages)


def _embed_missing(session: Session, job: ProcessingJob, doc: Document, embedder: Embedder) -> int:
    """Calcula y guarda los embeddings que faltan de `doc` para el modelo de `embedder` (reanudable, sin duplicados)."""
    s = get_settings()
    model = get_or_create_model_row(session, embedder)
    session.commit()

    missing_all = list(
        session.execute(
            select(DocumentChunk.id, DocumentChunk.text, DocumentChunk.modality, DocumentChunk.meta)
            .where(
                DocumentChunk.document_id == doc.id,
                ~select(ChunkEmbedding.chunk_id)
                .where(ChunkEmbedding.chunk_id == DocumentChunk.id, ChunkEmbedding.model_id == model.id)
                .exists(),
            )
            .order_by(DocumentChunk.chunk_index)
        )
    )
    missing = [(cid, text) for cid, text, modality, _ in missing_all if modality == "text"]
    # Fragmentos «de medio» (vector de la imagen): solo se pueden embeber con un modelo que admita medios; con otro se omiten (no cuentan como pendientes)
    media_missing = [(cid, modality, meta) for cid, _, modality, meta in missing_all if modality != "text"] if model.supports_media else []
    total = max(len(missing) + len(media_missing), 1)
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
    done = len(missing)
    if media_missing:
        done += _embed_media_chunks(session, job, doc, embedder, model, media_missing, offset=len(missing), total=total)
    log.info("Embed %s: %d vectores nuevos (modelo %s)", doc.id, done, model.name)
    return done


MEDIA_BATCH = 4  # elementos de medio por petición al servicio (una imagen ≈1-3 s en CPU)


def _media_item(doc: Document, modality: str, meta: dict | None = None) -> dict:
    """Referencia al archivo (dentro del almacenamiento compartido con el servicio de embeddings): la figura extraída de un PDF o la imagen subida."""
    if modality == "image":
        fig = (meta or {}).get("figure")
        path = storage.figure_path(doc.id, fig) if fig is not None else storage.resolve_storage_path(doc.storage_name)
        return {"modality": "image", "path": str(path)}
    raise PdfError(f"Modalidad de medio aún no soportada: {modality}")


def _embed_media_chunks(session: Session, job: ProcessingJob, doc: Document, embedder: Embedder, model: EmbeddingModel,
                        media_missing: list[tuple[int, str, dict]], *, offset: int, total: int) -> int:
    for start in range(0, len(media_missing), MEDIA_BATCH):
        part = media_missing[start : start + MEDIA_BATCH]
        vectors = embedder.embed_media([_media_item(doc, modality, meta) for _, modality, meta in part])
        assert vectors.shape[1] == model.dimension
        stmt = pg_insert(ChunkEmbedding).values(
            [{"chunk_id": cid, "model_id": model.id, "embedding": vec.tolist()} for (cid, _, _), vec in zip(part, vectors)]
        )
        session.execute(stmt.on_conflict_do_nothing(index_elements=["chunk_id", "model_id"]))
        job.progress = 0.4 + 0.6 * (offset + min(start + MEDIA_BATCH, len(media_missing))) / total
        session.commit()
    return len(media_missing)


def run_embed(session: Session, job: ProcessingJob, embedder: Embedder | None = None) -> None:
    """Indexa el documento con el modelo ACTIVO (el perfil de embeddings elegido en Ajustes) en el momento de ejecutar.

    Si el modelo activo cambia mientras se procesa, se repite con el nuevo (máx. 3 veces) para no dejar el documento «listo»
    sin vectores del modelo vigente.
    """
    doc = session.get(Document, job.document_id)
    if doc is None:
        raise PdfError("Documento inexistente")
    ready = False
    for _ in range(3):
        emb = embedder or get_embedder(session)
        _embed_missing(session, job, doc, emb)
        if embedder is not None or get_embedder(session) is emb:
            ready = True
            break
    if ready:
        doc.status, doc.error = "ready", None
    session.commit()


def active_model_id(session: Session, embedder: Embedder | None = None) -> int:
    embedder = embedder or get_embedder(session)
    return get_or_create_model_row(session, embedder).id


def model_coverage(session: Session, model_id: int) -> dict[uuid.UUID, tuple[int, int]]:
    """Por documento con fragmentos: (nº de fragmentos, nº con vector de ese modelo).

    Solo cuentan los fragmentos que el modelo PUEDE embeber: si no admite medios, los fragmentos de imagen/audio/vídeo no se cuentan como pendientes.
    """
    supports_media = bool(session.scalar(select(EmbeddingModel.supports_media).where(EmbeddingModel.id == model_id)))
    stmt = (
        select(DocumentChunk.document_id, func.count(DocumentChunk.id), func.count(ChunkEmbedding.chunk_id))
        .join(
            ChunkEmbedding,
            (ChunkEmbedding.chunk_id == DocumentChunk.id) & (ChunkEmbedding.model_id == model_id),
            isouter=True,
        )
        .group_by(DocumentChunk.document_id)
    )
    if not supports_media:
        stmt = stmt.where(DocumentChunk.modality == "text")
    rows = session.execute(stmt)
    return {did: (n, e) for did, n, e in rows}


def requeue_for_model(session: Session, model_id: int) -> dict:
    """Tras activar un modelo: los documentos ya indexados con él quedan listos y el resto se encola.

    Devuelve {documents, already_indexed, queued}. Los documentos encolados pasan a «processing» (no consultables hasta
    terminar: no se mezclan vectores de modelos distintos).
    """
    already = queued = 0
    for did, (n, e) in model_coverage(session, model_id).items():
        if e >= n:
            already += 1
            # Listo salvo que tenga trabajo pendiente (p. ej. un `ingest` en cola).
            busy = session.scalar(
                select(ProcessingJob.id).where(
                    ProcessingJob.document_id == did, ProcessingJob.status.in_(("queued", "running"))
                ).limit(1)
            )
            if busy is None:
                session.execute(
                    Document.__table__.update()
                    .where(Document.id == did, Document.status == "processing")
                    .values(status="ready", error=None)
                )
        else:
            queued += 1
            session.execute(Document.__table__.update().where(Document.id == did).values(status="processing", error=None))
            enqueue_job(session, did, "embed", {"model_id": model_id})
    session.commit()
    return {"documents": already + queued, "already_indexed": already, "queued": queued}


def requeue_all_for_active_model(session: Session) -> int:
    """Tras cambiar de modelo de embeddings (CLI): encola `embed` de los documentos sin vectores del modelo activo."""
    model_id = active_model_id(session)
    session.commit()
    return requeue_for_model(session, model_id)["queued"]


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
