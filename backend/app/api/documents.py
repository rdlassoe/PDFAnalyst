from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..models import Document, DocumentChunk, DocumentPage, ProcessingJob
from ..schemas import ChunkOut, DocumentOut, JobOut, UploadOut
from ..providers import LLMError
from ..services import media_ingest, storage, vision
from ..services.ingestion import delete_document, enqueue_job, register_upload
from ..services.pdf_extract import PdfError

router = APIRouter(tags=["documents"])


def doc_out(doc: Document) -> DocumentOut:
    meta = doc.meta or {}
    out = DocumentOut.model_validate(doc)
    out.chunk_count, out.ocr_pages = meta.get("chunk_count"), meta.get("ocr_pages")
    out.warnings = meta.get("warnings", [])
    fig = meta.get("figures") or {}
    out.figure_count, out.figures_described = fig.get("count"), fig.get("described")
    if doc.media_type == "image":
        out.thumbnail_url = f"/documents/{doc.id}/thumbnail"
        out.description = meta.get("description")
    return out


def get_doc_or_404(session: Session, doc_id: uuid.UUID) -> Document:
    doc = session.get(Document, doc_id)
    if doc is None:
        raise HTTPException(404, "Documento no encontrado")
    return doc


@router.post("/documents", response_model=UploadOut, status_code=201)
async def upload_document(file: UploadFile = File(...), session: Session = Depends(get_session)):
    limit = get_settings().max_upload_bytes
    data = await file.read(limit + 1)  # lee a lo sumo límite+1: evita cargar archivos gigantes enteros
    try:
        result = register_upload(session, file.filename or "documento.pdf", data)
    except PdfError as exc:
        raise HTTPException(422, str(exc)) from exc
    return UploadOut(document=doc_out(result.document), created=result.created)


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(session: Session = Depends(get_session)):
    docs = session.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [doc_out(d) for d in docs]


@router.get("/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: uuid.UUID, session: Session = Depends(get_session)):
    return doc_out(get_doc_or_404(session, doc_id))


@router.delete("/documents/{doc_id}", status_code=204)
def remove_document(doc_id: uuid.UUID, session: Session = Depends(get_session)):
    delete_document(session, get_doc_or_404(session, doc_id))


@router.post("/documents/{doc_id}/reprocess", response_model=JobOut, status_code=202)
def reprocess_document(doc_id: uuid.UUID, session: Session = Depends(get_session)):
    doc = get_doc_or_404(session, doc_id)
    doc.status, doc.error = "uploaded", None
    job = enqueue_job(session, doc.id, "ingest")
    session.commit()
    return job


@router.get("/documents/{doc_id}/jobs", response_model=list[JobOut])
def document_jobs(doc_id: uuid.UUID, session: Session = Depends(get_session)):
    get_doc_or_404(session, doc_id)
    return session.scalars(
        select(ProcessingJob).where(ProcessingJob.document_id == doc_id).order_by(ProcessingJob.id.desc()).limit(20)
    ).all()


@router.get("/documents/{doc_id}/file")
def download_file(doc_id: uuid.UUID, session: Session = Depends(get_session)):
    doc = get_doc_or_404(session, doc_id)
    try:
        path = storage.resolve_storage_path(doc.storage_name)
    except storage.StorageError as exc:
        raise HTTPException(500, "Archivo no disponible") from exc
    if not path.exists():
        raise HTTPException(404, "Archivo original no encontrado en el almacenamiento")
    return FileResponse(
        path,
        media_type=doc.mime or "application/pdf",  # el tipo sale de la firma validada al subir, nunca del cliente
        headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/documents/{doc_id}/thumbnail")
def thumbnail(doc_id: uuid.UUID, session: Session = Depends(get_session)):
    doc = get_doc_or_404(session, doc_id)
    if doc.media_type != "image":
        raise HTTPException(404, "Este documento no tiene miniatura")
    path = storage.thumb_path(doc.id)
    if not path.exists():
        raise HTTPException(404, "Miniatura no disponible (el documento aún se está procesando)")
    return FileResponse(path, media_type="image/jpeg", headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"})


class DescribeIn(BaseModel):
    question: str = Field(default="¿Qué ves en esta imagen?", min_length=1, max_length=2000)


@router.post("/documents/{doc_id}/describe")
def describe_document(doc_id: uuid.UUID, body: DescribeIn, session: Session = Depends(get_session)):
    """Pregunta al modelo de visión activo sobre esta imagen («¿qué ves?»). La respuesta es automática y puede equivocarse. Con un servicio
    externo, la imagen sale de la máquina (solo si el perfil tiene el consentimiento)."""
    doc = get_doc_or_404(session, doc_id)
    if doc.media_type != "image":
        raise HTTPException(422, "Solo se pueden hacer preguntas visuales sobre imágenes")
    try:
        res = vision.ask_about_images(session, body.question, [media_ingest.read_original(doc)])
    except vision.VisionDisabled as exc:
        raise HTTPException(409, str(exc)) from exc
    except vision.ImageError as exc:
        raise HTTPException(422, str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, f"El modelo de visión no pudo responder: {exc}") from exc
    return {"text": res.text, "model": res.model, "provider": res.provider, "external": res.external, "automatic": True}


@router.get("/documents/{doc_id}/pages/{page_number}")
def get_page_text(doc_id: uuid.UUID, page_number: int, session: Session = Depends(get_session)):
    get_doc_or_404(session, doc_id)
    page = session.scalar(
        select(DocumentPage).where(DocumentPage.document_id == doc_id, DocumentPage.page_number == page_number)
    )
    if page is None:
        raise HTTPException(404, "Página no encontrada")
    return {"page_number": page.page_number, "text": page.text, "ocr_used": page.ocr_used}


@router.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: int, session: Session = Depends(get_session)):
    row = session.execute(
        select(DocumentChunk, Document.filename)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id == chunk_id)
    ).first()
    if row is None:
        raise HTTPException(404, "Fragmento no encontrado")
    c, filename = row
    fig = (c.meta or {}).get("figure")
    return ChunkOut(
        id=c.id, document_id=c.document_id, filename=filename, page_start=c.page_start, page_end=c.page_end, text=c.text,
        modality=c.modality, source=c.source, figure_url=f"/chunks/{c.id}/figure" if fig is not None else None,
    )


@router.get("/chunks/{chunk_id}/figure")
def chunk_figure(chunk_id: int, session: Session = Depends(get_session)):
    """La figura (imagen incrustada o gráfico) a la que pertenece un fragmento de un PDF, tal como se ve en la página."""
    c = session.get(DocumentChunk, chunk_id)
    fig = (c.meta or {}).get("figure") if c is not None else None
    if c is None or fig is None:
        raise HTTPException(404, "Este fragmento no tiene figura")
    path = storage.figure_path(c.document_id, fig)
    if not path.exists():
        raise HTTPException(404, "Figura no disponible")
    return FileResponse(path, media_type="image/jpeg", headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"})
