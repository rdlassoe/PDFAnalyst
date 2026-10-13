from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..models import Document, DocumentChunk, DocumentPage, ProcessingJob
from ..schemas import ChunkOut, DocumentOut, JobOut, UploadOut
from ..services import storage
from ..services.ingestion import delete_document, enqueue_job, register_upload
from ..services.pdf_extract import PdfError

router = APIRouter(tags=["documents"])


def doc_out(doc: Document) -> DocumentOut:
    meta = doc.meta or {}
    out = DocumentOut.model_validate(doc)
    out.chunk_count, out.ocr_pages = meta.get("chunk_count"), meta.get("ocr_pages")
    out.warnings = meta.get("warnings", [])
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
        media_type="application/pdf",
        headers={"Content-Disposition": "inline", "X-Content-Type-Options": "nosniff"},
    )


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
    return ChunkOut(
        id=c.id, document_id=c.document_id, filename=filename, page_start=c.page_start, page_end=c.page_end, text=c.text
    )
