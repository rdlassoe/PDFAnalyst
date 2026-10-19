from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class JobOut(ORM):
    id: int
    kind: str
    status: str
    progress: float
    attempts: int
    error: str | None


class DocumentOut(ORM):
    id: uuid.UUID
    filename: str
    size_bytes: int
    page_count: int | None
    title: str | None
    author: str | None
    status: str
    error: str | None
    created_at: datetime
    media_type: str = "pdf"  # pdf | image | audio | video
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    figure_count: int | None = None  # figuras detectadas dentro de un PDF
    figures_described: int | None = None
    thumbnail_url: str | None = None
    description: str | None = None  # descripción automática (imágenes), si hay un modelo de visión
    chunk_count: int | None = None
    ocr_pages: int | None = None
    warnings: list[str] = []


class UploadOut(BaseModel):
    document: DocumentOut
    created: bool


class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None


class CollectionOut(ORM):
    id: uuid.UUID
    name: str
    description: str | None
    document_ids: list[uuid.UUID] = []


class SearchIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] | None = None
    collection_id: uuid.UUID | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    hybrid: bool | None = None


class HitOut(BaseModel):
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    page_start: int
    page_end: int
    text: str
    similarity: float | None
    vector_rank: int | None
    fts_rank: int | None
    coverage: float
    rrf: float
    relevant_by: list[str]
    modality: str = "text"
    source: str = "pdf_text"
    start_ms: int | None = None
    end_ms: int | None = None


class SearchOut(BaseModel):
    hits: list[HitOut]
    considered: int
    abstain_reason: str | None
    terms: list[str]


class ConversationOut(ORM):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[uuid.UUID] | None = None
    collection_id: uuid.UUID | None = None


class CitationOut(BaseModel):
    label: str
    chunk_id: int | None
    document_id: uuid.UUID | None
    filename: str | None
    page_start: int | None
    page_end: int | None
    score: float | None
    snippet: str | None
    modality: str | None = None
    source: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    status: str
    created_at: datetime
    citations: list[CitationOut] = []
    meta: dict = {}


class ChunkOut(BaseModel):
    id: int
    document_id: uuid.UUID
    filename: str
    page_start: int
    page_end: int
    text: str
    modality: str = "text"
    source: str = "pdf_text"
    figure_url: str | None = None  # el fragmento pertenece a una figura de un PDF
