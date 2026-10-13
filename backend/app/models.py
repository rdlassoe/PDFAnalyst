"""Esquema relacional de PDF Analyst (ver docs/ARCHITECTURE.md §3)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# Estados de documento y de job (cadenas simples + CHECK, más fáciles de migrar que ENUM nativos)
DOC_STATUSES = ("uploaded", "processing", "ready", "failed")
JOB_STATUSES = ("queued", "running", "done", "failed")
JOB_KINDS = ("ingest", "embed")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(f"status IN {DOC_STATUSES}", name="ck_documents_status"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)  # nombre original, solo para mostrar
    storage_name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)  # <uuid>.pdf
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="uploaded", nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    pages: Mapped[list[DocumentPage]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentPage(Base):
    __tablename__ = "document_pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number", name="uq_pages_doc_page"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)  # base 1, como PDF.js
    text: Mapped[str] = mapped_column(Text, nullable=False)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    document: Mapped[Document] = relationship(back_populates="pages")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_idx"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunks_document", "document_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Columna generada: 'simple' + f_unaccent (sin tildes, sin stemming). Se concatenan los tokens originales
    # y los normalizados (separadores -/_.,;:@# -> espacio) porque el parser parte «ISO-27001» en 'iso'+'-27001'.
    tsv = mapped_column(
        TSVECTOR, Computed("to_tsvector('simple', f_unaccent(text)) || to_tsvector('simple', regexp_replace(f_unaccent(text), '[-/_.,;:@#]+', ' ', 'g'))", persisted=True)
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class EmbeddingModel(Base):
    __tablename__ = "embedding_models"
    __table_args__ = (
        UniqueConstraint("provider", "name", name="uq_embedding_models_provider_name"),
        CheckConstraint("distance IN ('cosine')", name="ck_embedding_models_distance"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    distance: Mapped[str] = mapped_column(String(16), default="cosine", nullable=False)
    query_prefix: Mapped[str] = mapped_column(Text, default="", nullable=False)
    passage_prefix: Mapped[str] = mapped_column(Text, default="", nullable=False)
    normalize: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = _created()


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    model_id: Mapped[int] = mapped_column(
        ForeignKey("embedding_models.id", ondelete="CASCADE"), primary_key=True
    )
    # Sin dimensión fija: cada modelo tiene la suya; los índices son parciales por modelo (D6).
    embedding = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = _created()


class Collection(Base):
    __tablename__ = "collections"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class CollectionDocument(Base):
    __tablename__ = "collection_documents"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    added_at: Mapped[datetime] = _created()


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    title: Mapped[str] = mapped_column(String(300), default="Nueva conversación", nullable=False)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.id",
    )


MESSAGE_STATUSES = ("ok", "abstained", "unverified", "error")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint("role IN ('user','assistant')", name="ck_messages_role"),
        CheckConstraint(f"status IN {MESSAGE_STATUSES}", name="ck_messages_status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="ok", nullable=False)
    # Filtros de la consulta, modelos usados, chunks recuperados, citas descartadas…
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = _created()

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    citations: Mapped[list[MessageCitation]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="MessageCitation.rank",
    )


class MessageCitation(Base):
    __tablename__ = "message_citations"
    __table_args__ = (UniqueConstraint("message_id", "label", name="uq_citation_msg_label"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    # SET NULL: si se borra el documento, la respuesta histórica se conserva pero la cita queda huérfana.
    chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL")
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    label: Mapped[str] = mapped_column(String(16), nullable=False)  # "C1"
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    snapshot: Mapped[str | None] = mapped_column(Text)  # texto del chunk al momento de responder

    message: Mapped[Message] = relationship(back_populates="citations")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        CheckConstraint(f"status IN {JOB_STATUSES}", name="ck_jobs_status"),
        CheckConstraint(f"kind IN {JOB_KINDS}", name="ck_jobs_kind"),
        Index("ix_jobs_status_created", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # 0..1
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)  # p. ej. model_id
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


LLM_KINDS = ("ollama", "openai-compatible")


class LLMProfile(Base):
    """Proveedor de generación configurado desde Ajustes. La clave API se guarda CIFRADA (services/secrets.py)."""

    __tablename__ = "llm_profiles"
    __table_args__ = (CheckConstraint(f"kind IN {LLM_KINDS}", name="ck_llm_profiles_kind"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    api_key_enc: Mapped[str | None] = mapped_column(Text)  # token Fernet; nunca texto plano
    allow_external: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class AppSetting(Base):
    """Ajustes no secretos clave/valor (p. ej. el proveedor LLM activo)."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
