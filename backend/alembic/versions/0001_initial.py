"""esquema inicial: documentos, chunks, embeddings (pgvector), conversaciones, jobs

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

DOC_STATUSES = ("uploaded", "processing", "ready", "failed")
JOB_STATUSES = ("queued", "running", "done", "failed")
JOB_KINDS = ("ingest", "embed")
MESSAGE_STATUSES = ("ok", "abstained", "unverified", "error")


def _now():
    return sa.text("now()")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    # unaccent() no es IMMUTABLE; este envoltorio (patrón estándar) permite usarlo en columnas generadas.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION f_unaccent(text) RETURNS text
        LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
        AS $$ SELECT public.unaccent('public.unaccent', $1) $$
        """
    )

    op.create_table(
        "documents",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("storage_name", sa.String(64), nullable=False, unique=True),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("page_count", sa.Integer),
        sa.Column("title", sa.Text),
        sa.Column("author", sa.Text),
        sa.Column("status", sa.String(16), nullable=False, server_default="uploaded"),
        sa.Column("error", sa.Text),
        sa.Column("metadata", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint(f"status IN {DOC_STATUSES}", name="ck_documents_status"),
    )

    op.create_table(
        "document_pages",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("document_id", pg.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("ocr_used", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("document_id", "page_number", name="uq_pages_doc_page"),
    )

    op.create_table(
        "document_chunks",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("document_id", pg.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("page_start", sa.Integer, nullable=False),
        sa.Column("page_end", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column(
            "tsv",
            pg.TSVECTOR,
            sa.Computed("to_tsvector('simple', f_unaccent(text)) || to_tsvector('simple', regexp_replace(f_unaccent(text), '[-/_.,;:@#]+', ' ', 'g'))", persisted=True),
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunks_doc_idx"),
    )
    op.create_index("ix_chunks_tsv", "document_chunks", ["tsv"], postgresql_using="gin")
    op.create_index("ix_chunks_document", "document_chunks", ["document_id"])

    op.create_table(
        "embedding_models",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        sa.Column("dimension", sa.Integer, nullable=False),
        sa.Column("distance", sa.String(16), nullable=False, server_default="cosine"),
        sa.Column("query_prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("passage_prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("normalize", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.UniqueConstraint("provider", "name", name="uq_embedding_models_provider_name"),
        sa.CheckConstraint("distance IN ('cosine')", name="ck_embedding_models_distance"),
    )

    op.create_table(
        "chunk_embeddings",
        sa.Column("chunk_id", sa.BigInteger, sa.ForeignKey("document_chunks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model_id", sa.Integer, sa.ForeignKey("embedding_models.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
    )
    op.create_index("ix_chunk_embeddings_model", "chunk_embeddings", ["model_id"])

    op.create_table(
        "collections",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
    )
    op.create_table(
        "collection_documents",
        sa.Column("collection_id", pg.UUID(as_uuid=True), sa.ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("document_id", pg.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
    )

    op.create_table(
        "conversations",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(300), nullable=False, server_default="Nueva conversación"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("conversation_id", pg.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="ok"),
        sa.Column("metadata", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.CheckConstraint("role IN ('user','assistant')", name="ck_messages_role"),
        sa.CheckConstraint(f"status IN {MESSAGE_STATUSES}", name="ck_messages_status"),
    )
    op.create_index("ix_messages_conversation", "messages", ["conversation_id", "id"])

    op.create_table(
        "message_citations",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("message_id", sa.BigInteger, sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.BigInteger, sa.ForeignKey("document_chunks.id", ondelete="SET NULL")),
        sa.Column("document_id", pg.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="SET NULL")),
        sa.Column("label", sa.String(16), nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("score", sa.Float),
        sa.Column("page_start", sa.Integer),
        sa.Column("page_end", sa.Integer),
        sa.Column("snapshot", sa.Text),
        sa.UniqueConstraint("message_id", "label", name="uq_citation_msg_label"),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("document_id", pg.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("progress", sa.Float, nullable=False, server_default="0"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text),
        sa.Column("payload", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("locked_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(f"status IN {JOB_STATUSES}", name="ck_jobs_status"),
        sa.CheckConstraint(f"kind IN {JOB_KINDS}", name="ck_jobs_kind"),
    )
    op.create_index("ix_jobs_status_created", "processing_jobs", ["status", "created_at"])


def downgrade() -> None:
    for table in (
        "processing_jobs",
        "message_citations",
        "messages",
        "conversations",
        "collection_documents",
        "collections",
        "chunk_embeddings",
        "embedding_models",
        "document_chunks",
        "document_pages",
        "documents",
    ):
        op.drop_table(table)
    op.execute("DROP FUNCTION IF EXISTS f_unaccent(text)")
