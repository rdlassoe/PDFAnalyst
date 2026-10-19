"""medios: imágenes (y base para audio/vídeo) — tipo de documento, modalidad de cada fragmento, marcas temporales y soporte de medios por modelo/perfil

Revision ID: 0006
Revises: 0005
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # documents: qué es (pdf | image | audio | video) y sus propiedades
    op.add_column("documents", sa.Column("media_type", sa.String(16), nullable=False, server_default="pdf"))
    op.add_column("documents", sa.Column("mime", sa.String(64)))
    op.add_column("documents", sa.Column("width", sa.Integer))
    op.add_column("documents", sa.Column("height", sa.Integer))
    op.add_column("documents", sa.Column("duration_ms", sa.Integer))
    op.create_check_constraint("ck_documents_media_type", "documents", "media_type IN ('pdf', 'image', 'audio', 'video')")

    # document_chunks: modalidad del fragmento, de dónde sale su contenido y (audio/vídeo) su tramo de tiempo
    op.add_column("document_chunks", sa.Column("modality", sa.String(16), nullable=False, server_default="text"))
    op.add_column("document_chunks", sa.Column("source", sa.String(16), nullable=False, server_default="pdf_text"))
    op.add_column("document_chunks", sa.Column("start_ms", sa.Integer))
    op.add_column("document_chunks", sa.Column("end_ms", sa.Integer))
    op.add_column("document_chunks", sa.Column("meta", pg.JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")))
    op.create_check_constraint("ck_chunks_modality", "document_chunks", "modality IN ('text', 'image', 'audio', 'video', 'page_image')")
    op.create_check_constraint("ck_chunks_source", "document_chunks", "source IN ('pdf_text', 'ocr', 'asr', 'caption', 'media')")
    op.create_check_constraint("ck_chunks_time_order", "document_chunks", "start_ms IS NULL OR end_ms IS NULL OR end_ms > start_ms")

    # message_citations: la cita de un medio guarda su modalidad y tramo
    op.add_column("message_citations", sa.Column("modality", sa.String(16)))
    op.add_column("message_citations", sa.Column("source", sa.String(16)))
    op.add_column("message_citations", sa.Column("start_ms", sa.Integer))
    op.add_column("message_citations", sa.Column("end_ms", sa.Integer))

    # ¿el modelo/perfil de embeddings sabe embeber imágenes, audio y vídeo? (servicio local de EmbeddingGemma 2)
    op.add_column("embedding_models", sa.Column("supports_media", sa.Boolean, nullable=False, server_default=sa.false()))
    op.add_column("embedding_profiles", sa.Column("supports_media", sa.Boolean, nullable=False, server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("embedding_profiles", "supports_media")
    op.drop_column("embedding_models", "supports_media")
    for c in ("end_ms", "start_ms", "source", "modality"):
        op.drop_column("message_citations", c)
    op.drop_constraint("ck_chunks_time_order", "document_chunks")
    op.drop_constraint("ck_chunks_source", "document_chunks")
    op.drop_constraint("ck_chunks_modality", "document_chunks")
    for c in ("meta", "end_ms", "start_ms", "source", "modality"):
        op.drop_column("document_chunks", c)
    op.drop_constraint("ck_documents_media_type", "documents")
    for c in ("duration_ms", "height", "width", "mime", "media_type"):
        op.drop_column("documents", c)
