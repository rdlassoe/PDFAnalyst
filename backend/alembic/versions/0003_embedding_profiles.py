"""ajustes: perfiles de proveedor de embeddings (clave cifrada)

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embedding_profiles",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("base_url", sa.Text, nullable=False),
        sa.Column("model", sa.String(256), nullable=False),
        sa.Column("api_key_enc", sa.Text),
        sa.Column("allow_external", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("query_prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("passage_prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("dimensions", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("kind IN ('ollama', 'openai-compatible')", name="ck_embedding_profiles_kind"),
    )


def downgrade() -> None:
    op.drop_table("embedding_profiles")
