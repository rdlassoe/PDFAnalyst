"""ajustes: perfiles del modelo de visión (descripción de imágenes; clave cifrada)

Revision ID: 0005
Revises: 0004
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "vision_profiles",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("base_url", sa.Text, nullable=False),
        sa.Column("model", sa.String(256), nullable=False),
        sa.Column("api_key_enc", sa.Text),
        sa.Column("allow_external", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("kind IN ('ollama', 'openai-compatible')", name="ck_vision_profiles_kind"),
    )


def downgrade() -> None:
    op.drop_table("vision_profiles")
