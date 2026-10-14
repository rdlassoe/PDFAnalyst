"""embeddings: umbral de similitud propio de cada perfil (cada modelo reparte sus similitudes de forma distinta)

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("embedding_profiles", sa.Column("min_similarity", sa.Float))
    op.create_check_constraint(
        "ck_embedding_profiles_min_similarity", "embedding_profiles", "min_similarity IS NULL OR (min_similarity > 0 AND min_similarity <= 1)"
    )


def downgrade() -> None:
    op.drop_constraint("ck_embedding_profiles_min_similarity", "embedding_profiles")
    op.drop_column("embedding_profiles", "min_similarity")
