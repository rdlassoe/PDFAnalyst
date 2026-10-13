"""Índices HNSW parciales por modelo de embeddings (D6, D8).

Un índice por modelo porque cada modelo tiene su propia dimensión y `vector` sin typmod no es indexable:
    CREATE INDEX ... USING hnsw ((embedding::vector(N)) vector_cosine_ops) WHERE model_id = M
Las consultas de retrieval.py usan exactamente esa expresión, así que el planner lo usa solo.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..models import EmbeddingModel


def index_name(model_id: int) -> str:
    return f"ix_chunk_emb_hnsw_m{int(model_id)}"


def create_hnsw_index(engine: Engine, model: EmbeddingModel, m: int = 16, ef_construction: int = 64) -> str:
    name = index_name(model.id)
    ddl = (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON chunk_embeddings "
        f"USING hnsw ((embedding::vector({int(model.dimension)})) vector_cosine_ops) "
        f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)}) WHERE model_id = {int(model.id)}"
    )
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:  # CONCURRENTLY no admite transacción
        conn.execute(text(ddl))
    return name


def drop_hnsw_index(engine: Engine, model_id: int) -> None:
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name(model_id)}"))


def index_status(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) "
                "FROM pg_indexes WHERE indexname LIKE 'ix_chunk_emb_hnsw_m%'"
            )
        )
        return [{"index": n, "size": s} for n, s in rows]
