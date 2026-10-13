"""Fase 1 — aceptación: PostgreSQL + pgvector operativos, esquema migrado, vectores consultables."""

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DataError, IntegrityError

from app.models import ChunkEmbedding, Document, DocumentChunk, EmbeddingModel


def _doc(session, n=1):
    d = Document(sha256=f"{n:064x}", filename="x.pdf", storage_name=f"{n:032x}.pdf", size_bytes=1)
    session.add(d)
    session.flush()
    return d


def test_extensions_installed(session):
    names = set(session.scalars(text("SELECT extname FROM pg_extension")))
    assert {"vector", "unaccent"} <= names


def test_all_tables_exist(session):
    tables = set(session.scalars(text("SELECT tablename FROM pg_tables WHERE schemaname='public'")))
    expected = {
        "documents", "document_pages", "document_chunks", "embedding_models", "chunk_embeddings",
        "collections", "collection_documents", "conversations", "messages", "message_citations",
        "processing_jobs",
    }
    assert expected <= tables


def test_vector_roundtrip_and_cosine_order(session):
    d = _doc(session)
    m = EmbeddingModel(provider="test", name="m3", dimension=3)
    session.add(m)
    chunks = [DocumentChunk(document_id=d.id, chunk_index=i, page_start=1, page_end=1, text=f"t{i}") for i in range(3)]
    session.add_all(chunks)
    session.flush()
    vecs = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]]
    for c, v in zip(chunks, vecs):
        session.add(ChunkEmbedding(chunk_id=c.id, model_id=m.id, embedding=v))
    session.flush()

    rows = session.execute(
        text(
            "SELECT chunk_id, 1 - (embedding::vector(3) <=> CAST(:q AS vector(3))) AS sim "
            "FROM chunk_embeddings WHERE model_id = :m ORDER BY embedding::vector(3) <=> CAST(:q AS vector(3)) LIMIT 3"
        ),
        {"q": "[1,0,0]", "m": m.id},
    ).all()
    assert [r[0] for r in rows] == [chunks[0].id, chunks[1].id, chunks[2].id]
    assert rows[0][1] == pytest.approx(1.0, abs=1e-6)
    assert rows[2][1] == pytest.approx(0.0, abs=1e-6)


def test_dimension_mismatch_is_rejected_by_database(session):
    """Una consulta/vector con otra dimensión no puede mezclarse con la de un modelo (D6)."""
    d = _doc(session)
    m = EmbeddingModel(provider="test", name="m3", dimension=3)
    session.add(m)
    c = DocumentChunk(document_id=d.id, chunk_index=0, page_start=1, page_end=1, text="t")
    session.add(c)
    session.flush()
    session.add(ChunkEmbedding(chunk_id=c.id, model_id=m.id, embedding=[1, 0, 0]))
    session.flush()
    with pytest.raises(DataError):
        session.execute(
            text("SELECT embedding::vector(3) <=> CAST(:q AS vector(4)) FROM chunk_embeddings"),
            {"q": "[1,0,0,0]"},
        ).all()


def test_partial_hnsw_index_per_model_is_creatable_and_used(session):
    """El patrón de D6: índice HNSW parcial sobre embedding::vector(N) filtrado por model_id."""
    d = _doc(session)
    m = EmbeddingModel(provider="test", name="m8", dimension=8)
    session.add(m)
    chunks = [DocumentChunk(document_id=d.id, chunk_index=i, page_start=1, page_end=1, text=f"t{i}") for i in range(200)]
    session.add_all(chunks)
    session.flush()
    rng = np.random.default_rng(0)
    for c in chunks:
        v = rng.normal(size=8)
        session.add(ChunkEmbedding(chunk_id=c.id, model_id=m.id, embedding=(v / np.linalg.norm(v)).tolist()))
    session.flush()
    session.execute(text("SET LOCAL enable_seqscan = off"))
    session.execute(
        text(
            f"CREATE INDEX ix_test_hnsw ON chunk_embeddings USING hnsw ((embedding::vector(8)) vector_cosine_ops) "
            f"WHERE model_id = {int(m.id)}"
        )
    )
    plan = "\n".join(
        r[0]
        for r in session.execute(
            text(
                f"EXPLAIN SELECT chunk_id FROM chunk_embeddings WHERE model_id = {int(m.id)} "
                "ORDER BY embedding::vector(8) <=> CAST(:q AS vector(8)) LIMIT 5"
            ),
            {"q": "[1,0,0,0,0,0,0,0]"},
        )
    )
    assert "ix_test_hnsw" in plan, plan


def test_unaccent_fulltext_and_identifiers(session):
    d = _doc(session)
    session.add(DocumentChunk(document_id=d.id, chunk_index=0, page_start=1, page_end=1,
                              text="La política de contraseñas cumple el Art. 45 y la norma ISO-27001."))
    session.flush()
    def hits(q):
        return session.scalar(text("SELECT count(*) FROM document_chunks WHERE tsv @@ to_tsquery('simple', :q)"), {"q": q})
    assert hits("politica") == 1  # sin tilde encuentra «política»
    assert hits("contrasenas") == 1
    assert hits("45") == 1
    assert hits("27001") == 1 and hits("iso") == 1  # partes de un código compuesto


def test_cascade_delete_removes_pages_chunks_embeddings(session):
    d = _doc(session)
    m = EmbeddingModel(provider="test", name="m3", dimension=3)
    session.add(m)
    c = DocumentChunk(document_id=d.id, chunk_index=0, page_start=1, page_end=1, text="t")
    session.add(c)
    session.flush()
    session.add(ChunkEmbedding(chunk_id=c.id, model_id=m.id, embedding=[1, 2, 3]))
    session.flush()
    session.execute(text("DELETE FROM documents WHERE id=:i"), {"i": d.id})
    assert session.scalar(text("SELECT count(*) FROM chunk_embeddings")) == 0
    assert session.scalar(text("SELECT count(*) FROM document_chunks")) == 0


def test_duplicate_hash_rejected(session):
    _doc(session, 1)
    session.add(Document(sha256=f"{1:064x}", filename="y.pdf", storage_name=f"{2:032x}.pdf", size_bytes=1))
    with pytest.raises(IntegrityError):
        session.flush()
