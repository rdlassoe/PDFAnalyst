"""Fase 3 — convivencia de varios modelos de embeddings: sin mezclar vectores, reindexación, índices, purga."""

import hashlib

import numpy as np
import pytest
from sqlalchemy import func, select, text

from app import cli
from app.db import SessionLocal, engine
from app.models import ChunkEmbedding, Document, DocumentChunk, EmbeddingModel
from app.services import vector_index
from app.services.embeddings import Embedder, get_or_create_model_row
from app.services.ingestion import register_upload, requeue_all_for_active_model
from app.services.retrieval import retrieve
from app.worker import process_one

from . import corpus


class Fake16(Embedder):
    """SOLO PRUEBAS: vectores deterministas de 16 dims derivados del texto (mismo texto => mismo vector)."""

    provider, name, query_prefix, passage_prefix = "fake", "fake16", "", ""

    @property
    def dimension(self):
        return 16

    def _encode(self, texts):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")
            out.append(np.random.default_rng(seed).normal(size=16))
        return np.array(out, dtype="float32")


def _drain(s):
    while process_one(s):
        pass


@pytest.fixture
def ready_doc(session, embedder):
    doc = register_upload(session, "politica_seguridad.pdf", corpus.build("politica_seguridad.pdf")).document
    _drain(session)
    session.expire_all()
    assert session.get(Document, doc.id).status == "ready"
    return doc


def _counts(session):
    return dict(
        session.execute(
            select(EmbeddingModel.name, func.count(ChunkEmbedding.chunk_id))
            .join(ChunkEmbedding, ChunkEmbedding.model_id == EmbeddingModel.id, isouter=True)
            .group_by(EmbeddingModel.name)
        ).all()
    )


def test_reindex_after_model_change_keeps_old_vectors_and_queries_stay_isolated(session, ready_doc, embedder, monkeypatch):
    n_chunks = session.scalar(select(func.count()).select_from(DocumentChunk))
    fake = Fake16()
    monkeypatch.setattr("app.services.ingestion.get_embedder", lambda: fake)

    assert requeue_all_for_active_model(session) == 1
    session.expire_all()
    assert session.get(Document, ready_doc.id).status == "processing"  # no consultable hasta tener vectores nuevos
    r = retrieve(session, "contraseña", embedder=embedder)
    assert r.abstain_reason == "no_documents"

    _drain(session)
    session.expire_all()
    assert session.get(Document, ready_doc.id).status == "ready"
    counts = _counts(session)
    assert counts["intfloat/multilingual-e5-small"] == n_chunks and counts["fake16"] == n_chunks
    assert set(session.scalars(text("SELECT vector_dims(embedding) FROM chunk_embeddings WHERE model_id=2"))) == {16}

    # Consulta con el modelo nuevo: usa solo sus vectores (el texto idéntico del chunk da similitud ~1)
    chunk_text = session.scalar(select(DocumentChunk.text).where(DocumentChunk.chunk_index == 1))
    r = retrieve(session, chunk_text, embedder=fake, hybrid=False)
    assert r.hits[0].similarity == pytest.approx(1.0, abs=1e-4)
    # El modelo anterior sigue funcionando sobre sus propios vectores
    r2 = retrieve(session, "¿Cada cuánto debo cambiar la contraseña?", embedder=embedder, hybrid=False)
    assert r2.hits and r2.hits[0].page_start == 2


def test_hnsw_partial_index_create_use_and_drop(session, ready_doc, embedder):
    model = get_or_create_model_row(session, embedder)
    session.commit()
    session.close()
    name = vector_index.create_hnsw_index(engine, model)
    assert name in {i["index"] for i in vector_index.index_status(engine)}
    # idempotente
    vector_index.create_hnsw_index(engine, model)
    with SessionLocal() as s:
        s.execute(text("SET LOCAL enable_seqscan = off"))
        plan = "\n".join(
            r[0]
            for r in s.execute(
                text(
                    f"EXPLAIN SELECT chunk_id FROM chunk_embeddings WHERE model_id = {model.id} "
                    f"ORDER BY embedding::vector({model.dimension}) <=> CAST(:q AS vector({model.dimension})) LIMIT 3"
                ),
                {"q": "[" + ",".join(["0.05"] * model.dimension) + "]"},
            )
        )
        assert name in plan, plan
        # y la búsqueda real sigue devolviendo lo correcto con el índice presente
        r = retrieve(s, "¿Cada cuánto debo cambiar la contraseña?", embedder=embedder, hybrid=False)
        assert r.hits and r.hits[0].page_start == 2
    vector_index.drop_hnsw_index(engine, model.id)
    assert name not in {i["index"] for i in vector_index.index_status(engine)}


def test_purge_old_model_via_cli_and_refuse_active(session, ready_doc, embedder, monkeypatch, capsys):
    fake = Fake16()
    monkeypatch.setattr("app.services.ingestion.get_embedder", lambda: fake)
    requeue_all_for_active_model(session)
    _drain(session)
    monkeypatch.undo()  # el modelo activo vuelve a ser el real
    session.expire_all()
    old = session.scalar(select(EmbeddingModel.id).where(EmbeddingModel.name == "fake16"))
    active = session.scalar(select(EmbeddingModel.id).where(EmbeddingModel.name == embedder.name))

    assert cli.main(["purge-model", str(active)]) == 2  # el activo no se puede purgar
    assert cli.main(["purge-model", str(old)]) == 0
    session.expire_all()
    assert "fake16" not in _counts(session)
    assert _counts(session)[embedder.name] > 0
    assert cli.main(["models"]) == 0
    assert '"active": true' in capsys.readouterr().out
