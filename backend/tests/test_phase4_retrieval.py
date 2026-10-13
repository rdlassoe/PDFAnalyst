"""Fase 4 — búsqueda semántica, textual e híbrida, filtros y abstención (modelo de embeddings real)."""

import uuid

import pytest
from sqlalchemy import text

from app.models import Collection, CollectionDocument, Document
from app.services.ingestion import register_upload
from app.services.retrieval import fold, query_terms, retrieve, rrf_fuse, Hit
from app.worker import process_one

from . import corpus


@pytest.fixture(scope="module")
def indexed(embedder):
    """Indexa el corpus una sola vez para todo el módulo."""
    from app.db import SessionLocal
    from tests.conftest import _TABLES

    with SessionLocal() as s:
        s.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))
        s.commit()
        ids = {}
        for name in corpus.DOCS:
            ids[name] = register_upload(s, name, corpus.build(name)).document.id
        while process_one(s):
            pass
        statuses = {d.filename: d.status for d in s.query(Document)}
        assert set(statuses.values()) == {"ready"}, statuses
    return ids


@pytest.fixture
def db():
    from app.db import SessionLocal

    with SessionLocal() as s:
        yield s


def _top(r, n=1):
    return [(h.filename, h.page_start) for h in r.hits[:n]]


# ---------------------------------------------------------------- utilidades puras
def test_fold_and_terms():
    assert fold("Política DE Contraseñas") == "politica de contrasenas"
    assert query_terms("¿Cuál es el plazo para notificar incidentes?") == ["plazo", "notificar", "incidentes"]
    assert "sec" in query_terms("Procedimiento SEC-2024-017") and "017" in query_terms("Procedimiento SEC-2024-017")


def test_rrf_prefers_items_in_both_branches_and_is_deterministic():
    def h(i):
        return Hit(i, uuid.uuid4(), "f", 1, 1, "t")

    a, b, c, d = h(1), h(2), h(3), h(4)
    # vector: a, b, c  | texto: c, a, d  -> a (rangos 1 y 2) debe ganar; c (3 y 1) segundo
    fused = rrf_fuse([[a, b, c], [Hit(3, c.document_id, "f", 1, 1, "t"), Hit(1, a.document_id, "f", 1, 1, "t"), d]], k=60)
    assert [x.chunk_id for x in fused][:2] == [1, 3]
    assert fused[0].rrf == pytest.approx(1 / 61 + 1 / 62)


# ---------------------------------------------------------------- semántica (vector)
def test_semantic_paraphrase_finds_right_page(indexed, db):
    # «clave» y «renovar» no aparecen en el texto; el significado sí
    r = retrieve(db, "¿Cada cuánto tiempo debo renovar mi clave de acceso?", hybrid=False)
    assert r.hits, r
    assert _top(r) == [("politica_seguridad.pdf", 2)]
    assert r.hits[0].similarity > 0.8


def test_cross_language_query(indexed, db):
    r = retrieve(db, "How much did revenue increase in 2023?", hybrid=False)
    assert _top(r) == [("annual_report_2023.pdf", 1)]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("¿Cuántos días de vacaciones me corresponden?", ("manual_vacaciones.pdf", 1)),
        ("¿Qué ingredientes lleva la paella?", ("receta_paella.pdf", 1)),
        ("¿En cuántas horas hay que avisar de un incidente grave?", ("politica_seguridad.pdf", 3)),
        ("licencia por fallecimiento de un familiar", ("manual_vacaciones.pdf", 2)),
    ],
)
def test_top1_is_correct_for_known_questions(indexed, db, question, expected):
    assert _top(retrieve(db, question), 1) == [expected]


# ---------------------------------------------------------------- textual / híbrida
def test_exact_identifier_found_by_text_search(indexed, db):
    r = retrieve(db, "SEC-2024-017", hybrid=True)
    assert _top(r) == [("politica_seguridad.pdf", 2)]
    assert "keywords" in r.hits[0].relevant_by
    assert r.hits[0].fts_rank == 1


def test_article_number_lookup_hybrid(indexed, db):
    r = retrieve(db, "Article 45", hybrid=True)
    assert _top(r) == [("annual_report_2023.pdf", 2)]


def test_accent_insensitive_text_match(indexed, db):
    r = retrieve(db, "politica de contrasenas", hybrid=True)
    assert r.hits and r.hits[0].filename == "politica_seguridad.pdf"


def test_hybrid_flag_disables_text_branch(indexed, db):
    r = retrieve(db, "SEC-2024-017", hybrid=False)
    assert all(h.fts_rank is None for h in r.hits)


# ---------------------------------------------------------------- filtros
def test_document_filter_restricts_results(indexed, db):
    only_vac = [indexed["manual_vacaciones.pdf"]]
    r = retrieve(db, "contraseña de 14 caracteres", document_ids=only_vac)
    assert all(h.document_id in only_vac for h in r.hits)


def test_collection_filter(indexed, db):
    col = Collection(name=f"rrhh-{uuid.uuid4().hex[:6]}")
    db.add(col)
    db.flush()
    db.add(CollectionDocument(collection_id=col.id, document_id=indexed["manual_vacaciones.pdf"]))
    db.commit()
    r = retrieve(db, "¿Cuántos días de vacaciones tengo?", collection_id=col.id)
    assert _top(r) == [("manual_vacaciones.pdf", 1)]
    r2 = retrieve(db, "ingredientes de la paella", collection_id=col.id)
    assert all(h.filename == "manual_vacaciones.pdf" for h in r2.hits)


def test_empty_selection_reports_no_documents(indexed, db):
    r = retrieve(db, "cualquier cosa", document_ids=[uuid.uuid4()])
    assert r.hits == [] and r.abstain_reason == "no_documents"


# ---------------------------------------------------------------- abstención
@pytest.mark.parametrize(
    "question",
    [
        "¿Cuál es la capital de Mongolia?",
        "Explica cómo funciona la fotosíntesis en las plantas",
        "Who won the 1998 football world cup?",
    ],
)
def test_out_of_domain_questions_abstain(indexed, db, question):
    r = retrieve(db, question)
    assert r.hits == [], [(h.filename, h.similarity, h.relevant_by) for h in r.hits]
    assert r.abstain_reason == "no_relevant_chunks"


def test_does_not_pad_results_to_fixed_k(indexed, db):
    r = retrieve(db, "plazo para notificar incidentes graves", top_k=20)
    assert 1 <= len(r.hits) < 20
    # los 'relevantes' siempre tienen justificación explícita
    assert all(h.relevant_by for h in r.hits)


# ---------------------------------------------------------------- arnés de evaluación (app.evaluate)
def test_evaluation_harness_metrics_on_synthetic_corpus(indexed):
    from app.evaluate import evaluate

    qs = [
        {"question": "¿Cada cuánto tiempo debo renovar mi clave de acceso?", "document": "politica_seguridad.pdf", "pages": [2]},
        {"question": "SEC-2024-017", "document": "politica_seguridad.pdf", "pages": [2]},
        {"question": "¿En cuántas horas hay que avisar de un incidente grave?", "document": "politica_seguridad.pdf", "pages": [3]},
        {"question": "¿Cuántos días de vacaciones me corresponden?", "document": "manual_vacaciones.pdf", "pages": [1]},
        {"question": "licencia por fallecimiento de un familiar", "document": "manual_vacaciones.pdf", "pages": [2]},
        {"question": "¿Qué ingredientes lleva la paella?", "document": "receta_paella.pdf", "pages": [1]},
        {"question": "How much did revenue grow in 2023?", "document": "annual_report_2023.pdf", "pages": [1]},
        {"question": "Article 45", "document": "annual_report_2023.pdf", "pages": [2]},
        {"question": "¿Cuál es la capital de Mongolia?", "document": "-", "pages": [], "answerable": False},
        {"question": "Explica cómo funciona la fotosíntesis", "document": "-", "pages": [], "answerable": False},
    ]
    res = evaluate(qs, k=5)
    assert res["missing_documents"] == []
    h = res["modes"]["hybrid"]
    assert h["recall@1"] >= 0.85 and h["recall@5"] == 1.0 and h["mrr"] >= 0.9
    assert h["correct_abstention"] == 1.0 and h["answers_given"] == 1.0
    # el modo textual solo encuentra por palabras clave: debe resolver los identificadores exactos
    assert res["modes"]["fts"]["recall@5"] >= 0.5
    # la calibración del umbral debe poder separar preguntas con y sin respuesta en este corpus
    assert res["similarity"]["separable"] is True, res["similarity"]


# ---------------------------------------------------------------- ponderación por rareza (IDF)
def test_distinctive_terms_drop_common_words_but_keep_absent_ones(indexed, db):
    from app.services.retrieval import distinctive_terms, resolve_document_ids

    docs = resolve_document_ids(db)
    # «dias» aparece en 3+ fragmentos del mini-corpus (común); «sec» en uno; «zzzausente» en ninguno
    distinctive, matchable = distinctive_terms(db, ["dias", "sec", "zzzausente"], docs)
    assert "dias" not in distinctive
    assert distinctive == ["sec", "zzzausente"]  # el término raro ausente cuenta en el denominador
    assert matchable == ["sec"]  # pero no puede formar parte de la consulta


def test_all_common_terms_fall_back_to_present_terms(indexed, db):
    from app.services.retrieval import distinctive_terms, resolve_document_ids

    distinctive, matchable = distinctive_terms(db, ["dias"], resolve_document_ids(db))
    assert distinctive == matchable == ["dias"]


def test_missing_rare_term_blocks_keyword_relevance(indexed, db):
    """Muchos términos raros ausentes del corpus bajan la cobertura por debajo de 0.5: no hay respaldo textual."""
    r = retrieve(db, "SEC-2024-017 zzzaa qqqbb wwwcc xxxdd", mode="fts")
    assert r.hits == []  # solo 3 de 7 términos distintivos presentes (<50 %)
    ok = retrieve(db, "SEC-2024-017 zzzaa", mode="fts")  # 3 de 4 (75 %): sí cuenta
    assert ok.hits and ok.hits[0].page_start == 2
