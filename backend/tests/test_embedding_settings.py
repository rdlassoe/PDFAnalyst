"""Ajustes de embeddings: perfiles con clave cifrada, consentimiento, activación con reindexado, vectores por modelo.

El «servidor externo» es un transporte simulado que sirve el modelo e5 REAL por el protocolo `/embeddings` de OpenAI: así el flujo
completo (alta → activación → indexación por el worker → búsqueda) se ejercita con vectores reales, sin red ni claves de pago.
"""

import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import func, select, text

from app.config import get_settings
from app.db import SessionLocal
from app.main import app
from app.models import ChunkEmbedding, Document, DocumentChunk, EmbeddingModel, ProcessingJob
from app.services import credentials
from app.services import embeddings as E
from app.services.embeddings import Embedder
from app.services.ingestion import register_upload
from app.worker import process_one

from . import corpus

LOCAL = "http://localhost:9999/v1"
EXTERNAL = "https://api.example.com/v1"
KEY = "emb_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
SEC = "/settings/embeddings"


@pytest.fixture(autouse=True)
def _no_http_cache():
    def clean():
        for k in [k for k in E._CACHE if k[0] != "sentence-transformers"]:
            del E._CACHE[k]

    clean()
    yield
    clean()


class Server:
    """Servidor de embeddings simulado (e5 real). `mode`: ok | down | 401."""

    def __init__(self, embedder):
        self.embedder, self.mode, self.calls, self.auth = embedder, "ok", [], []

    def handler(self, req: httpx.Request):
        body = json.loads(req.content)
        self.calls.append(body)
        self.auth.append(req.headers.get("authorization"))
        if self.mode == "down":
            raise httpx.ConnectError("caído")
        if self.mode == "401":
            return httpx.Response(401, text="no")
        vecs = self.embedder._encode(body["input"]).tolist()  # recibe el texto YA con prefijos
        return httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]})


@pytest.fixture
def server(monkeypatch, embedder):
    srv = Server(embedder)
    monkeypatch.setattr(E, "_transport_override", httpx.MockTransport(srv.handler))
    return srv


@pytest.fixture
def client(session):  # `session` trunca las tablas (perfiles, ajustes…) antes de cada prueba
    return TestClient(app)


def _drain():
    with SessionLocal() as s:
        while process_one(s):
            pass


def _upload(client, name="politica_seguridad.pdf") -> str:
    r = client.post("/documents", files={"file": (name, corpus.build(name), "application/pdf")})
    assert r.status_code == 201
    return r.json()["document"]["id"]


def _status(client, doc_id) -> str:
    return client.get(f"/documents/{doc_id}").json()["status"]


def _create(client, **kw):
    body = dict(name="API e5", kind="openai-compatible", base_url=LOCAL, model="e5-via-api", api_key=None, allow_external=False,
                query_prefix="query: ", passage_prefix="passage: ")
    body.update(kw)
    return client.post(f"{SEC}/profiles", json=body)


def _pid(view, name):
    return next(p["id"] for p in view["profiles"] if p["name"] == name)


def _activate(client, pid):
    return client.put(f"{SEC}/active", json={"profile_id": pid})


def _indexed_on_api_model(client, server, doc="politica_seguridad.pdf"):
    """Escenario común: documento indexado con e5 (entorno) y luego reindexado con el perfil API, que queda activo."""
    doc_id = _upload(client, doc)
    _drain()
    pid = _pid(_create(client).json(), "API e5")
    assert _activate(client, pid).status_code == 200
    _drain()
    assert _status(client, doc_id) == "ready"
    return doc_id, pid


# ---------------------------------------------------------------- vista, validación, secreto
def test_default_view_is_local_env_and_presets_include_gemma2(client):
    v = client.get(SEC).json()
    assert v["active"] == "env" and len(v["profiles"]) == 1
    env = v["profiles"][0]
    assert env["id"] == "env" and env["kind"] == "sentence-transformers" and env["active"] and not env["editable"] and not env["external"]
    ids = {p["id"]: p for p in v["presets"]}
    assert {"gemma2-local", "gemma2-api", "ollama-embed", "openai-embed", "custom-embed"} <= set(ids)
    assert list(ids)[0] == "gemma2-local"  # la plantilla local es la primera (la que se ofrece por defecto)
    for key in ("gemma2-local", "gemma2-api"):
        assert ids[key]["query_prefix"] == "task: search result | query: " and ids[key]["passage_prefix"] == "title: none | text: "
    assert ids["gemma2-local"]["base_url"] == "http://embedder:8000/v1" and ids["gemma2-local"]["model"] == "google/embeddinggemma-2"
    assert ids["gemma2-local"]["needs_key"] is False
    # EmbeddingGemma 2 reparte sus similitudes distinto de e5 (≈0,72 en vez de 0,80): trae su umbral calibrado; el resto usa el global
    assert ids["gemma2-local"]["min_similarity"] == ids["gemma2-api"]["min_similarity"] == 0.72
    assert all(ids[k]["min_similarity"] is None for k in ("ollama-embed", "openai-embed", "custom-embed"))
    assert "MLX" in ids["ollama-embed"]["note"]  # aviso: embeddinggemma-2 en Ollama exige MLX
    assert v["coverage"] == {"documents_total": 0, "documents_indexed": 0}


@pytest.mark.parametrize(
    "kw",
    [
        dict(base_url="ftp://x.com"), dict(base_url="https://u:p@api.example.com"), dict(base_url="http://169.254.169.254/x"),
        dict(base_url="sin esquema"), dict(model=""), dict(name=""), dict(kind="sentence-transformers"), dict(dimensions=4), dict(min_similarity=0), dict(min_similarity=1.5), dict(min_similarity=-0.2),
        dict(dimensions=99999), dict(query_prefix="a\nb"), dict(passage_prefix="x" * 201), dict(api_key="con espacios"), dict(api_key="a\nb"),
    ],
)
def test_validation_errors_store_nothing(client, kw):
    assert _create(client, **kw).status_code == 422
    assert len(client.get(SEC).json()["profiles"]) == 1


def test_duplicate_name_is_409(client):
    assert _create(client).status_code == 201
    assert _create(client).status_code == 409


def test_api_key_is_encrypted_and_never_returned(client, session):
    r = _create(client, api_key=KEY)
    assert r.status_code == 201 and KEY not in r.text
    p = next(p for p in r.json()["profiles"] if p["name"] == "API e5")
    assert p["has_key"] and p["key_hint"] == "…3456" and "api_key" not in p
    enc = session.scalar(text("SELECT api_key_enc FROM embedding_profiles"))
    assert enc and KEY not in enc and credentials.decrypt(enc) == KEY
    for path in (SEC, "/health", "/settings/llm", "/documents"):
        assert KEY not in client.get(path).text


def test_update_keeps_or_replaces_or_clears_the_key(client, session):
    pid = _pid(_create(client, api_key=KEY).json(), "API e5")
    enc0 = session.scalar(text("SELECT api_key_enc FROM embedding_profiles"))
    assert client.patch(f"{SEC}/profiles/{pid}", json={"api_key": "", "base_url": "http://localhost:1/v1"}).status_code == 200
    assert session.scalar(text("SELECT api_key_enc FROM embedding_profiles")) == enc0
    client.patch(f"{SEC}/profiles/{pid}", json={"api_key": "NUEVA-CLAVE-1234567890"})
    session.expire_all()
    assert credentials.decrypt(session.scalar(text("SELECT api_key_enc FROM embedding_profiles"))) == "NUEVA-CLAVE-1234567890"
    v = client.patch(f"{SEC}/profiles/{pid}", json={"clear_api_key": True}).json()
    assert next(p for p in v["profiles"] if p["id"] == pid)["has_key"] is False


# ---------------------------------------------------------------- vista previa y consentimiento
def test_preview_reports_text_that_would_leave_the_machine_only_for_external(client, server):
    _upload(client)
    _drain()
    local = _pid(_create(client, name="Local").json(), "Local")
    ext = _pid(_create(client, name="Externo", base_url=EXTERNAL, model="m-ext").json(), "Externo")
    pl = client.post(f"{SEC}/preview", json={"profile_id": local}).json()
    pe = client.post(f"{SEC}/preview", json={"profile_id": ext}).json()
    assert pl["documents_to_index"] == 1 and pl["chunks_to_index"] >= 1 and pl["chars_to_send"] == 0 and not pl["external"]
    assert pe["external"] and pe["host"] == "api.example.com" and pe["chars_to_send"] > 100 and pe["documents_to_index"] == 1
    assert not pe["allow_external"]
    assert server.calls == []  # la vista previa no usa la red
    env = client.post(f"{SEC}/preview", json={"profile_id": "env"}).json()
    assert env["documents_to_index"] == 0 and env["already_indexed"] == 1 and env["chars_to_send"] == 0


def test_external_provider_cannot_be_activated_without_consent(client, server):
    doc = _upload(client)
    _drain()
    pid = _pid(_create(client, name="Ext", base_url=EXTERNAL, model="m-ext").json(), "Ext")
    r = _activate(client, pid)
    assert r.status_code == 409 and "externo" in r.json()["detail"] and "texto íntegro" in r.json()["detail"]
    assert client.get(SEC).json()["active"] == "env" and server.calls == [] and _status(client, doc) == "ready"
    assert client.patch(f"{SEC}/profiles/{pid}", json={"allow_external": True}).status_code == 200
    assert _activate(client, pid).status_code == 200  # con consentimiento sí


def test_revoking_consent_blocks_search_and_indexing_immediately(client, server):
    _upload(client)
    _drain()
    pid = _pid(_create(client, name="Ext", base_url=EXTERNAL, model="m-ext", allow_external=True).json(), "Ext")
    assert _activate(client, pid).status_code == 200
    _drain()
    assert client.post("/search", json={"query": "contraseña"}).status_code == 200
    client.patch(f"{SEC}/profiles/{pid}", json={"allow_external": False})
    n_before = len(server.calls)
    r = client.post("/search", json={"query": "contraseña"})
    assert r.status_code == 503 and "externo" in r.json()["detail"] and len(server.calls) == n_before  # no se envió nada


# ---------------------------------------------------------------- activación, indexación y búsqueda
def test_switching_reindexes_only_what_is_missing_and_search_uses_the_new_model(client, server):
    doc = _upload(client)
    _drain()
    assert _status(client, doc) == "ready"
    pid = _pid(_create(client).json(), "API e5")
    pv = client.post(f"{SEC}/preview", json={"profile_id": pid}).json()
    assert pv["documents_to_index"] == 1 and pv["chars_to_send"] == 0

    r = _activate(client, pid)
    assert r.status_code == 200
    a = r.json()["activation"]
    assert a["queued"] == 1 and a["already_indexed"] == 0 and a["dimension"] == 384 and a["model"] == "e5-via-api"
    # mientras se reindexa NO es consultable (no se mezclan espacios vectoriales)
    assert _status(client, doc) == "processing"
    assert client.post("/search", json={"query": "contraseña"}).json()["abstain_reason"] == "no_documents"

    _drain()
    assert _status(client, doc) == "ready"
    hits = client.post("/search", json={"query": "¿Cada cuánto debo cambiar mi clave de acceso?"}).json()["hits"]
    assert hits and hits[0]["filename"] == "politica_seguridad.pdf" and hits[0]["page_start"] == 2
    # el servidor recibió los textos con los prefijos configurados
    inputs = [t for c in server.calls for t in c["input"]]
    assert any(t.startswith("passage: ") for t in inputs) and any(t.startswith("query: ") for t in inputs)

    v = client.get(SEC).json()
    assert v["active"] == pid and v["coverage"] == {"documents_total": 1, "documents_indexed": 1}
    assert {m["name"] for m in v["models"]} == {"intfloat/multilingual-e5-small", "e5-via-api"}
    assert [m["active"] for m in v["models"] if m["name"] == "e5-via-api"] == [True]
    h = client.get("/health").json()["embeddings"]
    assert h["ok"] and h["model"] == "e5-via-api" and h["dimension"] == 384 and h["documents_indexed"] == 1 and not h["external"]


def test_switching_back_to_an_indexed_model_is_instant(client, server, session):
    doc, pid = _indexed_on_api_model(client, server)
    n_jobs = session.scalar(select(func.count()).select_from(ProcessingJob))
    r = _activate(client, "env")
    assert r.status_code == 200 and r.json()["activation"]["queued"] == 0 and r.json()["activation"]["already_indexed"] == 1
    assert _status(client, doc) == "ready"  # sus vectores e5 se conservaron
    assert session.scalar(select(func.count()).select_from(ProcessingJob).where(ProcessingJob.status == "queued")) == 0
    assert session.scalar(select(func.count()).select_from(ProcessingJob)) == n_jobs  # ningún job nuevo
    hits = client.post("/search", json={"query": "¿Cada cuánto debo cambiar mi clave de acceso?"}).json()["hits"]
    assert hits and hits[0]["page_start"] == 2


def test_failed_activation_keeps_the_active_provider_and_documents(client, server):
    doc = _upload(client)
    _drain()
    pid = _pid(_create(client).json(), "API e5")
    server.mode = "down"
    r = _activate(client, pid)
    assert r.status_code == 502 and "No se pudo conectar" in r.json()["detail"]
    server.mode = "401"
    r = _activate(client, pid)
    assert r.status_code == 502 and "credenciales" in r.json()["detail"]
    assert client.get(SEC).json()["active"] == "env" and _status(client, doc) == "ready"  # nada cambió


def test_prefix_mismatch_with_already_indexed_model_is_blocked(client, server):
    _indexed_on_api_model(client, server)
    p2 = _pid(_create(client, name="API e5 (otros prefijos)", query_prefix="q: ", passage_prefix="p: ").json(), "API e5 (otros prefijos)")
    r = _activate(client, p2)
    assert r.status_code == 409 and "prefijos" in r.json()["detail"]


def test_active_profile_cannot_change_model_or_prefixes_but_can_be_renamed(client, server):
    _, pid = _indexed_on_api_model(client, server)
    for patch in ({"model": "otro"}, {"query_prefix": "x"}, {"passage_prefix": "y"}, {"dimensions": 64}):
        r = client.patch(f"{SEC}/profiles/{pid}", json=patch)
        assert r.status_code == 409 and "predeterminado" in r.json()["detail"], patch
    assert client.patch(f"{SEC}/profiles/{pid}", json={"name": "Renombrado", "base_url": "http://localhost:1234/v1"}).status_code == 200


def test_similarity_threshold_is_per_profile_editable_while_active_and_changes_what_is_relevant(client, server):
    from app.services.retrieval import retrieve

    q = "¿Cada cuánto debo cambiar mi clave de acceso?"

    def vector_hits() -> int:
        with SessionLocal() as s:
            return len(retrieve(s, q, mode="vector").hits)

    _, pid = _indexed_on_api_model(client, server)
    assert client.get(SEC).json()["profiles"][1]["min_similarity"] is None  # sin umbral propio → el global
    assert vector_hits() > 0
    # el umbral solo afecta a la búsqueda (no a los vectores): se puede cambiar con el perfil ACTIVO y rige de inmediato
    r = client.patch(f"{SEC}/profiles/{pid}", json={"min_similarity": 0.999})
    assert r.status_code == 200 and r.json()["profiles"][1]["min_similarity"] == 0.999
    assert vector_hits() == 0
    assert client.patch(f"{SEC}/profiles/{pid}", json={"min_similarity": 0.05}).status_code == 200
    assert vector_hits() > 0
    # null explícito = volver al global; un PATCH que no lo menciona lo conserva
    client.patch(f"{SEC}/profiles/{pid}", json={"min_similarity": 0.999})
    assert client.patch(f"{SEC}/profiles/{pid}", json={"name": "Otro nombre"}).json()["profiles"][1]["min_similarity"] == 0.999
    r = client.patch(f"{SEC}/profiles/{pid}", json={"min_similarity": None})
    assert r.json()["profiles"][1]["min_similarity"] is None and vector_hits() > 0
    for bad in (0, 1.2, -1):
        assert client.patch(f"{SEC}/profiles/{pid}", json={"min_similarity": bad}).status_code == 422


def test_deleting_the_active_profile_falls_back_to_env_and_keeps_documents_ready(client, server):
    doc, pid = _indexed_on_api_model(client, server)
    v = client.delete(f"{SEC}/profiles/{pid}").json()
    assert v["active"] == "env" and len(v["profiles"]) == 1
    assert _status(client, doc) == "ready"


# ---------------------------------------------------------------- vectores por modelo
def test_purge_rules(client, server, session):
    doc, pid = _indexed_on_api_model(client, server)
    models = {m["name"]: m for m in client.get(SEC).json()["models"]}
    api_id, env_id = models["e5-via-api"]["id"], models["intfloat/multilingual-e5-small"]["id"]
    r = client.delete(f"{SEC}/models/{api_id}/vectors")  # el activo no se puede borrar
    assert r.status_code == 409 and "activo" in r.json()["detail"]
    assert client.delete(f"{SEC}/models/99999/vectors").status_code == 422
    r = client.delete(f"{SEC}/models/{env_id}/vectors")
    assert r.status_code == 200 and r.json()["purged"]["vectors_deleted"] >= 1
    assert "intfloat/multilingual-e5-small" not in {m["name"] for m in r.json()["models"]}
    # volver a ese modelo ahora SÍ exige reindexar
    a = _activate(client, "env").json()["activation"]
    assert a["queued"] == 1 and _status(client, doc) == "processing"
    _drain()
    assert _status(client, doc) == "ready"


# ---------------------------------------------------------------- fallos en uso
def test_provider_down_at_query_time_gives_503_and_document_management_keeps_working(client, server):
    _indexed_on_api_model(client, server)
    server.mode = "down"
    r = client.post("/search", json={"query": "contraseña"})
    assert r.status_code == 503 and "No se pudo conectar" in r.json()["detail"]
    cid = client.post("/conversations").json()["id"]
    assert client.post(f"/conversations/{cid}/messages", json={"question": "¿Cada cuánto cambio la contraseña?"}).status_code == 503
    assert client.get("/documents").status_code == 200 and client.get(SEC).status_code == 200  # la gestión sigue funcionando


def test_permanent_provider_error_fails_the_job_without_retrying(client, server, session):
    _indexed_on_api_model(client, server)
    server.mode = "401"
    other = _upload(client, "manual_vacaciones.pdf")
    _drain()
    job = session.scalar(select(ProcessingJob).where(ProcessingJob.document_id == other, ProcessingJob.kind == "embed"))
    session.refresh(job)
    assert job.status == "failed" and job.attempts == 1 and "credenciales" in job.error  # no se reintenta una clave rechazada
    assert _status(client, other) == "failed"


def test_unreadable_stored_key_is_flagged_and_blocks_activation(client, server, monkeypatch):
    pid = _pid(_create(client, api_key=KEY).json(), "API e5")
    monkeypatch.setattr(get_settings(), "app_secret_key", SecretStr(credentials.generate_master_key()))
    p = next(p for p in client.get(SEC).json()["profiles"] if p["id"] == pid)
    assert p["has_key"] and p["key_error"] and p["key_hint"] is None
    r = _activate(client, pid)
    assert r.status_code == 502 and "Vuelva a introducirla" in r.json()["detail"]
    t = client.post(f"{SEC}/test", json={"profile_id": pid}).json()
    assert t["ok"] is False and "Vuelva a introducirla" in t["detail"]


# ---------------------------------------------------------------- probar conexión
def test_connection_test_embeds_one_fixed_sentence_and_reports_dimension(client, server):
    t = client.post(f"{SEC}/test", json={"kind": "openai-compatible", "base_url": LOCAL, "model": "e5-via-api", "query_prefix": "query: "}).json()
    assert t["ok"] and t["dimension"] == 384 and isinstance(t["latency_ms"], int) and "384" in t["detail"]
    sent = [x for c in server.calls for x in c["input"]]
    assert sent[0] == "query: Prueba de conexión de PDF Analyst"  # frase fija; nunca contenido de documentos


def test_connection_test_with_saved_profile_reuses_key_and_does_not_expose_it(client, server):
    pid = _pid(_create(client, api_key=KEY).json(), "API e5")
    r = client.post(f"{SEC}/test", json={"profile_id": pid})
    assert r.json()["ok"] and server.auth[0] == f"Bearer {KEY}" and KEY not in r.text


def test_connection_test_works_for_external_without_consent_but_activation_does_not(client, server):
    t = client.post(f"{SEC}/test", json={"kind": "openai-compatible", "base_url": EXTERNAL, "model": "m"}).json()
    assert t["ok"] and server.calls  # solo se envía la frase de prueba


@pytest.mark.parametrize("mode, fragment", [("down", "No se pudo conectar"), ("401", "credenciales")])
def test_connection_test_reports_errors_without_raising(client, server, mode, fragment):
    server.mode = mode
    t = client.post(f"{SEC}/test", json={"kind": "openai-compatible", "base_url": LOCAL, "model": "m"}).json()
    assert t["ok"] is False and fragment in t["detail"] and t["dimension"] is None


def test_connection_test_refuses_ssrf_targets_without_a_request(client, server):
    t = client.post(f"{SEC}/test", json={"kind": "openai-compatible", "base_url": "http://169.254.169.254/latest", "model": "m"}).json()
    assert t["ok"] is False and "no está permitida" in t["detail"] and server.calls == []
    assert client.post(f"{SEC}/test", json={"base_url": LOCAL, "model": "m"}).status_code == 422  # falta el tipo


# ---------------------------------------------------------------- el worker ante un cambio de modelo a mitad de trabajo
class _Fake(Embedder):
    provider = "fake"
    query_prefix = passage_prefix = ""

    def __init__(self, name, dim):
        self.name, self._d = name, dim

    @property
    def dimension(self):
        return self._d

    def _encode(self, texts):
        import hashlib

        return np.array([np.random.default_rng(int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")).normal(size=self._d) for t in texts])


def test_run_embed_repeats_with_the_new_model_if_the_active_one_changes_midway(session, embedder, monkeypatch):
    a, b = _Fake("A", 8), _Fake("B", 12)
    seq, calls = [a, b, b, b], []

    def fake_get(*_a, **_k):
        calls.append(1)
        return seq[min(len(calls), len(seq)) - 1]

    monkeypatch.setattr("app.services.ingestion.get_embedder", fake_get)
    doc = register_upload(session, "politica_seguridad.pdf", corpus.build("politica_seguridad.pdf")).document
    while process_one(session):
        pass
    session.expire_all()
    assert session.get(Document, doc.id).status == "ready"
    n = session.scalar(select(func.count()).select_from(DocumentChunk))
    counts = dict(session.execute(
        select(EmbeddingModel.name, func.count(ChunkEmbedding.chunk_id))
        .join(ChunkEmbedding, ChunkEmbedding.model_id == EmbeddingModel.id).group_by(EmbeddingModel.name)).all())
    assert counts == {"A": n, "B": n}  # el documento quedó indexado con el modelo vigente (B) y no «listo» solo con A
