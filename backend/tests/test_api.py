"""API HTTP de extremo a extremo (sin LLM real): subida, procesamiento, búsqueda, chat, seguridad."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers import LLMProvider
from app.worker import process_one

from . import corpus
from .pdfs import make_pdf


@pytest.fixture
def client(session):  # `session` trunca tablas antes de cada test
    return TestClient(app)


def _drain():
    from app.db import SessionLocal

    with SessionLocal() as s:
        while process_one(s):
            pass


def _upload(client, name="politica_seguridad.pdf", data=None, filename=None):
    data = data if data is not None else corpus.build(name)
    return client.post("/documents", files={"file": (filename or name, data, "application/pdf")})


def test_health_reports_components_even_if_llm_down(client):
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 200
    assert body["database"]["ok"] and body["pgvector"]["ok"] and body["ok"]
    assert body["embeddings"]["model"]
    assert "generation" in body  # presente aunque Ollama no esté disponible


def test_upload_process_list_and_download(client, embedder):
    r = _upload(client)
    assert r.status_code == 201 and r.json()["created"] is True
    doc = r.json()["document"]
    assert doc["status"] == "uploaded"
    _drain()
    doc = client.get(f"/documents/{doc['id']}").json()
    assert doc["status"] == "ready" and doc["page_count"] == 3 and doc["chunk_count"] >= 1
    assert client.get("/documents").json()[0]["id"] == doc["id"]
    f = client.get(f"/documents/{doc['id']}/file")
    assert f.status_code == 200 and f.headers["content-type"] == "application/pdf" and f.content[:5] == b"%PDF-"
    assert f.headers["x-content-type-options"] == "nosniff"
    page = client.get(f"/documents/{doc['id']}/pages/2").json()
    assert "SEC-2024-017" in page["text"]
    assert client.get(f"/documents/{doc['id']}/pages/99").status_code == 404


def test_duplicate_upload_returns_same_document(client, embedder):
    a = _upload(client).json()
    b = _upload(client, filename="copia.pdf").json()
    assert b["created"] is False and a["document"]["id"] == b["document"]["id"]
    assert len(client.get("/documents").json()) == 1


def test_rejects_non_pdf_and_traversal_filename_is_harmless(client):
    r = client.post("/documents", files={"file": ("virus.pdf", b"MZ not a pdf", "application/pdf")})
    assert r.status_code == 422
    r = _upload(client, filename="../../../../windows/system32/evil.pdf")
    assert r.status_code == 201
    assert r.json()["document"]["filename"] == "evil.pdf"
    from app.services.storage import storage_root

    assert all(p.parent == storage_root() for p in storage_root().glob("**/*.pdf"))


def test_unknown_document_ids_are_404(client):
    import uuid

    u = uuid.uuid4()
    assert client.get(f"/documents/{u}").status_code == 404
    assert client.get(f"/documents/{u}/file").status_code == 404
    assert client.delete(f"/documents/{u}").status_code == 404
    assert client.get("/documents/no-es-uuid").status_code == 422


def test_search_endpoint_and_chunk_lookup(client, embedder):
    doc = _upload(client).json()["document"]
    _drain()
    s = client.post("/search", json={"query": "SEC-2024-017"}).json()
    assert s["hits"] and s["hits"][0]["page_start"] == 2 and s["hits"][0]["filename"] == "politica_seguridad.pdf"
    chunk = client.get(f"/chunks/{s['hits'][0]['chunk_id']}").json()
    assert chunk["document_id"] == doc["id"] and "SEC-2024-017" in chunk["text"]
    assert client.get("/chunks/99999999").status_code == 404
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_sql_injection_attempts_are_inert(client, embedder):
    _upload(client)
    _drain()
    for q in ["'; DROP TABLE documents; --", "x') OR 1=1 --", "a | b & !c <-> (", "\\x00"]:
        r = client.post("/search", json={"query": q})
        assert r.status_code == 200, (q, r.text)
    assert len(client.get("/documents").json()) == 1


def test_collections_crud_and_scoped_search(client, embedder):
    a = _upload(client, "politica_seguridad.pdf").json()["document"]
    b = _upload(client, "manual_vacaciones.pdf").json()["document"]
    _drain()
    col = client.post("/collections", json={"name": "RRHH"}).json()
    assert client.post("/collections", json={"name": "RRHH"}).status_code == 409
    assert client.put(f"/collections/{col['id']}/documents/{b['id']}").json()["document_ids"] == [b["id"]]
    hits = client.post("/search", json={"query": "contraseña cada 90 días", "collection_id": col["id"]}).json()["hits"]
    assert all(h["document_id"] == b["id"] for h in hits)
    assert client.delete(f"/collections/{col['id']}/documents/{b['id']}").json()["document_ids"] == []
    assert client.delete(f"/collections/{col['id']}").status_code == 204


def test_chat_flow_with_citations_and_listing(client, embedder, monkeypatch):
    doc = _upload(client).json()["document"]
    _drain()

    class Fixed(LLMProvider):
        name, model, external = "fixed", "t", False

        def chat(self, messages, *, temperature, max_tokens):
            return "Debe cambiarse cada 90 días [C1]."

        def health(self):
            return {"ok": True, "detail": ""}

    monkeypatch.setattr("app.services.answering.get_llm_provider", lambda *a, **k: Fixed())
    cid = client.post("/conversations").json()["id"]
    r = client.post(f"/conversations/{cid}/messages", json={"question": "¿Cada cuánto cambio la contraseña?"})
    assert r.status_code == 200
    m = r.json()
    assert m["status"] == "ok" and m["citations"][0]["label"] == "C1"
    c = m["citations"][0]
    assert c["document_id"] == doc["id"] and c["page_start"] == 2 and c["filename"] == "politica_seguridad.pdf"
    assert "90 días" in c["snippet"]
    # la cita permite abrir el fragmento exacto y el documento original
    assert client.get(f"/chunks/{c['chunk_id']}").json()["page_start"] == 2
    assert client.get(f"/documents/{c['document_id']}/file").status_code == 200
    msgs = client.get(f"/conversations/{cid}/messages").json()
    assert [x["role"] for x in msgs] == ["user", "assistant"] and msgs[1]["citations"]
    assert client.delete(f"/conversations/{cid}").status_code == 204
    assert client.get(f"/conversations/{cid}/messages").status_code == 404


def test_chat_survives_llm_outage_and_docs_keep_working(client, embedder, monkeypatch):
    _upload(client)
    _drain()
    from app.providers import LLMError

    class Down(LLMProvider):
        name, model, external = "down", "t", False

        def chat(self, *a, **k):
            raise LLMError("proveedor caído")

        def health(self):
            return {"ok": False, "detail": "caído"}

    monkeypatch.setattr("app.services.answering.get_llm_provider", lambda *a, **k: Down())
    cid = client.post("/conversations").json()["id"]
    r = client.post(f"/conversations/{cid}/messages", json={"question": "¿Cada cuánto cambio la contraseña?"})
    assert r.status_code == 200 and r.json()["status"] == "error"
    # la gestión documental sigue funcionando
    assert client.get("/documents").status_code == 200
    assert client.post("/search", json={"query": "contraseña"}).status_code == 200
    assert _upload(client, "manual_vacaciones.pdf").status_code == 201
