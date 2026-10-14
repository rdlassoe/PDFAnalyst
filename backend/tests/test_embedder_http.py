"""Adaptadores de embeddings por HTTP (OpenAI-compatible y Ollama): formato, reintentos, errores, consentimiento."""

import json

import httpx
import numpy as np
import pytest

from app.services import embeddings as E
from app.services.embeddings import (
    EmbedderConfig,
    EmbeddingError,
    OllamaEmbedder,
    OpenAICompatEmbedder,
    build_embedder,
    check_external_allowed,
    env_config,
)

URL = "http://localhost:9999/v1"
EXT = "https://api.example.com/v1"


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    """Sin caché de adaptadores HTTP entre pruebas (los locales pesados se conservan) y sin esperas reales."""
    for k in [k for k in E._CACHE if k[0] != "sentence-transformers"]:
        del E._CACHE[k]
    sleeps: list[float] = []
    monkeypatch.setattr(E.time, "sleep", lambda s: sleeps.append(s))
    yield sleeps
    for k in [k for k in E._CACHE if k[0] != "sentence-transformers"]:
        del E._CACHE[k]


def cfg(**kw) -> EmbedderConfig:
    base = dict(kind="openai-compatible", model="m", base_url=URL)
    base.update(kw)
    return EmbedderConfig(**base)


def serve(monkeypatch, handler):
    monkeypatch.setattr(E, "_transport_override", httpx.MockTransport(handler))


def ok_response(rows, *, reverse=False):
    data = [{"object": "embedding", "index": i, "embedding": r} for i, r in enumerate(rows)]
    return httpx.Response(200, json={"data": list(reversed(data)) if reverse else data})


# ---------------------------------------------------------------- formato OpenAI-compatible
def test_request_shape_headers_and_prefixes(monkeypatch):
    seen = []

    def handler(req: httpx.Request):
        seen.append((str(req.url), req.headers.get("authorization"), json.loads(req.content)))
        n = len(json.loads(req.content)["input"])
        return ok_response([[3.0, 4.0, 0.0, 0.0]] * n)

    serve(monkeypatch, handler)
    emb = build_embedder(cfg(api_key="sk-secreto", query_prefix="Q: ", passage_prefix="P: "))
    assert isinstance(emb, OpenAICompatEmbedder)
    v = emb.embed_query("hola")
    assert np.allclose(v, [0.6, 0.8, 0, 0]) and emb.dimension == 4  # normalizado y dimensión detectada
    emb.embed_passages(["uno", "dos"])
    url, auth, body = seen[0]
    assert url == "http://localhost:9999/v1/embeddings" and auth == "Bearer sk-secreto"
    assert body["model"] == "m" and body["encoding_format"] == "float" and "dimensions" not in body
    assert body["input"] == ["Q: hola"]  # prefijo de consulta
    assert seen[-1][2]["input"] == ["P: uno", "P: dos"]  # prefijo de documento


def test_results_are_ordered_by_index_not_by_arrival(monkeypatch):
    rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    serve(monkeypatch, lambda r: ok_response(rows[: len(json.loads(r.content)["input"])], reverse=True))
    out = build_embedder(cfg()).embed_passages(["a", "b", "c"])
    assert np.allclose(out[0], [1, 0]) and np.allclose(out[1], [0, 1]) and np.allclose(out[2], np.array([1, 1]) / np.sqrt(2))


def test_dimensions_are_requested_and_truncated_when_server_ignores_them(monkeypatch):
    bodies = []

    def handler(req):
        bodies.append(json.loads(req.content))
        return ok_response([[1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0]])  # el servidor ignora `dimensions` y devuelve 8

    serve(monkeypatch, handler)
    emb = build_embedder(cfg(dimensions=4))
    v = emb.embed_query("x")
    assert bodies[0]["dimensions"] == 4 and v.shape == (4,) and emb.dimension == 4
    assert np.isclose(np.linalg.norm(v), 1.0)  # re-normalizado tras truncar (Matryoshka)


# ---------------------------------------------------------------- reintentos y errores
def test_rate_limit_honours_retry_after_then_succeeds(monkeypatch, _isolation):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "7"}) if len(calls) < 3 else ok_response([[1.0, 0.0]])

    serve(monkeypatch, handler)
    assert build_embedder(cfg()).embed_query("x").shape == (2,)
    assert len(calls) == 4  # dos 429 + éxito de la consulta, y una llamada más para detectar la dimensión
    assert _isolation[:2] == [7.0, 7.0]


def test_transient_5xx_is_retried_and_gives_up_after_four_attempts(monkeypatch):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(503, text="x")

    serve(monkeypatch, handler)
    with pytest.raises(EmbeddingError, match="503") as e:
        build_embedder(cfg()).embed_query("x")
    assert len(calls) == 4 and not e.value.permanent


@pytest.mark.parametrize(
    "status, fragment",
    [(401, "credenciales"), (403, "credenciales"), (404, "no encontrados")],
)
def test_auth_and_notfound_are_permanent_and_not_retried_nor_leaking(monkeypatch, status, fragment):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(status, text="sk-secreto")

    serve(monkeypatch, handler)
    with pytest.raises(EmbeddingError, match=fragment) as e:
        build_embedder(cfg(api_key="sk-secreto")).embed_query("x")
    assert e.value.permanent and len(calls) == 1 and "sk-secreto" not in str(e.value)


def test_400_includes_provider_message_truncated(monkeypatch):
    serve(monkeypatch, lambda r: httpx.Response(400, json={"error": {"message": "input too long " + "x" * 500}}))
    with pytest.raises(EmbeddingError) as e:
        build_embedder(cfg()).embed_query("x")
    assert e.value.permanent and "input too long" in str(e.value) and len(str(e.value)) < 260


@pytest.mark.parametrize(
    "resp",
    [httpx.Response(200, text="no es json"), httpx.Response(200, json={"nope": 1}), httpx.Response(200, json={"data": [{"x": 1}]})],
)
def test_malformed_responses(monkeypatch, resp):
    serve(monkeypatch, lambda r: resp)
    with pytest.raises(EmbeddingError):
        build_embedder(cfg()).embed_query("x")


def test_vector_count_mismatch_and_nan(monkeypatch):
    serve(monkeypatch, lambda r: ok_response([[1.0, 0.0]]))  # 1 vector para 2 textos
    with pytest.raises(EmbeddingError, match="vectores para 2 textos"):
        build_embedder(cfg()).embed_passages(["a", "b"])
    serve(monkeypatch, lambda r: httpx.Response(200, content=b'{"data":[{"index":0,"embedding":[NaN, 1.0]}]}'))
    with pytest.raises(EmbeddingError):
        build_embedder(cfg(model="nan-model")).embed_query("x")


def test_network_failures(monkeypatch):
    def refuse(req):
        raise httpx.ConnectError("x")

    serve(monkeypatch, refuse)
    with pytest.raises(EmbeddingError, match="No se pudo conectar"):
        build_embedder(cfg(model="a")).embed_query("x")

    def slow(req):
        raise httpx.ReadTimeout("x")

    serve(monkeypatch, slow)
    with pytest.raises(EmbeddingError, match="tardó demasiado"):
        build_embedder(cfg(model="b")).embed_query("x")


# ---------------------------------------------------------------- consentimiento (servicio externo)
def test_external_provider_requires_explicit_consent():
    with pytest.raises(EmbeddingError, match="texto íntegro") as e:
        build_embedder(cfg(base_url=EXT))
    assert e.value.permanent and "externo" in str(e.value)
    assert build_embedder(cfg(base_url=EXT, allow_external=True)).external is True


def test_revoking_consent_blocks_even_a_cached_instance():
    assert build_embedder(cfg(base_url=EXT, allow_external=True))
    with pytest.raises(EmbeddingError, match="externo"):
        build_embedder(cfg(base_url=EXT, allow_external=False))  # misma configuración, sin permiso: bloqueada pese a la caché


@pytest.mark.parametrize(
    "url, external",
    [
        ("http://localhost:11434", False), ("http://192.168.1.5:8080/v1", False), ("http://10.0.0.7", False),
        ("https://api.example.com/v1", True), ("http://10.evil.com", True), ("http://8.8.8.8", True),
    ],
)
def test_external_detection_for_embedding_endpoints(url, external):
    assert cfg(base_url=url).external is external
    assert EmbedderConfig(kind="sentence-transformers", model="m").external is False  # el modelo local nunca es externo


def test_missing_base_url_and_unknown_kind():
    with pytest.raises(EmbeddingError, match="URL base"):
        build_embedder(cfg(base_url=""))
    with pytest.raises(EmbeddingError, match="desconocido"):
        build_embedder(cfg(kind="nada"))


def test_key_never_printed_and_cache_key_has_no_plaintext():
    c = cfg(api_key="sk-secreto")
    assert "sk-secreto" not in repr(c) and "sk-secreto" not in str(c.cache_key())


def test_same_config_gives_same_instance_and_different_key_a_different_one():
    a = build_embedder(cfg(model="x"))
    assert build_embedder(cfg(model="x")) is a
    assert build_embedder(cfg(model="x", api_key="otra")) is not a


# ---------------------------------------------------------------- Ollama
def test_ollama_embedder_payload_parse_and_model_missing(monkeypatch):
    bodies = []

    def handler(req):
        bodies.append((str(req.url), json.loads(req.content)))
        if json.loads(req.content)["model"] == "no-existe":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]] * len(json.loads(req.content)["input"])})

    serve(monkeypatch, handler)
    emb = build_embedder(EmbedderConfig(kind="ollama", model="embeddinggemma-2:270m", base_url="http://localhost:11434", query_prefix="Q|"))
    assert isinstance(emb, OllamaEmbedder)
    assert np.allclose(emb.embed_query("hola"), [0.6, 0.8])
    assert bodies[0] == ("http://localhost:11434/api/embed", {"model": "embeddinggemma-2:270m", "input": ["Q|hola"]})  # 1ª llamada: la consulta
    with pytest.raises(EmbeddingError, match="ollama pull") as e:
        build_embedder(EmbedderConfig(kind="ollama", model="no-existe", base_url="http://localhost:11434")).embed_query("x")
    assert e.value.permanent


def test_remote_ollama_is_external():
    with pytest.raises(EmbeddingError, match="externo"):
        build_embedder(EmbedderConfig(kind="ollama", model="m", base_url="https://ollama.example.com"))


# ---------------------------------------------------------------- entorno
def test_env_config_defaults_to_local_e5():
    c = env_config()
    assert c.kind == "sentence-transformers" and c.model == "intfloat/multilingual-e5-small" and not c.external
    assert c.query_prefix == "query: " and c.passage_prefix == "passage: " and c.label == "Entorno (.env)"


def test_check_external_allowed_message_names_the_host():
    with pytest.raises(EmbeddingError, match="api.example.com"):
        check_external_allowed(cfg(base_url=EXT))
