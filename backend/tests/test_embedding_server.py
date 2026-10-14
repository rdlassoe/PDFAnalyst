"""Servicio local de embeddings (`app.embedding_server`): API OpenAI, validación, carga solo-texto, y cadena completa con el cliente."""

import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import embedding_server as S
from app.services import embeddings as E
from app.services.embeddings import EmbedderConfig, build_embedder


class FakeEngine(S.Engine):
    """Sin modelo real: vectores deterministas de 16 dimensiones (el primer componente codifica la longitud del texto)."""

    def __init__(self, cfg=None, *, fail=False):
        super().__init__(cfg or S.ServerConfig(max_batch=3))
        self.fail, self.seen = fail, []

    @property
    def dimension(self):
        return 16

    def encode(self, texts):
        if self.fail:
            raise RuntimeError("no hay memoria")
        self.seen.append(list(texts))
        out = np.ones((len(texts), 16), dtype=np.float32)
        out[:, 0] = [len(t) for t in texts]
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def client(engine=None):
    return TestClient(S.create_app(engine or FakeEngine()))


# ---------------------------------------------------------------- API
def test_openai_format_order_and_normalization():
    eng = FakeEngine()
    r = client(eng).post("/v1/embeddings", json={"model": "m", "input": ["a", "bbbb"]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list" and body["model"] == "m"
    assert [d["index"] for d in body["data"]] == [0, 1] and all(d["object"] == "embedding" for d in body["data"])
    assert all(np.isclose(np.linalg.norm(d["embedding"]), 1.0) for d in body["data"])
    assert eng.seen == [["a", "bbbb"]]


def test_single_string_input_and_default_model_name():
    r = client().post("/v1/embeddings", json={"input": "hola"})
    assert r.status_code == 200 and len(r.json()["data"]) == 1 and r.json()["model"] == "google/embeddinggemma-2"


def test_dimensions_truncates_and_renormalizes_matryoshka():
    r = client().post("/v1/embeddings", json={"input": ["abc"], "dimensions": 8})
    v = np.array(r.json()["data"][0]["embedding"])
    assert v.shape == (8,) and np.isclose(np.linalg.norm(v), 1.0)
    r = client().post("/v1/embeddings", json={"input": ["abc"], "dimensions": 64})  # mayor que la dimensión real: se ignora
    assert len(r.json()["data"][0]["embedding"]) == 16


@pytest.mark.parametrize(
    "body, fragment",
    [
        ({"input": []}, "vacío"),
        ({"input": ["a", "b", "c", "d"]}, "máximo es 3"),
        ({"input": ["ok", "   "]}, "no vacías"),
        ({"input": [""]}, "no vacías"),
    ],
)
def test_input_validation_returns_openai_style_errors(body, fragment):
    r = client().post("/v1/embeddings", json=body)
    assert r.status_code == 400 and fragment in r.json()["error"]["message"]


def test_wrong_types_are_rejected_by_schema():
    assert client().post("/v1/embeddings", json={"input": 123}).status_code == 422
    assert client().post("/v1/embeddings", json={"input": ["a"], "dimensions": 2}).status_code == 422


def test_api_key_is_enforced_when_configured():
    c = client(FakeEngine(S.ServerConfig(api_key="secreta")))
    assert c.post("/v1/embeddings", json={"input": ["a"]}).status_code == 401
    assert c.post("/v1/embeddings", json={"input": ["a"]}, headers={"Authorization": "Bearer mala"}).status_code == 401
    assert c.post("/v1/embeddings", json={"input": ["a"]}, headers={"Authorization": "Bearer secreta"}).status_code == 200
    assert "secreta" not in c.post("/v1/embeddings", json={"input": ["a"]}).text


def test_model_failure_is_a_503_without_leaking_details():
    r = client(FakeEngine(fail=True)).post("/v1/embeddings", json={"input": ["a"]})
    assert r.status_code == 503 and "no está disponible" in r.json()["error"]["message"] and "memoria" not in r.text


def test_health_does_not_force_loading_and_models_lists_the_model():
    c = client()
    h = c.get("/health").json()
    assert h["ok"] and h["loaded"] is False and h["text_only"] is True and h["dtype"] == "float32"
    assert c.get("/v1/models").json()["data"][0]["id"] == "google/embeddinggemma-2"


# ---------------------------------------------------------------- configuración y carga del modelo
def test_float16_is_refused(monkeypatch):
    monkeypatch.setenv("EMBED_SERVER_DTYPE", "float16")
    with pytest.raises(SystemExit, match="float16"):
        S.ServerConfig.from_env()


def test_config_defaults_and_env(monkeypatch):
    for k in [k for k in os.environ if k.startswith("EMBED_SERVER_")]:
        monkeypatch.delenv(k)
    c = S.ServerConfig.from_env()
    assert (c.model, c.text_only, c.dtype, c.device, c.max_seq_length, c.max_batch, c.preload) == ("google/embeddinggemma-2", True, "float32", "cpu", 2048, 64, False)
    monkeypatch.setenv("EMBED_SERVER_TEXT_ONLY", "false")
    monkeypatch.setenv("EMBED_SERVER_MAX_BATCH", "8")
    assert S.ServerConfig.from_env().text_only is False and S.ServerConfig.from_env().max_batch == 8


class _RecorderST:
    calls = []

    def __init__(self, name, **kw):
        type(self).calls.append((name, kw))
        self.max_seq_length = 8192

    def get_embedding_dimension(self):
        return 768


@pytest.mark.parametrize("text_only", [True, False])
def test_text_only_loading_omits_vision_and_audio_encoders(monkeypatch, text_only):
    import sentence_transformers

    _RecorderST.calls = []
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _RecorderST)
    eng = S.Engine(S.ServerConfig(text_only=text_only, max_seq_length=1024))
    assert eng.dimension == 768
    name, kw = _RecorderST.calls[0]
    assert name == "google/embeddinggemma-2" and kw["device"] == "cpu"
    assert str(kw["model_kwargs"]["torch_dtype"]) == "torch.float32"  # nunca float16
    assert ("config_kwargs" in kw) is text_only
    if text_only:
        assert kw["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert eng._model.max_seq_length == 1024  # tope de longitud para acotar memoria
    eng.dimension
    assert len(_RecorderST.calls) == 1  # se carga una sola vez


def test_first_load_does_not_deadlock_and_concurrent_callers_load_once(monkeypatch):
    """Regresión: registrar la dimensión dentro del cerrojo de carga la pedía otra vez (cerrojo no reentrante) y colgaba el servicio."""
    import threading

    import sentence_transformers

    _RecorderST.calls = []
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _RecorderST)
    eng = S.Engine(S.ServerConfig())
    results = []
    threads = [threading.Thread(target=lambda: results.append(eng.dimension), daemon=True) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=10)
    assert not any(th.is_alive() for th in threads), "la primera carga se quedó bloqueada"
    assert results == [768] * 4 and len(_RecorderST.calls) == 1


def test_load_failure_is_recorded_and_reported_by_health(monkeypatch):
    import sentence_transformers

    def boom(*a, **k):
        raise OSError("sin disco")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
    eng = S.Engine(S.ServerConfig())
    with pytest.raises(OSError):
        eng.dimension
    h = TestClient(S.create_app(eng)).get("/health").json()
    assert h["ok"] is False and "OSError" in h["error"] and h["loaded"] is False


# ---------------------------------------------------------------- cadena completa: cliente de la app → servidor → modelo REAL (e5)
def _transport_to(app_client: TestClient) -> httpx.MockTransport:
    def handler(req: httpx.Request) -> httpx.Response:
        r = app_client.request(req.method, req.url.path, content=req.content, headers={"content-type": "application/json", **{k: v for k, v in req.headers.items() if k == "authorization"}})
        return httpx.Response(r.status_code, headers={"content-type": "application/json"}, content=r.content)

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _clean_cache():
    for k in [k for k in E._CACHE if k[0] != "sentence-transformers"]:
        del E._CACHE[k]
    yield
    for k in [k for k in E._CACHE if k[0] != "sentence-transformers"]:
        del E._CACHE[k]


def test_client_adapter_against_the_server_with_a_real_model(monkeypatch):
    """El adaptador de la app habla con el servidor real (código de carga de sentence-transformers incluido) y obtiene vectores semánticos."""
    eng = S.Engine(S.ServerConfig(model="intfloat/multilingual-e5-small", text_only=False, max_batch=16))
    monkeypatch.setattr(E, "_transport_override", _transport_to(TestClient(S.create_app(eng))))
    emb = build_embedder(EmbedderConfig(kind="openai-compatible", model="intfloat/multilingual-e5-small", base_url="http://embedder:8000/v1",
                                        query_prefix="query: ", passage_prefix="passage: "))
    assert not emb.external and emb.dimension == 384
    docs = emb.embed_passages(["El gato duerme en el sofá.", "Factura de electricidad de marzo."])
    q = emb.embed_query("¿Dónde descansa el felino?")
    assert docs.shape == (2, 384) and float(q @ docs[0]) > float(q @ docs[1])  # lo semánticamente cercano puntúa más
    assert eng.loaded and eng.load_seconds is not None


# ---------------------------------------------------------------- EmbeddingGemma 2 REAL (se omite si el modelo no está en la caché local)
def _cached(model_id: str) -> bool:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return (home / "hub" / ("models--" + model_id.replace("/", "--"))).exists()


gemma = pytest.mark.skipif(not _cached("google/embeddinggemma-2"), reason="google/embeddinggemma-2 no está descargado (docker compose run --rm embedder python -m app.embedding_server --download)")


@pytest.fixture(scope="module")
def gemma_engine():
    return S.Engine(S.ServerConfig(model="google/embeddinggemma-2", text_only=True))


@gemma
@pytest.mark.gemma2
def test_real_embeddinggemma2_text_only(gemma_engine):
    v = gemma_engine.encode(["El gato duerme en el sofá.", "Un felino descansa en el sillón.", "Factura de electricidad de marzo."])
    assert gemma_engine.dimension == 768 and v.shape == (3, 768) and np.isfinite(v).all()
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert float(v[0] @ v[1]) > float(v[0] @ v[2]) + 0.1  # paráfrasis claramente más cercana que un texto no relacionado
    # modo solo texto: sin los codificadores de visión/audio (≈270M parámetros frente a ≈740M)
    n_params = sum(p.numel() for p in gemma_engine._model.parameters())
    assert 200e6 < n_params < 350e6, n_params


@gemma
@pytest.mark.gemma2
def test_official_prompts_match_the_ones_in_our_preset(gemma_engine):
    """Los prefijos de la plantilla de Ajustes son exactamente los prompts oficiales del modelo."""
    from app.services.embedding_settings import GEMMA_PASSAGE_PREFIX, GEMMA_QUERY_PREFIX

    gemma_engine._load()
    prompts = gemma_engine._model.prompts
    assert prompts["Retrieval-query"] == GEMMA_QUERY_PREFIX and prompts["Retrieval-document"] == GEMMA_PASSAGE_PREFIX


@gemma
@pytest.mark.gemma2
def test_real_embeddinggemma2_retrieval_with_official_prefixes(gemma_engine):
    """Con los prefijos oficiales, la pregunta recupera el documento correcto entre varios."""
    from app.services.embedding_settings import GEMMA_PASSAGE_PREFIX, GEMMA_QUERY_PREFIX

    docs = ["La contraseña debe tener al menos 14 caracteres y cambiarse cada 90 días.",
            "Todo empleado tiene derecho a 15 días hábiles de vacaciones al año.",
            "Ingredientes de la paella: arroz bomba, pollo, conejo y azafrán."]
    D = gemma_engine.encode([GEMMA_PASSAGE_PREFIX + d for d in docs])
    for question, expected in [("¿Cada cuánto debo renovar mi clave de acceso?", 0), ("¿Cuántos días de descanso anual me corresponden?", 1), ("¿Qué lleva la paella?", 2)]:
        q = gemma_engine.encode([GEMMA_QUERY_PREFIX + question])[0]
        assert int(np.argmax(D @ q)) == expected, question
