"""Servicio local de embeddings (`app.embedding_server`): API OpenAI, validación, carga solo-texto, y cadena completa con el cliente."""

import json
import os
import threading
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


# ====================================================================================================== MEDIOS (imagen, audio, vídeo)
def _png(path: Path, size=(64, 48), color="red") -> Path:
    from PIL import Image

    Image.new("RGB", size, color).save(path)
    return path


def _wav(path: Path, seconds=3.0, rate=16000, channels=1, width=2) -> Path:
    import wave

    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes((np.sin(np.linspace(0, 200, n * channels)) * 8000).astype("<i2").tobytes() if width == 2 else b"\x00" * n * channels * width)
    return path


class MediaFake(FakeEngine):
    """Motor sin modelo para el endpoint de medios: registra qué recibe (tipos y formas) y devuelve vectores deterministas."""

    def __init__(self, cfg, *, fail=False):
        super().__init__(cfg, fail=fail)
        self.media_seen = []

    def embed_media(self, modality, inputs):
        if self.fail:
            raise RuntimeError("sin memoria para los codificadores")
        self.media_seen.append((modality, [x.size if hasattr(x, "size") and not isinstance(x, list) else (len(x), getattr(x[0], "size", None) if isinstance(x, list) else None) for x in inputs]))
        out = np.ones((len(inputs), 16), dtype=np.float32)
        out[:, 0] = np.arange(1, len(inputs) + 1)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


@pytest.fixture
def media(tmp_path):
    cfg = S.ServerConfig(media_root=str(tmp_path), media_max_batch=2, media_max_audio_s=5, media_max_frames=3, media_max_pixels=10_000, media_max_file_mb=1)
    eng = MediaFake(cfg)
    return TestClient(S.create_app(eng)), eng, tmp_path


def test_resolve_media_path_stays_inside_the_root(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    ok = _png(root / "a.png")
    outside = _png(tmp_path / "fuera.png")
    assert S.resolve_media_path(str(root), "a.png", 1) == ok.resolve()
    assert S.resolve_media_path(str(root), str(ok), 1) == ok.resolve()  # absoluta dentro de la raíz
    for bad in ("../fuera.png", str(outside), "/etc/passwd", "sub/../../fuera.png"):
        with pytest.raises(S.MediaError, match="fuera del directorio"):
            S.resolve_media_path(str(root), bad, 1)
    with pytest.raises(S.MediaError, match="no existe"):
        S.resolve_media_path(str(root), "nada.png", 1)
    (root / "carpeta").mkdir()
    with pytest.raises(S.MediaError, match="no existe"):  # un directorio no es un archivo de medio
        S.resolve_media_path(str(root), "carpeta", 1)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="sin enlaces simbólicos")
def test_symlink_pointing_outside_the_root_is_rejected(tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    outside = _png(tmp_path / "secreto.png")
    try:
        (root / "enlace.png").symlink_to(outside)
    except OSError:
        pytest.skip("no se pueden crear enlaces simbólicos aquí")
    with pytest.raises(S.MediaError, match="fuera del directorio"):
        S.resolve_media_path(str(root), "enlace.png", 1)


def test_resolve_media_path_enforces_the_size_limit(tmp_path):
    big = tmp_path / "grande.bin"
    big.write_bytes(b"0" * (2 * 1024 * 1024))
    with pytest.raises(S.MediaError, match="tamaño máximo"):
        S.resolve_media_path(str(tmp_path), "grande.bin", 1)


def test_load_image_checks_pixels_and_corruption(tmp_path):
    small = _png(tmp_path / "s.png", (50, 50))
    big = _png(tmp_path / "b.png", (200, 200))
    bad = tmp_path / "roto.png"
    bad.write_bytes(b"\x89PNG no es una imagen")
    assert S.load_image(small, 10_000).mode == "RGB"
    with pytest.raises(S.MediaError, match="demasiado grande"):
        S.load_image(big, 10_000)
    with pytest.raises(S.MediaError, match="No se pudo leer la imagen"):
        S.load_image(bad, 10_000)


def test_load_wav_slices_the_requested_window_and_validates_format(tmp_path):
    w = _wav(tmp_path / "a.wav", seconds=3.0)
    full = S.load_wav_16k_mono(w, None, None, 10)
    assert full.dtype == np.float32 and len(full) == 48000 and np.abs(full).max() <= 1.0
    win = S.load_wav_16k_mono(w, 1000, 2500, 10)
    assert len(win) == 24000  # 1,5 s a 16 kHz
    assert np.allclose(win, full[16000:40000])
    assert len(S.load_wav_16k_mono(w, 2000, 99999, 10)) == 16000  # el final se recorta al archivo
    with pytest.raises(S.MediaError, match="máximo"):
        S.load_wav_16k_mono(w, None, None, 2)
    with pytest.raises(S.MediaError, match="vacía"):
        S.load_wav_16k_mono(w, 5000, 6000, 10)
    with pytest.raises(S.MediaError, match="16 kHz"):
        S.load_wav_16k_mono(_wav(tmp_path / "44.wav", rate=44100), None, None, 10)
    with pytest.raises(S.MediaError, match="16 kHz"):
        S.load_wav_16k_mono(_wav(tmp_path / "est.wav", channels=2), None, None, 10)
    junk = tmp_path / "x.wav"
    junk.write_bytes(b"RIFFno es audio")
    with pytest.raises(S.MediaError, match="WAV no válido"):
        S.load_wav_16k_mono(junk, None, None, 10)


def test_media_endpoint_image_audio_video_happy_paths(media):
    c, eng, root = media
    _png(root / "a.png", (40, 30))
    _png(root / "b.png", (40, 30), "blue")
    _wav(root / "s.wav", 3.0)
    r = c.post("/v1/media/embeddings", json={"items": [{"modality": "image", "path": "a.png"}, {"modality": "image", "path": "b.png"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "image" and [d["index"] for d in body["data"]] == [0, 1] and body["model"] == "google/embeddinggemma-2"
    assert all(np.isclose(np.linalg.norm(d["embedding"]), 1.0) for d in body["data"])
    r = c.post("/v1/media/embeddings", json={"items": [{"modality": "audio", "path": "s.wav", "start_ms": 500, "end_ms": 2500}]})
    assert r.status_code == 200 and r.json()["modality"] == "audio" and eng.media_seen[-1][0] == "audio"
    assert eng.media_seen[-1][1] == [32000]  # el motor recibió el array de la ventana (2 s a 16 kHz = 32 000 muestras)
    r = c.post("/v1/media/embeddings", json={"items": [{"modality": "video", "paths": ["a.png", "b.png"]}]})
    assert r.status_code == 200 and r.json()["modality"] == "video" and eng.media_seen[-1][1][0][0] == 2  # 2 fotogramas


@pytest.mark.parametrize(
    "item, fragment",
    [
        ({"modality": "image"}, "Falta `path`"),
        ({"modality": "image", "path": "../../etc/passwd"}, "fuera del directorio"),
        ({"modality": "image", "path": "/etc/passwd"}, "fuera del directorio"),
        ({"modality": "image", "path": "no_existe.png"}, "no existe"),
        ({"modality": "video"}, "lista `paths`"),
        ({"modality": "video", "paths": ["a.png"] * 4}, "Demasiados fotogramas"),
        ({"modality": "audio", "path": "s.wav", "start_ms": 0, "end_ms": 9000}, "máximo"),
        ({"modality": "audio", "path": "a.png"}, "WAV no válido"),
        ({"modality": "image", "path": "gigante.png"}, "demasiado grande"),
    ],
)
def test_media_endpoint_rejects_bad_items_without_calling_the_model(media, item, fragment):
    c, eng, root = media
    _png(root / "a.png", (40, 30))
    _wav(root / "s.wav", 8.0)
    _png(root / "gigante.png", (300, 300))  # 90 000 píxeles > 10 000
    r = c.post("/v1/media/embeddings", json={"items": [item]})
    assert r.status_code == 400 and fragment in r.json()["error"]["message"], r.text
    assert eng.media_seen == []


def test_media_endpoint_batch_rules_and_schema(media):
    c, eng, root = media
    _png(root / "a.png")
    _wav(root / "s.wav")
    assert c.post("/v1/media/embeddings", json={"items": []}).status_code == 400
    r = c.post("/v1/media/embeddings", json={"items": [{"modality": "image", "path": "a.png"}] * 3})
    assert r.status_code == 400 and "máximo es 2" in r.json()["error"]["message"]
    r = c.post("/v1/media/embeddings", json={"items": [{"modality": "image", "path": "a.png"}, {"modality": "audio", "path": "s.wav"}]})
    assert r.status_code == 400 and "una modalidad" in r.json()["error"]["message"]
    assert c.post("/v1/media/embeddings", json={"items": [{"modality": "texto", "path": "a.png"}]}).status_code == 422  # modalidad desconocida
    assert eng.media_seen == []


def test_media_endpoint_requires_the_api_key_when_configured(tmp_path):
    _png(tmp_path / "a.png")
    c = TestClient(S.create_app(MediaFake(S.ServerConfig(media_root=str(tmp_path), api_key="secreta"))))
    body = {"items": [{"modality": "image", "path": "a.png"}]}
    assert c.post("/v1/media/embeddings", json=body).status_code == 401
    assert c.post("/v1/media/embeddings", json=body, headers={"Authorization": "Bearer secreta"}).status_code == 200


def test_media_model_failure_is_a_503_without_leaking_details(tmp_path):
    _png(tmp_path / "a.png")
    c = TestClient(S.create_app(MediaFake(S.ServerConfig(media_root=str(tmp_path)), fail=True)))
    r = c.post("/v1/media/embeddings", json={"items": [{"modality": "image", "path": "a.png"}]})
    assert r.status_code == 503 and "RuntimeError" in r.json()["error"]["message"] and "memoria" not in r.text


def test_media_config_defaults_and_env(monkeypatch):
    for k in [k for k in os.environ if k.startswith("EMBED_SERVER_")]:
        monkeypatch.delenv(k)
    c = S.ServerConfig.from_env()
    assert (c.media_root, c.media_idle_s, c.media_max_audio_s, c.media_max_frames, c.image_tokens, c.video_tokens) == ("/data/storage", 300.0, 300, 32, 280, 140)
    monkeypatch.setenv("EMBED_SERVER_MEDIA_IDLE_S", "60")
    monkeypatch.setenv("EMBED_SERVER_IMAGE_TOKENS", "140")
    monkeypatch.setenv("EMBED_SERVER_MEDIA_ROOT", "/otro")
    c = S.ServerConfig.from_env()
    assert (c.media_idle_s, c.image_tokens, c.media_root) == (60.0, 140, "/otro")


def test_health_reports_mode_and_media_settings():
    h = client().get("/health").json()
    assert h["mode"] is None and h["base_mode"] == "text" and h["media"]["idle_s"] == 300.0


# ---------------------------------------------------------------- modos de carga (sin modelo real)
class _ModeST(_RecorderST):
    """Registra con qué `config_kwargs` se carga cada vez y simula `model[0].processor`."""

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.kw = kw
        self.encoded = []

        class _P:
            image_processor = type("IP", (), {"max_soft_tokens": 280})()
            video_processor = type("VP", (), {"max_soft_tokens": 140})()
            image_seq_length = 280

        self._p = _P()

    def __getitem__(self, i):
        return type("M0", (), {"processor": self._p})()

    def encode(self, x, **kw):
        self.encoded.append((x, kw))
        n = len(x) if isinstance(x, list) else 1
        v = np.ones((n, 768), dtype=np.float32)
        return v if n > 1 else v[0]


def _mode_engine(monkeypatch, **cfg):
    import sentence_transformers

    _RecorderST.calls = []
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _ModeST)
    eng = S.Engine(S.ServerConfig(**cfg))
    eng._watchdog = threading.current_thread()  # vivo: así no se lanza el hilo de inactividad (volvería a cargar el modelo con el parche ya retirado)
    return eng


def _loaded_configs():
    return [c[1].get("config_kwargs") for c in _RecorderST.calls]


def test_modes_load_only_the_needed_encoders_and_free_the_previous_model(monkeypatch):
    eng = _mode_engine(monkeypatch)
    eng.encode(["hola"])
    assert eng.mode == "text" and _loaded_configs() == [{"vision_config": None, "audio_config": None}]
    eng.embed_media("image", ["IMG"])  # necesita el codificador de visión → recarga con texto+imagen
    assert eng.mode == "image" and _loaded_configs()[-1] == {"audio_config": None} and len(_RecorderST.calls) == 2
    eng.encode(["otra vez texto"])  # el texto funciona en cualquier modo: NO recarga
    eng.embed_media("video", [["F1", "F2"]])  # el vídeo usa el codificador de imagen: NO recarga
    assert len(_RecorderST.calls) == 2 and eng.mode == "image"
    eng.embed_media("audio", ["WAV"])  # cambio a texto+audio
    assert eng.mode == "audio" and _loaded_configs()[-1] == {"vision_config": None} and len(_RecorderST.calls) == 3


def test_full_base_mode_never_reloads_for_media(monkeypatch):
    eng = _mode_engine(monkeypatch, text_only=False)
    eng.embed_media("image", ["IMG"])
    eng.embed_media("audio", ["WAV"])
    assert eng.mode == "full" and len(_RecorderST.calls) == 1 and "config_kwargs" not in _RecorderST.calls[0][1]


def test_media_uses_the_configured_visual_token_budgets_and_no_prefix(monkeypatch):
    eng = _mode_engine(monkeypatch, image_tokens=140, video_tokens=70)
    eng.embed_media("image", ["IMG"])
    model = eng._model
    assert model._p.image_processor.max_soft_tokens == 140 and model._p.image_seq_length == 140
    assert model.encoded[-1][0] == ["IMG"]  # los medios se pasan tal cual, sin prefijos de texto ni diccionario
    eng.embed_media("video", [["F1", "F2"]])
    assert model._p.video_processor.max_soft_tokens == 70 and model.encoded[-1][0] == {"video": ["F1", "F2"]}


def test_idle_revert_returns_to_text_mode_and_frees_encoders(monkeypatch):
    eng = _mode_engine(monkeypatch, media_idle_s=100)
    now = [1000.0]
    eng._clock = lambda: now[0]
    eng.embed_media("image", ["IMG"])
    assert eng.mode == "image" and eng.revert_if_idle() is False  # recién usado
    now[0] += 99
    assert eng.revert_if_idle() is False
    now[0] += 2  # 101 s sin uso
    assert eng.revert_if_idle() is True
    assert eng.mode == "text" and _loaded_configs()[-1] == {"vision_config": None, "audio_config": None}
    assert eng.revert_if_idle() is False  # ya está en el modo base


def test_idle_revert_waits_if_an_inference_is_running_and_is_disabled_with_zero(monkeypatch):
    eng = _mode_engine(monkeypatch, media_idle_s=10)
    now = [0.0]
    eng._clock = lambda: now[0]
    eng.embed_media("image", ["IMG"])
    now[0] += 50
    with eng._encode_lock:  # inferencia en curso
        assert eng.revert_if_idle() is False and eng.mode == "image"
    eng2 = _mode_engine(monkeypatch, media_idle_s=0)
    eng2.embed_media("image", ["IMG"])
    eng2._clock = lambda: 10**9
    assert eng2.revert_if_idle() is False and eng2.mode == "image"  # 0 = nunca


# ---------------------------------------------------------------- con el modelo REAL (EmbeddingGemma 2 multimodal, CPU, float32)
def _shape_image(kind: str):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    if kind == "circulo":
        d.ellipse((220, 140, 420, 340), fill="red")
    elif kind == "cuadrado":
        d.rectangle((220, 140, 420, 340), fill="blue")
    else:
        for i, h in enumerate((60, 140, 220, 300)):
            d.rectangle((100 + i * 120, 400 - h, 180 + i * 120, 400), fill="green")
    return im


@pytest.fixture(scope="module")
def media_engine(tmp_path_factory):
    cfg = S.ServerConfig(model="google/embeddinggemma-2", media_root=str(tmp_path_factory.mktemp("media")), media_idle_s=100, image_tokens=140, video_tokens=70)
    return S.Engine(cfg)


@gemma
@pytest.mark.gemma2
def test_real_image_search_across_modalities_and_text_vectors_unchanged(gemma_engine, media_engine, tmp_path):
    texts = ["Los babilonios ahorraban una décima parte de sus ingresos", "política de contraseñas de la empresa"]
    before = gemma_engine.encode(texts)  # modo solo texto
    root = Path(media_engine.cfg.media_root)
    names = ["circulo", "cuadrado", "barras"]
    for n in names:
        _shape_image(n).save(root / f"{n}.png")
    app = TestClient(S.create_app(media_engine))
    r = app.post("/v1/media/embeddings", json={"items": [{"modality": "image", "path": f"{n}.png"} for n in names]})
    assert r.status_code == 200 and r.json()["mode"] == "image"
    V = np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)
    assert V.shape == (3, 768) and np.isfinite(V).all() and np.allclose(np.linalg.norm(V, axis=1), 1.0, atol=1e-4)
    # consultas en texto con el prefijo oficial: cada una encuentra su imagen; una consulta sin imagen correspondiente queda claramente por debajo
    from app.services.embedding_settings import GEMMA_QUERY_PREFIX

    Q = media_engine.encode([GEMMA_QUERY_PREFIX + q for q in ["un círculo rojo", "un cuadrado azul", "un gráfico de barras verdes", "un gato durmiendo"]])
    sim = Q @ V.T
    assert [int(np.argmax(sim[i])) for i in range(3)] == [0, 1, 2], sim.round(2)
    assert sim[3].max() < min(sim[0, 0], sim[1, 1], sim[2, 2]) - 0.05
    # M1: el vector de texto es el mismo con los codificadores de visión cargados que en modo solo texto
    after = media_engine.encode(texts)
    assert np.allclose(np.sum(before * after, axis=1), 1.0, atol=1e-3)


@gemma
@pytest.mark.gemma2
def test_real_audio_and_video_windows_and_idle_revert(media_engine):
    root = Path(media_engine.cfg.media_root)
    _wav(root / "a.wav", 6.0)
    for i in range(3):
        _shape_image(["circulo", "cuadrado", "barras"][i]).resize((160, 120)).save(root / f"f{i}.png")
    app = TestClient(S.create_app(media_engine))
    r = app.post("/v1/media/embeddings", json={"items": [{"modality": "audio", "path": "a.wav", "start_ms": 0, "end_ms": 4000}]})
    assert r.status_code == 200 and r.json()["mode"] == "audio"
    v = np.array(r.json()["data"][0]["embedding"])
    assert v.shape == (768,) and np.isfinite(v).all() and np.isclose(np.linalg.norm(v), 1.0, atol=1e-4)
    r = app.post("/v1/media/embeddings", json={"items": [{"modality": "video", "paths": ["f0.png", "f1.png", "f2.png"]}]})
    assert r.status_code == 200 and r.json()["mode"] == "image"  # el vídeo recarga con el codificador de imagen
    assert np.isfinite(r.json()["data"][0]["embedding"]).all()
    # sin uso durante media_idle_s el servicio vuelve al modo solo texto y sigue respondiendo consultas
    media_engine._clock = lambda: media_engine._last_media_use + 101
    assert media_engine.revert_if_idle() is True and media_engine.mode == "text"
    assert media_engine.encode(["hola"]).shape == (1, 768)
