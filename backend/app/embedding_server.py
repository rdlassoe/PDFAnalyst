"""Servicio LOCAL de embeddings con la API `/v1/embeddings` de OpenAI.

Sirve un modelo de sentence-transformers (por defecto **`google/embeddinggemma-2`**, Apache-2.0, cargado en modo SOLO TEXTO ≈270M parámetros)
desde un único proceso, para que la API y el worker no tengan cada uno su copia en memoria (importa con poca RAM). Todo ocurre en tu red de
Docker: ningún texto sale del equipo.

    python -m app.embedding_server              # arranca el servicio (puerto 8000)
    python -m app.embedding_server --download   # solo descarga el modelo a la caché y termina

Variables (EMBED_SERVER_*): MODEL, TEXT_ONLY (true), DTYPE (float32|bfloat16; **nunca float16**: EmbeddingGemma 2 no lo admite), DEVICE (cpu),
MAX_SEQ_LENGTH (2048), MAX_BATCH (64), THREADS (0 = por defecto de PyTorch), API_KEY (opcional: exige `Authorization: Bearer`), PRELOAD (false: se carga
en la primera petición; true: al arrancar, en segundo plano).

Los prefijos de consulta/documento (`task: search result | query: ` / `title: none | text: `) los añade el CLIENTE (perfil de embeddings): este servicio
embebe el texto tal cual lo recibe. Es independiente de la API de la aplicación y no accede a la base de datos.

MEDIOS (imagen, audio, vídeo): `POST /v1/media/embeddings` recibe **referencias a archivos** bajo EMBED_SERVER_MEDIA_ROOT (no el contenido por red). El servicio carga
los codificadores que hacen falta (modo `image` = texto+imagen, también para fotogramas de vídeo; modo `audio` = texto+audio) y, tras EMBED_SERVER_MEDIA_IDLE_S
segundos sin uso, **vuelve al modo base** (solo texto) para liberar memoria. Los medios no llevan prefijo. Siempre float32: en CPU sin AVX-512/AMX bfloat16 es ≈3-7× más lento.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("embedder")


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "si", "sí")


@dataclass(frozen=True)
class ServerConfig:
    model: str = "google/embeddinggemma-2"
    text_only: bool = True
    dtype: str = "float32"
    device: str = "cpu"
    max_seq_length: int = 2048
    max_batch: int = 64
    threads: int = 0
    api_key: str = ""
    preload: bool = False
    # medios
    media_root: str = "/data/storage"
    media_idle_s: float = 300.0  # sin uso de medios durante este tiempo → vuelve al modo base (0 = nunca)
    media_max_batch: int = 8
    media_max_audio_s: int = 300  # el modelo admite ≈327 s (25 tokens/s en 8192)
    media_max_frames: int = 32
    media_max_pixels: int = 40_000_000
    media_max_file_mb: int = 600
    image_tokens: int = 280  # presupuesto visual por imagen (70-1120); menos = más rápido
    video_tokens: int = 140  # por fotograma de vídeo

    @classmethod
    def from_env(cls) -> "ServerConfig":
        e = os.environ.get
        dtype = e("EMBED_SERVER_DTYPE", "float32").strip().lower()
        if dtype not in ("float32", "bfloat16"):
            raise SystemExit("EMBED_SERVER_DTYPE debe ser float32 o bfloat16 (EmbeddingGemma 2 no admite float16)")
        return cls(
            model=e("EMBED_SERVER_MODEL", cls.model),
            text_only=_bool("EMBED_SERVER_TEXT_ONLY", True),
            dtype=dtype,
            device=e("EMBED_SERVER_DEVICE", "cpu"),
            max_seq_length=int(e("EMBED_SERVER_MAX_SEQ_LENGTH", cls.max_seq_length)),
            max_batch=int(e("EMBED_SERVER_MAX_BATCH", cls.max_batch)),
            threads=int(e("EMBED_SERVER_THREADS", cls.threads)),
            api_key=e("EMBED_SERVER_API_KEY", ""),
            preload=_bool("EMBED_SERVER_PRELOAD", False),
            media_root=e("EMBED_SERVER_MEDIA_ROOT", cls.media_root),
            media_idle_s=float(e("EMBED_SERVER_MEDIA_IDLE_S", cls.media_idle_s)),
            media_max_batch=int(e("EMBED_SERVER_MEDIA_MAX_BATCH", cls.media_max_batch)),
            media_max_audio_s=int(e("EMBED_SERVER_MEDIA_MAX_AUDIO_S", cls.media_max_audio_s)),
            media_max_frames=int(e("EMBED_SERVER_MEDIA_MAX_FRAMES", cls.media_max_frames)),
            media_max_pixels=int(e("EMBED_SERVER_MEDIA_MAX_PIXELS", cls.media_max_pixels)),
            media_max_file_mb=int(e("EMBED_SERVER_MEDIA_MAX_FILE_MB", cls.media_max_file_mb)),
            image_tokens=int(e("EMBED_SERVER_IMAGE_TOKENS", cls.image_tokens)),
            video_tokens=int(e("EMBED_SERVER_VIDEO_TOKENS", cls.video_tokens)),
        )


# Codificadores que carga cada modo (el de texto siempre está) y qué modalidades cubre.
_MODE_CONFIG: dict[str, dict] = {
    "text": {"vision_config": None, "audio_config": None},
    "image": {"audio_config": None},
    "audio": {"vision_config": None},
    "full": {},
}
_COVERS = {"text": {"text"}, "image": {"text", "image"}, "audio": {"text", "audio"}, "full": {"text", "image", "audio"}}


def _trim_memory() -> None:
    """Devuelve a Linux la memoria liberada (glibc suele conservarla tras descargar un modelo grande)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class Engine:
    """Modelo cargado de forma perezosa y segura entre hilos. Reemplazable en pruebas (`encode`, `embed_media`, `dimension`)."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._model = None
        self.mode: str | None = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self._clock = time.monotonic
        self._last_media_use = 0.0
        self._watchdog: threading.Thread | None = None
        self.load_seconds: float | None = None
        self.load_error: str | None = None

    @property
    def base_mode(self) -> str:
        return "text" if self.cfg.text_only else "full"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _dim_of(model) -> int:
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(get_dim())

    def _load(self, modality: str | None = None):
        """Devuelve el modelo cargado. Con `modality` ("image" | "audio") garantiza que cubra esa modalidad, recargando con otro modo si hace falta
        (no se mantienen dos modelos a la vez: la memoria no da). Sin `modality` acepta lo que haya cargado (el texto funciona en todos los modos)."""
        m = self._model
        if m is not None and (modality is None or modality in _COVERS[self.mode or "text"]):  # camino rápido, sin cerrojo
            return m
        with self._load_lock:
            m = self._model
            if m is not None and (modality is None or modality in _COVERS[self.mode or "text"]):
                return m
            import torch
            from sentence_transformers import SentenceTransformer

            if m is not None:  # cambio de modo: se libera el modelo anterior ANTES de cargar el nuevo
                self._model, self.mode, m = None, None, None
                _trim_memory()
            target = self.base_mode if (modality is None or modality in _COVERS[self.base_mode]) else modality
            if self.cfg.threads > 0:
                torch.set_num_threads(self.cfg.threads)
            t0 = time.perf_counter()
            log.info("Cargando %s (modo %s, %s, %s)…", self.cfg.model, target, self.cfg.dtype, self.cfg.device)
            kwargs: dict = {"device": self.cfg.device, "model_kwargs": {"torch_dtype": getattr(torch, self.cfg.dtype)}}
            if _MODE_CONFIG[target]:  # omite los codificadores que el modo no necesita (texto ≈270M · +imagen ≈440M · +audio ≈570M · completo ≈740M)
                kwargs["config_kwargs"] = dict(_MODE_CONFIG[target])
            try:
                model = SentenceTransformer(self.cfg.model, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self.load_error = f"{exc.__class__.__name__}: {exc}"[:300]
                raise
            model.max_seq_length = min(self.cfg.max_seq_length, model.max_seq_length or self.cfg.max_seq_length)
            self._model, self.mode = model, target
            self.load_error = None
            self.load_seconds = round(time.perf_counter() - t0, 1)
            # OJO: no usar self.dimension aquí (volvería a pedir el cerrojo, que no es reentrante → interbloqueo).
            log.info("Modelo listo en %.1f s (modo %s, dimensión %s, longitud máx. %s)", self.load_seconds, target, self._dim_of(model), model.max_seq_length)
            return model

    @property
    def dimension(self) -> int:
        return self._dim_of(self._load())

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        with self._encode_lock:  # una inferencia a la vez: es CPU-bound y evita picos de memoria
            return np.asarray(
                model.encode(texts, batch_size=min(len(texts), 16), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32
            )

    # ------------------------------------------------------------------ medios
    def embed_media(self, modality: str, inputs: list) -> np.ndarray:
        """`inputs`: imágenes PIL (image) · arrays float32 mono 16 kHz (audio) · listas de fotogramas PIL (video). Devuelve (n, 768) normalizado."""
        need = "audio" if modality == "audio" else "image"  # los fotogramas de vídeo usan el codificador de imagen
        with self._encode_lock:
            model = self._load(need)
            self._touch_media()
            proc = model[0].processor
            if modality in ("image", "video"):
                tokens = self.cfg.image_tokens if modality == "image" else self.cfg.video_tokens
                target = proc.image_processor if modality == "image" else proc.video_processor
                target.max_soft_tokens = tokens
                if modality == "image" and hasattr(proc, "image_seq_length"):
                    proc.image_seq_length = tokens
            if modality == "image":
                vecs = model.encode(inputs, batch_size=1, normalize_embeddings=True, show_progress_bar=False)
            else:  # audio y vídeo: una entrada por llamada (un vector por ventana)
                key = None if modality == "audio" else "video"
                vecs = [model.encode(x if key is None else {key: x}, normalize_embeddings=True, show_progress_bar=False) for x in inputs]
            self._touch_media()
        arr = np.asarray(vecs, dtype=np.float32).reshape(len(inputs), -1)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)

    def _touch_media(self) -> None:
        self._last_media_use = self._clock()
        if self.cfg.media_idle_s > 0 and self.mode != self.base_mode and (self._watchdog is None or not self._watchdog.is_alive()):
            self._watchdog = threading.Thread(target=self._watch_idle, daemon=True, name="media-idle")
            self._watchdog.start()

    def _watch_idle(self) -> None:
        interval = max(1.0, min(30.0, self.cfg.media_idle_s / 2))
        while self.mode is not None and self.mode != self.base_mode:
            time.sleep(interval)
            try:
                self.revert_if_idle()
            except Exception:  # noqa: BLE001
                log.exception("No se pudo volver al modo base")

    def revert_if_idle(self) -> bool:
        """Si los codificadores de medios llevan `media_idle_s` sin usarse, vuelve al modo base (libera memoria). Devuelve si lo hizo."""
        if self.mode is None or self.mode == self.base_mode or self.cfg.media_idle_s <= 0:
            return False
        if self._clock() - self._last_media_use < self.cfg.media_idle_s:
            return False
        if not self._encode_lock.acquire(blocking=False):  # hay una inferencia en curso: se reintenta luego
            return False
        try:
            with self._load_lock:
                self._model, self.mode = None, None
            _trim_memory()
            log.info("Sin uso de medios durante %.0f s: se vuelve al modo %s", self.cfg.media_idle_s, self.base_mode)
            self._load()
        finally:
            self._encode_lock.release()
        return True


class EmbeddingsIn(BaseModel):
    model: str = Field(default="", max_length=256)
    input: list[str] | str
    encoding_format: str | None = None
    dimensions: int | None = Field(default=None, ge=8, le=8192)


class MediaItem(BaseModel):
    modality: Literal["image", "audio", "video"]
    path: str | None = Field(default=None, max_length=1024)  # imagen o audio (WAV PCM 16 kHz mono 16 bits)
    paths: list[str] | None = Field(default=None, max_length=256)  # vídeo: fotogramas ya extraídos (imágenes), en orden
    start_ms: int | None = Field(default=None, ge=0)  # audio: ventana dentro del archivo
    end_ms: int | None = Field(default=None, gt=0)


class MediaIn(BaseModel):
    items: list[MediaItem]


class MediaError(Exception):
    """Entrada de medios no válida (400)."""


def resolve_media_path(root: str, raw: str, max_mb: int) -> Path:
    """Resuelve `raw` (absoluta o relativa a `root`) y exige que quede DENTRO de `root` (sin `..` ni enlaces simbólicos que salgan), sea un archivo normal
    y no supere el tamaño máximo. El servicio no lee nada fuera de la raíz de medios."""
    base = Path(os.path.realpath(root))
    cand = Path(raw)
    resolved = Path(os.path.realpath(cand if cand.is_absolute() else base / cand))
    if resolved != base and base not in resolved.parents:
        raise MediaError("La ruta del medio queda fuera del directorio permitido")
    if not resolved.is_file():
        raise MediaError("El archivo de medio no existe")
    if resolved.stat().st_size > max_mb * 1024 * 1024:
        raise MediaError(f"El archivo supera el tamaño máximo ({max_mb} MB)")
    return resolved


def load_image(path: Path, max_pixels: int):
    """Abre una imagen con límite de píxeles (contra bombas de descompresión) y la devuelve en RGB."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            if im.width * im.height > max_pixels:
                raise MediaError(f"La imagen es demasiado grande ({im.width}×{im.height} píxeles; máximo {max_pixels})")
            im.load()
            return im.convert("RGB")
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001 - archivo corrupto o formato no admitido
        raise MediaError(f"No se pudo leer la imagen ({exc.__class__.__name__})") from exc


def load_wav_16k_mono(path: Path, start_ms: int | None, end_ms: int | None, max_seconds: int) -> np.ndarray:
    """Lee una ventana de un WAV PCM de 16 bits, mono, a 16 kHz (el formato que prepara la ingesta con ffmpeg) como float32 en [-1, 1]."""
    try:
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
                raise MediaError("El audio debe ser WAV PCM de 16 bits, mono, a 16 kHz (normalícelo con ffmpeg)")
            total_ms = int(w.getnframes() * 1000 / 16000)
            start = min(start_ms or 0, total_ms)
            end = min(end_ms if end_ms is not None else total_ms, total_ms)
            if end <= start:
                raise MediaError("La ventana de audio está vacía o fuera del archivo")
            if (end - start) > max_seconds * 1000:
                raise MediaError(f"La ventana de audio supera el máximo ({max_seconds} s)")
            w.setpos(int(start * 16))
            raw = w.readframes(int((end - start) * 16))
    except MediaError:
        raise
    except (wave.Error, EOFError) as exc:
        raise MediaError("No se pudo leer el audio (WAV no válido)") from exc
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def create_app(engine: Engine | None = None) -> FastAPI:
    cfg = engine.cfg if engine else ServerConfig.from_env()
    eng = engine or Engine(cfg)
    app = FastAPI(title="Servicio local de embeddings", docs_url=None, redoc_url=None)

    if cfg.preload:
        threading.Thread(target=lambda: _safe_preload(eng), daemon=True, name="preload").start()

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"message": message, "type": "invalid_request_error" if status < 500 else "server_error"}})

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        return error(exc.status_code, str(exc.detail))

    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingsIn, authorization: str | None = Header(default=None)):
        if cfg.api_key and authorization != f"Bearer {cfg.api_key}":
            raise HTTPException(401, "clave API inválida")
        texts = [body.input] if isinstance(body.input, str) else list(body.input)
        if not texts:
            raise HTTPException(400, "`input` está vacío")
        if len(texts) > cfg.max_batch:
            raise HTTPException(400, f"Demasiados textos en una petición ({len(texts)}); el máximo es {cfg.max_batch}")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise HTTPException(400, "Todos los textos deben ser cadenas no vacías")
        try:
            vecs = eng.encode(texts)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - p. ej. fallo al cargar el modelo
            log.exception("Fallo al embeber")
            raise HTTPException(503, f"El modelo no está disponible: {exc.__class__.__name__}") from exc
        if body.dimensions and body.dimensions < vecs.shape[1]:  # Matryoshka: truncar y re-normalizar
            vecs = vecs[:, : body.dimensions]
            vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12, None)
        return {
            "object": "list",
            "model": body.model or cfg.model,
            "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": cfg.model, "object": "model", "owned_by": "local"}]}

    FRAME_MAX_PIXELS = 4_000_000  # los fotogramas de vídeo los prepara la ingesta ya reducidos; un tope bajo acota la memoria de 32 a la vez

    def load_item(it: MediaItem):
        if it.modality == "video":
            if not it.paths:
                raise MediaError("Un vídeo necesita la lista `paths` de fotogramas")
            if len(it.paths) > cfg.media_max_frames:
                raise MediaError(f"Demasiados fotogramas ({len(it.paths)}); el máximo es {cfg.media_max_frames}")
            return [load_image(resolve_media_path(cfg.media_root, q, cfg.media_max_file_mb), min(cfg.media_max_pixels, FRAME_MAX_PIXELS)) for q in it.paths]
        if not it.path:
            raise MediaError("Falta `path`")
        path = resolve_media_path(cfg.media_root, it.path, cfg.media_max_file_mb)
        if it.modality == "image":
            return load_image(path, cfg.media_max_pixels)
        return load_wav_16k_mono(path, it.start_ms, it.end_ms, cfg.media_max_audio_s)

    @app.post("/v1/media/embeddings")
    def media_embeddings(body: MediaIn, authorization: str | None = Header(default=None)):
        """Vectores de imágenes, audio o fotogramas de vídeo, por referencia a archivos de la raíz de medios (una sola modalidad por petición)."""
        if cfg.api_key and authorization != f"Bearer {cfg.api_key}":
            raise HTTPException(401, "clave API inválida")
        items = body.items
        if not items:
            raise HTTPException(400, "`items` está vacío")
        if len(items) > cfg.media_max_batch:
            raise HTTPException(400, f"Demasiados elementos en una petición ({len(items)}); el máximo es {cfg.media_max_batch}")
        if len({it.modality for it in items}) != 1:
            raise HTTPException(400, "Una petición solo puede contener una modalidad")
        modality = items[0].modality
        try:
            inputs = [load_item(it) for it in items]
        except MediaError as exc:
            raise HTTPException(400, str(exc)) from exc
        t0 = time.perf_counter()
        try:
            vecs = eng.embed_media(modality, inputs)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - p. ej. fallo al cargar los codificadores o falta de memoria
            log.exception("Fallo al embeber medios")
            raise HTTPException(503, f"El modelo de medios no está disponible: {exc.__class__.__name__}") from exc
        return {
            "object": "list",
            "model": cfg.model,
            "modality": modality,
            "mode": eng.mode,
            "seconds": round(time.perf_counter() - t0, 2),
            "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
        }

    @app.get("/health")
    def health():
        """Estado sin forzar la carga del modelo (para el healthcheck del contenedor)."""
        return {"ok": eng.load_error is None, "model": cfg.model, "loaded": eng.loaded, "text_only": cfg.text_only, "dtype": cfg.dtype,
                "device": cfg.device, "load_seconds": eng.load_seconds, "error": eng.load_error, "mode": eng.mode, "base_mode": eng.base_mode,
                "media": {"root": cfg.media_root, "idle_s": cfg.media_idle_s}}

    return app


def _safe_preload(eng: Engine) -> None:
    try:
        eng._load()
    except Exception:  # noqa: BLE001
        log.exception("No se pudo precargar el modelo (se reintentará en la primera petición)")


def download() -> None:
    """Descarga el modelo a la caché de Hugging Face (HF_HOME) sin cargarlo en memoria."""
    from huggingface_hub import snapshot_download

    cfg = ServerConfig.from_env()
    path = snapshot_download(cfg.model)
    print(f"Modelo {cfg.model} descargado en {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if "--download" in sys.argv:
        download()
    else:
        import uvicorn

        uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), log_level="warning")
