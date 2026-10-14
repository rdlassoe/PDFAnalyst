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
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass

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
        )


class Engine:
    """Modelo cargado de forma perezosa y segura entre hilos. Reemplazable en pruebas (`encode`, `dimension`)."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()
        self.load_seconds: float | None = None
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _dim_of(model) -> int:
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(get_dim())

    def _load(self):
        if self._model is not None:  # camino rápido, sin cerrojo
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            import torch
            from sentence_transformers import SentenceTransformer

            if self.cfg.threads > 0:
                torch.set_num_threads(self.cfg.threads)
            t0 = time.perf_counter()
            log.info("Cargando %s (solo texto=%s, %s, %s)…", self.cfg.model, self.cfg.text_only, self.cfg.dtype, self.cfg.device)
            kwargs: dict = {"device": self.cfg.device, "model_kwargs": {"torch_dtype": getattr(torch, self.cfg.dtype)}}
            if self.cfg.text_only:  # omite los codificadores de visión y audio (≈740M → ≈270M parámetros)
                kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
            try:
                model = SentenceTransformer(self.cfg.model, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self.load_error = f"{exc.__class__.__name__}: {exc}"[:300]
                raise
            model.max_seq_length = min(self.cfg.max_seq_length, model.max_seq_length or self.cfg.max_seq_length)
            self._model = model
            self.load_error = None
            self.load_seconds = round(time.perf_counter() - t0, 1)
            # OJO: no usar self.dimension aquí (volvería a pedir el cerrojo, que no es reentrante → interbloqueo).
            log.info("Modelo listo en %.1f s (dimensión %s, longitud máx. %s)", self.load_seconds, self._dim_of(model), model.max_seq_length)
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


class EmbeddingsIn(BaseModel):
    model: str = Field(default="", max_length=256)
    input: list[str] | str
    encoding_format: str | None = None
    dimensions: int | None = Field(default=None, ge=8, le=8192)


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

    @app.get("/health")
    def health():
        """Estado sin forzar la carga del modelo (para el healthcheck del contenedor)."""
        return {"ok": eng.load_error is None, "model": cfg.model, "loaded": eng.loaded, "text_only": cfg.text_only, "dtype": cfg.dtype,
                "device": cfg.device, "load_seconds": eng.load_seconds, "error": eng.load_error}

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
