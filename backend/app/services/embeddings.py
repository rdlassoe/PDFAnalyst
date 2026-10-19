"""Proveedores de embeddings, independientes de los de generación (D5).

Tres adaptadores, todos construidos desde un `EmbedderConfig` explícito (el «entorno» o un perfil de Ajustes):
- `sentence-transformers`: modelo local dentro del proceso (solo vía entorno; no se descargan modelos arbitrarios desde la UI).
- `ollama`: `/api/embed` de un servidor Ollama.
- `openai-compatible`: `/embeddings` de cualquier servicio con esa API (servidor propio, proveedor externo…).

Cada adaptador devuelve vectores float32 normalizados (L2) para que la distancia coseno de pgvector sea consistente. La dimensión se
detecta del modelo real, nunca se asume. Un destino NO local es un servicio **externo**: recibe el texto íntegro de los fragmentos al
indexar y las preguntas al buscar, así que exige consentimiento explícito (`allow_external`).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import EmbeddingModel
from ..providers.base import is_local_url

log = logging.getLogger(__name__)

KINDS = ("sentence-transformers", "ollama", "openai-compatible")

# Pruebas: transporte HTTP inyectado en los adaptadores HTTP (permite simular un servidor sin red).
_transport_override: httpx.BaseTransport | None = None


class EmbeddingError(Exception):
    """Fallo del proveedor de embeddings. `permanent=True` = reintentar no sirve (credenciales, modelo inexistente…)."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


class EmbeddingConflict(EmbeddingError):
    """El modelo registrado es incompatible con la configuración (dimensión o prefijos distintos): vectores mezclables = no."""

    def __init__(self, message: str) -> None:
        super().__init__(message, permanent=True)


@dataclass(frozen=True)
class EmbedderConfig:
    kind: str
    model: str
    base_url: str = ""
    api_key: str = ""
    allow_external: bool = False
    query_prefix: str = ""
    passage_prefix: str = ""
    dimensions: int | None = None  # truncado Matryoshka (solo modelos MRL); se envía al servidor y se aplica también aquí
    min_similarity: float | None = None  # umbral coseno de relevancia de este modelo (None = RETRIEVAL_MIN_SIMILARITY)
    supports_media: bool = False  # el servicio sabe embeber imágenes/audio/vídeo (`/media/embeddings`); solo se usa si es LOCAL
    batch_size: int = 32
    timeout_s: float = 60.0
    device: str = "cpu"
    label: str = ""

    def __repr__(self) -> str:  # nunca imprimir la clave
        return (f"EmbedderConfig(kind={self.kind!r}, model={self.model!r}, base_url={self.base_url!r}, "
                f"api_key={'***' if self.api_key else ''})")

    @property
    def external(self) -> bool:
        return self.kind != "sentence-transformers" and not is_local_url(self.base_url)

    def cache_key(self) -> tuple:
        key_hash = hashlib.sha256(self.api_key.encode()).hexdigest()[:16] if self.api_key else ""
        return (self.kind, self.model, self.base_url, key_hash, self.query_prefix, self.passage_prefix,
                self.dimensions, self.min_similarity, self.supports_media, self.batch_size, self.device)


def check_external_allowed(cfg: EmbedderConfig) -> None:
    if cfg.kind in ("ollama", "openai-compatible") and not cfg.base_url:
        raise EmbeddingError("La URL base es obligatoria para este proveedor de embeddings", permanent=True)
    if cfg.external and not cfg.allow_external:
        host = urlparse(cfg.base_url).hostname
        raise EmbeddingError(
            f"El proveedor de embeddings {host} es externo: recibiría el texto íntegro de sus documentos al indexarlos y sus "
            "preguntas al buscar. Para permitirlo, marque «Permitir el envío del texto de los documentos a este servicio externo» "
            "en Ajustes (o defina EMBEDDING_ALLOW_EXTERNAL=true si usa el proveedor del entorno).",
            permanent=True,
        )


class Embedder(ABC):
    provider: str
    name: str
    query_prefix: str
    passage_prefix: str
    external: bool = False
    min_similarity: float | None = None
    supports_media: bool = False  # ¿embebe imágenes, audio y vídeo? (solo el servicio local de EmbeddingGemma 2)

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    def embed_media(self, items: list[dict]) -> np.ndarray:
        """Vectores de medios (imagen/audio/fotogramas de vídeo) por REFERENCIA a archivos del almacenamiento. Solo si `supports_media`."""
        raise EmbeddingError(
            f"El modelo de embeddings activo ({self.name}) no admite imágenes, audio ni vídeo. Active «EmbeddingGemma 2 (local)» en Ajustes.", permanent=True
        )

    @abstractmethod
    def _encode(self, texts: list[str]) -> np.ndarray: ...

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._check(self._encode([self.passage_prefix + t for t in texts]), len(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return self._check(self._encode([self.query_prefix + text]), 1)[0]

    def _check(self, arr: np.ndarray, expected_rows: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != expected_rows or arr.shape[1] != self.dimension:
            raise EmbeddingError(
                f"Forma de embeddings inesperada {arr.shape}; se esperaba ({expected_rows}, {self.dimension})"
            )
        if not np.isfinite(arr).all():
            raise EmbeddingError("El modelo devolvió valores no finitos (NaN/inf)")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class SentenceTransformerEmbedder(Embedder):
    provider = "sentence-transformers"
    external = False

    def __init__(self, cfg: EmbedderConfig) -> None:
        self.name = cfg.model
        self.query_prefix = cfg.query_prefix
        self.passage_prefix = cfg.passage_prefix
        self.min_similarity = cfg.min_similarity
        self._batch = cfg.batch_size
        self._device = cfg.device
        self._model = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                log.info("Cargando modelo de embeddings %s en %s", self.name, self._device)
                self._model = SentenceTransformer(self.name, device=self._device)
                get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
                self._dim = int(get_dim())
        return self._model

    @property
    def dimension(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(texts, batch_size=self._batch, normalize_embeddings=True, show_progress_bar=False)


class _HttpEmbedder(Embedder):
    """Base de los adaptadores HTTP: consentimiento, reintentos y errores sin filtrar claves ni contenido."""

    path: str

    def __init__(self, cfg: EmbedderConfig, transport: httpx.BaseTransport | None = None) -> None:
        if not cfg.base_url:
            raise EmbeddingError("La URL base es obligatoria para este proveedor de embeddings", permanent=True)
        check_external_allowed(cfg)
        self._cfg = cfg
        self.external = cfg.external
        self.name = cfg.model
        self.query_prefix = cfg.query_prefix
        self.passage_prefix = cfg.passage_prefix
        self.min_similarity = cfg.min_similarity
        # Los medios (voz, caras, documentos…) NUNCA salen de la máquina: un servicio externo no los recibe aunque el perfil lo marque.
        self.supports_media = bool(cfg.supports_media) and not self.external
        self._base = cfg.base_url.rstrip("/")
        self._transport = transport
        self._dim: int | None = None

    # ---- a implementar por cada protocolo
    @abstractmethod
    def _payload(self, texts: list[str]) -> dict: ...

    @abstractmethod
    def _parse(self, data: dict) -> list[list[float]]: ...

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._cfg.api_key}"} if self._cfg.api_key else {}

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self._encode(["dimension probe"]).shape[1])
        return self._dim

    def _post_json(self, payload: dict, path: str | None = None) -> dict:
        url = f"{self._base}{path or self.path}"
        host = urlparse(self._base).netloc
        attempts = 4
        for attempt in range(attempts):
            last = attempt == attempts - 1
            try:
                with httpx.Client(timeout=self._cfg.timeout_s, transport=self._transport or _transport_override,
                                  headers=self._headers()) as c:
                    r = c.post(url, json=payload)
            except httpx.ConnectError as exc:
                raise EmbeddingError(f"No se pudo conectar con {host}") from exc
            except httpx.TimeoutException as exc:
                raise EmbeddingError(f"{host} tardó demasiado en responder") from exc
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Error de red con {host}: {exc.__class__.__name__}") from exc

            if r.status_code == 429:
                if last:
                    raise EmbeddingError("El proveedor limitó las peticiones (429); reintente más tarde")
                try:
                    wait = min(float(r.headers.get("retry-after", "")), 30.0)
                except ValueError:
                    wait = float(2**attempt)
                time.sleep(wait)
                continue
            if r.status_code in (500, 502, 503, 504):
                if last:
                    raise EmbeddingError(f"El proveedor respondió {r.status_code} de forma repetida")
                time.sleep(1.0 * (attempt + 1))
                continue
            if r.status_code in (401, 403):
                raise EmbeddingError("El proveedor rechazó las credenciales (revise la clave API)", permanent=True)
            if r.status_code == 404:
                raise EmbeddingError(
                    f"Modelo o ruta no encontrados en el proveedor (modelo «{self._cfg.model}»; ¿falta `ollama pull`?)",
                    permanent=True,
                )
            if r.status_code in (400, 413, 422):
                raise EmbeddingError(f"El proveedor rechazó la petición ({r.status_code}): {self._error_message(r)}", permanent=True)
            if r.status_code >= 300:
                raise EmbeddingError(f"El proveedor respondió {r.status_code}")
            try:
                return r.json()
            except ValueError as exc:
                raise EmbeddingError("Respuesta del proveedor no válida (no es JSON)") from exc
        raise EmbeddingError("Reintentos agotados")  # pragma: no cover

    @staticmethod
    def _error_message(r: httpx.Response) -> str:
        try:
            body = r.json()
            msg = body.get("error", body)
            msg = msg.get("message", msg) if isinstance(msg, dict) else msg
            return str(msg)[:160]
        except Exception:  # noqa: BLE001
            return "sin detalle"

    def _encode(self, texts: list[str]) -> np.ndarray:
        data = self._post_json(self._payload(texts))
        try:
            rows = self._parse(data)
            arr = np.array(rows, dtype=np.float32)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingError("Respuesta del proveedor con un formato de embeddings inesperado") from exc
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise EmbeddingError(f"El proveedor devolvió {arr.shape[0] if arr.ndim else 0} vectores para {len(texts)} textos")
        dims = self._cfg.dimensions
        if dims and arr.shape[1] > dims:  # Matryoshka: truncar (el servidor puede ignorar `dimensions`); se re-normaliza en _check
            arr = arr[:, :dims]
        return arr


class OllamaEmbedder(_HttpEmbedder):
    """Embeddings vía Ollama (`/api/embed`)."""

    provider = "ollama"
    path = "/api/embed"

    def _payload(self, texts: list[str]) -> dict:
        p: dict = {"model": self._cfg.model, "input": texts}
        if self._cfg.dimensions:
            p["dimensions"] = self._cfg.dimensions
        return p

    def _parse(self, data: dict) -> list[list[float]]:
        return data["embeddings"]


class OpenAICompatEmbedder(_HttpEmbedder):
    """Embeddings vía cualquier servicio con la API `/embeddings` de OpenAI."""

    provider = "openai-compatible"
    path = "/embeddings"

    def _payload(self, texts: list[str]) -> dict:
        p: dict = {"model": self._cfg.model, "input": texts, "encoding_format": "float"}
        if self._cfg.dimensions:
            p["dimensions"] = self._cfg.dimensions
        return p

    def _parse(self, data: dict) -> list[list[float]]:
        items = sorted(data["data"], key=lambda d: d.get("index", 0))
        return [it["embedding"] for it in items]

    def embed_media(self, items: list[dict]) -> np.ndarray:
        """`items`: {"modality": "image"|"audio"|"video", "path": …} (o "paths" para vídeo), rutas dentro del almacenamiento compartido con el servicio."""
        if not self.supports_media:
            return super().embed_media(items)
        data = self._post_json({"items": items}, path="/media/embeddings")
        try:
            arr = np.array(self._parse(data), dtype=np.float32)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise EmbeddingError("Respuesta del servicio de medios con un formato inesperado") from exc
        return self._check(arr, len(items))


# ---------------------------------------------------------------------------------------------- fábrica y caché
_CACHE: dict[tuple, Embedder] = {}
_CACHE_LOCK = threading.Lock()


def build_embedder(cfg: EmbedderConfig) -> Embedder:
    """Devuelve el adaptador de esa configuración (en caché: los modelos locales son pesados). Puede lanzar EmbeddingError."""
    if cfg.kind not in KINDS:
        raise EmbeddingError(f"Proveedor de embeddings desconocido: {cfg.kind}", permanent=True)
    check_external_allowed(cfg)  # también antes de consultar la caché: revocar el permiso bloquea al instante
    key = cfg.cache_key()
    with _CACHE_LOCK:
        emb = _CACHE.get(key)
        if emb is None:
            if cfg.kind == "sentence-transformers":
                emb = SentenceTransformerEmbedder(cfg)
            elif cfg.kind == "ollama":
                emb = OllamaEmbedder(cfg)
            else:
                emb = OpenAICompatEmbedder(cfg)
            _CACHE[key] = emb
        return emb


def env_config() -> EmbedderConfig:
    """Configuración del perfil «entorno» (variables EMBEDDING_*): local por defecto."""
    s = get_settings()
    kind = s.embedding_provider
    return EmbedderConfig(
        kind=kind,
        model=s.embedding_model,
        base_url=s.ollama_base_url if kind == "ollama" else "",
        allow_external=s.embedding_allow_external,
        query_prefix=s.embedding_query_prefix,
        passage_prefix=s.embedding_passage_prefix,
        batch_size=s.embedding_batch_size,
        timeout_s=s.embedding_timeout_s,
        device=s.embedding_device,
        label="Entorno (.env)",
    )


def get_embedder(session: Session | None = None) -> Embedder:
    """Embedder ACTIVO: el perfil elegido en Ajustes (o el entorno). Sin sesión, el del entorno."""
    if session is None:
        return build_embedder(env_config())
    from . import embedding_settings  # import tardío: embedding_settings depende de este módulo

    return build_embedder(embedding_settings.active_config(session))


def get_or_create_model_row(session: Session, embedder: Embedder) -> EmbeddingModel:
    """Registra el modelo. Si ya existe con otra dimensión o prefijos, se rechaza (vectores incompatibles)."""
    row = session.scalar(
        select(EmbeddingModel).where(
            EmbeddingModel.provider == embedder.provider, EmbeddingModel.name == embedder.name
        )
    )
    dim = embedder.dimension
    if row is None:
        row = EmbeddingModel(
            provider=embedder.provider,
            name=embedder.name,
            dimension=dim,
            distance="cosine",
            query_prefix=embedder.query_prefix,
            passage_prefix=embedder.passage_prefix,
            normalize=True,
            supports_media=embedder.supports_media,
        )
        session.add(row)
        session.flush()
    elif row.dimension != dim:
        raise EmbeddingConflict(
            f"El modelo {row.name} está registrado con dimensión {row.dimension} pero ahora produce {dim}. "
            "Elimine sus vectores en Ajustes («Vectores indexados») o use otro modelo."
        )
    elif (row.query_prefix, row.passage_prefix) != (embedder.query_prefix, embedder.passage_prefix):
        raise EmbeddingConflict(
            "Los prefijos de consulta/pasaje difieren de los usados al indexar este modelo; los vectores existentes "
            "serían incompatibles. Restaure los prefijos originales o elimine los vectores de ese modelo en Ajustes."
        )
    if row.supports_media != embedder.supports_media:  # la capacidad del perfil puede cambiar sin invalidar vectores
        row.supports_media = embedder.supports_media
        session.flush()
    return row
