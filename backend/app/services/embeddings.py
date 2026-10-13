"""Proveedores de embeddings, independientes de los de generación (D5).

Cada proveedor devuelve vectores float32 normalizados (L2) para que la distancia coseno de pgvector
sea consistente. La dimensión se detecta del modelo real, nunca se asume.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from functools import lru_cache

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..models import EmbeddingModel

log = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


class Embedder(ABC):
    provider: str
    name: str
    query_prefix: str
    passage_prefix: str

    @property
    @abstractmethod
    def dimension(self) -> int: ...

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

    def __init__(self, s: Settings) -> None:
        self.name = s.embedding_model
        self.query_prefix = s.embedding_query_prefix
        self.passage_prefix = s.embedding_passage_prefix
        self._batch = s.embedding_batch_size
        self._device = s.embedding_device
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
        return model.encode(
            texts, batch_size=self._batch, normalize_embeddings=True, show_progress_bar=False
        )


class OllamaEmbedder(Embedder):
    """Embeddings vía Ollama (`/api/embed`). Requiere un modelo de embeddings (p. ej. nomic-embed-text)."""

    provider = "ollama"

    def __init__(self, s: Settings) -> None:
        self.name = s.embedding_model
        self.query_prefix = s.embedding_query_prefix
        self.passage_prefix = s.embedding_passage_prefix
        self._base = s.ollama_base_url.rstrip("/")
        self._timeout = s.llm_timeout_s
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = int(self._encode(["dimension probe"]).shape[1])
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        try:
            r = httpx.post(
                f"{self._base}/api/embed",
                json={"model": self.name, "input": texts},
                timeout=self._timeout,
            )
            r.raise_for_status()
            return np.array(r.json()["embeddings"], dtype=np.float32)
        except (httpx.HTTPError, KeyError) as exc:
            raise EmbeddingError(f"Ollama no pudo generar embeddings con '{self.name}': {exc}") from exc


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbedder(s)
    if s.embedding_provider == "ollama":
        return OllamaEmbedder(s)
    raise EmbeddingError(f"Proveedor de embeddings desconocido: {s.embedding_provider}")


def get_or_create_model_row(session: Session, embedder: Embedder) -> EmbeddingModel:
    """Registra el modelo activo. Si ya existe con otra dimensión, se rechaza (vectores incompatibles)."""
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
        )
        session.add(row)
        session.flush()
    elif row.dimension != dim:
        raise EmbeddingError(
            f"El modelo {row.name} está registrado con dimensión {row.dimension} pero ahora produce {dim}. "
            "Registre el nuevo modelo con otro nombre o reconstruya los embeddings."
        )
    elif (row.query_prefix, row.passage_prefix) != (embedder.query_prefix, embedder.passage_prefix):
        raise EmbeddingError(
            "Los prefijos de consulta/pasaje difieren de los usados al indexar; los vectores existentes "
            "serían incompatibles. Reconstruya los embeddings (ver docs/OPERATIONS.md)."
        )
    return row
