"""Ajustes del modelo de embeddings editables desde la interfaz.

Igual que con los LLM (services/llm_settings.py):
- Perfil implícito **«env»**: lo que dicen las variables EMBEDDING_* (por defecto, `multilingual-e5-small` local). Siempre existe,
  no es editable y es el activo mientras no se elija otro.
- Perfiles creados en Ajustes (`embedding_profiles`): la clave API se guarda cifrada (services/credentials.py) y NUNCA se devuelve.
El perfil activo se guarda en `app_settings["active_embedding"]`.

Diferencias importantes con los LLM:
1. **Indexación según el modelo seleccionado**: solo se indexa con el modelo activo. Al activar otro, los documentos que aún no
   tengan sus vectores se reencolan (quedan «processing», no consultables, hasta terminar: no se mezclan espacios vectoriales).
   Volver a un modelo ya indexado es instantáneo (sus vectores se conservan).
2. Un proveedor externo recibe **el texto íntegro de los fragmentos de todos los documentos** al indexar, y las preguntas al buscar:
   exige consentimiento explícito, y antes de activarlo se puede obtener una vista previa de cuánto texto se enviaría.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import replace
from urllib.parse import urlparse

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import AppSetting, ChunkEmbedding, DocumentChunk, EmbeddingModel, EmbeddingProfile
from . import credentials, vector_index
from .embeddings import (
    EmbedderConfig,
    EmbeddingConflict,
    EmbeddingError,
    build_embedder,
    env_config,
    get_or_create_model_row,
)
from .ingestion import embedding_counts, model_coverage, requeue_for_model
from .net_validation import SettingsError, clean_api_key, validate_base_url

log = logging.getLogger(__name__)

ENV_ID = "env"
ACTIVE_KEY = "active_embedding"
KINDS = ("ollama", "openai-compatible")
GEMMA_QUERY_PREFIX = "task: search result | query: "
GEMMA_PASSAGE_PREFIX = "title: none | text: "
# Calibrado con eval/ (Babilonia, 22 preguntas): preguntas con respuesta ≥ 0,722, sin respuesta ≤ 0,714 → punto medio redondeado.
GEMMA_MIN_SIMILARITY = 0.72

# Plantillas para el formulario. Las URL/modelos son orientativos: verifíquelos con su proveedor.
PRESETS = [
    {"id": "gemma2-local", "label": "EmbeddingGemma 2 (local, servicio «embedder» incluido)", "kind": "openai-compatible",
     "base_url": "http://embedder:8000/v1", "model": "google/embeddinggemma-2", "query_prefix": GEMMA_QUERY_PREFIX,
     "passage_prefix": GEMMA_PASSAGE_PREFIX, "dimensions": None, "min_similarity": GEMMA_MIN_SIMILARITY, "supports_media": True, "needs_key": False, "key_url": None,
     "note": "Se ejecuta en su equipo (CPU, solo texto, ≈1,7 GB de RAM medidos, un único proceso): ningún texto sale de él. Apache-2.0. La primera vez hay que descargar "
             "≈1,5 GB del modelo desde Hugging Face (`docker compose run --rm embedder python -m app.embedding_server --download`). Prefijos oficiales del modelo."},
    {"id": "gemma2-api", "label": "EmbeddingGemma 2 (servidor externo compatible con OpenAI)", "kind": "openai-compatible", "base_url": "",
     "model": "google/embeddinggemma-2", "query_prefix": GEMMA_QUERY_PREFIX, "passage_prefix": GEMMA_PASSAGE_PREFIX, "dimensions": None,
     "min_similarity": GEMMA_MIN_SIMILARITY, "supports_media": False, "needs_key": True, "key_url": None,
     "note": "Indique la URL base del servicio que lo aloje (p. ej. su propio servidor con llama.cpp `llama-server --embeddings`). "
             "No se ha confirmado ningún proveedor alojado de EmbeddingGemma 2; el nombre del modelo depende del servicio. "
             "Los prefijos son los oficiales del modelo; algunos servicios ya los aplican: verifíquelo."},
    {"id": "ollama-embed", "label": "Ollama (otro servidor o local)", "kind": "ollama", "base_url": "http://host.docker.internal:11434",
     "model": "", "query_prefix": "", "passage_prefix": "", "dimensions": None, "min_similarity": None, "needs_key": False, "key_url": None,
     "note": "Para otros modelos de embeddings de Ollama. OJO: `embeddinggemma-2` en Ollama exige MLX (solo Apple Silicon) y NO funciona en Windows/Linux con CUDA. "
             "Prefijos vacíos: compruebe con `ollama show <modelo> --template` si la plantilla ya los aplica. Un Ollama remoto se trata como servicio externo."},
    {"id": "openai-embed", "label": "OpenAI (API de embeddings)", "kind": "openai-compatible", "base_url": "https://api.openai.com/v1",
     "model": "", "query_prefix": "", "passage_prefix": "", "dimensions": None, "min_similarity": None, "needs_key": True,
     "key_url": "https://platform.openai.com/api-keys", "note": "Servicio externo. Indique el modelo de embeddings (el nombre cambia con el tiempo)."},
    {"id": "custom-embed", "label": "Otro servicio compatible con OpenAI", "kind": "openai-compatible", "base_url": "", "model": "",
     "query_prefix": "", "passage_prefix": "", "dimensions": None, "min_similarity": None, "needs_key": True, "key_url": None,
     "note": "URL base = la que precede a /embeddings. Si el modelo exige prefijos, indíquelos."},
]


# ------------------------------------------------------------------------------ configuraciones
def profile_config(p: EmbeddingProfile) -> EmbedderConfig:
    s = get_settings()
    key = ""
    if p.api_key_enc:
        try:
            key = credentials.decrypt(p.api_key_enc)
        except credentials.SecretError as exc:
            raise EmbeddingError(str(exc), permanent=True) from exc
    return EmbedderConfig(
        kind=p.kind, model=p.model, base_url=p.base_url, api_key=key, allow_external=p.allow_external,
        query_prefix=p.query_prefix, passage_prefix=p.passage_prefix, dimensions=p.dimensions, min_similarity=p.min_similarity, supports_media=p.supports_media,
        batch_size=s.embedding_batch_size, timeout_s=s.embedding_timeout_s, label=p.name,
    )


def get_active_id(session: Session) -> str:
    row = session.get(AppSetting, ACTIVE_KEY)
    return row.value if row else ENV_ID


def _get_profile(session: Session, profile_id: str) -> EmbeddingProfile:
    try:
        pid = uuid.UUID(str(profile_id))
    except ValueError:
        raise SettingsError("Proveedor de embeddings no encontrado") from None
    p = session.get(EmbeddingProfile, pid)
    if p is None:
        raise SettingsError("Proveedor de embeddings no encontrado")
    return p


def resolve(session: Session, profile_id: str) -> EmbedderConfig:
    return env_config() if profile_id == ENV_ID else profile_config(_get_profile(session, profile_id))


def active_config(session: Session) -> EmbedderConfig:
    try:
        return resolve(session, get_active_id(session))
    except SettingsError:  # el perfil activo fue eliminado: se vuelve al entorno
        return env_config()


def active_model_row(session: Session) -> EmbeddingModel | None:
    """Fila del modelo activo si ya fue registrado (consulta solo la BD; no usa la red)."""
    cfg = active_config(session)
    return session.scalar(select(EmbeddingModel).where(EmbeddingModel.provider == cfg.kind, EmbeddingModel.name == cfg.model))


# ------------------------------------------------------------------------------ validación y CRUD
def _validate(kind: str, base_url: str, model: str, name: str | None, dimensions: int | None, qp: str, pp: str,
              min_similarity: float | None = None) -> str:
    if kind not in KINDS:
        raise SettingsError(f"Tipo de proveedor no soportado: {kind}")
    if name is not None and not name.strip():
        raise SettingsError("El nombre es obligatorio")
    if not model.strip():
        raise SettingsError("El modelo es obligatorio")
    if dimensions is not None and not (8 <= dimensions <= 8192):
        raise SettingsError("Las dimensiones deben estar entre 8 y 8192 (déjelo vacío para usar todas)")
    if min_similarity is not None and not (0 < min_similarity <= 1):
        raise SettingsError("El umbral de similitud debe estar entre 0 y 1 (déjelo vacío para usar el global)")
    for label, v in (("consulta", qp), ("documento", pp)):
        if len(v) > 200 or "\n" in v or "\r" in v:
            raise SettingsError(f"El prefijo de {label} debe ser una sola línea de hasta 200 caracteres")
    return validate_base_url(base_url)


def create_profile(session: Session, *, name: str, kind: str, base_url: str, model: str, api_key: str | None, allow_external: bool,
                   query_prefix: str = "", passage_prefix: str = "", dimensions: int | None = None,
                   min_similarity: float | None = None, supports_media: bool = False) -> EmbeddingProfile:
    base = _validate(kind, base_url, model, name, dimensions, query_prefix, passage_prefix, min_similarity)
    if session.scalar(select(EmbeddingProfile).where(EmbeddingProfile.name == name.strip())):
        raise SettingsError("Ya existe un proveedor de embeddings con ese nombre")
    key = clean_api_key(api_key) if api_key else ""
    p = EmbeddingProfile(name=name.strip(), kind=kind, base_url=base, model=model.strip(),
                         api_key_enc=credentials.encrypt(key) if key else None, allow_external=bool(allow_external),
                         query_prefix=query_prefix, passage_prefix=passage_prefix, dimensions=dimensions, min_similarity=min_similarity,
                         supports_media=bool(supports_media))
    session.add(p)
    session.commit()
    return p


_UNSET = object()


def update_profile(session: Session, profile_id: str, *, name: str | None = None, kind: str | None = None, base_url: str | None = None,
                   model: str | None = None, api_key: str | None = None, clear_api_key: bool = False, allow_external: bool | None = None,
                   query_prefix: str | None = None, passage_prefix: str | None = None, dimensions: object = _UNSET,
                   min_similarity: object = _UNSET, supports_media: bool | None = None) -> EmbeddingProfile:
    p = _get_profile(session, profile_id)
    new = dict(
        kind=kind or p.kind, model=model if model is not None else p.model,
        query_prefix=query_prefix if query_prefix is not None else p.query_prefix,
        passage_prefix=passage_prefix if passage_prefix is not None else p.passage_prefix,
        dimensions=p.dimensions if dimensions is _UNSET else dimensions,
    )
    if get_active_id(session) == str(p.id) and any(getattr(p, k) != v for k, v in new.items()):
        raise SettingsError(
            "Este proveedor es el predeterminado: no se puede cambiar su modelo, prefijos ni dimensiones sin invalidar los vectores "
            "indexados. Active otro proveedor primero (o cree uno nuevo)."
        )
    p.base_url = _validate(new["kind"], base_url if base_url is not None else p.base_url, new["model"], name,
                           new["dimensions"], new["query_prefix"], new["passage_prefix"],
                           p.min_similarity if min_similarity is _UNSET else min_similarity)
    if min_similarity is not _UNSET:  # solo afecta a la búsqueda, no a los vectores: se puede cambiar con el perfil activo
        p.min_similarity = min_similarity
    if supports_media is not None:  # capacidad del servicio, no cambia los vectores de texto
        p.supports_media = bool(supports_media)
    for k, v in new.items():
        setattr(p, k, v.strip() if k == "model" else v)
    if name is not None and name.strip() != p.name:
        if session.scalar(select(EmbeddingProfile).where(EmbeddingProfile.name == name.strip())):
            raise SettingsError("Ya existe un proveedor de embeddings con ese nombre")
        p.name = name.strip()
    if clear_api_key:
        p.api_key_enc = None
    elif api_key:  # None o "" = conservar la actual
        p.api_key_enc = credentials.encrypt(clean_api_key(api_key))
    if allow_external is not None:
        p.allow_external = bool(allow_external)
    session.commit()
    return p


def _set_setting(session: Session, key: str, value: str) -> None:
    row = session.get(AppSetting, key)
    if row:
        row.value = value
    else:
        session.add(AppSetting(key=key, value=value))


def delete_profile(session: Session, profile_id: str) -> None:
    p = _get_profile(session, profile_id)
    if get_active_id(session) == str(p.id):
        # Se vuelve al entorno; si su modelo ya está indexado, los documentos quedan listos al instante.
        _set_setting(session, ACTIVE_KEY, ENV_ID)
        session.commit()
        try:
            row = get_or_create_model_row(session, build_embedder(env_config()))
            session.commit()
            requeue_for_model(session, row.id)
        except EmbeddingError as exc:
            log.warning("No se pudo reindexar con el modelo del entorno tras eliminar el perfil activo: %s", exc)
    session.delete(p)
    session.commit()


# ------------------------------------------------------------------------------ prueba, vista previa, activación
def probe(cfg: EmbedderConfig) -> dict:
    """Comprueba conexión y credenciales embebiendo UNA frase fija de prueba (no se envía contenido de documentos)."""
    if cfg.kind != "sentence-transformers":
        try:
            validate_base_url(cfg.base_url)
        except SettingsError as exc:
            return {"ok": False, "detail": str(exc), "dimension": None, "latency_ms": None}
    try:
        emb = build_embedder(replace(cfg, allow_external=True))  # la prueba no envía documentos: no exige el permiso
        t0 = time.perf_counter()
        vec = emb.embed_query("Prueba de conexión de PDF Analyst")
        ms = round((time.perf_counter() - t0) * 1000)
    except EmbeddingError as exc:
        return {"ok": False, "detail": str(exc), "dimension": None, "latency_ms": None}
    except Exception as exc:  # noqa: BLE001 - la prueba nunca debe lanzar
        return {"ok": False, "detail": f"No se pudo cargar el modelo: {exc.__class__.__name__}", "dimension": None, "latency_ms": None}
    return {"ok": True, "detail": f"Conexión correcta: vectores de {len(vec)} dimensiones en {ms} ms",
            "dimension": int(len(vec)), "latency_ms": ms}


def preview(session: Session, profile_id: str) -> dict:
    """Qué implicaría activar este perfil: documentos a (re)indexar y cuánto texto se enviaría. No usa la red."""
    cfg = resolve(session, profile_id)
    row = session.scalar(select(EmbeddingModel).where(EmbeddingModel.provider == cfg.kind, EmbeddingModel.name == cfg.model))
    model_id = row.id if row else -1  # -1: aún sin vectores
    cov = model_coverage(session, model_id)
    pending = [d for d, (n, e) in cov.items() if e < n]
    missing_filter = ~select(ChunkEmbedding.chunk_id).where(
        ChunkEmbedding.chunk_id == DocumentChunk.id, ChunkEmbedding.model_id == model_id
    ).exists()
    chunks, chars = session.execute(
        select(func.count(DocumentChunk.id), func.coalesce(func.sum(func.length(DocumentChunk.text)), 0)).where(missing_filter)
    ).one()
    return {
        "name": cfg.label or cfg.model, "kind": cfg.kind, "model": cfg.model,
        "host": urlparse(cfg.base_url).netloc if cfg.base_url else "local (en este equipo)",
        "external": cfg.external, "allow_external": cfg.allow_external,
        "documents_total": len(cov), "documents_to_index": len(pending), "already_indexed": len(cov) - len(pending),
        "chunks_to_index": int(chunks), "chars_to_send": int(chars) if cfg.external else 0,
    }


def activate(session: Session, profile_id: str) -> dict:
    """Hace predeterminado el perfil y reencola los documentos sin vectores de su modelo.

    Antes de cambiar nada se comprueba el acceso (consentimiento, conexión, credenciales, compatibilidad con los vectores existentes):
    si algo falla, el proveedor activo no cambia. Lanza SettingsError / EmbeddingError / EmbeddingConflict.
    """
    cfg = resolve(session, profile_id)
    emb = build_embedder(cfg)                      # consentimiento
    row = get_or_create_model_row(session, emb)    # conexión + dimensión + compatibilidad de prefijos
    session.commit()
    _set_setting(session, ACTIVE_KEY, profile_id)
    session.commit()
    summary = requeue_for_model(session, row.id)
    summary.update(model_id=row.id, model=cfg.model, dimension=row.dimension)
    return summary


def purge_vectors(session: Session, model_id: int) -> dict:
    """Borra los vectores (y el registro) de un modelo que NO está activo. Libera espacio; volver a él exigirá reindexar."""
    row = session.get(EmbeddingModel, model_id)
    if row is None:
        raise SettingsError("Modelo no encontrado")
    active = active_model_row(session)
    if active is not None and active.id == model_id:
        raise SettingsError("No se pueden borrar los vectores del modelo activo")
    n = session.scalar(select(func.count()).select_from(ChunkEmbedding).where(ChunkEmbedding.model_id == model_id)) or 0
    session.commit()  # cerrar la transacción abierta antes de DDL concurrente
    from ..db import engine

    vector_index.drop_hnsw_index(engine, model_id)
    session.query(ChunkEmbedding).filter(ChunkEmbedding.model_id == model_id).delete()
    session.delete(row)
    session.commit()
    return {"model_id": model_id, "vectors_deleted": int(n)}


# ------------------------------------------------------------------------------ vistas (sin secretos)
def _view(*, id_: str, name: str, kind: str, base_url: str, model: str, has_key: bool, key_hint: str | None, key_error: bool,
          allow_external: bool, editable: bool, active: bool, qp: str, pp: str, dims: int | None, external: bool,
          min_similarity: float | None = None, supports_media: bool = False) -> dict:
    return {"id": id_, "name": name, "kind": kind, "base_url": base_url, "model": model, "has_key": has_key, "key_hint": key_hint,
            "key_error": key_error, "allow_external": allow_external, "external": external, "editable": editable, "active": active,
            "query_prefix": qp, "passage_prefix": pp, "dimensions": dims,
            "min_similarity": min_similarity, "supports_media": supports_media}


def list_view(session: Session) -> dict:
    active = get_active_id(session)
    profiles = list(session.scalars(select(EmbeddingProfile).order_by(EmbeddingProfile.created_at)))
    if active != ENV_ID and not any(str(p.id) == active for p in profiles):
        active = ENV_ID
    e = env_config()
    out = [_view(id_=ENV_ID, name=e.label, kind=e.kind, base_url=e.base_url, model=e.model, has_key=False, key_hint=None, key_error=False,
                 allow_external=e.allow_external, editable=False, active=active == ENV_ID, qp=e.query_prefix, pp=e.passage_prefix,
                 dims=None, external=e.external)]
    for p in profiles:
        hint, err = None, False
        if p.api_key_enc:
            try:
                hint = credentials.hint(credentials.decrypt(p.api_key_enc))
            except credentials.SecretError:
                err = True
        cfg = EmbedderConfig(kind=p.kind, model=p.model, base_url=p.base_url)
        out.append(_view(id_=str(p.id), name=p.name, kind=p.kind, base_url=p.base_url, model=p.model, has_key=bool(p.api_key_enc),
                         key_hint=hint, key_error=err, allow_external=p.allow_external, editable=True, active=str(p.id) == active,
                         qp=p.query_prefix, pp=p.passage_prefix, dims=p.dimensions, external=cfg.external, min_similarity=p.min_similarity, supports_media=p.supports_media))
    row = active_model_row(session)
    cov = model_coverage(session, row.id if row else -1)
    indexed = sum(1 for n, e_ in cov.values() if e_ >= n)
    models = [dict(m, active=bool(row and m["id"] == row.id)) for m in embedding_counts(session)]
    return {"active": active, "profiles": out, "presets": PRESETS, "models": models,
            "coverage": {"documents_total": len(cov), "documents_indexed": indexed}}
