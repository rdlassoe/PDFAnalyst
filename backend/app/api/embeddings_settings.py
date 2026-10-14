from __future__ import annotations

from dataclasses import replace
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_session
from ..services import credentials, embedding_settings as svc
from ..services.embeddings import EmbedderConfig, EmbeddingConflict, EmbeddingError
from ..services.net_validation import SettingsError, clean_api_key

router = APIRouter(prefix="/settings/embeddings", tags=["settings"])

Kind = Literal["ollama", "openai-compatible"]


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: Kind
    base_url: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=256)
    api_key: str | None = Field(default=None, max_length=512, repr=False)
    allow_external: bool = False
    query_prefix: str = Field(default="", max_length=200)
    passage_prefix: str = Field(default="", max_length=200)
    dimensions: int | None = Field(default=None, ge=8, le=8192)
    min_similarity: float | None = Field(default=None, gt=0, le=1)


class ProfilePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    kind: Kind | None = None
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    api_key: str | None = Field(default=None, max_length=512, repr=False)  # None/"" = conservar
    clear_api_key: bool = False
    allow_external: bool | None = None
    query_prefix: str | None = Field(default=None, max_length=200)
    passage_prefix: str | None = Field(default=None, max_length=200)
    dimensions: int | None = Field(default=None, ge=8, le=8192)
    min_similarity: float | None = Field(default=None, gt=0, le=1)  # null explícito = volver al umbral global


class ProfileRef(BaseModel):
    profile_id: str = Field(max_length=64)


class TestIn(BaseModel):
    """Prueba un borrador (sin guardarlo) o un perfil existente. La clave del borrador no se almacena."""

    profile_id: str | None = Field(default=None, max_length=64)
    kind: Kind | None = None
    base_url: str | None = Field(default=None, max_length=500)
    model: str | None = Field(default=None, max_length=256)
    api_key: str | None = Field(default=None, max_length=512, repr=False)
    query_prefix: str | None = Field(default=None, max_length=200)
    dimensions: int | None = Field(default=None, ge=8, le=8192)


def _wrap(fn):
    """Traduce las excepciones del servicio a respuestas HTTP (409 conflicto/consentimiento, 422 entrada, 502 proveedor)."""
    try:
        return fn()
    except SettingsError as exc:
        msg = str(exc)
        raise HTTPException(409 if "Ya existe" in msg or "predeterminado" in msg or "activo" in msg else 422, msg) from exc
    except EmbeddingConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(409 if "externo" in str(exc) else 502, str(exc)) from exc
    except credentials.SecretError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("")
def get_view(session: Session = Depends(get_session)):
    return svc.list_view(session)


@router.post("/profiles", status_code=201)
def create(body: ProfileIn, session: Session = Depends(get_session)):
    _wrap(lambda: svc.create_profile(session, **body.model_dump()))
    return svc.list_view(session)


@router.patch("/profiles/{profile_id}")
def update(profile_id: str, body: ProfilePatch, session: Session = Depends(get_session)):
    _wrap(lambda: svc.update_profile(session, profile_id, **body.model_dump(exclude_unset=True)))
    return svc.list_view(session)


@router.delete("/profiles/{profile_id}")
def delete(profile_id: str, session: Session = Depends(get_session)):
    _wrap(lambda: svc.delete_profile(session, profile_id))
    return svc.list_view(session)


@router.post("/preview")
def preview(body: ProfileRef, session: Session = Depends(get_session)):
    """Cuántos documentos se (re)indexarían y cuánto texto se enviaría a un servicio externo. No usa la red."""
    return _wrap(lambda: svc.preview(session, body.profile_id))


@router.put("/active")
def activate(body: ProfileRef, session: Session = Depends(get_session)):
    """Activa el perfil: comprueba el acceso, cambia el predeterminado y reencola lo que falte por indexar."""
    summary = _wrap(lambda: svc.activate(session, body.profile_id))
    return {"activation": summary, **svc.list_view(session)}


@router.post("/test")
def test_connection(body: TestIn, session: Session = Depends(get_session)):
    """Embebe una frase fija para validar URL, credenciales y dimensión (no envía documentos)."""

    def build() -> EmbedderConfig:
        base = svc.resolve(session, body.profile_id) if body.profile_id else None
        kind = body.kind or (base.kind if base else None)
        if kind is None:
            raise SettingsError("Indique el tipo de proveedor")
        key = clean_api_key(body.api_key) if body.api_key else (base.api_key if base else "")
        cfg = EmbedderConfig(
            kind=kind,
            model=body.model or (base.model if base else ""),
            base_url=body.base_url if body.base_url is not None else (base.base_url if base else ""),
            api_key=key,
            query_prefix=body.query_prefix if body.query_prefix is not None else (base.query_prefix if base else ""),
            passage_prefix=base.passage_prefix if base else "",
            dimensions=body.dimensions if body.dimensions is not None else (base.dimensions if base else None),
        )
        if not cfg.model:
            raise SettingsError("Indique el modelo")
        return cfg

    try:
        cfg = _wrap(build)
    except HTTPException as exc:
        if exc.status_code in (409, 500, 502):  # clave guardada ilegible, etc.
            return {"ok": False, "detail": exc.detail, "dimension": None, "latency_ms": None}
        raise
    return svc.probe(replace(cfg))


@router.delete("/models/{model_id}/vectors")
def purge(model_id: int, session: Session = Depends(get_session)):
    """Borra los vectores de un modelo que no está activo (libera espacio; volver a él exigirá reindexar)."""
    result = _wrap(lambda: svc.purge_vectors(session, model_id))
    return {"purged": result, **svc.list_view(session)}
