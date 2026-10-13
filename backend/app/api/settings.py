from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..db import get_session
from ..providers import LLMError, ProviderConfig
from ..services import credentials, llm_settings as svc

router = APIRouter(prefix="/settings/llm", tags=["settings"])

Kind = Literal["ollama", "openai-compatible"]


class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: Kind
    base_url: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=256)
    api_key: str | None = Field(default=None, max_length=512, repr=False)
    allow_external: bool = False


class ProfilePatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    kind: Kind | None = None
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    api_key: str | None = Field(default=None, max_length=512, repr=False)  # None/"" = conservar la actual
    clear_api_key: bool = False
    allow_external: bool | None = None


class ActiveIn(BaseModel):
    profile_id: str = Field(max_length=64)


class TestIn(BaseModel):
    """Prueba un borrador (sin guardarlo) o un perfil existente. La clave del borrador no se almacena."""

    profile_id: str | None = Field(default=None, max_length=64)
    kind: Kind | None = None
    base_url: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=512, repr=False)


def _wrap(fn):
    try:
        return fn()
    except svc.SettingsError as exc:
        msg = str(exc)
        raise HTTPException(409 if "externo" in msg or "Ya existe" in msg else 422, msg) from exc
    except credentials.SecretError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("")
def get_settings_view(session: Session = Depends(get_session)):
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


@router.put("/active")
def activate(body: ActiveIn, session: Session = Depends(get_session)):
    _wrap(lambda: svc.set_active(session, body.profile_id))
    return svc.list_view(session)


@router.post("/test")
def test_connection(body: TestIn, session: Session = Depends(get_session)):
    """Lista los modelos del proveedor para validar URL y credenciales (no envía documentos)."""

    def build() -> ProviderConfig:
        base = svc.resolve(session, body.profile_id) if body.profile_id else None
        kind = body.kind or (base.kind if base else None)
        url = body.base_url if body.base_url is not None else (base.base_url if base else "")
        if kind is None:
            raise svc.SettingsError("Indique el tipo de proveedor")
        key = svc._clean_key(body.api_key) if body.api_key else (base.api_key if base else "")
        return ProviderConfig(kind=kind, base_url=url, model=base.model if base else "", api_key=key)

    try:
        cfg = _wrap(build)
    except LLMError as exc:  # clave guardada ilegible
        return {"ok": False, "detail": str(exc), "models": []}
    return svc.probe(cfg)
