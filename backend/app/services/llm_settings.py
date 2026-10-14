"""Ajustes del proveedor de generación (LLM) editables desde la interfaz.

Fuentes de configuración:
- Perfil implícito **"env"**: lo que dicen las variables de entorno (por defecto, Ollama local). Siempre existe, no es editable
  y es el que se usa mientras no se elija otro (comportamiento previo preservado).
- Perfiles creados en Ajustes (`llm_profiles`): la clave API se guarda cifrada (services/credentials.py) y NUNCA se devuelve.
El perfil activo se guarda en `app_settings["active_llm"]`.

Consentimiento: un destino externo solo puede activarse/usarse si el perfil tiene `allow_external=True`, que se marca con una
acción explícita en la interfaz.
"""

from __future__ import annotations

import re
import uuid
from urllib.parse import urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import AppSetting, LLMProfile
from ..providers import LLMError, LLMProvider, ProviderConfig, build_provider, is_local_url
from . import credentials
from .net_validation import SettingsError, clean_api_key, validate_base_url

ENV_ID = "env"
ACTIVE_KEY = "active_llm"
KINDS = ("ollama", "openai-compatible")

# Plantillas para el formulario. Las URL son las publicadas por cada proveedor; verifíquelas en su documentación.
PRESETS = [
    {"id": "ollama-local", "label": "Ollama (local)", "kind": "ollama", "base_url": "http://host.docker.internal:11434",
     "needs_key": False, "key_url": None, "note": "Se ejecuta en su equipo; los datos no salen de él."},
    {"id": "groq", "label": "Groq", "kind": "openai-compatible", "base_url": "https://api.groq.com/openai/v1",
     "needs_key": True, "key_url": "https://console.groq.com/keys", "note": "Servicio externo."},
    {"id": "gemini", "label": "Google Gemini API", "kind": "openai-compatible",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "needs_key": True,
     "key_url": "https://aistudio.google.com/apikey", "note": "Servicio externo (endpoint compatible con OpenAI)."},
    {"id": "openai", "label": "OpenAI", "kind": "openai-compatible", "base_url": "https://api.openai.com/v1",
     "needs_key": True, "key_url": "https://platform.openai.com/api-keys", "note": "Servicio externo."},
    {"id": "lmstudio", "label": "LM Studio / llama.cpp / vLLM (local)", "kind": "openai-compatible",
     "base_url": "http://host.docker.internal:1234/v1", "needs_key": False, "key_url": None,
     "note": "Servidor local compatible con OpenAI."},
    {"id": "custom-openai", "label": "Otro proveedor compatible con OpenAI", "kind": "openai-compatible", "base_url": "",
     "needs_key": True, "key_url": None, "note": "Indique la URL base (la que precede a /chat/completions)."},
    {"id": "custom-ollama", "label": "Ollama en otra dirección", "kind": "ollama", "base_url": "",
     "needs_key": False, "key_url": None, "note": "Un Ollama remoto se trata como servicio externo."},
]

# Re-exportados desde net_validation (los usan los routers y las pruebas)
_clean_key = clean_api_key


# ------------------------------------------------------------------------------ configuraciones
def env_config() -> ProviderConfig:
    s = get_settings()
    kind = s.llm_provider if s.llm_provider in KINDS else "ollama"
    base = s.ollama_base_url if kind == "ollama" else s.llm_base_url
    key = s.llm_api_key.get_secret_value() if s.llm_api_key else ""
    return ProviderConfig(kind=kind, base_url=base, model=s.llm_model, api_key=key, allow_external=s.llm_allow_external,
                          timeout_s=s.llm_timeout_s, disable_thinking=s.llm_disable_thinking, label="Entorno (.env)")


def _decrypt(p: LLMProfile) -> str:
    if not p.api_key_enc:
        return ""
    try:
        return credentials.decrypt(p.api_key_enc)
    except credentials.SecretError as exc:
        raise LLMError(str(exc)) from exc


def profile_config(p: LLMProfile) -> ProviderConfig:
    s = get_settings()
    return ProviderConfig(kind=p.kind, base_url=p.base_url, model=p.model, api_key=_decrypt(p), allow_external=p.allow_external,
                          timeout_s=s.llm_timeout_s, disable_thinking=s.llm_disable_thinking, label=p.name)


def get_active_id(session: Session) -> str:
    row = session.get(AppSetting, ACTIVE_KEY)
    return row.value if row else ENV_ID


def resolve(session: Session, profile_id: str) -> ProviderConfig:
    if profile_id == ENV_ID:
        return env_config()
    p = _get_profile(session, profile_id)
    return profile_config(p)


def active_config(session: Session) -> ProviderConfig:
    active = get_active_id(session)
    try:
        return resolve(session, active)
    except SettingsError:  # el perfil activo fue eliminado: se vuelve al entorno
        return env_config()


def get_llm_provider(session: Session) -> LLMProvider:
    """Proveedor de generación activo (entorno o perfil de Ajustes). Puede lanzar LLMError."""
    return build_provider(active_config(session))


# ------------------------------------------------------------------------------ CRUD
def _get_profile(session: Session, profile_id: str) -> LLMProfile:
    try:
        pid = uuid.UUID(str(profile_id))
    except ValueError:
        raise SettingsError("Proveedor no encontrado") from None
    p = session.get(LLMProfile, pid)
    if p is None:
        raise SettingsError("Proveedor no encontrado")
    return p


def _validate_fields(kind: str, base_url: str, model: str, name: str | None = None) -> str:
    if kind not in KINDS:
        raise SettingsError(f"Tipo de proveedor no soportado: {kind}")
    if name is not None and not name.strip():
        raise SettingsError("El nombre es obligatorio")
    if not model.strip():
        raise SettingsError("El modelo es obligatorio (use «Cargar modelos» para ver los disponibles)")
    return validate_base_url(base_url)


def create_profile(session: Session, *, name: str, kind: str, base_url: str, model: str, api_key: str | None,
                   allow_external: bool) -> LLMProfile:
    base = _validate_fields(kind, base_url, model, name)
    if session.scalar(select(LLMProfile).where(LLMProfile.name == name.strip())):
        raise SettingsError("Ya existe un proveedor con ese nombre")
    key = _clean_key(api_key) if api_key else ""
    p = LLMProfile(name=name.strip(), kind=kind, base_url=base, model=model.strip(),
                   api_key_enc=credentials.encrypt(key) if key else None, allow_external=bool(allow_external))
    session.add(p)
    session.commit()
    return p


def update_profile(session: Session, profile_id: str, *, name: str | None = None, kind: str | None = None,
                   base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                   clear_api_key: bool = False, allow_external: bool | None = None) -> LLMProfile:
    p = _get_profile(session, profile_id)
    new_kind = kind or p.kind
    p.base_url = _validate_fields(new_kind, base_url if base_url is not None else p.base_url,
                                  model if model is not None else p.model, name)
    p.kind = new_kind
    if model is not None:
        p.model = model.strip()
    if name is not None and name.strip() != p.name:
        if session.scalar(select(LLMProfile).where(LLMProfile.name == name.strip())):
            raise SettingsError("Ya existe un proveedor con ese nombre")
        p.name = name.strip()
    if clear_api_key:
        p.api_key_enc = None
    elif api_key:  # None o "" = conservar la clave actual
        p.api_key_enc = credentials.encrypt(_clean_key(api_key))
    if allow_external is not None:
        p.allow_external = bool(allow_external)
    session.commit()
    return p


def delete_profile(session: Session, profile_id: str) -> None:
    p = _get_profile(session, profile_id)
    if get_active_id(session) == str(p.id):
        _set_setting(session, ACTIVE_KEY, ENV_ID)
    session.delete(p)
    session.commit()


def _set_setting(session: Session, key: str, value: str) -> None:
    row = session.get(AppSetting, key)
    if row:
        row.value = value
    else:
        session.add(AppSetting(key=key, value=value))


def set_active(session: Session, profile_id: str) -> None:
    if profile_id != ENV_ID:
        p = _get_profile(session, profile_id)
        cfg_external = not is_local_url(p.base_url)
        if cfg_external and not p.allow_external:
            raise SettingsError(
                "Es un servicio externo: marque «Permitir el envío de fragmentos a este servicio externo» antes de activarlo"
            )
    _set_setting(session, ACTIVE_KEY, profile_id)
    session.commit()


# ------------------------------------------------------------------------------ vistas (sin secretos)
def _view(*, id_: str, name: str, kind: str, base_url: str, model: str, has_key: bool, key_hint: str | None,
          key_error: bool, allow_external: bool, editable: bool, active: bool) -> dict:
    return {"id": id_, "name": name, "kind": kind, "base_url": base_url, "model": model, "has_key": has_key,
            "key_hint": key_hint, "key_error": key_error, "allow_external": allow_external,
            "external": not is_local_url(base_url), "editable": editable, "active": active}


def list_view(session: Session) -> dict:
    active = get_active_id(session)
    profiles = list(session.scalars(select(LLMProfile).order_by(LLMProfile.created_at)))
    if active != ENV_ID and not any(str(p.id) == active for p in profiles):
        active = ENV_ID
    e = env_config()
    out = [_view(id_=ENV_ID, name=e.label, kind=e.kind, base_url=e.base_url, model=e.model, has_key=bool(e.api_key),
                 key_hint=None, key_error=False, allow_external=e.allow_external, editable=False, active=active == ENV_ID)]
    for p in profiles:
        hint, err = None, False
        if p.api_key_enc:
            try:
                hint = credentials.hint(credentials.decrypt(p.api_key_enc))
            except credentials.SecretError:
                err = True
        out.append(_view(id_=str(p.id), name=p.name, kind=p.kind, base_url=p.base_url, model=p.model,
                         has_key=bool(p.api_key_enc), key_hint=hint, key_error=err, allow_external=p.allow_external,
                         editable=True, active=str(p.id) == active))
    return {"active": active, "profiles": out, "presets": PRESETS}


# ------------------------------------------------------------------------------ prueba de conexión
def probe(cfg: ProviderConfig) -> dict:
    """Comprueba conexión y credenciales listando modelos. No envía contenido de documentos. Nunca lanza."""
    try:
        base = validate_base_url(cfg.base_url)
    except SettingsError as exc:
        return {"ok": False, "detail": str(exc), "models": []}
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    url = f"{base}/api/tags" if cfg.kind == "ollama" else f"{base}/models"
    try:
        r = httpx.get(url, headers=headers, timeout=8.0, follow_redirects=False)
    except httpx.ConnectError:
        return {"ok": False, "detail": f"No se pudo conectar con {urlparse(base).netloc}", "models": []}
    except httpx.TimeoutException:
        return {"ok": False, "detail": "El servidor tardó demasiado en responder", "models": []}
    except httpx.HTTPError as exc:
        return {"ok": False, "detail": f"Error de red: {exc.__class__.__name__}", "models": []}
    if r.status_code in (401, 403):
        return {"ok": False, "detail": "Credenciales rechazadas (revise la clave API)", "models": []}
    if r.status_code == 404:
        return {"ok": False, "detail": "Ruta no encontrada: revise la URL base", "models": []}
    if r.status_code >= 300:
        return {"ok": False, "detail": f"El servidor respondió {r.status_code}", "models": []}
    try:
        data = r.json()
        if cfg.kind == "ollama":
            models = sorted(m["name"] for m in data.get("models", []))
        else:
            models = sorted(re.sub(r"^models/", "", m["id"]) for m in data.get("data", []))
    except (ValueError, KeyError, TypeError, AttributeError):
        return {"ok": False, "detail": "La respuesta no tiene el formato esperado para este tipo de proveedor", "models": []}
    return {"ok": True, "detail": f"Conexión correcta ({len(models)} modelos disponibles)", "models": models}
