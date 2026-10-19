"""Ajustes del modelo de visión (descripción de imágenes) editables desde la interfaz.

Es independiente del modelo de chat: puede usarse la misma cuenta (Gemini, Groq…) con OTRO modelo, el que admita imágenes. Por defecto está **desactivado**
(`none`): las imágenes se indexan con su vector y su OCR, pero sin descripción. Los perfiles guardan la clave API CIFRADA (services/credentials.py) y nunca
la devuelven. Un destino externo recibe las IMÁGENES: solo puede activarse con consentimiento explícito (`allow_external`).
El perfil activo se guarda en `app_settings["active_vision"]`.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import AppSetting, VisionProfile
from ..providers import LLMError, ProviderConfig, is_local_url
from . import credentials
from .llm_settings import probe  # noqa: F401 - la prueba de conexión (listar modelos) es la misma que la del LLM
from .net_validation import SettingsError, clean_api_key, validate_base_url

NONE_ID = "none"
ACTIVE_KEY = "active_vision"
ANSWER_KEY = "vision_answer"  # «1»/«0»: responder mirando las figuras con el modelo de visión (por defecto sí)
KINDS = ("ollama", "openai-compatible")

# Plantillas. Las URL son las publicadas por cada proveedor; los modelos son SUGERENCIAS (cambian con frecuencia): use «Cargar modelos» y verifique en su documentación.
PRESETS = [
    {"id": "gemini-vision", "label": "Google Gemini API", "kind": "openai-compatible",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-3.8-flash",
     "needs_key": True, "key_url": "https://aistudio.google.com/apikey",
     "note": "Servicio EXTERNO: cada imagen se envía a Google. Tiene capa gratuita con límites por minuto y por día; en ella, según la página de precios de Google, "
             "el contenido puede usarse para mejorar sus productos (no en la de pago): no la use con imágenes privadas. Los modelos cambian: `gemini-2.5-flash` ya no está disponible para cuentas nuevas; "
             "elija uno de «Cargar modelos». Si Google responde «Your project has been denied access», el proyecto de su clave no tiene permiso para generar (cree otra clave en un proyecto nuevo o revise el estado del proyecto)."},
    {"id": "groq-vision", "label": "Groq", "kind": "openai-compatible", "base_url": "https://api.groq.com/openai/v1", "model": "",
     "needs_key": True, "key_url": "https://console.groq.com/keys",
     "note": "Servicio EXTERNO: cada imagen se envía a Groq. Elija un modelo con visión en «Cargar modelos» (la documentación de Groq lista cuáles admiten imágenes y "
             "no indica si están en el plan gratuito: compruébelo con «Probar con una imagen de prueba»)."},
    {"id": "ollama-vision", "label": "Ollama (local)", "kind": "ollama", "base_url": "http://host.docker.internal:11434", "model": "qwen3-vl:4b",
     "needs_key": False, "key_url": None,
     "note": "Local: las imágenes no salen de su equipo. Requiere un modelo con visión (p. ej. `ollama pull qwen3-vl:4b`, ≈3,3 GB; `qwen3-vl:2b` ≈1,9 GB). "
             "La GPU la comparte con el modelo de chat: al alternar hay unos segundos de espera."},
    {"id": "custom-vision", "label": "Otro proveedor compatible con OpenAI", "kind": "openai-compatible", "base_url": "", "model": "",
     "needs_key": True, "key_url": None, "note": "Indique la URL base (la que precede a /chat/completions) y un modelo con visión."},
]


# ------------------------------------------------------------------------------ configuración
def _decrypt(p: VisionProfile) -> str:
    if not p.api_key_enc:
        return ""
    try:
        return credentials.decrypt(p.api_key_enc)
    except credentials.SecretError as exc:
        raise LLMError(str(exc)) from exc


def profile_config(p: VisionProfile) -> ProviderConfig:
    s = get_settings()
    return ProviderConfig(kind=p.kind, base_url=p.base_url, model=p.model, api_key=_decrypt(p), allow_external=p.allow_external,
                          timeout_s=s.llm_timeout_s, label=p.name)


def get_active_id(session: Session) -> str:
    row = session.get(AppSetting, ACTIVE_KEY)
    return row.value if row else NONE_ID


def _get_profile(session: Session, profile_id: str) -> VisionProfile:
    try:
        pid = uuid.UUID(str(profile_id))
    except ValueError:
        raise SettingsError("Modelo de visión no encontrado") from None
    p = session.get(VisionProfile, pid)
    if p is None:
        raise SettingsError("Modelo de visión no encontrado")
    return p


def resolve(session: Session, profile_id: str) -> ProviderConfig:
    return profile_config(_get_profile(session, profile_id))


def active_config(session: Session) -> ProviderConfig | None:
    """Configuración del perfil activo, o None si no hay modelo de visión (desactivado, o el perfil activo fue eliminado)."""
    active = get_active_id(session)
    if active == NONE_ID:
        return None
    try:
        return resolve(session, active)
    except SettingsError:
        return None


# ------------------------------------------------------------------------------ CRUD
def _validate(kind: str, base_url: str, model: str, name: str | None) -> str:
    if kind not in KINDS:
        raise SettingsError(f"Tipo de proveedor no soportado: {kind}")
    if name is not None and not name.strip():
        raise SettingsError("El nombre es obligatorio")
    if not model.strip():
        raise SettingsError("El modelo es obligatorio (use «Cargar modelos» y elija uno con visión)")
    return validate_base_url(base_url)


def create_profile(session: Session, *, name: str, kind: str, base_url: str, model: str, api_key: str | None, allow_external: bool) -> VisionProfile:
    base = _validate(kind, base_url, model, name)
    if session.scalar(select(VisionProfile).where(VisionProfile.name == name.strip())):
        raise SettingsError("Ya existe un modelo de visión con ese nombre")
    key = clean_api_key(api_key) if api_key else ""
    p = VisionProfile(name=name.strip(), kind=kind, base_url=base, model=model.strip(), api_key_enc=credentials.encrypt(key) if key else None,
                      allow_external=bool(allow_external))
    session.add(p)
    session.commit()
    return p


def update_profile(session: Session, profile_id: str, *, name: str | None = None, kind: str | None = None, base_url: str | None = None,
                   model: str | None = None, api_key: str | None = None, clear_api_key: bool = False,
                   allow_external: bool | None = None) -> VisionProfile:
    p = _get_profile(session, profile_id)
    new_kind = kind or p.kind
    p.base_url = _validate(new_kind, base_url if base_url is not None else p.base_url, model if model is not None else p.model, name)
    p.kind = new_kind
    if model is not None:
        p.model = model.strip()
    if name is not None and name.strip() != p.name:
        if session.scalar(select(VisionProfile).where(VisionProfile.name == name.strip())):
            raise SettingsError("Ya existe un modelo de visión con ese nombre")
        p.name = name.strip()
    if clear_api_key:
        p.api_key_enc = None
    elif api_key:  # None o "" = conservar la actual
        p.api_key_enc = credentials.encrypt(clean_api_key(api_key))
    if allow_external is not None:
        p.allow_external = bool(allow_external)
        if not p.allow_external and not is_local_url(p.base_url) and get_active_id(session) == str(p.id):
            _set_setting(session, ACTIVE_KEY, NONE_ID)  # retirar el consentimiento desactiva el modelo al instante
    session.commit()
    return p


def delete_profile(session: Session, profile_id: str) -> None:
    p = _get_profile(session, profile_id)
    if get_active_id(session) == str(p.id):
        _set_setting(session, ACTIVE_KEY, NONE_ID)
    session.delete(p)
    session.commit()


def _set_setting(session: Session, key: str, value: str) -> None:
    row = session.get(AppSetting, key)
    if row:
        row.value = value
    else:
        session.add(AppSetting(key=key, value=value))


def set_active(session: Session, profile_id: str) -> None:
    """Activa un perfil, o `none` para desactivar la descripción de imágenes."""
    if profile_id != NONE_ID:
        p = _get_profile(session, profile_id)
        if not is_local_url(p.base_url) and not p.allow_external:
            raise SettingsError(
                "Es un servicio externo: marque «Permitir el envío de las imágenes a este servicio externo» antes de activarlo"
            )
    _set_setting(session, ACTIVE_KEY, profile_id)
    session.commit()


# ------------------------------------------------------------------------------ vistas (sin secretos)
def get_answer_with_images(session: Session) -> bool:
    row = session.get(AppSetting, ANSWER_KEY)
    return row is None or row.value != "0"


def set_answer_with_images(session: Session, enabled: bool) -> None:
    _set_setting(session, ANSWER_KEY, "1" if enabled else "0")
    session.commit()


def list_view(session: Session) -> dict:
    active = get_active_id(session)
    profiles = list(session.scalars(select(VisionProfile).order_by(VisionProfile.created_at)))
    if active != NONE_ID and not any(str(p.id) == active for p in profiles):
        active = NONE_ID
    out = []
    for p in profiles:
        hint, err = None, False
        if p.api_key_enc:
            try:
                hint = credentials.hint(credentials.decrypt(p.api_key_enc))
            except credentials.SecretError:
                err = True
        out.append({"id": str(p.id), "name": p.name, "kind": p.kind, "base_url": p.base_url, "model": p.model, "has_key": bool(p.api_key_enc),
                    "key_hint": hint, "key_error": err, "allow_external": p.allow_external, "external": not is_local_url(p.base_url),
                    "editable": True, "active": str(p.id) == active})
    return {"active": active, "profiles": out, "presets": PRESETS, "answer_with_images": get_answer_with_images(session)}
