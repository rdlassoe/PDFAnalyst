"""Cifrado en reposo de claves API (Fernet: AES-128-CBC + HMAC-SHA256, autenticado).

Modelo de amenaza: un volcado/copia de la base de datos NO debe revelar las claves. Por eso la clave maestra vive FUERA de
PostgreSQL: en `APP_SECRET_KEY` (entorno) o, si no existe, en `<dir de datos>/secret.key` (creada con permisos 0600 la primera
vez). Quien tenga acceso al contenedor/volumen de la aplicación sí puede descifrarlas: esto no protege frente a un
atacante con ejecución de código en el servidor, solo frente a fugas de la base de datos y de sus copias de seguridad.
"""

from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from ..config import get_settings


class SecretError(Exception):
    """La clave guardada no puede descifrarse (clave maestra perdida/cambiada) o el material es inválido."""


def _load_or_create_key_file(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # O_EXCL: si dos procesos (api y worker) arrancan a la vez, solo uno crea el archivo; el otro lo lee.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    except FileExistsError:
        key = path.read_bytes().strip()
        if not key:  # carrera: el otro proceso aún no escribió
            raise SecretError(f"{path} está vacío; reintente en unos segundos") from None
        return key
    with os.fdopen(fd, "wb") as f:
        key = Fernet.generate_key()
        f.write(key)
    return key


def _fernet() -> Fernet:
    s = get_settings()
    raw = s.app_secret_key.get_secret_value() if s.app_secret_key else None
    key = raw.encode() if raw else _load_or_create_key_file(s.secret_key_path)
    try:
        return Fernet(key)
    except (ValueError, TypeError) as exc:
        raise SecretError("La clave maestra no es válida (debe ser base64 urlsafe de 32 bytes)") from exc


def encrypt(plain: str) -> str:
    return _fernet().encrypt(plain.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise SecretError(
            "No se puede descifrar la clave guardada (la clave maestra cambió o se perdió). Vuelva a introducirla en Ajustes."
        ) from exc


def hint(plain: str) -> str:
    """Pista no sensible para la UI: solo los 4 últimos caracteres y solo si la clave es suficientemente larga."""
    return f"…{plain[-4:]}" if len(plain) >= 12 else "••••"


def generate_master_key() -> str:
    return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")
