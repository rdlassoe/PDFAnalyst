"""Almacenamiento local de PDFs originales.

Los nombres en disco los genera el servidor (`<uuid>.pdf`); el nombre que envía el usuario solo se
guarda como etiqueta de visualización y NUNCA participa en rutas (previene path traversal).
"""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path

from ..config import get_settings

_STORAGE_NAME_RE = re.compile(r"^[0-9a-f]{32}\.pdf$")
PDF_MAGIC = b"%PDF-"


class StorageError(Exception):
    pass


def storage_root() -> Path:
    root = get_settings().storage_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_storage_name() -> str:
    return f"{uuid.uuid4().hex}.pdf"


def resolve_storage_path(storage_name: str) -> Path:
    """Convierte un nombre almacenado en ruta absoluta, garantizando que queda dentro del directorio."""
    if not _STORAGE_NAME_RE.match(storage_name):
        raise StorageError("Nombre de almacenamiento inválido")
    root = storage_root()
    path = (root / storage_name).resolve()
    if path.parent != root:
        raise StorageError("Ruta fuera del directorio autorizado")
    return path


def sanitize_display_name(name: str | None) -> str:
    """Solo para mostrar: quita directorios y caracteres de control."""
    base = (name or "documento.pdf").replace("\\", "/").split("/")[-1]
    base = re.sub(r"[\x00-\x1f\x7f]", "", base).strip()
    return (base or "documento.pdf")[:255]


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_pdf(storage_name: str, data: bytes) -> Path:
    path = resolve_storage_path(storage_name)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)  # escritura atómica
    return path


def delete_pdf(storage_name: str) -> None:
    try:
        resolve_storage_path(storage_name).unlink(missing_ok=True)
    except StorageError:
        pass
