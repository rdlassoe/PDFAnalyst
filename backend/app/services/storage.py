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

_STORAGE_NAME_RE = re.compile(r"^[0-9a-f]{32}\.(pdf|png|jpg|webp)$")
PDF_MAGIC = b"%PDF-"
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}  # sin SVG ni GIF animado: no se interpretan


class StorageError(Exception):
    pass


def storage_root() -> Path:
    root = get_settings().storage_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_storage_name(ext: str = "pdf") -> str:
    if ext != "pdf" and ext not in IMAGE_TYPES:
        raise StorageError("Extensión no admitida")
    return f"{uuid.uuid4().hex}.{ext}"


def detect_image_type(data: bytes) -> str | None:
    """Extensión (png|jpg|webp) según la FIRMA del archivo (nunca por el nombre ni el tipo que declare el cliente); None si no es una imagen admitida."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


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


write_blob, delete_blob = write_pdf, delete_pdf  # el nombre histórico es «pdf», pero sirven para cualquier archivo almacenado


def thumb_path(doc_id: uuid.UUID, create: bool = True) -> Path:
    """Miniatura de un medio: <raíz>/thumbs/<uuid>.jpg (el nombre sale del id del documento, nunca del usuario)."""
    d = storage_root() / "thumbs"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d / f"{doc_id.hex}.jpg"


def delete_thumb(doc_id: uuid.UUID) -> None:
    thumb_path(doc_id, create=False).unlink(missing_ok=True)


def figure_dir(doc_id: uuid.UUID, create: bool = False) -> Path:
    """Figuras extraídas de un PDF: <raíz>/figures/<uuid del documento>/ (nombres generados por el servidor)."""
    d = storage_root() / "figures" / doc_id.hex
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def figure_path(doc_id: uuid.UUID, n: int, create: bool = False) -> Path:
    return figure_dir(doc_id, create) / f"f{int(n):03d}.jpg"


def delete_figures(doc_id: uuid.UUID) -> None:
    import shutil

    shutil.rmtree(figure_dir(doc_id), ignore_errors=True)
