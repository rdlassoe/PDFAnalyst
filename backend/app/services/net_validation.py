"""Validaciones compartidas por los módulos de Ajustes (LLM y embeddings): URL de proveedor y claves API."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

_BLOCKED_HOSTS = {"metadata.google.internal", "metadata", "instance-data"}
_CTRL = re.compile(r"[\x00-\x1f\x7f\s]")


class SettingsError(ValueError):
    """Entrada inválida (se traduce a 422/409 en la API)."""


def validate_base_url(url: str) -> str:
    """Solo http(s), sin credenciales embebidas y sin direcciones de enlace local / metadatos de nube (anti-SSRF)."""
    url = (url or "").strip()
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise SettingsError("La URL debe empezar por http:// o https:// e incluir el servidor")
    if p.username or p.password:
        raise SettingsError("No incluya usuario ni contraseña en la URL; use el campo de clave API")
    host = p.hostname.lower()
    blocked = host in _BLOCKED_HOSTS
    try:
        ip = ipaddress.ip_address(host)
        blocked = blocked or ip.is_link_local or ip.is_multicast or ip.is_unspecified  # 169.254.x.x (metadatos de nube)
    except ValueError:
        pass
    if blocked:
        raise SettingsError("Esa dirección no está permitida como proveedor (metadatos de red/enlace local)")
    return url.rstrip("/")


def clean_api_key(key: str) -> str:
    """Rechaza espacios/saltos de línea (inyección de cabeceras)."""
    key = key.strip()
    if _CTRL.search(key):
        raise SettingsError("La clave API contiene espacios o caracteres de control")
    return key
