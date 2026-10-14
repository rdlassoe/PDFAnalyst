"""Interfaz común de proveedores de generación (D13). Cada adaptador implementa `chat` y `health`."""

from __future__ import annotations

import ipaddress
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse


class LLMError(Exception):
    """Fallo del proveedor (red, modelo ausente, respuesta inválida). Nunca debe tumbar la API."""


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMProvider(ABC):
    name: str
    model: str
    #: ¿Los datos salen de esta máquina? Se muestra en la UI y exige consentimiento informado (Fase 7).
    external: bool = False

    @abstractmethod
    def chat(self, messages: list[ChatMessage], *, temperature: float, max_tokens: int) -> str: ...

    @abstractmethod
    def health(self) -> dict:
        """Debe devolver {'ok': bool, 'detail': str}; nunca lanzar."""


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama", "embedder", "db"}


def is_local_url(url: str) -> bool:
    """¿El destino está en esta máquina o en la red privada del despliegue? Si no, se considera servicio EXTERNO.

    Los nombres de dominio solo cuentan como locales si están en una lista cerrada; los rangos privados se reconocen
    únicamente cuando el host ES una IP (un dominio como «10.evil.com» no es local).
    """
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host in _LOCAL_HOSTS or host.endswith(".localhost")
    return ip.is_loopback or ip.is_private


@dataclass(frozen=True)
class ProviderConfig:
    """Configuración resuelta (entorno o perfil de Ajustes) con la que se construye un proveedor."""

    kind: str  # "ollama" | "openai-compatible"
    base_url: str
    model: str
    api_key: str = ""
    allow_external: bool = False
    timeout_s: float = 180.0
    disable_thinking: bool = True  # solo Ollama
    label: str = ""

    def __repr__(self) -> str:  # nunca imprimir la clave
        return f"ProviderConfig(kind={self.kind!r}, base_url={self.base_url!r}, model={self.model!r}, api_key={'***' if self.api_key else ''})"

    @property
    def external(self) -> bool:
        return not is_local_url(self.base_url)


def check_external_allowed(cfg: ProviderConfig) -> None:
    if cfg.external and not cfg.allow_external:
        raise LLMError(
            f"El proveedor {urlparse(cfg.base_url).hostname} es externo: los fragmentos recuperados saldrían de esta máquina. "
            "Para permitirlo, marque «Permitir el envío de fragmentos a este servicio externo» en Ajustes "
            "(o defina LLM_ALLOW_EXTERNAL=true si usa el proveedor del entorno)."
        )
