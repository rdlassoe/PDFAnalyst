from __future__ import annotations

from .base import ChatMessage, LLMError, LLMProvider, ProviderConfig, is_local_url
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider


def build_provider(cfg: ProviderConfig) -> LLMProvider:
    """Construye el adaptador adecuado. Lanza LLMError si falta configuración o si es externo sin consentimiento."""
    if cfg.kind == "ollama":
        return OllamaProvider(cfg)
    if cfg.kind == "openai-compatible":
        return OpenAICompatProvider(cfg)
    raise LLMError(f"Tipo de proveedor no soportado: {cfg.kind}")


__all__ = [
    "ChatMessage", "LLMError", "LLMProvider", "ProviderConfig", "build_provider", "is_local_url",
    "OllamaProvider", "OpenAICompatProvider",
]
