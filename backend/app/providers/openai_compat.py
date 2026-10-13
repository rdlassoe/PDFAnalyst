"""Adaptador para cualquier servidor con la API `/chat/completions` de OpenAI.

Cubre servidores locales (llama.cpp, LM Studio, vLLM, el endpoint /v1 de Ollama) y servicios externos
(OpenAI, Groq, Gemini con su endpoint compatible) cambiando solo `LLM_BASE_URL`, `LLM_MODEL` y `LLM_API_KEY`.
La clave llega ya descifrada en `ProviderConfig` (en memoria); nunca se registra ni se devuelve por la API.
"""

from __future__ import annotations

import httpx

from .base import ChatMessage, LLMError, LLMProvider, ProviderConfig, check_external_allowed, is_local_url  # noqa: F401


class OpenAICompatProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(self, cfg: ProviderConfig, transport: httpx.BaseTransport | None = None) -> None:
        if not cfg.base_url:
            raise LLMError("La URL base es obligatoria para un proveedor compatible con OpenAI")
        check_external_allowed(cfg)
        self.external = cfg.external
        self.model = cfg.model
        self._base = cfg.base_url.rstrip("/")
        self._key = cfg.api_key
        self._timeout = cfg.timeout_s
        self._transport = transport

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._key}"} if self._key else {}

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout, transport=self._transport, headers=self._headers())

    def chat(self, messages: list[ChatMessage], *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        try:
            with self._client() as c:
                r = c.post(f"{self._base}/chat/completions", json=payload)
            if r.status_code in (401, 403):
                raise LLMError("El proveedor rechazó las credenciales (revise LLM_API_KEY)")
            if r.status_code == 404:
                raise LLMError(f"Modelo o ruta no encontrados en el proveedor (modelo '{self.model}')")
            if r.status_code == 429:
                raise LLMError("El proveedor limitó las peticiones (429); reintente en unos instantes")
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except LLMError:
            raise
        except httpx.ConnectError as exc:
            raise LLMError(f"No se pudo conectar con {self._base}") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("El proveedor tardó demasiado en responder") from exc
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Respuesta inválida del proveedor: {exc.__class__.__name__}") from exc

    def health(self) -> dict:
        try:
            with self._client() as c:
                r = c.get(f"{self._base}/models", timeout=5.0)
            if r.status_code in (401, 403):
                return {"ok": False, "detail": "credenciales rechazadas"}
            r.raise_for_status()
            return {"ok": True, "detail": "listo"}
        except Exception as exc:  # noqa: BLE001 - health nunca lanza
            return {"ok": False, "detail": f"proveedor no disponible: {exc.__class__.__name__}"}
