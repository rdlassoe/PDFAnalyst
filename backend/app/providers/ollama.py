from __future__ import annotations

import time

import httpx

from .base import ChatMessage, LLMError, LLMProvider, ProviderConfig, check_external_allowed


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, cfg: ProviderConfig) -> None:
        check_external_allowed(cfg)  # un Ollama remoto también envía los fragmentos fuera de esta máquina
        self.external = cfg.external
        self.model = cfg.model
        self._base = cfg.base_url.rstrip("/")
        self._timeout = cfg.timeout_s
        self._no_think = cfg.disable_thinking

    def chat(self, messages: list[ChatMessage], *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [self._message(m) for m in messages],
            "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192},
        }
        if self._no_think:
            payload["think"] = False
        try:
            r = self._post(payload)
            if r.status_code == 400 and "think" in r.text.lower():  # el modelo no admite el parámetro
                payload.pop("think", None)
                r = self._post(payload)
            if r.status_code == 404:
                raise LLMError(
                    f"El modelo '{self.model}' no está instalado en Ollama. Ejecute: ollama pull {self.model}"
                )
            r.raise_for_status()
            content = r.json()["message"]["content"]
        except LLMError:
            raise
        except httpx.ConnectError as exc:
            raise LLMError(f"No se pudo conectar con Ollama en {self._base}. ¿Está en ejecución?") from exc
        except httpx.TimeoutException as exc:
            raise LLMError("Ollama tardó demasiado en responder") from exc
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise LLMError(f"Respuesta inválida de Ollama: {exc}") from exc
        return content or ""

    @staticmethod
    def _message(m: ChatMessage) -> dict:
        out: dict = {"role": m.role, "content": m.content}
        if m.images:  # Ollama recibe las imágenes en base64 dentro del mensaje
            out["images"] = [img.b64() for img in m.images]
        return out

    def _post(self, payload: dict) -> httpx.Response:
        """POST con reintentos ante fallos transitorios (500/502/503 al cargar o descargar el modelo de la GPU)."""
        last: httpx.Response | None = None
        for attempt in range(3):
            last = httpx.post(f"{self._base}/api/chat", json=payload, timeout=self._timeout)
            if last.status_code not in (500, 502, 503):
                return last
            time.sleep(2.0 * (attempt + 1))
        assert last is not None
        return last

    def health(self) -> dict:
        try:
            r = httpx.get(f"{self._base}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = {m["name"] for m in r.json().get("models", [])}
            has = self.model in names or f"{self.model}:latest" in names
            return {
                "ok": has,
                "detail": "listo" if has else f"modelo '{self.model}' no instalado (ollama pull {self.model})",
            }
        except Exception as exc:  # noqa: BLE001 - health nunca lanza
            return {"ok": False, "detail": f"Ollama no disponible: {exc.__class__.__name__}"}
