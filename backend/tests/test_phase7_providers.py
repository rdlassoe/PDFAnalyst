"""Fase 7 — proveedor openai-compatible: forma de las peticiones, errores, secreto y bloqueo de externos.
Verificación real: se ejecuta contra el endpoint /v1 de Ollama (un servidor genuinamente compatible).
NO se ha verificado contra OpenAI/Groq/Gemini (requieren claves y envían datos fuera de la máquina)."""

import json

import httpx
import pytest

from app.providers import ChatMessage, LLMError, ProviderConfig, build_provider, is_local_url
from app.providers.openai_compat import OpenAICompatProvider


def _cfg(**kw) -> ProviderConfig:
    base = dict(kind="openai-compatible", base_url="http://localhost:9999/v1", model="m")
    base.update(kw)
    return ProviderConfig(**base)


def _transport(handler):
    return httpx.MockTransport(handler)


MSG = [ChatMessage("system", "s"), ChatMessage("user", "hola")]


def test_request_shape_and_bearer_header():
    seen = {}

    def handler(req: httpx.Request):
        seen["url"], seen["auth"], seen["body"] = str(req.url), req.headers.get("authorization"), json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " respuesta [C1] "}}]})

    p = OpenAICompatProvider(_cfg(api_key="sk-secreto"), _transport(handler))
    assert p.chat(MSG, temperature=0.1, max_tokens=50) == "respuesta [C1]"
    assert seen["url"] == "http://localhost:9999/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secreto"
    assert seen["body"]["model"] == "m" and seen["body"]["stream"] is False and seen["body"]["max_tokens"] == 50
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "status, fragment",
    [(401, "credenciales"), (403, "credenciales"), (404, "no encontrados"), (429, "limitó")],
)
def test_http_errors_become_llmerror_without_leaking_key(status, fragment):
    p = OpenAICompatProvider(_cfg(api_key="sk-secreto"), _transport(lambda r: httpx.Response(status, text="sk-secreto")))
    with pytest.raises(LLMError, match=fragment) as e:
        p.chat(MSG, temperature=0, max_tokens=5)
    assert "sk-secreto" not in str(e.value)


def test_malformed_and_unreachable():
    p = OpenAICompatProvider(_cfg(), _transport(lambda r: httpx.Response(200, json={"nope": 1})))
    with pytest.raises(LLMError, match="inválida"):
        p.chat(MSG, temperature=0, max_tokens=5)

    def boom(req):
        raise httpx.ConnectError("x")

    with pytest.raises(LLMError, match="No se pudo conectar"):
        OpenAICompatProvider(_cfg(), _transport(boom)).chat(MSG, temperature=0, max_tokens=5)


def test_health_never_raises_and_hides_key():
    ok = OpenAICompatProvider(_cfg(), _transport(lambda r: httpx.Response(200, json={"data": []}))).health()
    assert ok["ok"] is True
    bad = OpenAICompatProvider(_cfg(api_key="sk-x"), _transport(lambda r: httpx.Response(401))).health()
    assert bad == {"ok": False, "detail": "credenciales rechazadas"}


def test_api_key_is_never_printed():
    c = _cfg(api_key="sk-secreto")
    assert "sk-secreto" not in repr(c) and "sk-secreto" not in str(c)


def test_external_provider_is_blocked_until_explicitly_allowed():
    ext = "https://api.groq.com/openai/v1"
    assert not is_local_url(ext) and is_local_url("http://host.docker.internal:11434/v1")
    with pytest.raises(LLMError, match="externo"):
        OpenAICompatProvider(_cfg(base_url=ext))
    p = OpenAICompatProvider(_cfg(base_url=ext, allow_external=True))
    assert p.external is True


@pytest.mark.parametrize(
    "url, local",
    [
        ("http://localhost:11434", True), ("http://127.0.0.1:1234/v1", True), ("http://[::1]:8000", True),
        ("http://192.168.1.50:1234", True), ("http://10.1.2.3/v1", True), ("http://172.20.0.5", True),
        ("http://ollama:11434", True), ("http://host.docker.internal:11434", True),
        ("https://api.groq.com/openai/v1", False), ("http://8.8.8.8/v1", False),
        ("http://10.evil.com/v1", False),  # un DOMINIO que empieza por «10.» NO es local
        ("http://192.168.evil.com/v1", False), ("http://localhost.evil.com/v1", False), ("not a url", False),
    ],
)
def test_is_local_url_cannot_be_spoofed(url, local):
    assert is_local_url(url) is local


def test_remote_ollama_also_requires_consent():
    with pytest.raises(LLMError, match="externo"):
        build_provider(ProviderConfig(kind="ollama", base_url="https://ollama.example.com", model="m"))
    assert build_provider(ProviderConfig(kind="ollama", base_url="https://ollama.example.com", model="m", allow_external=True)).external


def test_missing_base_url_and_unknown_kind():
    with pytest.raises(LLMError, match="URL base"):
        OpenAICompatProvider(_cfg(base_url=""))
    with pytest.raises(LLMError, match="no soportado"):
        build_provider(_cfg(kind="nada"))
    assert isinstance(build_provider(_cfg()), OpenAICompatProvider)


def test_blocked_external_provider_gives_error_message_not_crash(session, monkeypatch, embedder):
    """Con un proveedor externo sin consentimiento, la consulta responde con un aviso y la API sigue viva."""
    from app.models import Conversation
    from app.services import llm_settings
    from app.services.answering import answer_question
    from app.services.ingestion import register_upload
    from app.worker import process_one

    from . import corpus

    register_upload(session, "politica_seguridad.pdf", corpus.build("politica_seguridad.pdf"))
    while process_one(session):
        pass
    monkeypatch.setattr(llm_settings, "env_config", lambda: _cfg(base_url="https://api.openai.com/v1"))
    conv = Conversation()
    session.add(conv)
    session.flush()
    res = answer_question(session, conv, "¿Cada cuánto debo cambiar la contraseña?")
    assert res.assistant_message.status == "error" and "externo" in res.assistant_message.content


@pytest.mark.ollama
def test_real_server_ollama_v1_endpoint(session):
    """Contra un servidor compatible real: el endpoint /v1 de Ollama."""
    import os

    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/v1"
    p = OpenAICompatProvider(_cfg(base_url=base, model=os.environ.get("LLM_MODEL", "qwen3:8b")))
    h = p.health()
    if not h["ok"]:
        pytest.skip(h["detail"])
    out = p.chat([ChatMessage("user", "Responde solo con la palabra: hola")], temperature=0, max_tokens=200)
    assert out and "hola" in out.lower()
