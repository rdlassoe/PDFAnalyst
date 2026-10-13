"""Módulo Ajustes: proveedores LLM configurables desde la API/UI, con claves cifradas y consentimiento explícito."""

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text

from app.config import get_settings
from app.main import app
from app.services import credentials, llm_settings

KEY = "gsk_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
GROQ = "https://api.groq.com/openai/v1"


@pytest.fixture
def client(session):  # `session` trunca las tablas (incluidas llm_profiles y app_settings)
    return TestClient(app)


def _create(client, **kw):
    body = dict(name="Groq", kind="openai-compatible", base_url=GROQ, model="llama-x", api_key=KEY, allow_external=False)
    body.update(kw)
    return client.post("/settings/llm/profiles", json=body)


def _profile(view, name):
    return next(p for p in view["profiles"] if p["name"] == name)


# ---------------------------------------------------------------- cifrado
def test_encrypt_roundtrip_and_ciphertext_is_not_plaintext():
    tok = credentials.encrypt(KEY)
    assert KEY not in tok and credentials.decrypt(tok) == KEY
    assert credentials.encrypt(KEY) != tok  # Fernet usa IV aleatorio


def test_decrypt_fails_cleanly_with_a_different_master_key(monkeypatch):
    tok = credentials.encrypt(KEY)
    monkeypatch.setattr(get_settings(), "app_secret_key", SecretStr(credentials.generate_master_key()))
    with pytest.raises(credentials.SecretError, match="Vuelva a introducirla"):
        credentials.decrypt(tok)


def test_invalid_master_key_is_reported(monkeypatch):
    monkeypatch.setattr(get_settings(), "app_secret_key", SecretStr("no-es-una-clave"))
    with pytest.raises(credentials.SecretError, match="no es válida"):
        credentials.encrypt("x")


def test_master_key_file_is_created_once_with_private_permissions(monkeypatch, tmp_path):
    import os
    import stat

    monkeypatch.setattr(get_settings(), "app_secret_key", None)
    monkeypatch.setattr(get_settings(), "storage_dir", tmp_path / "storage")
    path = get_settings().secret_key_path
    assert path.parent == tmp_path and not path.exists()
    tok = credentials.encrypt("abc")
    assert path.exists() and credentials.decrypt(tok) == "abc"  # la 2ª llamada reutiliza el mismo archivo
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_hint_reveals_at_most_four_chars():
    assert credentials.hint(KEY) == "…3456" and credentials.hint("corta") == "••••"


# ---------------------------------------------------------------- API: vista y alta
def test_default_is_env_ollama_and_presets_are_offered(client):
    v = client.get("/settings/llm").json()
    assert v["active"] == "env" and len(v["profiles"]) == 1
    env = v["profiles"][0]
    assert env["id"] == "env" and env["kind"] == "ollama" and env["active"] and not env["editable"] and not env["external"]
    assert {p["id"] for p in v["presets"]} >= {"ollama-local", "groq", "gemini", "openai", "lmstudio", "custom-openai"}


def test_api_key_is_encrypted_in_db_and_never_returned(client, session):
    r = _create(client)
    assert r.status_code == 201
    assert KEY not in r.text and "api_key" not in r.text.replace("has_key", "")
    p = _profile(r.json(), "Groq")
    assert p["has_key"] and p["key_hint"] == "…3456" and p["external"] and not p["allow_external"]
    enc = session.scalar(text("SELECT api_key_enc FROM llm_profiles"))
    assert enc and KEY not in enc and credentials.decrypt(enc) == KEY
    # y tampoco en ningún otro endpoint
    for path in ("/settings/llm", "/health", "/documents"):
        assert KEY not in client.get(path).text


def test_validation_errors(client):
    assert _create(client, base_url="ftp://x.com").status_code == 422
    assert _create(client, base_url="https://user:pw@api.groq.com/v1").status_code == 422
    assert _create(client, base_url="http://169.254.169.254/latest").status_code == 422  # metadatos de nube
    assert _create(client, base_url="http://metadata.google.internal/").status_code == 422
    assert _create(client, base_url="sin esquema").status_code == 422
    assert _create(client, api_key="clave con espacios").status_code == 422
    assert _create(client, api_key="abc\ndef").status_code == 422  # inyección de cabeceras
    assert _create(client, kind="otro").status_code == 422
    assert _create(client, model="").status_code == 422
    assert _create(client, name="").status_code == 422
    assert client.get("/settings/llm").json()["profiles"][1:] == []  # nada se guardó


def test_duplicate_name_is_409(client):
    assert _create(client).status_code == 201
    assert _create(client).status_code == 409


# ---------------------------------------------------------------- consentimiento y activación
def test_external_provider_cannot_be_activated_without_consent(client):
    pid = _profile(_create(client).json(), "Groq")["id"]
    r = client.put("/settings/llm/active", json={"profile_id": pid})
    assert r.status_code == 409 and "externo" in r.json()["detail"]
    assert client.get("/settings/llm").json()["active"] == "env"
    # tras el consentimiento explícito sí se puede
    assert client.patch(f"/settings/llm/profiles/{pid}", json={"allow_external": True}).status_code == 200
    v = client.put("/settings/llm/active", json={"profile_id": pid}).json()
    assert v["active"] == pid and _profile(v, "Groq")["active"]


def test_local_provider_activates_without_consent_and_can_return_to_env(client):
    pid = _profile(_create(client, name="LM Studio", base_url="http://host.docker.internal:1234/v1", api_key=None).json(), "LM Studio")["id"]
    assert client.put("/settings/llm/active", json={"profile_id": pid}).json()["active"] == pid
    assert client.put("/settings/llm/active", json={"profile_id": "env"}).json()["active"] == "env"


def test_activating_unknown_profile_fails(client):
    assert client.put("/settings/llm/active", json={"profile_id": "no-existe"}).status_code == 422
    import uuid

    assert client.put("/settings/llm/active", json={"profile_id": str(uuid.uuid4())}).status_code == 422


def test_active_profile_is_what_the_system_uses(client, session):
    pid = _profile(_create(client, allow_external=True, model="mi-modelo").json(), "Groq")["id"]
    client.put("/settings/llm/active", json={"profile_id": pid})
    session.expire_all()
    p = llm_settings.get_llm_provider(session)
    assert (p.name, p.model, p.external, p._key) == ("openai-compatible", "mi-modelo", True, KEY)  # clave descifrada en memoria
    client.put("/settings/llm/active", json={"profile_id": "env"})
    p = llm_settings.get_llm_provider(session)
    assert p.name == "ollama" and p.external is False


def test_revoking_consent_blocks_use_with_clear_message(client, session):
    from app.providers import LLMError

    pid = _profile(_create(client, allow_external=True).json(), "Groq")["id"]
    client.put("/settings/llm/active", json={"profile_id": pid})
    client.patch(f"/settings/llm/profiles/{pid}", json={"allow_external": False})
    session.expire_all()
    with pytest.raises(LLMError, match="externo"):
        llm_settings.get_llm_provider(session)


# ---------------------------------------------------------------- edición y borrado
def test_update_keeps_key_unless_replaced_or_cleared(client, session):
    pid = _profile(_create(client).json(), "Groq")["id"]
    enc0 = session.scalar(text("SELECT api_key_enc FROM llm_profiles"))
    client.patch(f"/settings/llm/profiles/{pid}", json={"model": "otro-modelo", "api_key": ""})  # "" = conservar
    assert session.scalar(text("SELECT api_key_enc FROM llm_profiles")) == enc0
    assert _profile(client.get("/settings/llm").json(), "Groq")["model"] == "otro-modelo"
    client.patch(f"/settings/llm/profiles/{pid}", json={"api_key": "NUEVA-CLAVE-1234567890"})
    session.expire_all()
    assert credentials.decrypt(session.scalar(text("SELECT api_key_enc FROM llm_profiles"))) == "NUEVA-CLAVE-1234567890"
    v = client.patch(f"/settings/llm/profiles/{pid}", json={"clear_api_key": True}).json()
    assert _profile(v, "Groq")["has_key"] is False
    assert session.scalar(text("SELECT api_key_enc FROM llm_profiles")) is None


def test_update_validates_and_rename_conflict(client):
    a = _profile(_create(client, name="A").json(), "A")["id"]
    _create(client, name="B")
    assert client.patch(f"/settings/llm/profiles/{a}", json={"name": "B"}).status_code == 409
    assert client.patch(f"/settings/llm/profiles/{a}", json={"base_url": "ftp://x"}).status_code == 422


def test_deleting_active_profile_returns_to_env(client, session):
    pid = _profile(_create(client, allow_external=True).json(), "Groq")["id"]
    client.put("/settings/llm/active", json={"profile_id": pid})
    v = client.delete(f"/settings/llm/profiles/{pid}").json()
    assert v["active"] == "env" and len(v["profiles"]) == 1
    assert session.scalar(text("SELECT count(*) FROM llm_profiles")) == 0
    assert client.delete(f"/settings/llm/profiles/{pid}").status_code == 422


def test_lost_master_key_is_flagged_not_crashing(client, monkeypatch):
    _create(client)
    monkeypatch.setattr(get_settings(), "app_secret_key", SecretStr(credentials.generate_master_key()))
    p = _profile(client.get("/settings/llm").json(), "Groq")
    assert p["has_key"] and p["key_error"] and p["key_hint"] is None


# ---------------------------------------------------------------- prueba de conexión
class _Resp:
    def __init__(self, status=200, data=None):
        self.status_code, self._d = status, data

    def json(self):
        if self._d is None:
            raise ValueError
        return self._d


def test_connection_lists_models_for_both_kinds(client, monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None, follow_redirects=None):
        seen["url"], seen["auth"], seen["redir"] = url, (headers or {}).get("Authorization"), follow_redirects
        if url.endswith("/api/tags"):
            return _Resp(data={"models": [{"name": "qwen3:8b"}, {"name": "llama3.1:8b"}]})
        return _Resp(data={"data": [{"id": "models/gemini-2.0-flash"}, {"id": "llama-70b"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    r = client.post("/settings/llm/test", json={"kind": "ollama", "base_url": "http://localhost:11434"}).json()
    assert r["ok"] and r["models"] == ["llama3.1:8b", "qwen3:8b"] and seen["url"] == "http://localhost:11434/api/tags"
    r = client.post("/settings/llm/test", json={"kind": "openai-compatible", "base_url": GROQ, "api_key": KEY}).json()
    assert r["ok"] and r["models"] == ["gemini-2.0-flash", "llama-70b"]  # sin el prefijo «models/»
    assert seen["auth"] == f"Bearer {KEY}" and seen["redir"] is False and seen["url"] == f"{GROQ}/models"


def test_connection_with_saved_profile_reuses_stored_key_without_exposing_it(client, monkeypatch):
    pid = _profile(_create(client).json(), "Groq")["id"]
    seen = {}

    def fake_get(url, headers=None, **kw):
        seen["auth"] = (headers or {}).get("Authorization")
        return _Resp(data={"data": [{"id": "m1"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    r = client.post("/settings/llm/test", json={"profile_id": pid})
    assert r.json()["ok"] and seen["auth"] == f"Bearer {KEY}" and KEY not in r.text


@pytest.mark.parametrize(
    "resp, fragment",
    [(_Resp(401), "Credenciales"), (_Resp(404), "URL base"), (_Resp(500), "500"), (_Resp(200, None), "formato")],
)
def test_connection_errors_are_reported_not_raised(client, monkeypatch, resp, fragment):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: resp)
    r = client.post("/settings/llm/test", json={"kind": "openai-compatible", "base_url": GROQ}).json()
    assert r["ok"] is False and fragment in r["detail"] and r["models"] == []


def test_connection_network_failures(client, monkeypatch):
    def refuse(*a, **k):
        raise httpx.ConnectError("x")

    monkeypatch.setattr(httpx, "get", refuse)
    assert "No se pudo conectar" in client.post("/settings/llm/test", json={"kind": "ollama", "base_url": "http://nada:1"}).json()["detail"]

    def slow(*a, **k):
        raise httpx.ReadTimeout("x")

    monkeypatch.setattr(httpx, "get", slow)
    assert "tardó" in client.post("/settings/llm/test", json={"kind": "ollama", "base_url": "http://nada:1"}).json()["detail"]


def test_connection_refuses_ssrf_targets_without_making_a_request(client, monkeypatch):
    def must_not_be_called(*a, **k):
        raise AssertionError("no debe haber petición de red")

    monkeypatch.setattr(httpx, "get", must_not_be_called)
    r = client.post("/settings/llm/test", json={"kind": "openai-compatible", "base_url": "http://169.254.169.254/latest/meta-data"}).json()
    assert r["ok"] is False and "no está permitida" in r["detail"]
    r = client.post("/settings/llm/test", json={"kind": "ollama", "base_url": "file:///etc/passwd"}).json()
    assert r["ok"] is False


def test_connection_with_unreadable_stored_key_says_so(client, monkeypatch):
    pid = _profile(_create(client).json(), "Groq")["id"]
    monkeypatch.setattr(get_settings(), "app_secret_key", SecretStr(credentials.generate_master_key()))
    r = client.post("/settings/llm/test", json={"profile_id": pid}).json()
    assert r["ok"] is False and "Vuelva a introducirla" in r["detail"]


@pytest.mark.ollama
def test_real_ollama_connection_lists_installed_models(client):
    import os

    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    r = client.post("/settings/llm/test", json={"kind": "ollama", "base_url": base}).json()
    if not r["ok"]:
        pytest.skip(r["detail"])
    assert any(m.startswith("qwen3") for m in r["models"])
