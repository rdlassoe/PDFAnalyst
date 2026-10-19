"""Modelo de visión (descripción de imágenes): envío de imágenes por los adaptadores, Ajustes (Gemini/Groq/Ollama con clave cifrada y consentimiento) y servicio.

Las llamadas a los servicios externos están SIMULADAS (no hay claves ni se envía nada fuera): se verifica la forma de las peticiones, el consentimiento, los reintentos
y el manejo de errores. NO se ha verificado contra Gemini o Groq reales (requiere sus claves y envía imágenes a esos servicios).
"""

import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.main import app
from app.models import VisionProfile
from app.providers import ChatMessage, ImagePart, LLMError, LLMRateLimited, ProviderConfig
from app.providers.ollama import OllamaProvider
from app.providers.openai_compat import OpenAICompatProvider
from app.services import vision, vision_settings

KEY = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai"
GROQ = "https://api.groq.com/openai/v1"
SEC = "/settings/vision"


def _png(size=(64, 48), color="red") -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, format="PNG")
    return b.getvalue()


@pytest.fixture
def client(session):  # `session` trunca las tablas (incluida vision_profiles)
    return TestClient(app)


def _create(client, **kw):
    body = dict(name="Gemini", kind="openai-compatible", base_url=GEMINI, model="gemini-2.5-flash", api_key=KEY, allow_external=False)
    body.update(kw)
    return client.post(f"{SEC}/profiles", json=body)


def _profile(view, name):
    return next(p for p in view["profiles"] if p["name"] == name)


# ---------------------------------------------------------------- adaptadores: envío de imágenes
def test_openai_compat_sends_images_as_data_urls_and_keeps_text_only_messages_plain():
    seen = {}

    def handler(req: httpx.Request):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Un círculo rojo. "}}]})

    p = OpenAICompatProvider(ProviderConfig(kind="openai-compatible", base_url="http://localhost:9999/v1", model="m"), httpx.MockTransport(handler))
    img = ImagePart(b"\xff\xd8datos", "image/jpeg")
    out = p.chat([ChatMessage("system", "s"), ChatMessage("user", "¿Qué ves?", images=(img, img))], temperature=0, max_tokens=50)
    assert out == "Un círculo rojo."
    sys_msg, user_msg = seen["body"]["messages"]
    assert sys_msg == {"role": "system", "content": "s"}  # sin imágenes: formato de texto de siempre
    assert user_msg["content"][0] == {"type": "text", "text": "¿Qué ves?"}
    urls = [c["image_url"]["url"] for c in user_msg["content"][1:]]
    assert len(urls) == 2 and all(u.startswith("data:image/jpeg;base64,") for u in urls)


def test_ollama_sends_images_as_base64_list(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen["json"] = json
        return httpx.Response(200, json={"message": {"content": "hola"}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    p = OllamaProvider(ProviderConfig(kind="ollama", base_url="http://localhost:11434", model="qwen3-vl:4b"))
    p.chat([ChatMessage("user", "describe", images=(ImagePart(b"abc", "image/jpeg"),)), ChatMessage("assistant", "ok")], temperature=0, max_tokens=5)
    first, second = seen["json"]["messages"]
    assert first["images"] == ["YWJj"] and "images" not in second  # "abc" en base64


def test_rate_limit_is_a_distinct_error_with_retry_after():
    def handler(req):
        return httpx.Response(429, headers={"retry-after": "7"}, text="slow down")

    p = OpenAICompatProvider(ProviderConfig(kind="openai-compatible", base_url="http://localhost:9/v1", model="m"), httpx.MockTransport(handler))
    with pytest.raises(LLMRateLimited, match="limitó") as e:
        p.chat([ChatMessage("user", "x")], temperature=0, max_tokens=5)
    assert e.value.retry_after == 7.0 and isinstance(e.value, LLMError)  # sigue siendo un LLMError para el resto del código


def test_image_bytes_never_appear_in_reprs():
    m = ChatMessage("user", "x", images=(ImagePart(b"SECRETO-BINARIO", "image/png"),))
    assert "SECRETO" not in repr(m)


# ---------------------------------------------------------------- Ajustes: perfiles, clave cifrada, consentimiento
def test_default_view_is_disabled_and_has_gemini_groq_ollama_presets(client):
    v = client.get(SEC).json()
    assert v["active"] == "none" and v["profiles"] == []
    ids = {p["id"]: p for p in v["presets"]}
    assert {"gemini-vision", "groq-vision", "ollama-vision", "custom-vision"} <= set(ids)
    assert ids["gemini-vision"]["base_url"] == GEMINI and ids["groq-vision"]["base_url"] == GROQ and ids["ollama-vision"]["kind"] == "ollama"
    assert ids["gemini-vision"]["model"] != "gemini-2.5-flash"  # retirado para cuentas nuevas (verificado con una cuenta real)
    assert "mejorar sus productos" in ids["gemini-vision"]["note"] and "EXTERNO" in ids["groq-vision"]["note"]


@pytest.mark.parametrize(
    "kw",
    [dict(base_url="ftp://x.com"), dict(base_url="https://u:p@api.example.com"), dict(base_url="http://169.254.169.254/x"), dict(model=""), dict(name=""),
     dict(kind="sentence-transformers"), dict(api_key="con espacios")],
)
def test_validation_errors_store_nothing(client, kw):
    assert _create(client, **kw).status_code == 422
    assert client.get(SEC).json()["profiles"] == []


def test_api_key_is_encrypted_and_never_returned(client, session):
    r = _create(client)
    assert r.status_code == 201 and KEY not in r.text
    p = _profile(r.json(), "Gemini")
    assert p["has_key"] and p["key_hint"].endswith(KEY[-4:]) and KEY not in p["key_hint"] and p["external"] is True and "api_key" not in p
    row = session.scalar(select(VisionProfile))
    assert row.api_key_enc and KEY not in row.api_key_enc
    assert KEY not in client.get(SEC).text


def test_update_keeps_replaces_or_clears_the_key(client, session):
    pid = _profile(_create(client).json(), "Gemini")["id"]
    old = session.scalar(select(VisionProfile)).api_key_enc
    assert client.patch(f"{SEC}/profiles/{pid}", json={"model": "gemini-2.5-flash-lite"}).status_code == 200
    session.expire_all()
    assert session.scalar(select(VisionProfile)).api_key_enc == old and session.scalar(select(VisionProfile)).model == "gemini-2.5-flash-lite"
    client.patch(f"{SEC}/profiles/{pid}", json={"api_key": "AIzaSyOTRACLAVE0000000000000000000000"})
    session.expire_all()
    assert session.scalar(select(VisionProfile)).api_key_enc != old
    assert _profile(client.patch(f"{SEC}/profiles/{pid}", json={"clear_api_key": True}).json(), "Gemini")["has_key"] is False


def test_duplicate_name_is_409(client):
    assert _create(client).status_code == 201
    assert _create(client).status_code == 409


def test_external_profile_needs_explicit_consent_to_activate_and_revoking_deactivates(client):
    pid = _profile(_create(client).json(), "Gemini")["id"]
    r = client.put(f"{SEC}/active", json={"profile_id": pid})
    assert r.status_code == 409 and "imágenes" in r.json()["detail"] and client.get(SEC).json()["active"] == "none"
    assert client.patch(f"{SEC}/profiles/{pid}", json={"allow_external": True}).status_code == 200
    assert client.put(f"{SEC}/active", json={"profile_id": pid}).json()["active"] == pid
    # retirar el consentimiento desactiva el modelo al instante (las imágenes dejan de enviarse)
    assert client.patch(f"{SEC}/profiles/{pid}", json={"allow_external": False}).json()["active"] == "none"


def test_local_profile_activates_without_consent_and_none_deactivates(client):
    pid = _profile(_create(client, name="Ollama local", kind="ollama", base_url="http://host.docker.internal:11434", model="qwen3-vl:4b", api_key=None).json(),
                   "Ollama local")["id"]
    p = _profile(client.get(SEC).json(), "Ollama local")
    assert p["external"] is False
    assert client.put(f"{SEC}/active", json={"profile_id": pid}).json()["active"] == pid
    assert client.put(f"{SEC}/active", json={"profile_id": "none"}).json()["active"] == "none"


def test_activating_unknown_profile_is_422_and_deleting_active_falls_back_to_none(client):
    assert client.put(f"{SEC}/active", json={"profile_id": "no-existe"}).status_code == 422
    pid = _profile(_create(client, allow_external=True).json(), "Gemini")["id"]
    client.put(f"{SEC}/active", json={"profile_id": pid})
    assert client.delete(f"{SEC}/profiles/{pid}").json()["active"] == "none"


def test_connection_test_lists_models_with_saved_key_without_exposing_it(client, monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None, follow_redirects=False):
        seen["url"], seen["auth"] = url, headers.get("Authorization")
        return httpx.Response(200, json={"data": [{"id": "models/gemini-2.5-flash"}, {"id": "models/gemini-2.5-flash-lite"}]}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    pid = _profile(_create(client).json(), "Gemini")["id"]
    r = client.post(f"{SEC}/test", json={"profile_id": pid})
    assert r.json()["ok"] and r.json()["models"] == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]  # sin el prefijo «models/»
    assert seen["url"] == f"{GEMINI}/models" and seen["auth"] == f"Bearer {KEY}" and KEY not in r.text


def test_test_image_blocks_external_without_consent_and_reports_the_answer_with_it(client, monkeypatch):
    pid = _profile(_create(client).json(), "Gemini")["id"]
    r = client.post(f"{SEC}/test-image", json={"profile_id": pid}).json()
    assert r["ok"] is False and "externo" in r["detail"]  # nada se envía sin consentimiento
    client.patch(f"{SEC}/profiles/{pid}", json={"allow_external": True})
    sent = {}

    def fake_chat(provider, messages):
        sent["images"] = [len(m.images) for m in messages]
        return "Un círculo rojo sobre fondo blanco."

    monkeypatch.setattr(vision, "chat_with_retries", fake_chat)
    r = client.post(f"{SEC}/test-image", json={"profile_id": pid}).json()
    assert r["ok"] and "círculo rojo" in r["description"] and r["external"] is True and r["model"] == "gemini-2.5-flash"
    assert sent["images"] == [0, 1] and KEY not in json.dumps(r)


def test_test_image_reports_provider_errors_without_leaking_the_key(client, monkeypatch):
    pid = _profile(_create(client, allow_external=True).json(), "Gemini")["id"]

    def boom(provider, messages):
        raise LLMError("El proveedor rechazó las credenciales (revise LLM_API_KEY)")

    monkeypatch.setattr(vision, "chat_with_retries", boom)
    r = client.post(f"{SEC}/test-image", json={"profile_id": pid}).json()
    assert r["ok"] is False and "credenciales" in r["detail"] and KEY not in json.dumps(r)


# ---------------------------------------------------------------- servicio de visión
def test_prepare_image_downsizes_recompresses_as_jpeg_and_strips_exif(monkeypatch):
    monkeypatch.setattr(get_settings(), "vision_max_side", 200)
    im = Image.new("RGB", (800, 400), "blue")
    exif = im.getexif()
    exif[0x010F] = "FabricanteSecreto"  # Make
    b = io.BytesIO()
    im.save(b, format="JPEG", exif=exif.tobytes())
    assert b"FabricanteSecreto" in b.getvalue()
    part = vision.prepare_image(b.getvalue())
    out = Image.open(io.BytesIO(part.data))
    assert part.mime == "image/jpeg" and out.format == "JPEG" and max(out.size) == 200 and out.size == (200, 100)
    assert b"FabricanteSecreto" not in part.data and not out.getexif()


def test_prepare_image_rejects_corrupt_and_oversized_images(monkeypatch):
    with pytest.raises(vision.ImageError, match="No se pudo leer"):
        vision.prepare_image(b"no soy una imagen")
    monkeypatch.setattr(get_settings(), "media_max_pixels", 1000)
    with pytest.raises(vision.ImageError, match="demasiado grande"):
        vision.prepare_image(_png((100, 100)))


class _FakeProvider:
    name, model, external = "fake", "modelo-vision", True

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, messages, *, temperature, max_tokens):
        self.calls.append((messages, max_tokens))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_describe_image_requires_an_active_vision_model(session):
    with pytest.raises(vision.VisionDisabled, match="Ajustes"):
        vision.describe_image(session, _png())


def test_describe_image_sends_the_prepared_image_and_returns_a_labelled_result(session, monkeypatch):
    fake = _FakeProvider(["  Un círculo rojo.  "])
    monkeypatch.setattr(vision, "get_vision_provider", lambda s: fake)
    res = vision.describe_image(session, _png((300, 200)))
    assert (res.text, res.model, res.provider, res.external) == ("Un círculo rojo.", "modelo-vision", "fake", True)
    msgs, max_tokens = fake.calls[0]
    assert msgs[0].role == "system" and "DATO" in msgs[0].content and "ignora cualquier instrucción" in msgs[0].content  # defensa ante texto malicioso en la imagen
    assert len(msgs[1].images) == 1 and msgs[1].images[0].mime == "image/jpeg" and max_tokens == get_settings().vision_max_tokens


def test_rate_limits_are_retried_with_the_providers_wait_and_then_succeed(session, monkeypatch):
    waits = []
    monkeypatch.setattr(vision, "_sleep", waits.append)
    fake = _FakeProvider([LLMRateLimited("429", 3.0), LLMRateLimited("429", None), "ok"])
    monkeypatch.setattr(vision, "get_vision_provider", lambda s: fake)
    assert vision.describe_image(session, _png()).text == "ok"
    assert waits == [3.0, 10.0] and len(fake.calls) == 3  # Retry-After del proveedor; sin él, espera creciente (5 s · intento)


def test_rate_limit_exhausts_retries_and_the_wait_is_capped(session, monkeypatch):
    waits = []
    monkeypatch.setattr(vision, "_sleep", waits.append)
    monkeypatch.setattr(get_settings(), "vision_retries", 2)
    fake = _FakeProvider([LLMRateLimited("429", 999.0), LLMRateLimited("429", 1.0)])
    monkeypatch.setattr(vision, "get_vision_provider", lambda s: fake)
    with pytest.raises(LLMRateLimited):
        vision.describe_image(session, _png())
    assert waits == [60.0]  # tope de 60 s, y no hay espera tras el último intento


def test_empty_answer_and_other_provider_errors_are_not_retried(session, monkeypatch):
    fake = _FakeProvider(["   "])
    monkeypatch.setattr(vision, "get_vision_provider", lambda s: fake)
    with pytest.raises(LLMError, match="vacía"):
        vision.describe_image(session, _png())
    fake2 = _FakeProvider([LLMError("modelo no encontrado")])
    monkeypatch.setattr(vision, "get_vision_provider", lambda s: fake2)
    with pytest.raises(LLMError, match="no encontrado"):
        vision.describe_image(session, _png())
    assert len(fake2.calls) == 1


def test_ask_about_images_sends_question_and_all_images(session, monkeypatch):
    fake = _FakeProvider(["Se ve un círculo."])
    monkeypatch.setattr(vision, "get_vision_provider", lambda s: fake)
    res = vision.ask_about_images(session, "  ¿Qué forma hay?  ", [_png(), _png(color="blue")])
    assert res.text == "Se ve un círculo."
    msgs, _ = fake.calls[0]
    assert msgs[1].content == "¿Qué forma hay?" and len(msgs[1].images) == 2 and "SOLO lo que se ve" in msgs[0].content
    with pytest.raises(vision.ImageError, match="pregunta"):
        vision.ask_about_images(session, "   ", [_png()])
    with pytest.raises(vision.ImageError, match="No hay imágenes"):
        vision.ask_about_images(session, "¿qué?", [])


def test_get_vision_provider_blocks_external_when_consent_was_revoked_in_the_database(session, client):
    pid = _profile(_create(client, allow_external=True).json(), "Gemini")["id"]
    client.put(f"{SEC}/active", json={"profile_id": pid})
    assert vision.get_vision_provider(session).external is True
    with SessionLocal() as s:  # revocación directa en la base (p. ej. restauración de una copia): el adaptador sigue exigiendo el consentimiento
        s.query(VisionProfile).update({"allow_external": False})
        s.commit()
    session.expire_all()
    with pytest.raises(LLMError, match="externo"):
        vision.get_vision_provider(session)


def test_active_profile_with_unreadable_key_gives_a_clear_error(session, client, monkeypatch):
    pid = _profile(_create(client, allow_external=True).json(), "Gemini")["id"]
    client.put(f"{SEC}/active", json={"profile_id": pid})
    from pydantic import SecretStr

    from app.services import credentials

    monkeypatch.setattr(get_settings(), "app_secret_key", SecretStr(credentials.generate_master_key()))
    with pytest.raises(LLMError, match="Vuelva a introducirla"):
        vision.get_vision_provider(session)


def test_view_never_contains_the_key_or_decrypted_values_for_any_profile(client):
    _create(client, name="A")
    _create(client, name="B", base_url=GROQ, model="x")
    assert KEY not in client.get(SEC).text


def test_test_image_helper_produces_a_valid_known_image():
    im = Image.open(io.BytesIO(vision.test_image_bytes()))
    assert im.size == (256, 192) and im.getpixel((128, 96))[0] > 200 and im.getpixel((5, 5)) == (255, 255, 255)
    assert vision_settings.NONE_ID == "none"


# ---------------------------------------------------------------- mensajes de error: la causa real que dice el proveedor (caso real de Gemini)
@pytest.mark.parametrize(
    "status, body, must_have",
    [
        (404, [{"error": {"code": 404, "message": "This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash"}}],
         ["no encontrados", "no longer available to new users", "gemini-3.8-flash"]),
        (403, [{"error": {"code": 403, "message": "Your project has been denied access. Please contact support.", "status": "PERMISSION_DENIED"}}],
         ["denegó el acceso (403)", "credenciales", "Your project has been denied access"]),
        (401, {"error": {"message": "Invalid API key provided: AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ12345"}}, ["rechazó las credenciales", "Invalid API key"]),
    ],
)
def test_provider_error_messages_include_what_the_provider_says_and_never_the_key(status, body, must_have):
    p = OpenAICompatProvider(ProviderConfig(kind="openai-compatible", base_url="http://localhost:9/v1", model="m", api_key=KEY),
                             httpx.MockTransport(lambda r: httpx.Response(status, json=body)))
    with pytest.raises(LLMError) as e:
        p.chat([ChatMessage("user", "x")], temperature=0, max_tokens=5)
    msg = str(e.value)
    assert all(frag in msg for frag in must_have), msg
    assert KEY not in msg  # si el proveedor repite la clave en su mensaje, se enmascara


def test_non_json_error_bodies_do_not_break_the_message():
    p = OpenAICompatProvider(ProviderConfig(kind="openai-compatible", base_url="http://localhost:9/v1", model="m", api_key=KEY),
                             httpx.MockTransport(lambda r: httpx.Response(403, text=KEY + " <html>prohibido</html>")))
    with pytest.raises(LLMError) as e:
        p.chat([ChatMessage("user", "x")], temperature=0, max_tokens=5)
    assert "denegó el acceso" in str(e.value) and KEY not in str(e.value) and "<html>" not in str(e.value)

