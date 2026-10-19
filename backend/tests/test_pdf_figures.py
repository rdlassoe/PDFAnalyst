"""Figuras dentro de PDF: detección (imágenes incrustadas, diapositivas, gráficos vectoriales), descarte de logos/iconos, ingesta (OCR, descripción, vector),
y respuesta MIRANDO las figuras con el modelo de visión.

El modelo de visión y el vector de la imagen están SIMULADOS (sin claves ni red). La detección, el renderizado, la ingesta, la base de datos y el flujo de respuesta son reales.
"""

import io
import json

import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import func, select

from app.config import get_settings
from app.main import app
from app.models import Conversation, Document, DocumentChunk, ProcessingJob
from app.providers import LLMError, LLMRateLimited
from app.services import answering, ingestion, pdf_figures, storage, vision, vision_settings
from app.services.ingestion import register_upload, run_embed
from app.services.pdf_extract import ocr_available
from app.services.retrieval import retrieve
from app.worker import process_one

from .test_images import MediaE5


def _png(size=(300, 225), kind="circle", text=None) -> bytes:
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    if text:
        d.text((10, size[1] // 3), text, fill="black", font=ImageFont.load_default(size=46))
    elif kind == "circle":
        d.ellipse((size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4), fill="red")
    elif kind == "bars":
        for i, h in enumerate((40, 90, 140, 190)):
            d.rectangle((30 + i * 60, size[1] - h, 70 + i * 60, size[1] - 5), fill="green")
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def make_figures_pdf(*, slide_text: str | None = None) -> bytes:
    """5 páginas: (1) texto + figura + logo · (2) texto + icono diminuto + logo · (3) diapositiva = imagen a página completa + logo
    (4) gráfico VECTORIAL (muchos trazos) + logo · (5) solo texto."""
    doc = fitz.open()
    logo_xref = 0
    for n in range(1, 6):
        page = doc.new_page()  # A4: 595 x 842 pt
        if n != 3:
            page.insert_text((72, 72), f"Pagina {n}: informe de ventas del trimestre con texto suficiente para indexar.", fontsize=11)
        if n == 1:
            page.insert_image(fitz.Rect(100, 120, 400, 345), stream=_png(kind="circle"))
        if n == 2:
            page.insert_image(fitz.Rect(72, 120, 102, 150), stream=_png((40, 40)))  # icono: diminuto
        if n == 3:
            page.insert_image(page.rect, stream=_png((1000, 1400), text=slide_text or "DIAPOSITIVA"))
        if n == 4:
            for i in range(60):  # barras de un gráfico dibujado con vectores
                page.draw_rect(fitz.Rect(80 + i * 8, 600 - (i % 7) * 45 - 60, 85 + i * 8, 600), fill=(0.2, 0.4, 0.8))  # ≈ 24 % de la página
        if n <= 4:  # el MISMO logo en 4 páginas
            rect = fitz.Rect(400, 20, 560, 90)
            if logo_xref == 0:
                logo_xref = page.insert_image(rect, stream=_png((320, 140), kind="bars"))
            else:
                page.insert_image(rect, xref=logo_xref)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def client(session):
    return TestClient(app)


def _scan(tmp_path, data=None):
    p = tmp_path / "f.pdf"
    p.write_bytes(data or make_figures_pdf())
    return pdf_figures.extract_figures(p)


# ---------------------------------------------------------------- detector
def test_detects_figures_slides_and_vector_charts_and_ignores_logos_and_icons(tmp_path):
    scan = _scan(tmp_path)
    assert [(f.page, f.kind) for f in scan.figures] == [(1, "embedded"), (3, "embedded"), (4, "vector")]
    assert scan.skipped["repetidas"] == 4  # el logo, en 4 páginas
    assert scan.skipped["pequeñas"] == 1  # el icono
    for f in scan.figures:
        im = Image.open(io.BytesIO(f.data))
        assert im.format == "JPEG" and max(im.size) <= get_settings().vision_max_side and (f.width, f.height) == im.size
        x0, y0, x1, y1 = f.bbox
        assert 0 <= x0 < x1 <= 595.1 and 0 <= y0 < y1 <= 842.1 and 0 < f.area_ratio <= 1
    slide = scan.figures[1]
    assert slide.area_ratio > 0.9  # una página que ES una imagen (diapositiva) se conserva


def test_a_logo_repeated_fewer_times_than_the_threshold_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "pdf_figure_repeat_pages", 5)
    monkeypatch.setattr(get_settings(), "pdf_figure_min_area", 0.01)
    scan = _scan(tmp_path)
    assert sum(1 for f in scan.figures if f.page <= 4 and f.kind == "embedded" and f.bbox[0] >= 399) == 4  # el logo cuenta ahora como figura
    assert not scan.skipped.get("repetidas")


def test_per_document_cap_keeps_the_largest_and_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "pdf_figures_max", 2)
    scan = _scan(tmp_path)
    assert len(scan.figures) == 2 and scan.skipped["por el límite"] == 1
    assert {f.page for f in scan.figures} == {3, 4}  # diapositiva y gráfico (los de mayor área)
    assert "2 figuras más grandes" in scan.warnings[0] and "PDF_FIGURES_MAX" in scan.warnings[0]


def test_vector_chart_detection_respects_thresholds(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "pdf_vector_min_paths", 500)
    assert not any(f.kind == "vector" for f in _scan(tmp_path).figures)
    monkeypatch.setattr(get_settings(), "pdf_vector_min_paths", 40)
    monkeypatch.setattr(get_settings(), "pdf_vector_min_area", 0.9)
    assert not any(f.kind == "vector" for f in _scan(tmp_path).figures)


def test_text_only_pdf_has_no_figures_and_corrupt_input_does_not_raise(tmp_path):
    from .pdfs import make_pdf

    assert _scan(tmp_path, make_pdf(["solo texto", "más texto"])).figures == []
    bad = tmp_path / "x.pdf"
    bad.write_bytes(b"%PDF-1.7 esto no es un pdf")
    scan = pdf_figures.extract_figures(bad)
    assert scan.figures == [] and scan.warnings


def test_identical_images_on_two_pages_share_a_fingerprint(tmp_path):
    doc = fitz.open()
    xref = 0
    for _ in range(2):  # 2 páginas < umbral de repetición (3): se conservan las dos, con la misma huella
        page = doc.new_page()
        page.insert_text((72, 72), "texto", fontsize=11)
        if xref == 0:
            xref = page.insert_image(fitz.Rect(100, 120, 400, 345), stream=_png())
        else:
            page.insert_image(fitz.Rect(100, 120, 400, 345), xref=xref)
    p = tmp_path / "d.pdf"
    p.write_bytes(doc.tobytes())
    figs = pdf_figures.extract_figures(p).figures
    assert len(figs) == 2 and figs[0].sha == figs[1].sha and figs[0].page != figs[1].page


# ---------------------------------------------------------------- ingesta
def _upload_and_ingest(session, data=None, name="informe.pdf"):
    for j in session.scalars(select(ProcessingJob).where(ProcessingJob.status == "queued")):
        j.status = "done"  # trabajos de documentos anteriores del mismo test (p. ej. su `embed`): no son lo que se prueba aquí
    session.commit()
    res = register_upload(session, name, data or make_figures_pdf())
    assert process_one(session)  # ingest
    session.refresh(res.document)
    return res.document


def _chunks(session, doc):
    return session.scalars(select(DocumentChunk).where(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index)).all()


def test_ingest_without_vision_adds_a_media_chunk_per_figure_with_page_position_and_files(session):
    doc = _upload_and_ingest(session)
    chunks = _chunks(session, doc)
    media = [c for c in chunks if c.modality == "image"]
    assert [(c.page_start, c.meta["figure"], c.meta["kind"]) for c in media] == [(1, 0, "embedded"), (3, 1, "embedded"), (4, 2, "vector")]
    assert all(c.source == "media" and c.text == "" and len(c.meta["bbox"]) == 4 for c in media)
    assert all(c.modality == "text" and c.source == "pdf_text" for c in chunks if c.meta.get("figure") is None)  # el texto del PDF sigue como siempre
    assert [c.page_start for c in chunks if c.source == "pdf_text"] == [1, 2, 4, 5]
    for n in range(3):
        assert storage.figure_path(doc.id, n).exists()
    f = doc.meta["figures"]
    assert (f["count"], f["described"], f["skipped"]["repetidas"]) == (3, 0, 4) and doc.meta["chunk_count"] == len(chunks)
    assert doc.status == "processing"


@pytest.mark.skipif(not ocr_available(), reason="Tesseract no está instalado")
def test_ocr_reads_the_text_inside_a_figure_slide(session, monkeypatch):
    monkeypatch.setattr(get_settings(), "ocr_enabled", True)
    doc = _upload_and_ingest(session, make_figures_pdf(slide_text="INGRESOS 2026"))
    ocr = [c for c in _chunks(session, doc) if c.source == "ocr" and c.meta.get("figure") is not None]
    assert ocr and "INGRESOS" in ocr[0].text.upper() and ocr[0].page_start == 3 and ocr[0].modality == "text"


def test_ingest_with_vision_stores_automatic_descriptions_per_figure_and_describes_identical_images_once(session, monkeypatch):
    calls = []

    def fake(sess, data):
        calls.append(len(data))
        return vision.VisionResult(text=f"Descripción {len(calls)}", model="modelo-vision", provider="fake", external=False)

    monkeypatch.setattr(vision, "describe_image", fake)
    doc = _upload_and_ingest(session)
    caps = [c for c in _chunks(session, doc) if c.source == "caption"]
    assert [(c.page_start, c.text, c.meta["figure"]) for c in caps] == [(1, "Descripción 1", 0), (3, "Descripción 2", 1), (4, "Descripción 3", 2)]
    assert all(c.modality == "text" and c.meta["automatic"] is True and c.meta["model"] == "modelo-vision" for c in caps)
    assert doc.meta["figures"]["described"] == 3 and doc.meta["figures"]["vision_model"] == "modelo-vision" and doc.meta["warnings"] == []

    # el mismo PDF con una imagen idéntica en dos páginas: una sola llamada al modelo
    calls.clear()
    d2 = fitz.open()
    xref = 0
    for _ in range(2):
        page = d2.new_page()
        page.insert_text((72, 72), "texto distinto", fontsize=11)
        xref = page.insert_image(fitz.Rect(100, 120, 400, 345), stream=_png(kind="bars")) if xref == 0 else (page.insert_image(fitz.Rect(100, 120, 400, 345), xref=xref) or xref)
    doc2 = _upload_and_ingest(session, d2.tobytes(), name="dos.pdf")
    assert len(calls) == 1 and sum(1 for c in _chunks(session, doc2) if c.source == "caption") == 2


def test_rate_limit_stops_describing_the_rest_and_the_document_is_still_indexed(session, monkeypatch):
    calls = []

    def fake(sess, data):
        calls.append(1)
        if len(calls) == 2:
            raise LLMRateLimited("429", 1.0)
        return vision.VisionResult(text="ok", model="m", provider="p", external=False)

    monkeypatch.setattr(vision, "describe_image", fake)
    doc = _upload_and_ingest(session)
    assert len(calls) == 2  # no se insiste con el resto: no gastar cuota ni tiempo
    assert sum(1 for c in _chunks(session, doc) if c.source == "caption") == 1
    assert sum(1 for c in _chunks(session, doc) if c.modality == "image") == 3  # las 3 figuras tienen su vector igualmente
    assert any("límite de peticiones" in w for w in doc.meta["warnings"]) and doc.meta["figures"]["described"] == 1


def test_repeated_provider_failures_stop_after_three_and_leave_a_warning(session, monkeypatch):
    calls = []

    def fake(sess, data):
        calls.append(1)
        raise LLMError("modelo no encontrado")

    monkeypatch.setattr(vision, "describe_image", fake)
    doc = _upload_and_ingest(session)
    assert len(calls) == 3 and doc.meta["figures"]["described"] == 0
    assert any("No se pudo describir" in w for w in doc.meta["warnings"]) and any("varias veces" in w for w in doc.meta["warnings"])


def test_no_vision_model_is_silent_and_figures_can_be_disabled(session, monkeypatch):
    doc = _upload_and_ingest(session)  # sin modelo de visión activo: no es un fallo
    assert doc.meta["warnings"] == [] and doc.meta["figures"]["count"] == 3
    monkeypatch.setattr(get_settings(), "pdf_figures_enabled", False)
    doc2 = _upload_and_ingest(session, make_figures_pdf(slide_text="OTRA"), name="otro.pdf")
    assert doc2.meta["figures"] == {"enabled": False} and not any(c.modality == "image" for c in _chunks(session, doc2))


def test_a_pdf_that_is_only_images_is_accepted_even_without_text(session, monkeypatch):
    d = fitz.open()
    page = d.new_page()
    page.insert_image(page.rect, stream=_png((800, 1100), kind="bars"))  # página sin texto ni OCR
    doc = _upload_and_ingest(session, d.tobytes(), name="solo_imagen.pdf")
    assert doc.status == "processing" and doc.meta["figures"]["count"] == 1  # antes: «No se extrajo texto utilizable»
    assert [c.modality for c in _chunks(session, doc)] == ["image"]


def test_reprocessing_replaces_figures_and_chunks_without_duplicates(session, monkeypatch):
    doc = _upload_and_ingest(session)
    first = len(_chunks(session, doc))
    for j in session.scalars(select(ProcessingJob)):
        j.status = "done"
    ingestion.enqueue_job(session, doc.id, "ingest")
    session.commit()
    assert process_one(session)
    assert len(_chunks(session, doc)) == first
    assert sorted(p.name for p in storage.figure_dir(doc.id).iterdir()) == ["f000.jpg", "f001.jpg", "f002.jpg"]


def test_deleting_the_document_removes_its_figures(session):
    doc = _upload_and_ingest(session)
    d = storage.figure_dir(doc.id)
    assert d.exists()
    ingestion.delete_document(session, doc)
    assert not d.exists()


# ---------------------------------------------------------------- vectores de las figuras y recuperación
def _indexed(session, monkeypatch, embedder, *, captions=True):
    if captions:
        n = []

        def fake(sess, data):
            n.append(1)
            return vision.VisionResult(text=f"Gráfico de barras de ventas {len(n)}", model="modelo-vision", provider="fake", external=False)

        monkeypatch.setattr(vision, "describe_image", fake)
    doc = _upload_and_ingest(session)
    fake_emb = MediaE5(embedder, lambda item: "gráfico de barras de ventas")
    job = session.scalar(select(ProcessingJob).where(ProcessingJob.document_id == doc.id, ProcessingJob.kind == "embed"))
    run_embed(session, job, embedder=fake_emb)
    session.refresh(doc)
    return doc, fake_emb


def test_figure_vectors_are_computed_from_the_extracted_figure_files(session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder)
    sent = [i for call in emb.media_calls for i in call]
    assert len(sent) == 3 and all(i["modality"] == "image" for i in sent)
    assert [i["path"] for i in sent] == [str(storage.figure_path(doc.id, n)) for n in range(3)]  # por referencia a cada figura, no al PDF
    assert doc.status == "ready"


def test_search_returns_figure_hits_with_their_figure_metadata(session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder)
    r = retrieve(session, "gráfico de barras de ventas", embedder=emb, mode="vector")
    figs = [h for h in r.hits if h.meta.get("figure") is not None]
    assert figs and {h.modality for h in figs} <= {"image", "text"}
    assert all(h.page_start in (1, 3, 4) and len(h.meta["bbox"]) == 4 for h in figs)
    assert any(h.source == "caption" for h in figs) and any(h.source == "media" for h in figs)


# ---------------------------------------------------------------- responder mirando las figuras
class LLM:
    name, model, external = "fake", "m", False

    def __init__(self, reply="Se ve un gráfico de barras [C1]."):
        self.reply, self.messages = reply, None

    def chat(self, messages, *, temperature, max_tokens):
        self.messages = messages
        return self.reply


class EchoLLM(LLM):
    """Cita todas las etiquetas [C#] que aparecen en <fuentes>: así se comprueba que la cita de una imagen adjunta se guarda."""

    def chat(self, messages, *, temperature, max_tokens):
        import re

        self.messages = messages
        labels = re.findall(r'<fuente id="(C\d+)"', messages[1].content)
        return "Según las fuentes: " + " ".join(f"[{lab}]" for lab in labels)


def _activate_vision(client):
    body = dict(name="Visión local", kind="ollama", base_url="http://localhost:11434", model="qwen3-vl:4b", api_key=None, allow_external=False)
    pid = next(p for p in client.post("/settings/vision/profiles", json=body).json()["profiles"])["id"]
    assert client.put("/settings/vision/active", json={"profile_id": pid}).status_code == 200
    return pid


def _ask(session, embedder, emb, llm, question="¿Qué muestra el gráfico de barras de ventas?", **kw):
    conv = Conversation()
    session.add(conv)
    session.flush()
    result = retrieve(session, question, embedder=emb)
    return answering.answer_question(session, conv, question, provider=llm, retrieval=result, **kw)


def test_figures_are_attached_to_the_vision_answer_with_their_source_labels(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder)
    _activate_vision(client)
    llm = LLM()
    res = _ask(session, embedder, emb, llm)
    user = llm.messages[1]
    assert 1 <= len(user.images) <= get_settings().vision_answer_max_images and all(i.mime == "image/jpeg" for i in user.images)
    assert "Se adjuntan" in user.content and "imagen 1 = fuente C" in user.content
    m = res.assistant_message
    assert m.status == "ok" and m.meta["generation"]["vision"]["images"] == len(user.images)
    assert set(m.meta["generation"]["vision"]["labels"]) <= {f"C{i}" for i in range(1, 12)}


def test_a_figure_without_text_is_added_as_an_attached_image_source(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder, captions=False)  # sin descripciones: solo vectores de las figuras
    _activate_vision(client)
    llm = EchoLLM()
    res = _ask(session, embedder, emb, llm, question="gráfico de barras de ventas")
    assert res.assistant_message.status == "ok"
    user = llm.messages[1]
    assert 'tipo="imagen_adjunta"' in user.content and answering.FIGURE_PLACEHOLDER in user.content and len(user.images) >= 1
    c = next(c for c in res.citations if c.snapshot == answering.FIGURE_PLACEHOLDER)  # la cita de una figura que solo se vio (sin texto)
    assert c.modality == "image" and c.source == "figure_image" and c.page_start in (1, 3, 4)


def test_no_images_attached_when_vision_answers_are_off_or_there_is_no_vision_model(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder)
    llm = LLM()
    _ask(session, embedder, emb, llm)  # sin modelo de visión activo
    assert llm.messages[1].images == ()
    pid = _activate_vision(client)
    assert client.put("/settings/vision/answer", json={"enabled": False}).json()["answer_with_images"] is False
    llm2 = LLM()
    _ask(session, embedder, emb, llm2)  # activo, pero «responder mirando las imágenes» desactivado
    assert llm2.messages[1].images == ()
    assert client.put("/settings/vision/answer", json={"enabled": True}).json()["answer_with_images"] is True
    assert client.get("/settings/vision").json()["answer_with_images"] is True and pid


def test_the_number_of_attached_images_is_capped(client, session, monkeypatch, embedder):
    monkeypatch.setattr(get_settings(), "vision_answer_max_images", 1)
    doc, emb = _indexed(session, monkeypatch, embedder)
    _activate_vision(client)
    llm = LLM()
    _ask(session, embedder, emb, llm)
    assert len(llm.messages[1].images) == 1


def test_vision_failure_falls_back_to_the_text_llm_and_drops_attached_image_sources(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder, captions=False)
    _activate_vision(client)

    class Broken:
        name, model, external = "visionprov", "vm", True

        def chat(self, messages, *, temperature, max_tokens):
            raise LLMError("El proveedor no admite imágenes")

        def health(self):
            return {"ok": True}

    text_llm = LLM("No hay texto suficiente para responder.")
    monkeypatch.setattr(vision, "get_vision_provider", lambda s: Broken())
    monkeypatch.setattr(answering, "get_llm_provider", lambda s: text_llm)
    conv = Conversation()
    session.add(conv)
    session.flush()
    q = "gráfico de barras de ventas"
    res = answering.answer_question(session, conv, q, retrieval=retrieve(session, q, embedder=emb))  # sin provider explícito: elige el modelo de visión
    m = res.assistant_message
    assert "no admite imágenes" in m.meta["generation"]["vision_error"] and "vision" not in m.meta["generation"]
    assert m.meta["generation"]["provider"] == "fake"  # respondió el LLM de texto
    assert text_llm.messages[1].images == () and "imagen_adjunta" not in text_llm.messages[1].content


def test_missing_consent_for_the_vision_model_falls_back_to_text(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder)
    _activate_vision(client)

    def blocked(session_):
        raise LLMError("El proveedor api.groq.com es externo: las imágenes saldrían de esta máquina")

    text_llm = LLM("Respuesta con las descripciones [C1].")
    monkeypatch.setattr(vision, "get_vision_provider", blocked)
    monkeypatch.setattr(answering, "get_llm_provider", lambda s: text_llm)
    conv = Conversation()
    session.add(conv)
    session.flush()
    q = "gráfico de barras de ventas"
    res = answering.answer_question(session, conv, q, retrieval=retrieve(session, q, embedder=emb))
    assert res.assistant_message.status == "ok" and "externo" in res.assistant_message.meta["generation"]["vision_error"]
    assert text_llm.messages[1].images == ()  # nada salió hacia el servicio externo


def test_related_media_has_one_entry_per_figure_with_its_page(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder, captions=False)
    llm = LLM()
    res = _ask(session, embedder, emb, llm, question="gráfico de barras de ventas")
    rel = res.assistant_message.meta["related_media"]
    assert len({(r["document_id"], r["figure"]) for r in rel}) == len(rel) and all(r["page"] in (1, 3, 4) and r["figure"] in (0, 1, 2) for r in rel)


# ---------------------------------------------------------------- API
def test_chunk_endpoints_expose_the_figure(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder)
    chunks = _chunks(session, doc)
    fig_chunk = next(c for c in chunks if c.modality == "image" and c.meta["figure"] == 1)
    text_chunk = next(c for c in chunks if c.modality == "text" and c.source == "pdf_text")
    out = client.get(f"/chunks/{fig_chunk.id}").json()
    assert out["figure_url"] == f"/chunks/{fig_chunk.id}/figure" and out["modality"] == "image" and out["page_start"] == 3
    img = client.get(f"/chunks/{fig_chunk.id}/figure")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg" and Image.open(io.BytesIO(img.content)).format == "JPEG"
    assert client.get(f"/chunks/{text_chunk.id}").json()["figure_url"] is None
    assert client.get(f"/chunks/{text_chunk.id}/figure").status_code == 404 and client.get("/chunks/999999/figure").status_code == 404


def test_document_out_reports_figure_counts(client, session, monkeypatch, embedder):
    doc, emb = _indexed(session, monkeypatch, embedder)
    d = next(x for x in client.get("/documents").json() if x["id"] == str(doc.id))
    assert d["media_type"] == "pdf" and d["figure_count"] == 3 and d["figures_described"] == 3


def test_reprocess_endpoint_requeues_a_pdf_so_figures_are_read(client, session):
    doc = _upload_and_ingest(session)
    r = client.post(f"/documents/{doc.id}/reprocess")
    assert r.status_code == 202 and r.json()["kind"] == "ingest"
