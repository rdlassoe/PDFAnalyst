"""Ingesta de imágenes: validación por firma, miniatura sin EXIF, OCR, descripción con el modelo de visión y fragmentos.

Una imagen produce hasta TRES fragmentos (ver docs/plans/PLAN_MULTIMODAL.md, M2/M7), cada uno con su propio vector:
- **de medio** (`modality='image'`, `source='media'`, texto vacío): su vector lo calcula EmbeddingGemma 2 (servicio local) a partir de la imagen; permite
  encontrarla con una pregunta en texto. Solo se embebe con un modelo que admita medios.
- **de OCR** (`source='ocr'`): el texto que Tesseract lee en la imagen (si hay).
- **de descripción** (`source='caption'`): lo que dice el modelo de visión que ve (si hay uno activo). Es AUTOMÁTICA y puede equivocarse; se guarda marcada.
Los dos de texto se buscan por palabras y por vector de texto, y son los únicos que se entregan al LLM de texto.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image
from sqlalchemy import delete, insert
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import Document, DocumentChunk, ProcessingJob
from ..providers import LLMError, LLMRateLimited
from . import pdf_figures, storage, vision
from .pdf_extract import PdfError, clean_text, ocr_available

log = logging.getLogger(__name__)

OCR_MIN_CHARS_IMAGE = 8  # una señal con «STOP» ya es texto útil; en PDF se exige más porque ahí el OCR compite con el texto real


class MediaError(PdfError):
    """El archivo de medio no es válido (se informa como 422 en la API, igual que un PDF no válido)."""


@dataclass(frozen=True)
class ImageInfo:
    ext: str
    mime: str
    width: int
    height: int


def validate_image_bytes(data: bytes) -> ImageInfo:
    """Valida por FIRMA (no por nombre ni tipo declarado), tamaño, píxeles y que Pillow pueda decodificarla entera. Lanza MediaError."""
    s = get_settings()
    ext = storage.detect_image_type(data)
    if ext is None:
        raise MediaError("Formato no admitido: suba un PDF o una imagen PNG, JPEG o WebP")
    if len(data) > s.media_max_image_mb * 1024 * 1024:
        raise MediaError(f"La imagen supera el tamaño máximo ({s.media_max_image_mb} MB)")
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            if w * h > s.media_max_pixels:
                raise MediaError(f"La imagen es demasiado grande ({w}×{h} píxeles; máximo {s.media_max_pixels // 1_000_000} megapíxeles)")
            im.load()  # decodifica entera: detecta archivos truncados o corruptos
    except MediaError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"No se pudo leer la imagen: archivo dañado o no válido ({exc.__class__.__name__})") from exc
    return ImageInfo(ext=ext, mime=storage.IMAGE_TYPES[ext], width=w, height=h)


def make_thumbnail(data: bytes, dest, max_px: int) -> None:
    """Miniatura JPEG (re-codificada: no conserva EXIF ni GPS)."""
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
    rgb.thumbnail((max_px, max_px))
    tmp = dest.with_suffix(".tmp")
    rgb.save(tmp, format="JPEG", quality=80)
    tmp.replace(dest)


def ocr_image(data: bytes) -> str:
    s = get_settings()
    if not (s.ocr_enabled and ocr_available()):
        return ""
    try:
        import pytesseract

        with Image.open(io.BytesIO(data)) as im:
            text = pytesseract.image_to_string(im.convert("RGB"), lang=s.ocr_langs)
    except Exception as exc:  # noqa: BLE001 - el OCR es opcional: un fallo no impide indexar la imagen
        log.warning("OCR de imagen falló: %s", exc.__class__.__name__)
        return ""
    return clean_text(text)


def register_image(session: Session, filename: str, data: bytes, enqueue):
    """Valida, evita duplicados por hash, guarda y encola el procesamiento. Devuelve (documento, creado)."""
    from sqlalchemy import select

    info = validate_image_bytes(data)
    digest = storage.sha256_of(data)
    existing = session.scalar(select(Document).where(Document.sha256 == digest))
    if existing:
        if existing.status == "failed":
            existing.status, existing.error = "uploaded", None
            enqueue(session, existing.id, "ingest")
            session.commit()
        return existing, False
    storage_name = storage.new_storage_name(info.ext)
    storage.write_blob(storage_name, data)
    doc = Document(
        sha256=digest, filename=storage.sanitize_display_name(filename), storage_name=storage_name, size_bytes=len(data), page_count=1,
        media_type="image", mime=info.mime, width=info.width, height=info.height, status="uploaded",
    )
    try:
        session.add(doc)
        session.flush()
        enqueue(session, doc.id, "ingest")
        session.commit()
    except Exception:
        session.rollback()
        storage.delete_blob(storage_name)
        raise
    return doc, True


def run_ingest_image(session: Session, job: ProcessingJob, doc: Document, enqueue) -> None:
    s = get_settings()
    path = storage.resolve_storage_path(doc.storage_name)
    data = path.read_bytes()
    info = validate_image_bytes(data)
    warnings: list[str] = []

    make_thumbnail(data, storage.thumb_path(doc.id), s.media_thumb_px)
    job.progress = 0.15
    session.commit()

    ocr_text = ocr_image(data)
    if len(ocr_text) < OCR_MIN_CHARS_IMAGE:
        ocr_text = ""
    job.progress = 0.3
    session.commit()

    caption, caption_meta = None, None
    try:
        res = vision.describe_image(session, data)
        caption = res.text
        caption_meta = {"model": res.model, "provider": res.provider, "external": res.external, "automatic": True}
    except vision.VisionDisabled:
        pass  # sin modelo de visión: se indexa solo con vector y OCR
    except (LLMError, vision.ImageError) as exc:  # un fallo del proveedor no impide indexar la imagen: queda el aviso y se puede reprocesar
        warnings.append(f"No se pudo describir la imagen: {exc}")
        log.warning("Descripción de %s falló: %s", doc.id, exc)
    job.progress = 0.4
    session.commit()

    rows: list[dict] = [
        {"chunk_index": 0, "page_start": 1, "page_end": 1, "text": "", "modality": "image", "source": "media", "meta": {}},
    ]
    if ocr_text:
        rows.append({"chunk_index": len(rows), "page_start": 1, "page_end": 1, "text": ocr_text, "modality": "text", "source": "ocr", "meta": {}})
    if caption:
        rows.append({"chunk_index": len(rows), "page_start": 1, "page_end": 1, "text": caption, "modality": "text", "source": "caption", "meta": caption_meta})

    # Reemplazo atómico de los fragmentos (los vectores se eliminan por cascada)
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
    session.execute(insert(DocumentChunk), [{"document_id": doc.id, **r} for r in rows])
    doc.mime, doc.width, doc.height, doc.page_count = info.mime, info.width, info.height, 1
    doc.meta = {
        **(doc.meta or {}),
        "chunk_count": len(rows),
        "ocr_pages": 0,
        "warnings": warnings,
        "ocr_chars": len(ocr_text),
        "description": caption,
        "description_meta": caption_meta,
    }
    enqueue(session, doc.id, "embed")
    session.commit()
    log.info("Ingest imagen %s: %dx%d, OCR %d car., descripción %s", doc.id, info.width, info.height, len(ocr_text), "sí" if caption else "no")


def read_original(doc: Document) -> bytes:
    return storage.resolve_storage_path(doc.storage_name).read_bytes()


MAX_VISION_FAILURES = 3  # tras tantos fallos seguidos del proveedor se deja de llamar (no gastar cuota ni tiempo)


def process_pdf_figures(session: Session, job: ProcessingJob, doc: Document, path) -> tuple[list[dict], dict, list[str]]:
    """Figuras de un PDF: las localiza, las guarda, lee su texto (OCR) y las describe (modelo de visión, si hay uno activo).

    Devuelve (filas de fragmentos SIN `chunk_index`, información para `documents.metadata`, avisos). Por cada figura hay un fragmento de medio (vector de la imagen)
    y, si hay contenido, uno de OCR y uno de descripción, todos con su página y con `meta.figure` (n.º de figura) y su posición en la página.
    Nunca falla por culpa del modelo de visión: sin él, o ante errores/límites del proveedor, las figuras se indexan solo con su vector y su OCR.
    """
    s = get_settings()
    if not s.pdf_figures_enabled:
        return [], {"enabled": False}, []
    scan = pdf_figures.extract_figures(path)
    warnings = list(scan.warnings)
    storage.delete_figures(doc.id)  # reprocesado: las figuras anteriores salen
    rows: list[dict] = []
    cache: dict[str, tuple[str, dict]] = {}
    blocked = False  # no seguir llamando al modelo de visión en este documento
    failures = described = ocr_count = 0
    model_used: str | None = None
    total = len(scan.figures)
    for n, fig in enumerate(scan.figures):
        storage.figure_path(doc.id, n, create=True).write_bytes(fig.data)
        meta = {"figure": n, "kind": fig.kind, "bbox": list(fig.bbox), "width": fig.width, "height": fig.height}
        base = {"page_start": fig.page, "page_end": fig.page}
        rows.append({**base, "text": "", "modality": "image", "source": "media", "meta": meta})
        ocr_text = ocr_image(fig.data)
        if len(ocr_text) >= OCR_MIN_CHARS_IMAGE:
            rows.append({**base, "text": ocr_text, "modality": "text", "source": "ocr", "meta": dict(meta)})
            ocr_count += 1
        if not blocked:
            try:
                if fig.sha not in cache:
                    res = vision.describe_image(session, fig.data)
                    cache[fig.sha] = (res.text, {"model": res.model, "provider": res.provider, "external": res.external, "automatic": True})
                text, cmeta = cache[fig.sha]
                rows.append({**base, "text": text, "modality": "text", "source": "caption", "meta": {**meta, **cmeta}})
                described += 1
                model_used = cmeta["model"]
                failures = 0
            except vision.VisionDisabled:
                blocked = True  # sin modelo de visión: se indexa solo con vector y OCR (no es un fallo)
            except LLMRateLimited:
                blocked = True
                warnings.append("Se agotó el límite de peticiones del modelo de visión: las figuras restantes se indexaron sin descripción. Reprocese el documento más tarde.")
            except (LLMError, vision.ImageError) as exc:
                failures += 1
                if failures == 1:
                    warnings.append(f"No se pudo describir alguna figura: {exc}")
                if failures >= MAX_VISION_FAILURES:
                    blocked = True
                    warnings.append("El modelo de visión falló varias veces seguidas: las figuras restantes se indexaron sin descripción.")
        job.progress = min(0.4, 0.4 * (n + 1) / max(total, 1))
        job.locked_at = datetime.now(timezone.utc)  # latido: una ingesta larga no debe darse por huérfana
        session.commit()
    info = {"enabled": True, "count": total, "described": described, "with_ocr": ocr_count, "skipped": scan.skipped, "vision_model": model_used}
    return rows, info, warnings
