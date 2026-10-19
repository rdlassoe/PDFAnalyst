"""Detección y extracción de FIGURAS de un PDF: imágenes incrustadas y gráficos dibujados con vectores.

Hasta ahora el extractor solo leía texto (y OCR de páginas casi sin texto): una foto, un esquema o un gráfico dentro de un PDF no se veía. Aquí se localizan
las figuras de cada página para que el resto de la ingesta las lea (OCR), las describa (modelo de visión), las vectorice (EmbeddingGemma 2) y las pueda adjuntar
al responder. Criterios, ajustados con PDF reales (diapositivas exportadas como imagen, escaneos, manuales con figuras y gráficos vectoriales):

- Se conservan las imágenes que ocupan una parte relevante de la página (incluida una página entera que ES una imagen, como una diapositiva) y no son diminutas.
- Se descartan: imágenes pequeñas (iconos, viñetas), muy alargadas (líneas), y **repetidas en varias páginas** (fondos, logos, marcas de agua).
- Los gráficos vectoriales (muchos trazos agrupados) no son imágenes incrustadas: se renderiza su región como figura.
- Cada figura se guarda **renderizada tal como se ve en la página** (con máscaras, CMYK, rotación…) a un lado mayor acotado, y sin EXIF.
- Hay un tope de figuras por documento (coste de CPU/API); las más grandes tienen prioridad y se avisa de las que se omiten.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from ..config import get_settings

log = logging.getLogger(__name__)

MAX_ASPECT = 10.0  # más alargada que 10:1 = línea o separador
MAX_RENDER_DPI = 200
MIN_RENDER_DPI = 72


@dataclass
class Figure:
    page: int  # base 1
    bbox: tuple[float, float, float, float]  # puntos PDF (x0, y0, x1, y1)
    kind: str  # "embedded" | "vector"
    data: bytes = field(repr=False)  # JPEG renderizado
    width: int = 0
    height: int = 0
    sha: str = ""  # huella del contenido (para describir una sola vez imágenes idénticas)
    area_ratio: float = 0.0  # fracción de la página que ocupa


@dataclass
class FigureScan:
    figures: list[Figure]
    skipped: dict[str, int]
    warnings: list[str]


def _render(page: "fitz.Page", rect: "fitz.Rect", max_side: int) -> tuple[bytes, int, int]:
    """Renderiza `rect` de la página a JPEG con el lado mayor ≈ `max_side` px (sin ampliar más de MAX_RENDER_DPI)."""
    longest = max(rect.width, rect.height, 1.0)
    dpi = int(max(MIN_RENDER_DPI, min(MAX_RENDER_DPI, 72.0 * max_side / longest)))
    pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue(), img.width, img.height


def _covered(rect: "fitz.Rect", others: list["fitz.Rect"]) -> float:
    """Fracción de `rect` cubierta por la unión (aproximada por la suma, acotada) de `others`."""
    if rect.is_empty or rect.get_area() == 0:
        return 0.0
    total = 0.0
    for o in others:
        inter = rect & o
        if not inter.is_empty:
            total += inter.get_area()
    return min(1.0, total / rect.get_area())


def extract_figures(path: Path) -> FigureScan:
    """Figuras de un PDF. Nunca lanza por un PDF raro: devuelve lo que pudo y un aviso."""
    s = get_settings()
    skipped: dict[str, int] = defaultdict(int)
    warnings: list[str] = []
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        return FigureScan([], {}, [f"No se pudieron buscar figuras: {exc.__class__.__name__}"])

    candidates: list[dict] = []
    pages_of_xref: dict[int, set[int]] = defaultdict(set)
    with doc:
        # ---- 1) imágenes incrustadas
        per_page: dict[int, list[dict]] = {}
        for index in range(doc.page_count):
            page = doc[index]
            parea = max(page.rect.width * page.rect.height, 1.0)
            try:
                infos = page.get_image_info(xrefs=True)
            except Exception:  # noqa: BLE001
                continue
            items = []
            for im in infos:
                rect = fitz.Rect(im["bbox"]) & page.rect
                if rect.is_empty or im.get("xref", 0) <= 0:
                    continue
                pages_of_xref[im["xref"]].add(index + 1)
                items.append({"page": index + 1, "xref": im["xref"], "rect": rect, "w": im["width"], "h": im["height"],
                              "area": rect.get_area() / parea})
            per_page[index] = items

        for index, items in per_page.items():
            for it in items:
                w, h, rect = it["w"], it["h"], it["rect"]
                if len(pages_of_xref[it["xref"]]) >= s.pdf_figure_repeat_pages:
                    skipped["repetidas"] += 1  # fondo, logo o marca de agua: se repite en varias páginas
                elif min(w, h) < s.pdf_figure_min_px or it["area"] < s.pdf_figure_min_area:
                    skipped["pequeñas"] += 1
                elif max(rect.width / max(rect.height, 1), rect.height / max(rect.width, 1)) > MAX_ASPECT:
                    skipped["alargadas"] += 1
                else:
                    candidates.append({"kind": "embedded", "page": it["page"], "rect": rect, "area": it["area"], "xref": it["xref"]})

        # ---- 2) gráficos vectoriales: región con muchos trazos que no es ya una imagen incrustada
        for index in range(doc.page_count):
            page = doc[index]
            try:
                drawings = page.get_drawings()
            except Exception:  # noqa: BLE001
                continue
            if len(drawings) < s.pdf_vector_min_paths:
                continue
            rects = [fitz.Rect(d["rect"]) for d in drawings if d.get("rect") is not None]
            if not rects:
                continue
            union = fitz.Rect(rects[0])
            for r in rects[1:]:
                union |= r
            union = (union + (-6, -6, 6, 6)) & page.rect
            parea = max(page.rect.width * page.rect.height, 1.0)
            if union.is_empty or union.get_area() / parea < s.pdf_vector_min_area:
                continue
            mine = [c["rect"] for c in candidates if c["page"] == index + 1]
            if _covered(union, mine) >= 0.6:
                continue  # ya está cubierta por imágenes incrustadas
            candidates.append({"kind": "vector", "page": index + 1, "rect": union, "area": union.get_area() / parea, "xref": 0})

        # ---- 3) tope por documento: las más grandes primero; se avisa de las omitidas
        if len(candidates) > s.pdf_figures_max:
            candidates.sort(key=lambda c: -c["area"])
            skipped["por el límite"] += len(candidates) - s.pdf_figures_max
            candidates = candidates[: s.pdf_figures_max]
        candidates.sort(key=lambda c: (c["page"], c["rect"].y0, c["rect"].x0))

        # ---- 4) renderizado
        figures: list[Figure] = []
        for c in candidates:
            try:
                data, w, h = _render(doc[c["page"] - 1], c["rect"], s.vision_max_side)
            except Exception as exc:  # noqa: BLE001
                skipped["no renderizables"] += 1
                log.warning("No se pudo renderizar una figura de la página %d: %s", c["page"], exc)
                continue
            r = c["rect"]
            sha = hashlib.sha256(data).hexdigest()
            if c["xref"]:  # misma imagen en dos páginas → misma huella (se describe una sola vez)
                try:
                    sha = hashlib.sha256(doc.extract_image(c["xref"])["image"]).hexdigest()
                except Exception:  # noqa: BLE001
                    pass
            figures.append(Figure(page=c["page"], bbox=(round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1)), kind=c["kind"],
                                  data=data, width=w, height=h, sha=sha, area_ratio=round(c["area"], 3)))

    if skipped.get("por el límite"):
        warnings.append(f"Solo se procesaron las {s.pdf_figures_max} figuras más grandes; se omitieron {skipped['por el límite']} (PDF_FIGURES_MAX).")
    return FigureScan(figures, dict(skipped), warnings)
