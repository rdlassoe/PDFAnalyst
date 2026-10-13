"""Fragmentación consciente de páginas.

Los chunks NUNCA cruzan el límite de una página: así cada cita apunta a una única página real y el visor
puede abrirla directamente (decisión de diseño: precisión de la cita > continuidad entre páginas).
Dentro de una página, los párrafos se agrupan hasta ~`target` caracteres sin cortarlos (los párrafos más
largos que `target` se parten por oraciones) y se solapan `overlap` caracteres entre chunks consecutivos.
`page_start`/`page_end` se mantienen en el esquema por si en el futuro se admiten chunks multipágina.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")
_PARA_SPLIT = re.compile(r"\n\s*\n")
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)
PARAGRAPH_SEP = "\n\n"


@dataclass(frozen=True)
class Segment:
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    index: int
    page_start: int
    page_end: int
    text: str


def _hard_split(text: str, limit: int) -> list[str]:
    """Parte `text` en trozos <= limit cortando en espacios cuando es posible."""
    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:  # sin espacio razonable: corte duro
            cut = limit
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def split_long_paragraph(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    buf = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        if len(sentence) > limit:
            if buf:
                pieces.append(buf)
                buf = ""
            pieces.extend(_hard_split(sentence, limit))
        elif buf and len(buf) + 1 + len(sentence) > limit:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip()
    if buf:
        pieces.append(buf)
    return pieces


def _segments(pages: Iterable[tuple[int, str]], limit: int) -> list[Segment]:
    segs: list[Segment] = []
    for number, text in pages:
        for para in _PARA_SPLIT.split(text):
            para = para.strip()
            if para:
                segs.extend(Segment(number, piece) for piece in split_long_paragraph(para, limit))
    return segs


def _joined_len(segs: Sequence[Segment]) -> int:
    return sum(len(s.text) for s in segs) + 2 * max(len(segs) - 1, 0)


def _tail_for_overlap(segs: Sequence[Segment], overlap: int) -> list[Segment]:
    """Últimos segmentos cuya suma ronda `overlap` caracteres (nunca el chunk completo)."""
    tail: list[Segment] = []
    total = 0
    for seg in reversed(segs[1:]):  # excluye el primero para garantizar progreso
        if total + len(seg.text) > overlap and tail:
            break
        if len(seg.text) > overlap:
            break
        tail.insert(0, seg)
        total += len(seg.text) + 2
    return tail


def _chunk_segments(segs: Sequence[Segment], target: int, overlap: int) -> list[list[Segment]]:
    groups: list[list[Segment]] = []
    cur: list[Segment] = []
    fresh = 0  # segmentos añadidos desde la última emisión (el resto es solapamiento)
    for seg in segs:
        if cur and _joined_len(cur) + 2 + len(seg.text) > target:
            if fresh:
                groups.append(cur)
                cur = _tail_for_overlap(cur, overlap)
            else:
                cur = []
            fresh = 0
            # Si aun con el solapamiento no cabe, se descarta el solapamiento.
            if cur and _joined_len(cur) + 2 + len(seg.text) > target:
                cur = []
        cur.append(seg)
        fresh += 1
    if cur and fresh:
        groups.append(cur)
    return groups


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    target: int = 1200,
    overlap: int = 200,
    min_alnum: int = 20,
) -> list[Chunk]:
    """`pages`: iterable de (número_de_página, texto_con_párrafos_separados_por_línea_en_blanco)."""
    if target < 200:
        raise ValueError("target demasiado pequeño")
    overlap = min(overlap, target // 3)

    texts: list[tuple[int, str]] = []
    for number, text in pages:  # una página a la vez: ningún chunk cruza páginas
        segs = _segments([(number, text)], target)
        for group in _chunk_segments(segs, target, overlap):
            texts.append((number, PARAGRAPH_SEP.join(s.text for s in group)))

    # Descarta ruido (números de página sueltos, etc.) y numera de forma consecutiva.
    kept = [(n, t) for n, t in texts if len(_ALNUM.findall(t)) >= min_alnum]
    return [Chunk(i, n, n, t) for i, (n, t) in enumerate(kept)]
