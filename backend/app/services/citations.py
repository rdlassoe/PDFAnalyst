"""Validación de citas (D11).

El LLM solo conoce etiquetas efímeras [C1]…[Cn] asignadas por el servidor a los fragmentos recuperados en
ESA consulta. Una etiqueta fuera de ese conjunto es una cita inventada: se detecta aquí, nunca se acepta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# [C1]  [C1, C2]  [c1; C3]
_GROUP = re.compile(r"\[\s*(C\d+(?:\s*[,;]\s*C\d+)*)\s*\]", re.IGNORECASE)
_LABEL = re.compile(r"C\d+", re.IGNORECASE)
NOT_FOUND_MARKER = "NO_ENCONTRADO"


def labels_in(answer: str) -> list[str]:
    """Etiquetas citadas, en orden de primera aparición, normalizadas a mayúsculas (sin repetidos)."""
    seen: dict[str, None] = {}
    for group in _GROUP.finditer(answer):
        for lab in _LABEL.findall(group.group(1)):
            seen.setdefault(lab.upper())
    return list(seen)


@dataclass(frozen=True)
class CitationCheck:
    valid: list[str]  # citadas y existentes, en orden de aparición
    invalid: list[str]  # citadas pero NO recuperadas en esta consulta

    @property
    def ok(self) -> bool:
        return bool(self.valid) and not self.invalid


def check_citations(answer: str, allowed_labels: set[str]) -> CitationCheck:
    cited = labels_in(answer)
    return CitationCheck(
        valid=[c for c in cited if c in allowed_labels],
        invalid=[c for c in cited if c not in allowed_labels],
    )


def strip_invalid(answer: str, allowed_labels: set[str]) -> str:
    """Elimina las etiquetas inválidas; si un grupo queda vacío se quita completo."""

    def repl(m: re.Match) -> str:
        keep = [lab.upper() for lab in _LABEL.findall(m.group(1)) if lab.upper() in allowed_labels]
        return f"[{', '.join(keep)}]" if keep else ""

    cleaned = _GROUP.sub(repl, answer)
    return re.sub(r"[ \t]+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", cleaned)).strip()


_NOT_FOUND_RE = re.compile(rf"^\W*{NOT_FOUND_MARKER}", re.IGNORECASE)


def is_not_found(answer: str) -> bool:
    """Solo cuenta si la respuesta COMIENZA con el marcador (no si lo menciona en prosa)."""
    return bool(_NOT_FOUND_RE.match(answer.strip()))
