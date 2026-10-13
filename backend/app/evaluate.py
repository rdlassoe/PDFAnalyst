"""Evaluación de la recuperación sobre documentos ya cargados.

    python -m app.evaluate eval/questions.jsonl [--k 5] [--json salida.json]

Cada línea del JSONL:
  {"id": "q1", "question": "...", "document": "informe.pdf", "pages": [12],
   "answerable": true}
- `document` y `pages` (base 1) indican dónde está la respuesta; con `"answerable": false` la pregunta no tiene
  respuesta en el corpus y el sistema debe abstenerse.
Métricas por modo (vector / fts / hybrid): recall@1/3/k, MRR, y para la abstención: tasa de respuesta correcta
(preguntas con respuesta que no se abstienen) y de abstención correcta (preguntas sin respuesta que sí se abstienen).
También sugiere un umbral `RETRIEVAL_MIN_SIMILARITY`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from sqlalchemy import select

from .db import SessionLocal
from .models import Document
from .services.retrieval import retrieve

MODES = ("vector", "fts", "hybrid")


def _rank_of(hits, filename: str, pages: set[int]) -> int | None:
    for i, h in enumerate(hits, 1):
        if h.filename == filename and pages & set(range(h.page_start, h.page_end + 1)):
            return i
    return None


def evaluate(questions: list[dict], k: int = 5) -> dict:
    out: dict = {"modes": {}, "questions": len(questions), "missing_documents": []}
    with SessionLocal() as s:
        ready = {d.filename: d.id for d in s.scalars(select(Document).where(Document.status == "ready"))}
        usable = []
        for q in questions:
            if q.get("answerable", True) and q["document"] not in ready:
                out["missing_documents"].append(q["document"])
            else:
                usable.append(q)

        for mode in MODES:
            ranks, answered_ok, abstained_ok = [], [], []
            top_sims_ans, top_sims_unans = [], []
            for q in usable:
                r = retrieve(s, q["question"], mode=mode, top_k=k)
                if q.get("answerable", True):
                    rank = _rank_of(r.ranked, q["document"], set(q["pages"]))
                    ranks.append(rank)
                    answered_ok.append(bool(r.hits))
                    if rank and mode == "vector":
                        sim = r.ranked[rank - 1].similarity
                        if sim is not None:
                            top_sims_ans.append(sim)
                else:
                    abstained_ok.append(not r.hits)
                    if mode == "vector" and r.ranked and r.ranked[0].similarity is not None:
                        top_sims_unans.append(max(h.similarity or 0 for h in r.ranked))
            n = len(ranks) or 1
            out["modes"][mode] = {
                "answerable": len(ranks),
                "recall@1": sum(1 for r in ranks if r and r <= 1) / n,
                "recall@3": sum(1 for r in ranks if r and r <= 3) / n,
                f"recall@{k}": sum(1 for r in ranks if r and r <= k) / n,
                "mrr": sum(1 / r for r in ranks if r) / n,
                "answers_given": sum(answered_ok) / n if ranks else None,
                "unanswerable": len(abstained_ok),
                "correct_abstention": (sum(abstained_ok) / len(abstained_ok)) if abstained_ok else None,
            }
            if mode == "vector":
                out["similarity"] = {
                    "answerable_correct_hit": _stats(top_sims_ans),
                    "unanswerable_max": _stats(top_sims_unans),
                }
                if top_sims_ans and top_sims_unans:
                    lo, hi = max(top_sims_unans), min(top_sims_ans)
                    out["similarity"]["suggested_threshold"] = (
                        round((lo + hi) / 2, 3) if lo < hi else None
                    )
                    out["similarity"]["separable"] = lo < hi
    return out


def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {"n": len(xs), "min": round(min(xs), 4), "median": round(statistics.median(xs), 4), "max": round(max(xs), 4)}


def _print(res: dict, k: int) -> None:
    print(f"Preguntas: {res['questions']}   documentos faltantes: {sorted(set(res['missing_documents'])) or 'ninguno'}")
    cols = ["recall@1", "recall@3", f"recall@{k}", "mrr", "answers_given", "correct_abstention"]
    print(f"{'modo':8} " + " ".join(f"{c:>18}" for c in cols))
    for mode, m in res["modes"].items():
        cells = [("–" if m[c] is None else f"{m[c]:.3f}") for c in cols]
        print(f"{mode:8} " + " ".join(f"{c:>18}" for c in cells))
    print("Similitud coseno (modo vector):", json.dumps(res.get("similarity", {}), ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    questions = [json.loads(ln) for ln in args.questions.read_text(encoding="utf-8").splitlines() if ln.strip()]
    res = evaluate(questions, args.k)
    _print(res, args.k)
    if args.json:
        args.json.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
