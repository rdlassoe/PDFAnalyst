"""Recuperación: vectorial (pgvector) + textual (FTS de PostgreSQL) + fusión RRF.

Decisiones (docs/ARCHITECTURE.md D7–D10):
- Distancia coseno (`<=>`) sobre vectores normalizados; similitud = 1 - distancia.
- La consulta usa la expresión `embedding::vector(N)` y filtra `model_id`, de modo que un índice HNSW
  parcial por modelo se usa automáticamente cuando existe (ver services/vector_index.py).
- RRF: score = Σ 1/(k + rank_i) sobre las ramas. No requiere normalizar scores heterogéneos.
- Un chunk es "relevante" si su similitud coseno supera el umbral O si cubre suficientes términos de la
  pregunta (rama textual). Si ninguno lo es, se devuelve vacío => el sistema se abstiene.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass, field

from sqlalchemy import bindparam, select, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models import CollectionDocument, Document, EmbeddingModel
from .embeddings import Embedder, get_embedder, get_or_create_model_row

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = frozenset(
    """
    a al algo ante aqui as asi aun aunque cada como con contra cual cuales cuando cuanto cuantos de del desde
    donde dos el ella ellas ellos en entre era eran es esa esas ese eso esos esta estan estar este esto estos
    fue ha han hasta hay la las le les lo los mas me mi mis muy ni no nos nuestro o os otra otro para pero por
    porque que quien quienes se ser si sin sobre son su sus tambien te tiene tienen todo tu tus un una uno unos
    y ya yo
    about after all also an and any are as at be been but by can did do does for from had has have how if in
    into is it its may more most no not of on or other our should so some such than that the their then there
    these they this to was we were what when where which who whom why will with would you your
    """.split()
)


def fold(text_: str) -> str:
    """Minúsculas y sin tildes (equivalente a lower(f_unaccent(x)) de la BD)."""
    nfkd = unicodedata.normalize("NFKD", text_)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch)).lower()


def tokenize(text_: str) -> list[str]:
    return _TOKEN.findall(fold(text_))


def query_terms(question: str) -> list[str]:
    """Términos informativos de la pregunta (sin palabras vacías, sin duplicados, orden estable)."""
    seen: dict[str, None] = {}
    for tok in tokenize(question):
        if tok in _STOPWORDS:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        seen.setdefault(tok)
    return list(seen)


@dataclass
class Hit:
    chunk_id: int
    document_id: uuid.UUID
    filename: str
    page_start: int
    page_end: int
    text: str
    similarity: float | None = None
    vector_rank: int | None = None
    fts_rank: int | None = None
    fts_score: float | None = None
    coverage: float = 0.0
    rrf: float = 0.0
    relevant_by: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    hits: list[Hit]
    candidates_considered: int
    abstain_reason: str | None
    model_id: int
    terms: list[str]
    document_ids: list[uuid.UUID]
    ranked: list[Hit] = field(default_factory=list)  # lista fusionada completa, antes del filtro de relevancia


def _vec_literal(vec) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


def resolve_document_ids(
    session: Session,
    document_ids: list[uuid.UUID] | None = None,
    collection_id: uuid.UUID | None = None,
) -> list[uuid.UUID]:
    """Documentos consultables (estado `ready`), restringidos por selección y/o colección."""
    stmt = select(Document.id).where(Document.status == "ready")
    if document_ids:
        stmt = stmt.where(Document.id.in_(document_ids))
    if collection_id:
        stmt = stmt.where(
            Document.id.in_(
                select(CollectionDocument.document_id).where(CollectionDocument.collection_id == collection_id)
            )
        )
    return list(session.scalars(stmt))


def _hit_from_row(r) -> Hit:
    return Hit(
        chunk_id=r.id,
        document_id=r.document_id,
        filename=r.filename,
        page_start=r.page_start,
        page_end=r.page_end,
        text=r.text,
        similarity=getattr(r, "sim", None),
    )


def vector_search(session: Session, model: EmbeddingModel, qvec, doc_ids: list[uuid.UUID], limit: int) -> list[Hit]:
    dim = int(model.dimension)  # entero validado: se interpola porque un typmod no admite parámetros
    if len(qvec) != dim:
        raise ValueError(f"Dimensión de consulta {len(qvec)} != modelo {dim}")
    if not doc_ids:
        return []
    # Recorrido iterativo (pgvector >= 0.8) para que el filtro por documento no recorte resultados
    # cuando el índice HNSW está en uso; sin índice es una búsqueda exacta y esto no tiene efecto.
    session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    stmt = text(
        f"""
        SELECT c.id, c.document_id, c.page_start, c.page_end, c.text, d.filename,
               1 - (ce.embedding::vector({dim}) <=> CAST(:q AS vector({dim}))) AS sim
        FROM chunk_embeddings ce
        JOIN document_chunks c ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE ce.model_id = :m AND c.document_id = ANY(:docs)
        ORDER BY ce.embedding::vector({dim}) <=> CAST(:q AS vector({dim}))
        LIMIT :n
        """
    ).bindparams(bindparam("docs", type_=ARRAY(PG_UUID(as_uuid=True))))
    rows = session.execute(stmt, {"q": _vec_literal(qvec), "m": model.id, "docs": doc_ids, "n": limit}).all()
    hits = [_hit_from_row(r) for r in rows]
    hits.sort(key=lambda h: -(h.similarity or 0.0))  # relaxed_order puede desordenar ligeramente
    for rank, h in enumerate(hits, 1):
        h.vector_rank = rank
    return hits


def distinctive_terms(
    session: Session, terms: list[str], doc_ids: list[uuid.UUID]
) -> tuple[list[str], list[str]]:
    """Pondera por rareza (idea de IDF) usando la frecuencia de documento real en el corpus consultado.

    Devuelve (`distinctive`, `matchable`):
    - `distinctive`: términos poco frecuentes (aparecen en <= max(2, ratio·N) fragmentos) INCLUIDOS los que no
      aparecen en ninguno: un término raro ausente del corpus es justo la señal de que no hay evidencia.
    - `matchable`: los distinctivos con df > 0, con los que se construye la consulta `tsquery`.
    Si todos los términos son comunes, se usan todos los que existan (mejor eso que nada).
    Palabras como «oro» o «bolsa» en un libro sobre oro y bolsas no deben dominar el ranking textual.
    """
    if not terms or not doc_ids:
        return terms, terms
    terms = terms[:24]
    filters = ", ".join(
        f"count(*) FILTER (WHERE tsv @@ to_tsquery('simple', :t{i})) AS d{i}" for i in range(len(terms))
    )
    stmt = text(
        f"SELECT count(*) AS n, {filters} FROM document_chunks WHERE document_id = ANY(:docs)"
    ).bindparams(bindparam("docs", type_=ARRAY(PG_UUID(as_uuid=True))))
    params = {f"t{i}": t for i, t in enumerate(terms)} | {"docs": doc_ids}
    row = session.execute(stmt, params).one()
    n = max(int(row.n), 1)
    df = {t: int(getattr(row, f"d{i}")) for i, t in enumerate(terms)}
    limit = max(2.0, get_settings().retrieval_common_term_ratio * n)
    distinctive = [t for t in terms if df[t] <= limit]
    if not any(df[t] > 0 for t in distinctive):  # todo común (o ausente): usar los presentes en el corpus
        present = [t for t in terms if df[t] > 0]
        return (present or terms), present
    return distinctive, [t for t in distinctive if df[t] > 0]


def fts_search(
    session: Session,
    model: EmbeddingModel,
    qvec,
    terms: list[str],
    doc_ids: list[uuid.UUID],
    limit: int,
    coverage_terms: list[str] | None = None,
) -> list[Hit]:
    """`terms` arma la consulta; `coverage_terms` (por defecto los mismos) es el denominador de la cobertura."""
    if not terms or not doc_ids:
        return []
    dim = int(model.dimension)
    tsq = " | ".join(terms)  # términos ya reducidos a [^\W_]+ : no hay sintaxis de tsquery inyectable
    stmt = text(
        f"""
        SELECT c.id, c.document_id, c.page_start, c.page_end, c.text, d.filename,
               ts_rank_cd(c.tsv, q.tsq) AS fts_score,
               1 - (ce.embedding::vector({dim}) <=> CAST(:q AS vector({dim}))) AS sim
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN (SELECT to_tsquery('simple', :tsq) AS tsq) q
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id AND ce.model_id = :m
        WHERE c.tsv @@ q.tsq AND c.document_id = ANY(:docs)
        ORDER BY fts_score DESC, c.id
        LIMIT :n
        """
    ).bindparams(bindparam("docs", type_=ARRAY(PG_UUID(as_uuid=True))))
    rows = session.execute(
        stmt, {"q": _vec_literal(qvec), "tsq": tsq, "m": model.id, "docs": doc_ids, "n": limit}
    ).all()
    hits = []
    term_set = set(coverage_terms or terms)
    for rank, r in enumerate(rows, 1):
        h = _hit_from_row(r)
        h.fts_rank, h.fts_score = rank, float(r.fts_score)
        h.coverage = len(term_set & set(tokenize(r.text))) / len(term_set)
        hits.append(h)
    return hits


def rrf_fuse(branches: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion. Devuelve hits únicos (por chunk) ordenados por score RRF descendente."""
    merged: dict[int, Hit] = {}
    for branch in branches:
        for rank, h in enumerate(branch, 1):
            cur = merged.get(h.chunk_id)
            if cur is None:
                cur = merged[h.chunk_id] = h
                cur.rrf = 0.0
            else:  # combinar metadatos de ambas ramas
                cur.vector_rank = cur.vector_rank or h.vector_rank
                if h.fts_rank is not None:
                    cur.fts_rank, cur.fts_score, cur.coverage = h.fts_rank, h.fts_score, h.coverage
                if cur.similarity is None:
                    cur.similarity = h.similarity
            cur.rrf += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda h: (-h.rrf, h.chunk_id))


def retrieve(
    session: Session,
    question: str,
    *,
    document_ids: list[uuid.UUID] | None = None,
    collection_id: uuid.UUID | None = None,
    top_k: int | None = None,
    hybrid: bool | None = None,
    mode: str | None = None,
    embedder: Embedder | None = None,
) -> RetrievalResult:
    """`mode`: "vector" | "fts" | "hybrid" (anula `hybrid`; "fts" y los otros dos sirven para evaluación)."""
    s = get_settings()
    embedder = embedder or get_embedder(session)
    top_k = top_k or s.retrieval_top_k
    if mode is None:
        mode = "hybrid" if (s.hybrid_enabled if hybrid is None else hybrid) else "vector"
    if mode not in ("vector", "fts", "hybrid"):
        raise ValueError(f"modo de recuperación desconocido: {mode}")
    use_vec, use_fts = mode in ("vector", "hybrid"), mode in ("fts", "hybrid")
    model = get_or_create_model_row(session, embedder)
    doc_ids = resolve_document_ids(session, document_ids, collection_id)
    terms = query_terms(question)

    if not doc_ids:
        return RetrievalResult([], 0, "no_documents", model.id, terms, doc_ids)

    qvec = embedder.embed_query(question)
    vec_hits = vector_search(session, model, qvec, doc_ids, s.retrieval_candidates) if use_vec else []
    if use_fts:
        terms, matchable = distinctive_terms(session, terms, doc_ids)
        fts_hits = fts_search(session, model, qvec, matchable, doc_ids, s.retrieval_candidates, coverage_terms=terms)
    else:
        fts_hits = []

    fused = rrf_fuse([b for b, on in ((vec_hits, use_vec), (fts_hits, use_fts)) if on], s.rrf_k)
    considered = len(fused)

    # Cada modelo reparte sus similitudes de forma distinta (e5 las comprime hacia 0,8): el umbral puede ser propio del perfil.
    min_sim = embedder.min_similarity if embedder.min_similarity is not None else s.retrieval_min_similarity
    kept: list[Hit] = []
    for h in fused:
        if use_vec and h.similarity is not None and h.similarity >= min_sim:
            h.relevant_by.append("similarity")
        if use_fts and h.fts_rank is not None and h.coverage >= s.retrieval_fts_min_coverage:
            h.relevant_by.append("keywords")
        if h.relevant_by:
            kept.append(h)
    kept = kept[:top_k]
    return RetrievalResult(
        hits=kept,
        candidates_considered=considered,
        abstain_reason=None if kept else "no_relevant_chunks",
        model_id=model.id,
        terms=terms,
        document_ids=doc_ids,
        ranked=fused,
    )
