from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_session
from ..providers import LLMError
from ..services import embedding_settings
from ..services.ingestion import embedding_counts, model_coverage
from ..services.llm_settings import get_llm_provider

router = APIRouter(tags=["system"])


@router.get("/health")
def health(session: Session = Depends(get_session)):
    """Cada componente se informa por separado: un proveedor caído no hace fallar la gestión documental."""
    s = get_settings()
    out: dict = {"database": {"ok": False}, "pgvector": {"ok": False}}
    try:
        out["database"] = {"ok": session.scalar(text("SELECT 1")) == 1}
        ver = session.scalar(text("SELECT extversion FROM pg_extension WHERE extname='vector'"))
        out["pgvector"] = {"ok": ver is not None, "version": ver}
    except Exception as exc:  # noqa: BLE001
        out["database"] = {"ok": False, "detail": exc.__class__.__name__}
    try:
        # Sin llamar al proveedor (sería lento y, si es externo, enviaría datos en cada sondeo): solo configuración y cobertura.
        cfg = embedding_settings.active_config(session)
        row = embedding_settings.active_model_row(session)
        cov = model_coverage(session, row.id if row else -1)
        out["embeddings"] = {
            "ok": True, "provider": cfg.kind, "model": cfg.model, "external": cfg.external,
            "local": not cfg.external, "dimension": row.dimension if row else None,
            "documents_total": len(cov), "documents_indexed": sum(1 for n, e in cov.values() if e >= n),
        }
    except Exception as exc:  # noqa: BLE001
        out["embeddings"] = {"ok": False, "detail": str(exc), "provider": None, "model": None, "local": None, "external": None}
    try:
        provider = get_llm_provider(session)
        h = provider.health()
        out["generation"] = {"provider": provider.name, "model": provider.model, "external": provider.external, **h}
    except Exception as exc:  # noqa: BLE001
        out["generation"] = {"ok": False, "detail": str(exc)}
    out["ok"] = out["database"]["ok"] and out["pgvector"]["ok"]
    return out


@router.get("/embedding-models")
def embedding_models(session: Session = Depends(get_session)):
    return embedding_counts(session)
