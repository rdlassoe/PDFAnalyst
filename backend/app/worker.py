"""Worker: consume `processing_jobs` con FOR UPDATE SKIP LOCKED (varios workers son seguros)."""

from __future__ import annotations

import logging
import signal
import time
import traceback
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .config import get_settings
from .db import SessionLocal
from .models import Document, ProcessingJob
from .services.ingestion import run_embed, run_ingest

log = logging.getLogger("pdfanalyst.worker")
_stop = False


def _handle_signal(*_):
    global _stop
    _stop = True


def recover_stale_jobs(session: Session) -> int:
    """Jobs 'running' cuyo worker murió: vuelven a la cola (el intento ya fue contado)."""
    limit = datetime.now(timezone.utc) - timedelta(minutes=get_settings().job_stale_minutes)
    res = session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.status == "running", ProcessingJob.locked_at < limit)
        .values(status="queued", locked_at=None)
    )
    session.commit()
    return res.rowcount or 0


def claim_job(session: Session) -> ProcessingJob | None:
    job = session.scalar(
        select(ProcessingJob)
        .where(ProcessingJob.status == "queued")
        .order_by(ProcessingJob.created_at, ProcessingJob.id)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    if job is None:
        return None
    job.status = "running"
    job.attempts += 1
    job.locked_at = datetime.now(timezone.utc)
    job.error = None
    session.commit()
    return job


def process_one(session: Session) -> bool:
    """Procesa un job si hay alguno. Devuelve True si había trabajo."""
    job = claim_job(session)
    if job is None:
        return False
    s = get_settings()
    log.info("Job %s (%s) documento=%s intento=%d", job.id, job.kind, job.document_id, job.attempts)
    try:
        if job.kind == "ingest":
            run_ingest(session, job)
        elif job.kind == "embed":
            run_embed(session, job)
        else:  # no debería ocurrir: CHECK en BD
            raise ValueError(f"Tipo de job desconocido: {job.kind}")
        job.status, job.progress = "done", 1.0
        job.finished_at = datetime.now(timezone.utc)
        session.commit()
    except Exception as exc:
        session.rollback()
        log.error("Job %s falló: %s\n%s", job.id, exc, traceback.format_exc())
        job = session.get(ProcessingJob, job.id)
        retry = job.attempts < s.job_max_attempts and not _is_permanent(exc)
        job.status = "queued" if retry else "failed"
        job.error = f"{type(exc).__name__}: {exc}"[:2000]
        job.locked_at = None
        if not retry:
            job.finished_at = datetime.now(timezone.utc)
            doc = session.get(Document, job.document_id)
            if doc:
                doc.status, doc.error = "failed", job.error
        session.commit()
    return True


def _is_permanent(exc: Exception) -> bool:
    """Errores que no mejoran reintentando (PDF sin texto, cifrado, etc.)."""
    from .services.pdf_extract import PdfError

    return isinstance(exc, PdfError)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    s = get_settings()
    log.info("Worker iniciado (poll %.1fs)", s.worker_poll_seconds)
    last_recover = 0.0
    while not _stop:
        with SessionLocal() as session:
            if time.monotonic() - last_recover > 60:
                n = recover_stale_jobs(session)
                if n:
                    log.warning("%d jobs huérfanos devueltos a la cola", n)
                last_recover = time.monotonic()
            worked = process_one(session)
        if not worked:
            time.sleep(s.worker_poll_seconds)
    log.info("Worker detenido")


if __name__ == "__main__":
    main()
