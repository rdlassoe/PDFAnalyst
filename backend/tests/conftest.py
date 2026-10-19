"""Fixtures de prueba. Usa una BD aparte (`<db>_test`) en el mismo servidor y ejecuta las migraciones
reales, de modo que cada ejecución también valida Alembic (upgrade/downgrade)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import psycopg
import pytest

# --- Debe ir ANTES de importar `app` (el engine se crea al importar) -------------------------------
from app.config import Settings  # noqa: E402  (lee .env solo para host/puerto/credenciales)

_base = Settings()
TEST_DB = f"{_base.postgres_db}_test"
_tmp_storage = tempfile.mkdtemp(prefix="pdfanalyst_test_")
os.environ["DATABASE_URL"] = (
    f"postgresql+psycopg://{_base.postgres_user}:{_base.postgres_password}"
    f"@{_base.postgres_host}:{_base.postgres_port}/{TEST_DB}"
)
os.environ["STORAGE_DIR"] = _tmp_storage
os.environ["OCR_ENABLED"] = os.environ.get("OCR_ENABLED", "false")

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import SessionLocal, engine  # noqa: E402


def _admin_conninfo() -> str:
    return (
        f"host={_base.postgres_host} port={_base.postgres_port} dbname=postgres "
        f"user={_base.postgres_user} password={_base.postgres_password}"
    )


def _alembic_cfg() -> Config:
    backend = Path(__file__).resolve().parents[1]
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    return cfg


@pytest.fixture(scope="session", autouse=True)
def _database():
    with psycopg.connect(_admin_conninfo(), autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (TEST_DB,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    engine.dispose()
    cfg = _alembic_cfg()
    command.downgrade(cfg, "base")  # prueba del camino de bajada
    command.upgrade(cfg, "head")
    yield
    engine.dispose()


_TABLES = (
    "message_citations, messages, conversations, collection_documents, collections, "
    "chunk_embeddings, document_chunks, document_pages, processing_jobs, documents, embedding_models, llm_profiles, embedding_profiles, vision_profiles, app_settings"
)


@pytest.fixture
def session():
    with SessionLocal() as s:
        s.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))
        s.commit()
        yield s
        s.rollback()


@pytest.fixture(scope="session")
def embedder():
    """Modelo REAL de embeddings (se descarga una vez, ~120 MB)."""
    from app.services.embeddings import get_embedder

    return get_embedder()


@pytest.fixture(autouse=True)
def _clean_storage():
    yield
    import shutil

    for f in Path(_tmp_storage).glob("*"):
        shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
