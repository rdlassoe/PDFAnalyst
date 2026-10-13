from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings

engine = create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def get_session() -> Iterator[Session]:
    """Dependencia de FastAPI: una sesión por petición."""
    with SessionLocal() as session:
        yield session
