"""Configuración central. Todo se lee de variables de entorno (nunca de la BD ni del frontend)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_ROOT_ENV), ".env"), extra="ignore", case_sensitive=False
    )

    # --- Base de datos ---
    database_url: str = ""
    postgres_user: str = "pdfanalyst"
    postgres_password: str = ""
    postgres_db: str = "pdfanalyst"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    # --- Almacenamiento de PDFs originales ---
    storage_dir: Path = Path("data/storage")
    max_upload_mb: int = 100

    # --- Embeddings ---
    embedding_provider: str = "sentence-transformers"  # | "ollama"
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"  # "cuda" si hay GPU y torch con CUDA

    # --- OCR ---
    ocr_enabled: bool = True
    ocr_langs: str = "spa+eng"
    ocr_min_chars: int = 30  # páginas con menos texto se consideran escaneadas
    ocr_dpi: int = 200

    # --- Chunking ---
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 200

    # --- Recuperación ---
    retrieval_candidates: int = 30  # candidatos por rama (vector / texto)
    retrieval_top_k: int = 6
    retrieval_min_similarity: float = 0.80  # coseno; CALIBRAR con eval/ (E5 comprime scores altos)
    retrieval_fts_min_coverage: float = 0.5  # fracción de términos de la pregunta presentes
    retrieval_common_term_ratio: float = 0.10  # un término en > ratio·N fragmentos es «común» y no guía la búsqueda textual
    rrf_k: int = 60
    hybrid_enabled: bool = True

    # --- Generación ---
    llm_provider: str = "ollama"
    llm_model: str = "qwen3:8b"
    llm_disable_thinking: bool = True  # modelos con "razonamiento" (qwen3, deepseek-r1): responder directo
    ollama_base_url: str = "http://localhost:11434"
    # Proveedor "openai-compatible" (llama.cpp, LM Studio, vLLM, OpenAI, Groq, Gemini…)
    llm_base_url: str = ""
    llm_api_key: SecretStr | None = None  # clave del perfil «entorno»; las de Ajustes se guardan cifradas en BD (services/credentials.py). Nunca se expone
    llm_allow_external: bool = False  # consentimiento explícito para enviar fragmentos fuera de esta máquina
    llm_temperature: float = 0.1
    llm_max_tokens: int = 700
    llm_timeout_s: float = 180.0
    context_max_tokens: int = 3500

    # --- Secretos (claves API guardadas desde Ajustes) ---
    # Clave maestra Fernet (urlsafe base64 de 32 bytes). Si se omite, se genera en <dir de datos>/secret.key (0600).
    app_secret_key: SecretStr | None = None

    # --- Worker ---
    worker_poll_seconds: float = 2.0
    job_max_attempts: int = 3
    job_stale_minutes: int = 30

    # --- Web ---
    cors_origins: str = "http://localhost:3000"

    @model_validator(mode="after")
    def _build_database_url(self) -> "Settings":
        if not self.database_url:
            self.database_url = (
                f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self

    @property
    def secret_key_path(self) -> Path:
        return self.storage_dir.resolve().parent / "secret.key"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
