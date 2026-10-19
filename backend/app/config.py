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
    embedding_allow_external: bool = False  # consentimiento para un Ollama remoto del perfil «entorno» (envía el texto de los documentos)
    embedding_timeout_s: float = 120.0

    # --- Medios (imágenes; audio y vídeo en fases posteriores) ---
    media_max_image_mb: int = 20
    media_max_pixels: int = 40_000_000  # contra bombas de descompresión (≈ 40 megapíxeles)
    media_thumb_px: int = 320  # lado mayor de las miniaturas
    vision_max_side: int = 1568  # las imágenes se reducen a este lado mayor antes de enviarlas al modelo de visión
    vision_max_tokens: int = 500
    vision_retries: int = 3  # ante límites de tasa (429) del proveedor

    # --- Figuras dentro de PDF (imágenes incrustadas y gráficos vectoriales) ---
    pdf_figures_enabled: bool = True
    pdf_figures_max: int = 60  # por documento (las más grandes primero); cada una cuesta OCR, vector y, con modelo de visión, una descripción
    pdf_figure_min_px: int = 100  # lado menor mínimo de una imagen incrustada
    pdf_figure_min_area: float = 0.02  # fracción mínima de la página
    pdf_figure_repeat_pages: int = 3  # una imagen en ≥ N páginas es fondo/logo y se ignora
    pdf_vector_min_paths: int = 40  # trazos vectoriales para considerar un gráfico
    pdf_vector_min_area: float = 0.15
    vision_answer_max_images: int = 3  # figuras que se adjuntan al modelo de visión al responder

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
    retrieval_min_similarity_media: float = 0.65  # coseno texto→imagen/audio/vídeo (otra escala que texto→texto); medido con imágenes sintéticas: correctas 0,74-0,84, ajenas ≤ 0,62. CALIBRAR con imágenes reales
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
