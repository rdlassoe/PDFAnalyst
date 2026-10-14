FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Tesseract (Apache-2.0) con idiomas español e inglés para el OCR opcional
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch solo-CPU (mucho más pequeño que la build con CUDA). Para GPU, cambie el índice.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
RUN python - <<'PY'
import tomllib, subprocess, sys
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
PY

# torchvision (CPU, misma versión que torch): el procesador multimodal de EmbeddingGemma 2 lo importa aunque se use solo texto; sin él el modelo no carga.
# Va en su propia capa para no invalidar la caché de torch y de las dependencias.
RUN pip install torchvision --index-url https://download.pytorch.org/whl/cpu

ARG WITH_DEV=false
RUN if [ "$WITH_DEV" = "true" ]; then pip install "pytest>=8.2"; fi

COPY alembic.ini ./
COPY alembic ./alembic
COPY app ./app

RUN useradd --create-home --uid 1000 appuser && mkdir -p /data && chown appuser /data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
