"""Servidor de EMBEDDINGS de desarrollo, con la API `/v1/embeddings` de OpenAI, para probar el módulo de Ajustes sin servicios de pago.

Sirve el modelo local (por defecto `intfloat/multilingual-e5-small`) y registra en la salida cuántos textos/caracteres recibe, de modo
que se puede comprobar QUÉ saldría del equipo hacia un servicio externo real. Si se define STUB_API_KEY exige `Authorization: Bearer`.

Uso (contenedor desechable en la red del compose; reutiliza la caché de modelos del volumen `appdata`):

  docker run --rm -d --name emb-stub --network pdfanalyst_default -e HF_HOME=/data/models -e STUB_API_KEY=clave-de-prueba-123456 \\
    -v pdfanalyst_appdata:/data -v "$PWD/scripts":/stub:ro pdfanalyst-api python /stub/dev_embedding_server.py
  # En Ajustes → Embeddings: URL base http://emb-stub:9999/v1 (un nombre de host se trata como servicio EXTERNO: pide consentimiento)
  docker logs -f emb-stub ; docker rm -f emb-stub

Solo para desarrollo y pruebas: no está pensado para exponerse ni como servicio de producción.
"""

from __future__ import annotations

import os
import threading

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL = os.environ.get("STUB_MODEL", "intfloat/multilingual-e5-small")
API_KEY = os.environ.get("STUB_API_KEY", "")

app = FastAPI(title="Servidor de embeddings de desarrollo")
_model = SentenceTransformer(MODEL, device="cpu")
_lock = threading.Lock()
_stats = {"requests": 0, "texts": 0, "chars": 0}


class EmbeddingsIn(BaseModel):
    model: str
    input: list[str] | str
    encoding_format: str | None = None
    dimensions: int | None = None


@app.post("/v1/embeddings")
def embeddings(body: EmbeddingsIn, authorization: str | None = Header(default=None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(401, "clave API inválida")
    texts = [body.input] if isinstance(body.input, str) else body.input
    with _lock:
        _stats["requests"] += 1
        _stats["texts"] += len(texts)
        _stats["chars"] += sum(len(t) for t in texts)
        print(f"[emb-stub] petición #{_stats['requests']}: {len(texts)} textos, {sum(len(t) for t in texts)} caracteres "
              f"| acumulado: {_stats['texts']} textos, {_stats['chars']} caracteres | primer texto: {texts[0][:60]!r}", flush=True)
    vecs = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    return {"object": "list", "model": body.model,
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
            "usage": {"prompt_tokens": 0, "total_tokens": 0}}


@app.get("/stats")
def stats():
    return _stats


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999, log_level="warning")
