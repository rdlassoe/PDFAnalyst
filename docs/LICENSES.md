# Licencias de dependencias y modelos

Revisado el 2026-10-13. Las columnas "Fuente" indican cómo se verificó: **metadatos** = leído del paquete instalado
(`scripts/py_licenses.py`); **ficha** = leído del modelo descargado; **paquete** = `ollama show --license`;
**conocida** = licencia publicada por el proyecto, no comprobada automáticamente (verifique al actualizar versiones).

## Licencia del proyecto: AGPL-3.0-or-later

Motivo: PyMuPDF se distribuye bajo **AGPL-3.0 o licencia comercial de Artifex**. Usarlo en un servicio accesible por red
obliga a publicar el código fuente completo bajo AGPL-compatible. Es software libre, pero **no** es permisivo: si
necesitara distribuir PDF Analyst como producto cerrado, debe (a) comprar la licencia comercial o (b) sustituir PyMuPDF
por `pypdf` (BSD-3, ya incluido como respaldo) + `pdfminer.six` (MIT), perdiendo calidad en el orden de lectura.

## Backend (Python)

| Componente | Licencia | Fuente |
|------------|----------|--------|
| FastAPI | MIT | metadatos |
| Uvicorn | BSD-3-Clause | metadatos |
| python-multipart | Apache-2.0 | metadatos |
| SQLAlchemy | MIT | metadatos |
| Alembic | MIT | metadatos |
| psycopg 3 | LGPL-3.0 (enlace dinámico; compatible con AGPL) | metadatos |
| Pydantic / pydantic-settings | MIT | metadatos |
| pgvector (cliente Python) | MIT | metadatos |
| **PyMuPDF** (texto, OCR y, ahora, detección y renderizado de las figuras de los PDF) | **AGPL-3.0 / comercial** | metadatos |
| pypdf | BSD-3-Clause | metadatos |
| pytesseract | Apache-2.0 | metadatos |
| Pillow | MIT-CMU (HPND) | metadatos |
| sentence-transformers | Apache-2.0 | metadatos |
| PyTorch (CPU) | BSD-3 y otras permisivas (Apache-2.0, etc.) | metadatos |
| torchvision (CPU; lo importa el procesador de EmbeddingGemma 2) | BSD | metadatos |
| transformers · huggingface_hub (descarga del modelo) | Apache-2.0 | metadatos |
| uvicorn (servicio `embedder`) | BSD-3-Clause | metadatos |
| NumPy | BSD-3-Clause (+ 0BSD, MIT, Zlib, CC0 en componentes) | metadatos |
| httpx | BSD-3-Clause | metadatos |
| cryptography (cifrado de claves) | Apache-2.0 OR BSD-3-Clause | metadatos |
| pytest (solo desarrollo) | MIT | metadatos |

## Frontend

| Componente | Licencia | Fuente |
|------------|----------|--------|
| Next.js, React, React DOM | MIT | conocida |
| Tailwind CSS, PostCSS, Autoprefixer | MIT | conocida |
| Lucide Icons (`lucide-react`) | ISC | conocida |
| PDF.js (`pdfjs-dist`) | Apache-2.0 | conocida |
| TypeScript | Apache-2.0 | conocida |

## Infraestructura y herramientas

| Componente | Licencia | Fuente / nota |
|------------|----------|---------------|
| PostgreSQL 16 | PostgreSQL License | conocida |
| pgvector (extensión, v0.8.7 en uso) | PostgreSQL License | conocida |
| `unaccent` (contrib de PostgreSQL) | PostgreSQL License | conocida |
| Tesseract OCR | Apache-2.0 | conocida |
| Ollama | MIT | conocida |
| Docker Engine / Compose | Apache-2.0 | conocida. **Docker Desktop** es software propietario con condiciones comerciales para organizaciones grandes; en Linux basta Docker Engine. |

## Modelos (pesos) — se verifican por separado del código

| Modelo | Uso | Licencia de los pesos | Fuente |
|--------|-----|-----------------------|--------|
| `intfloat/multilingual-e5-small` | Embeddings (por defecto) | MIT | ficha |
| `qwen3:8b` (Ollama) | Generación (por defecto) | Apache-2.0 | paquete |
| `google/embeddinggemma-2` | Embeddings (opcional, local, servicio `embedder`) | Apache-2.0 según los metadatos de la ficha de Hugging Face (el repositorio no incluye un archivo LICENSE y su README enlaza una página de licencia de Google: **léala antes de redistribuir los pesos**) | ficha |

Condiciones de uso/redistribución: ambos permiten uso comercial y redistribución con aviso de licencia. **No** hay
dependencia obligatoria de servicios de pago: los modelos se ejecutan localmente (EmbeddingGemma 2 es opcional y también local; requiere descargar ≈1,5 GB de Hugging Face la primera vez).

### Modelos que NO deben asumirse como "open source"

- Familia **Llama** (p. ej. `llama3.1:8b`): licencia comunitaria propia de Meta con restricciones de uso y
  redistribución (no es OSI). Presente en el equipo de desarrollo pero **no** usado por defecto.
- Cualquier modelo cuyos pesos sean descargables no es por ello de código abierto: revise la licencia concreta
  (`ollama show --license <modelo>` o la ficha de Hugging Face) antes de recomendarlo o redistribuirlo.
- Al cambiar `EMBEDDING_MODEL` o `LLM_MODEL`, actualice esta tabla.

## Proveedores externos opcionales

OpenAI, Groq, Gemini y similares se conectan mediante el adaptador `openai-compatible`; son servicios de terceros con sus
propios términos y **envían los fragmentos recuperados fuera de esta máquina**. Usados como **modelo de visión** (Ajustes → Modelo de visión) reciben además las **imágenes de los PDF** (reducidas y sin EXIF) y, con «responder mirando», la pregunta y los fragmentos;
los modelos de visión que ofrecen (p. ej. `qwen/qwen3.8-27b` en Groq, `gemini-3.8-flash`) son servicios, no pesos que se distribuyan con el proyecto: sus términos de uso y de datos son los de cada proveedor (en la capa gratuita de Gemini, según su página de precios, el contenido puede usarse para mejorar los productos de Google). Un modelo de visión **local** de Ollama (p. ej. `qwen3-vl:4b`) no se incluye ni se ha descargado: si se adopta, verifique su licencia con `ollama show --license` antes de redistribuirlo. Están bloqueados salvo
`LLM_ALLOW_EXTERNAL=true`. Revise la política de retención/entrenamiento de datos de cada proveedor antes de usarlos con
documentos sensibles. Esta integración se ha probado con un servidor compatible real (endpoint `/v1` de Ollama) y con
transportes simulados para la lógica de errores; **Groq** se probó además con visión real (descripción y respuesta con imágenes). Gemini y OpenAI **no** se han probado contra su servicio comercial.
