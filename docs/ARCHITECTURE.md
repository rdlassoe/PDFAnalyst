# PDF Analyst — Arquitectura y decisiones técnicas

Estado: Fases 1–7 implementadas; Fase 8 parcial (ver README §Estado). Documento vivo: cada decisión indica su motivo
y cuándo revisarla. El historial cronológico está en [CHANGELOG.md](CHANGELOG.md).

Cambios respecto al diseño inicial (decididos al implementar y verificar con datos):

- **Módulo Ajustes (D14 revisada):** perfiles de proveedor LLM editables desde la interfaz, con claves cifradas y consentimiento
  explícito para servicios externos; el perfil «env» conserva el comportamiento por variables de entorno.
- **Los chunks nunca cruzan páginas.** Con chunks multipágina las citas apuntaban a rangos ("pp. 1-3") y el visor no podía
  abrir una página concreta. Se prioriza la precisión de la cita sobre la continuidad entre páginas.
- **FTS con dos juegos de tokens** (originales + separadores `-/_.,;:@#` normalizados): el parser de PostgreSQL parte
  `ISO-27001` en `iso` + `-27001`, de modo que buscar `27001` no coincidía.
- **FTS ponderado por rareza (IDF):** con un libro real, palabras muy frecuentes del dominio («oro», «bolsa») dominaban el
  ranking textual. Solo los términos poco frecuentes guían la consulta (ver D9).
- **La abstención fiable la aporta el LLM, no el umbral de similitud** (ver D10): con datos reales el umbral coseno no separa
  preguntas con y sin respuesta.
- **La regeneración tras una cita inválida incluye la respuesta fallida del modelo**; sin ella, el modelo respondía a la
  instrucción de corrección en lugar de a la pregunta (detectado con el modelo real).
- **Regla 6 del prompt:** las preguntas sobre el propio asistente o ajenas a los fragmentos se responden con `NO_ENCONTRADO`
  (una meta-pregunta había filtrado el prompt del sistema citando un fragmento irrelevante).
- **Reintentos con Ollama:** 500/502/503 transitorios al cargar el modelo se reintentan 3 veces.
- **Embeddings configurables (D17) y EmbeddingGemma 2 local (D18):** el modelo de embeddings se elige en Ajustes (local, Ollama o API compatible con OpenAI, incluso un servicio externo; el compose incluye el servicio local `embedder` con EmbeddingGemma 2), con
  indexación únicamente según el modelo seleccionado, consentimiento explícito (un servicio externo recibe el texto íntegro de los documentos), vista previa de lo que se enviaría
  y reindexado en segundo plano. No hay modo «combinado» por ahora.
- **Frontend (D16):** sistema de notificaciones flotantes y diálogo de confirmación propios, eliminación de conversaciones con
  «Deshacer», diseño adaptable (pestañas y visor deslizante en móvil). Detalle en [FRONTEND.md](FRONTEND.md).
- Ollama del host por defecto (en Docker Desktop `host.docker.internal` alcanza servicios en loopback); el contenedor
  `ollama` es opcional (perfil `ollama`).

## 1. Visión general

```
┌────────────┐   HTTP/JSON    ┌───────────────────────────┐        ┌──────────────────────────┐
│  Frontend  │ ─────────────▶ │  Backend FastAPI (api)    │ ─────▶ │ PostgreSQL 16 + pgvector │
│  Next.js   │ ◀───────────── │  - upload / consulta      │  SQL   │  documentos, páginas,    │
│  PDF.js    │   (sin acceso  │  - retrieval híbrido      │        │  chunks, embeddings,     │
│  Avisos    │    a la BD)    │  - generación + citas     │        │  conversaciones, jobs,   │
│  Ajustes   │                │  - Ajustes (LLM, claves)  │        │  perfiles LLM (cifrados) │
└────────────┘                └──────────┬────────────────┘        └──────────▲───────────────┘
                                         │ encola job (fila en BD)            │
                              ┌──────────▼────────────────┐                   │
                              │  Worker (mismo código)    │───────────────────┘
                              │  extracción → OCR →       │
                              │  chunking → embeddings    │
                              └──────────┬────────────────┘
                                         │
              ┌──────────────────────────┼─────────────────────────┐
              ▼                          ▼                         ▼
      Volumen local PDFs        Embeddings, a elegir en     LLM de generación, a elegir en Ajustes:
      (data/storage)            Ajustes: e5 en proceso      Ollama (por defecto) · Groq · Gemini ·
      + secret.key (0600)       (entorno) o servicio        OpenAI · servidor compatible
      + caché de modelos        local `embedder` (HTTP,
                                EmbeddingGemma 2, CPU)
```

El servicio `embedder` es un contenedor aparte con una sola copia del modelo; la API y el worker lo llaman por HTTP dentro de la red de Docker (sin puerto en el host).

Componentes desacoplados: el frontend solo habla con la API; solo la API y el worker
conocen las credenciales de PostgreSQL.

## 2. Decisiones técnicas

| # | Decisión | Motivo | Revisar cuando |
|---|----------|--------|----------------|
| D1 | PostgreSQL 16 + pgvector como único almacén (datos + vectores + texto completo) | Requisito. Una sola transacción para chunk+embedding, sin sincronizar un segundo almacén. | Si se superan ~5–10 M de vectores. |
| D2 | SQLAlchemy 2.0 **síncrono** + psycopg 3 | El trabajo pesado (PDF, embeddings) es CPU-bound; FastAPI ejecuta endpoints `def` en un pool de hilos. Menos complejidad que async. | Si la API necesita miles de conexiones concurrentes. |
| D3 | Cola de trabajos en PostgreSQL (`processing_jobs` + `FOR UPDATE SKIP LOCKED`) | Procesamiento asíncrono sin añadir Redis/Celery. Reintentos idempotentes y estado visible en la misma BD. | Si se necesitan múltiples colas con prioridades complejas. |
| D4 | Embeddings por defecto (perfil «entorno»): `intfloat/multilingual-e5-small` vía sentence-transformers (384 dim, MIT); **sustituible desde Ajustes (D17)** | Multilingüe (español/inglés), ligero (~120 MB), CPU suficiente, licencia MIT. Requiere prefijos `query:` / `passage:` — se guardan en `embedding_models`. | Evaluar `BAAI/bge-m3` (MIT, 1024 dim) si la precisión no basta. |
| D5 | Un modelo de generación **no** se usa para embeddings salvo configuración explícita (los perfiles de embeddings y de LLM son independientes) | Los modelos de chat no producen vectores compatibles. Proveedores de embeddings y de generación son configuraciones independientes. | — |
| D6 | Tabla única `chunk_embeddings` con columna `vector` **sin dimensión fija** + índices parciales por modelo | Permite varios modelos/dimensiones coexistiendo sin mezclar vectores: `CREATE INDEX … ((embedding::vector(384))) WHERE model_id = N`. Cada consulta filtra por `model_id`. | — |
| D7 | Métrica: coseno (`<=>`) con vectores normalizados | Es la métrica con la que se entrenan los modelos E5/BGE. Se registra por modelo en `embedding_models.distance`. | Al cambiar de modelo. |
| D8 | Búsqueda exacta por defecto; HNSW por modelo bajo demanda (`reindex`) | Con <100 k chunks la búsqueda exacta es rápida y 100 % precisa. HNSW (recall ajustable, mejor latencia) se crea con un comando documentado cuando el volumen lo justifique. IVFFlat se descarta: requiere datos previos para entrenar listas y degrada con inserciones. | Medir con `eval/` al crecer el corpus. |
| D9 | Recuperación híbrida: vector + FTS (`tsvector`, config `simple` + `f_unaccent`) fusionados con **RRF** (k=60). La consulta textual usa solo los términos **poco frecuentes** (df ≤ max(2, 10 % de los fragmentos)); un término raro ausente del corpus cuenta en el denominador de la cobertura | RRF no requiere normalizar scores heterogéneos. Config `simple` conserva identificadores y códigos sin stemming erróneo en documentos multilingües; `f_unaccent` permite buscar sin tildes. La ponderación por rareza (idea de IDF) subió el recall@1 textual de 0,33 a 0,60 y el recall@5 híbrido a 1,00 sobre un libro real (`eval/RESULTS.md`). | Si el corpus es monolingüe, evaluar `spanish`. |
| D10 | Doble barrera de abstención: (1) filtro de relevancia previo al LLM — un fragmento pasa si su similitud coseno ≥ umbral **o** cubre ≥ 50 % de los términos distintivos; si ninguno pasa no se llama al LLM; (2) el LLM responde `NO_ENCONTRADO` si los fragmentos no contienen la respuesta | Un score RRF es relativo, no indica relevancia absoluta. **Medido con datos reales:** el umbral coseno **no separa** preguntas con y sin respuesta (máx. sin respuesta 0,861 > mín. con respuesta 0,819), por lo que la barrera (2) es la que da la abstención fiable (7/7 en la evaluación). | Recalibrar con `eval/` al cambiar de modelo de embeddings o de corpus. |
| D11 | Citas: el LLM solo ve etiquetas efímeras `[C1]…[Cn]`; el servidor las mapea a `chunk_id` reales | El modelo no puede inventar una referencia que el servidor acepte. Etiquetas desconocidas → cita inválida → regeneración controlada (1 intento) y, si persiste, se eliminan y se marca el mensaje. | — |
| D12 | PDFs originales en disco (`data/storage/<uuid>.pdf`), nombre generado por el servidor | Evita path traversal: nunca se usa el nombre del usuario como ruta. BD guarda ruta relativa, hash SHA-256 y metadatos. | — |
| D13 | Interfaz de proveedor LLM (`LLMProvider`, dos métodos: `chat`, `health`) construida desde un `ProviderConfig` explícito: adaptadores **Ollama** y **OpenAI-compatible** (cubre llama.cpp, vLLM, LM Studio, OpenAI, Groq, Gemini con su endpoint compatible). Un destino no local se considera **externo** (`is_local_url` analiza IPs; un dominio como `10.evil.com` no es local) y exige consentimiento | Un adaptador HTTP genérico cubre casi todos; solo se añade uno nativo si hay una necesidad demostrada. OpenCode **no** se incluye hasta verificar una API real. | Si aparece un proveedor sin API compatible. |
| D14 | **(revisada)** Claves API de proveedores: entorno (perfil «env») o Ajustes, donde se guardan **cifradas** (Fernet) en `llm_profiles`; clave maestra fuera de la BD (`APP_SECRET_KEY` o `secret.key` 0600); nunca se devuelven al frontend ni se registran | El usuario necesita cambiar de proveedor desde la interfaz; se mantiene que un volcado/copia de la BD no revele secretos. Consentimiento explícito por perfil para destinos externos. | Si se añade multiusuario: gestor de secretos externo y permisos por usuario. |
| D18 | **EmbeddingGemma 2 se ejecuta en local en un servicio propio (`embedder`).** Contenedor con `app/embedding_server.py` (FastAPI, API `/v1/embeddings` de OpenAI, solo red interna), modelo `google/embeddinggemma-2` por sentence-transformers cargado **solo texto** (`vision_config`/`audio_config` = None: ≈271M parámetros), fp32 en CPU, carga perezosa y una inferencia a la vez. API y worker lo usan como un perfil más (plantilla «EmbeddingGemma 2 (local)», prefijos oficiales). **Umbral de similitud por perfil** (`embedding_profiles.min_similarity`, migración 0004): editable con el perfil activo, sin reindexar | Una sola copia del modelo en memoria (el worker y la API no la duplican); reutiliza todo el módulo de perfiles/activación/consentimiento (D17) sin un camino especial; con la RAM disponible (≈3,5 GB libres, GPU ocupada por el LLM) cabe: ≈1,9 GiB medidos. Ollama no sirve: sus compilaciones de embeddinggemma-2 exigen MLX (Apple Silicon). El umbral por perfil hace falta porque Gemma separa las similitudes (con respuesta ≥ 0,72; sin respuesta ≤ 0,71) y con el 0,80 de e5 descartaría la mitad de las respuestas por similitud | Si la memoria es crítica: `bfloat16`, o un servidor remoto (entonces es externo y exige consentimiento). Si se quiere GPU: imagen con CUDA y `EMBED_SERVER_DEVICE=cuda`. |
| D17 | **Embeddings seleccionables con indexación solo del modelo activo.** Perfil «env» + perfiles en `embedding_profiles` (clave cifrada, mismos controles que D14); `activate` comprueba acceso, dimensión y prefijos *antes* de cambiar y luego reencola solo los documentos sin vectores del modelo; los vectores de los demás modelos se conservan (volver es instantáneo) y se pueden borrar. El worker resuelve el modelo activo al ejecutar y repite si cambia a mitad de trabajo. Adaptadores: local en proceso, Ollama `/api/embed`, compatible con OpenAI `/embeddings` (reintentos 429/5xx, errores permanentes sin reintento) | Con un proveedor externo se envía **todo el texto de los documentos** (no solo los fragmentos relevantes, como con un LLM): consentimiento explícito por perfil, vista previa del volumen enviado y bloqueo inmediato al revocarlo. Indexar solo con el modelo activo evita costes y envíos duplicados. | Si se necesita comparar modelos en caliente o un modo combinado (RRF de varios espacios vectoriales): indexación multi-modelo. |
| D15 | Prompts defensivos frente a inyección en documentos | El contenido recuperado se entrega delimitado como datos no confiables (los delimitadores dentro del texto se neutralizan); el sistema nunca ejecuta instrucciones del documento ni tiene herramientas con efectos. Probado con un PDF malicioso y con meta-preguntas (regla 6). | Ampliar el conjunto adversarial. |
| D16 | Frontend: notificaciones flotantes y confirmación propias (`components/Toaster.tsx`), sin biblioteca externa; acciones destructivas con confirmación, y la eliminación de conversaciones es **diferida con «Deshacer»** (el borrado real ocurre al cerrarse el aviso, o con `keepalive` al cerrar la página) | Retroalimentación consistente y accesible sin dependencias extra ni `alert()/confirm()` del navegador; deshacer es más seguro que preguntar. El borrado diferido es seguro por defecto: si la pestaña se cierra, en el peor caso no se borra. | Si se necesitara deshacer persistente (soft-delete en backend). |

## 3. Modelo de datos

Migraciones Alembic: `0001` (documentos, chunks, embeddings, conversaciones, jobs), `0002` (ajustes de LLM), `0003` (ajustes de embeddings) y `0004` (umbral de similitud por perfil de embeddings).

- `documents` — id (uuid), sha256 (único), filename (solo para mostrar), storage_name (`<uuid>.pdf`, generado por el servidor), size_bytes, page_count, title, author, status (`uploaded|processing|ready|failed`), error, metadata (JSONB: chunk_count, ocr_pages, warnings).
- `document_pages` — document_id, page_number, text, ocr_used. Se conserva por separado porque el visor y las citas necesitan el texto por página.
- `document_chunks` — id (bigint), document_id, chunk_index, page_start, page_end (siempre iguales: los chunks no cruzan páginas), text, `tsv` (columna generada con `f_unaccent` + GIN).
- `embedding_models` — provider, name, dimension, distance, query_prefix, passage_prefix, normalize.
- `chunk_embeddings` — PK (chunk_id, model_id), `embedding vector` (sin dimensión fija; índices HNSW parciales por modelo).
- `collections`, `collection_documents` — agrupación temática.
- `conversations`, `messages` (status `ok|abstained|unverified|error`, metadata con la traza de recuperación y el proveedor usado), `message_citations` (chunk_id/document_id con `SET NULL` al borrar el documento, label, rank, score, páginas y `snapshot` del texto citado).
- `processing_jobs` — kind (`ingest` | `embed`), status, progress, attempts, error, locked_at. Cola con `FOR UPDATE SKIP LOCKED`.
- `llm_profiles` — name (único), kind (`ollama|openai-compatible`), base_url, model, `api_key_enc` (token Fernet; nunca texto plano), allow_external.
- `embedding_profiles` — name (único), kind (`ollama|openai-compatible`), base_url, model, `api_key_enc` (Fernet), allow_external, query_prefix, passage_prefix, dimensions (truncado Matryoshka opcional), `min_similarity` (umbral coseno de relevancia propio del modelo; NULL = `RETRIEVAL_MIN_SIMILARITY`). Solo la *configuración de acceso*: el registro de espacios vectoriales sigue en `embedding_models`.
- `app_settings` — clave/valor no secretos (`active_llm` y `active_embedding`: `env` o el uuid del perfil activo).

No se crea tabla de usuarios: la ejecución local es monousuario. Si se añade multiusuario, se añadirá `owner_id` a
`documents`/`collections`/`conversations`/`llm_profiles` y se aplicarán filtros obligatorios en cada consulta (ver §6 y
OPERATIONS §8).

## 4. Pipeline de ingesta

1. `POST /documents` valida: extensión, *magic bytes* `%PDF-`, tamaño máximo, apertura correcta con PyMuPDF.
2. SHA-256 → si ya existe, devuelve el documento existente (sin duplicar; si estaba `failed` se reencola).
3. Se guarda el fichero con nombre generado y se crea `documents` + `processing_jobs(ingest)`.
4. Worker: extrae texto por página (PyMuPDF). Si una página tiene muy poco texto y el OCR está habilitado → Tesseract sobre la página renderizada.
5. Chunking consciente de páginas: párrafos → ventanas de ~1200 caracteres con solapamiento de 200, **sin cruzar el límite de página** (cada cita apunta a una única página real).
6. Embeddings por lotes con el modelo activo → `chunk_embeddings`.
7. Estado final `ready` o `failed` + mensaje de error.

**Idempotencia:** reprocesar un documento borra sus páginas/chunks y los reinserta en una única
transacción; un fallo a mitad no deja datos parciales. El job se reintenta hasta `MAX_ATTEMPTS`.

## 5. Pipeline de consulta

1. Embedding de la pregunta con el **modelo activo** (prefijo configurado; si es externo, la pregunta se envía a ese servicio); se resuelven los documentos consultables (`ready` = indexados con ese modelo) según la selección o colección.
2. Rama vectorial: top-N por coseno (`<=>`). Rama textual: se calculan los términos distintivos (frecuencia de documento real) y
   se consulta `to_tsquery('simple', t1 | t2 | …)` con `ts_rank_cd`; la cobertura es la fracción de términos distintivos presentes.
3. Fusión RRF → un fragmento es relevante si `similitud ≥ umbral` o `cobertura ≥ 0,5` (D10); se toman los top-K relevantes sin
   rellenar hasta un número fijo.
4. Si no queda ninguno → respuesta de abstención **sin llamar al LLM**.
5. Contexto con etiquetas `[C1..Ck]`, recortado por presupuesto de tokens **por chunk completo** (no se cortan evidencias a la mitad);
   el texto va delimitado como dato no confiable.
6. El proveedor activo (resuelto desde Ajustes) genera la respuesta → validación de citas (D11; una regeneración controlada que
   incluye la respuesta fallida) → persistencia en `messages` y `message_citations` con la traza de lo recuperado.
7. Estados del mensaje: `ok`, `abstained` (sin evidencia o `NO_ENCONTRADO`), `unverified` (sin citas válidas tras regenerar), `error`
   (proveedor caído/bloqueado; la API responde 200 con el aviso y la gestión documental sigue operativa).

## 6. Seguridad y privacidad

- Predeterminado 100 % local: embeddings en proceso, generación en Ollama. Un proveedor externo exige consentimiento explícito
  por perfil y envía únicamente la pregunta y los fragmentos recuperados; cada respuesta indica proveedor, modelo y si el contexto salió.
- **Claves API (D14):** cifradas con Fernet en `llm_profiles`; clave maestra en `APP_SECRET_KEY` o `/data/secret.key` (0600, creada con
  `O_EXCL`); la API devuelve solo `has_key` y los 4 últimos caracteres; no se registran. El archivo de la clave maestra **no** entra
  en las copias de seguridad (una copia filtrada no revela claves). Alcance: protege frente a fugas de la BD o de sus copias, no
  frente a un atacante con ejecución de código en el servidor.
- **Servicio `embedder` (local, D18):** solo en la red interna de Docker (sin puerto en el host), sin acceso a la base de datos; recibe textos y devuelve vectores. `embedder` figura en la lista de hosts locales (`is_local_url`), por lo que no exige consentimiento externo; un nombre de host propio con ese nombre en otra red sería tratado como local (por eso solo se acepta como nombre de servicio del compose). Clave opcional (`EMBED_SERVER_API_KEY`).
- **Embeddings externos:** envían el texto íntegro de los fragmentos (al indexar) y las preguntas (al buscar). Consentimiento por perfil (`allow_external`), vista previa del volumen
  (`POST /settings/embeddings/preview`), bloqueo inmediato al revocarlo (incluso con el adaptador en caché) y errores permanentes sin reintentos. «Probar» solo envía una frase fija.
- **Anti-SSRF en Ajustes:** solo `http(s)`, sin credenciales en la URL, bloqueo de enlace local/metadatos de nube, sin seguir
  redirecciones; «Probar» solo lista modelos (no envía documentos). Claves con espacios/saltos de línea rechazadas (inyección de cabeceras).
- SQL siempre parametrizado (SQLAlchemy Core/ORM; los pocos `text()` usan parámetros enlazados; las dimensiones de vector, que
  no admiten parámetros, son enteros validados). La consulta `tsquery` se arma solo con tokens `[^\W_]+`.
- Subidas: límite de tamaño, verificación de firma `%PDF-`, apertura de prueba, nombre en disco generado, sin rutas del usuario.
- CORS restringido al origen del frontend (métodos GET/POST/PUT/PATCH/DELETE, cabecera `Content-Type`). Todos los puertos se
  publican solo en `127.0.0.1`.
- Texto de documentos tratado como no confiable (D15).
- Copias de seguridad: `scripts/backup.ps1|sh` (pg_dump + copia de `storage`) y restauración documentada en `docs/OPERATIONS.md`.
- Sin autenticación de usuarios: cualquiera con acceso a la aplicación puede cambiar Ajustes. No exponer fuera de loopback.

## 7. Fases y criterios de aceptación

| Fase | Entregable | Aceptación (reproducible) |
|------|-----------|---------------------------|
| 1 | Estructura, config, Docker Compose, PostgreSQL+pgvector, migraciones Alembic | `docker compose up db` → `alembic upgrade head` → `SELECT extname FROM pg_extension` incluye `vector`; test inserta y consulta un vector. |
| 2 | Ingesta PDF, extracción, OCR opcional | Subir un PDF crea documento+páginas; subirlo otra vez no duplica; un PDF corrupto es rechazado; PDF escaneado produce texto con OCR. |
| 3 | Chunking, embeddings, índices | Todos los chunks tienen embedding del modelo activo con la dimensión correcta; insertar dimensión incorrecta falla; reintentar no duplica. |
| 4 | Búsqueda semántica, textual, híbrida | Sobre el corpus de `eval/`, recall@5 medido para vector / FTS / híbrido; las consultas con códigos exactos mejoran con FTS. |
| 5 | Ollama + respuestas con citas | Toda cita devuelta existe en los chunks recuperados; pregunta fuera de corpus → abstención; cita inventada → detectada en test. |
| 6 | UI completa + visor PDF | Flujo subir → preguntar → clic en cita abre el PDF en la página correcta. |
| 7 | Proveedores externos opcionales | Cambiar de proveedor por config; con el proveedor caído, la gestión documental sigue funcionando. |
| 8 | Evaluación, seguridad, rendimiento, docs finales | Informe de precisión, pruebas de inyección y de path traversal, `docs/LICENSES.md` completo. |

**Estado de verificación por fase** (detalle en README §Estado y `eval/RESULTS.md`):

| Fase | Estado | Evidencia |
|------|--------|-----------|
| 1–5 | Verificadas | 240 pruebas automáticas en Docker (migraciones con downgrade/upgrade, OCR real, embeddings reales, HNSW parcial, abstención, citas inventadas, Ollama real) |
| 6 | Verificada manualmente | Flujo completo en navegador; notificaciones, deshacer, móvil. **Sin pruebas automáticas de frontend** |
| 7 | Parcial | Ollama y servidor compatible (endpoint `/v1` de Ollama) probados; Groq/Gemini/OpenAI **no** probados con servicio real |
| 7b | Verificada (local) / Parcial (externos) | Embeddings seleccionables: **EmbeddingGemma 2 local verificado** de extremo a extremo (indexación de 274 fragmentos, evaluación A/B frente a e5, 15/15 y 7/7 con el LLM real, interfaz en el navegador). Los proveedores por API **externos** se verificaron con un servidor de desarrollo (e5 por `/embeddings`), no con un servicio real |
| 8 | Parcial | Evaluación con un libro real (22 preguntas); pruebas de inyección y de rutas; falta rendimiento a escala y más corpus |

## 8. Licencias (resumen; detalle en `docs/LICENSES.md`)

**Atención — PyMuPDF es AGPL-3.0.** Es software libre, pero su licencia exige que, si PDF Analyst
se ofrece como servicio en red, se publique el código fuente completo bajo AGPL-compatible. Por
eso el proyecto se licencia bajo **AGPL-3.0**. Si en el futuro se necesitara distribuir bajo una
licencia más permisiva, habría que sustituir PyMuPDF por `pypdf` (BSD) + `pdfminer.six` (MIT)
o adquirir licencia comercial de Artifex. Los **pesos** de cada modelo se verifican por separado
del código (p. ej. `multilingual-e5-small`: MIT, verificado en su ficha; `qwen3:8b`: Apache-2.0, verificado con
`ollama show --license`; Llama: licencia propia con restricciones de uso). La tabla completa está en `docs/LICENSES.md`.

## 9. Frontend

Resumen; la guía completa (componentes, API de notificaciones, convenciones, lista de verificación manual) está en
[FRONTEND.md](FRONTEND.md).

- Next.js 15 (App Router) + React 19 + TypeScript estricto + Tailwind 3; iconos Lucide; PDF.js para el visor.
- Páginas: `/` (documentos · conversación · visor) y `/ajustes` (proveedor LLM). Estado local en React; sin librería global.
- El navegador llama directamente a la API (`NEXT_PUBLIC_API_URL`, incorporada en el build); nunca ve credenciales de PostgreSQL ni claves API.
- Diseño adaptable: tres columnas en escritorio; pestañas «Documentos/Conversación» y visor deslizante por debajo de 1024 px.

## 10. API HTTP (resumen; OpenAPI interactivo en `/docs`)

| Grupo | Endpoints |
|-------|-----------|
| Sistema | `GET /health`, `GET /embedding-models` |
| Documentos | `POST /documents` (multipart), `GET /documents`, `GET|DELETE /documents/{id}`, `POST /documents/{id}/reprocess`, `GET /documents/{id}/jobs`, `GET /documents/{id}/file`, `GET /documents/{id}/pages/{n}`, `GET /chunks/{id}` |
| Colecciones | `POST|GET /collections`, `DELETE /collections/{id}`, `PUT|DELETE /collections/{id}/documents/{doc_id}` |
| Búsqueda y chat | `POST /search` (recuperación sin LLM, para depurar/evaluar), `POST|GET /conversations`, `DELETE /conversations/{id}`, `GET|POST /conversations/{id}/messages` |
| Ajustes embeddings | `GET /settings/embeddings`, `POST /settings/embeddings/profiles`, `PATCH|DELETE /settings/embeddings/profiles/{id}`, `POST /settings/embeddings/preview`, `PUT /settings/embeddings/active`, `POST /settings/embeddings/test`, `DELETE /settings/embeddings/models/{id}/vectors` |
| Ajustes LLM | `GET /settings/llm`, `POST /settings/llm/profiles`, `PATCH|DELETE /settings/llm/profiles/{id}`, `PUT /settings/llm/active`, `POST /settings/llm/test` |

Códigos relevantes: `422` entrada inválida (PDF corrupto, URL no permitida), `409` conflicto (nombre duplicado, proveedor externo sin
consentimiento). Un fallo del proveedor LLM no es un error HTTP: la conversación recibe un mensaje de estado `error` (el manejador
global que convierte `LLMError` en `502` queda como red de seguridad). Ningún endpoint devuelve claves API.
