# Operación de PDF Analyst

Todos los comandos se ejecutan desde la raíz del proyecto.

## 1. Estado y diagnóstico

```bash
curl http://localhost:8000/health                      # base de datos, pgvector, embeddings, generación
docker compose logs -f worker                          # progreso del procesamiento
docker compose exec api python -m app.cli models       # modelos de embeddings y nº de vectores de cada uno
curl http://localhost:8000/settings/llm                # proveedor LLM activo y perfiles (sin claves)
docker compose ps                                      # estado de los servicios
```

La interfaz muestra además el estado de cada componente en la cabecera y avisa con notificaciones flotantes cuando termina el
procesamiento de un documento, cuando la API deja de responder (y cuando se recupera) o cuando el modelo de generación cae o vuelve.

Un documento pasa por `uploaded → processing → ready` (o `failed` con el motivo en `error`). Los fallos transitorios se
reintentan automáticamente (máx. 3 intentos); los permanentes (PDF cifrado, sin texto) no. Desde la interfaz, el botón
de reintento (o volver a subir el mismo archivo) reencola el procesamiento sin duplicar datos.

## 2. OCR

Se aplica página por página cuando la extracción de texto devuelve menos de `OCR_MIN_CHARS` (30) caracteres.
Requiere Tesseract (incluido en la imagen Docker con idiomas `spa` y `eng`). Para más idiomas añada el paquete
`tesseract-ocr-<lang>` en `backend/Dockerfile` y ajuste `OCR_LANGS` (p. ej. `spa+eng+fra`).
Desactívelo con `OCR_ENABLED=false`; un PDF sin capa de texto fallará entonces con un mensaje explícito.

## 3. Índices vectoriales (pgvector)

Por defecto la búsqueda es **exacta** (sin índice): 100 % de recall y suficiente hasta decenas de miles de fragmentos.
Cuando la latencia de consulta deje de ser aceptable (referencia: > ~50 000 fragmentos), cree el índice HNSW del
modelo activo:

```bash
docker compose exec api python -m app.cli index-create --m 16 --ef-construction 64
docker compose exec api python -m app.cli index-status
docker compose exec api python -m app.cli index-drop     # volver a búsqueda exacta
```

- Es un índice **parcial por modelo** sobre `embedding::vector(N)` (`WHERE model_id = M`), creado con
  `CREATE INDEX CONCURRENTLY` (no bloquea escrituras). Las consultas ya usan esa misma expresión: no hay que cambiar nada.
- Cada consulta activa `hnsw.iterative_scan = relaxed_order` (pgvector ≥ 0.8) para que el filtro por documento/colección
  no devuelva menos resultados de los esperados.
- IVFFlat se descartó: necesita datos previos para entrenar las listas y su recall se degrada al insertar después.
- Valide siempre el recall del índice contra la búsqueda exacta con el conjunto de `eval/` antes de adoptarlo.

## 4. Cambiar el modelo de embeddings (reindexación)

Los vectores de modelos distintos **nunca se mezclan**: cada fila de `chunk_embeddings` lleva su `model_id` y cada consulta filtra por el modelo activo.
Se indexa **solo con el modelo seleccionado**; los vectores de los demás se conservan hasta que se borren.

**Desde la interfaz (recomendado):** Ajustes → «Modelo de embeddings» (README §«Modelo de embeddings»). Añadir el proveedor, *Probar conexión*, *Usar por defecto*
y confirmar el diálogo con la vista previa (documentos y texto que se enviaría). Qué ocurre al activar:

1. Se comprueba el acceso **antes de cambiar nada**: consentimiento (si es externo), conexión, credenciales, dimensión y compatibilidad de prefijos con los vectores ya
   indexados de ese modelo. Si algo falla, el modelo activo no cambia.
2. Los documentos que ya tienen todos los vectores de ese modelo quedan `ready` al instante; el resto pasa a `processing` y se encola un job `embed` (el worker usa el modelo activo
   *en el momento de ejecutar*; si cambia a mitad de trabajo, repite con el nuevo). Mientras tanto **no son consultables**.
3. Volver a un modelo ya indexado es inmediato. Para liberar espacio: «Vectores indexados» → *Borrar vectores* (solo de modelos no activos; volver a ellos exigirá reindexar).

**Por línea de comandos / sin interfaz** (proveedor del entorno): edite `EMBEDDING_PROVIDER` (`sentence-transformers` | `ollama`), `EMBEDDING_MODEL` y, si el modelo lo requiere,
`EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` en `.env` → `docker compose up -d api worker` → `docker compose exec api python -m app.cli reindex` (encola solo lo que falte) →
cuando todo esté `ready`, `python -m app.cli index-create` si usa HNSW y `python -m app.cli purge-model <id_antiguo>` (el modelo activo no se puede purgar).
Un Ollama remoto en el entorno exige `EMBEDDING_ALLOW_EXTERNAL=true`.

**Privacidad:** un proveedor externo recibe el texto íntegro de los fragmentos de todos los documentos al indexar y cada pregunta al buscar. Las claves se guardan cifradas
(misma clave maestra que las del LLM, §7: `db.dump` no las revela y `secret.key` no se copia en las copias de seguridad). **Prefijos:** un modelo usa siempre los mismos al indexar y
buscar; el sistema rechaza (409) una activación con prefijos o dimensión distintos de los registrados y no permite editar el modelo/prefijos/dimensiones de un proveedor en uso.

**EmbeddingGemma 2 en local** (servicio `embedder` incluido en el compose): §13. **Sin servicios de pago para probar un proveedor externo:** §12.

## 5. Copias de seguridad y recuperación

```bash
./scripts/backup.sh                       # PowerShell: .\scripts\backup.ps1
./scripts/restore.sh backups/<fecha>      # PowerShell: .\scripts\restore.ps1 -Src backups\<fecha>
```

La copia contiene `db.dump` (`pg_dump -Fc`: documentos, fragmentos, **embeddings**, conversaciones, citas) y
`storage.tgz` (PDF originales). La restauración es destructiva y pide confirmación; se recomienda probarla periódicamente
en otro entorno. Los modelos descargados (volumen `appdata`, carpeta `models`) no se incluyen: se vuelven a descargar.

`db.dump` incluye los perfiles de proveedor LLM con sus claves **cifradas**; la clave maestra (`/data/secret.key` o `APP_SECRET_KEY`)
**no** se copia a propósito (§7): tras restaurar en otro entorno, las claves guardadas aparecerán como «Clave ilegible» en Ajustes y
bastará con volver a introducirlas. Si desea poder restaurarlas tal cual, guarde `secret.key` aparte y por un canal seguro.

## 6. Actualizar el esquema

Las migraciones están en `backend/alembic/versions` y se aplican al arrancar mediante el servicio `migrate`
(`alembic upgrade head`). Haga una copia de seguridad antes de actualizar versiones de la aplicación.

**Importante:** `migrate`, `api`, `worker` y `frontend` son imágenes distintas; tras actualizar el código use
`docker compose up -d --build` para reconstruirlas todas (si solo reconstruye `api`, la migración nueva no se aplica).
Versiones: `0001` esquema inicial · `0002` ajustes de LLM (`llm_profiles`, `app_settings`) · `0003` ajustes de embeddings (`embedding_profiles`).

## 7. Proveedor de generación (Ajustes)

Guía de uso y plantillas: **README §"Proveedor de generación: Ajustes en la interfaz"**. Resumen operativo:

- Dos fuentes: el perfil **«Entorno (.env)»** (variables `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`,
  `LLM_ALLOW_EXTERNAL`, `OLLAMA_BASE_URL`; el compose las pasa a los contenedores) y los **perfiles de Ajustes** (tabla
  `llm_profiles`). El activo se guarda en `app_settings["active_llm"]`; por defecto, «env» (Ollama). El cambio no requiere reinicio.
- **Claves:** cifradas con Fernet en `llm_profiles.api_key_enc`. Clave maestra: `APP_SECRET_KEY` (base64 urlsafe de 32 bytes;
  genere una con `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`) o, si no existe,
  el archivo `/data/secret.key` (0600), creado en el primer uso en el volumen `appdata`.
  - **Copias de seguridad:** `backup.sh` guarda `db.dump` (con las claves **cifradas**) y `storage.tgz` (PDF), pero NO `secret.key`
    a propósito: así una copia filtrada no revela claves. Tras restaurar en otro entorno, vuelva a introducir las claves en Ajustes
    (o copie `secret.key` por un canal seguro, aparte de la copia).
  - **Rotar la clave maestra:** cambie `APP_SECRET_KEY` (o borre `secret.key`) y vuelva a introducir cada clave en Ajustes; las
    antiguas quedan ilegibles y se marcan como tales.
- Un destino que no sea local (`localhost`, `host.docker.internal`, IP privada, servicio `ollama`) es **externo** y exige el permiso
  explícito del perfil (casilla en Ajustes) o `LLM_ALLOW_EXTERNAL=true` para el perfil del entorno. Aplica también a un Ollama remoto.
- Errores del proveedor: 401/403 → revisar clave; 404 → modelo o ruta incorrectos; 429 → límite de peticiones del servicio. Se
  muestran como mensaje en la conversación, nunca como caída de la API. `/health` informa del proveedor activo.
- Ollama: si el modelo no está descargado, «Probar» lo indica (`ollama pull <modelo>` en el host). Los modelos con "razonamiento"
  (qwen3, deepseek-r1) se invocan con el razonamiento desactivado (`LLM_DISABLE_THINKING=true`). Los 500/502/503 transitorios al
  cargar el modelo se reintentan 3 veces.
- **Privacidad de servicios externos:** reciben la pregunta y los fragmentos recuperados; cada proveedor tiene sus propias políticas
  de retención y de uso de datos. Cambiar de proveedor **no** requiere reindexar: los embeddings son independientes.
- Endpoints (documentados en `/docs`): `GET /settings/llm`, `POST /settings/llm/profiles`, `PATCH|DELETE /settings/llm/profiles/{id}`,
  `PUT /settings/llm/active`, `POST /settings/llm/test`. Ningún endpoint devuelve claves.
- Tras actualizar a esta versión: `docker compose up -d --build` (aplica la migración 0002 y reconstruye todas las imágenes).

## 8. Multiusuario (no implementado)

La versión actual es de un solo usuario en una máquina de confianza. Para varios usuarios: (1) autenticación en un
proxy inverso o en la API; (2) columna `owner_id` en `documents`, `collections`, `conversations` y `llm_profiles` (las claves API serían por usuario); (3) filtro obligatorio
por propietario en `resolve_document_ids` y en todos los endpoints; (4) pruebas de aislamiento entre usuarios.
Hasta entonces, no exponga los puertos fuera de `127.0.0.1`.

## 9. Solución de problemas

| Síntoma | Causa probable / solución |
|---------|---------------------------|
| `ports are not available: 127.0.0.1:5432` en Windows | Puerto reservado/ocupado. Use `POSTGRES_HOST_PORT` (por defecto 5433). |
| `/health` → generación `ok: false` | Ollama sin el modelo: `ollama pull qwen3:8b` (en el host). |
| Documento `failed: No se extrajo texto…` | PDF escaneado con OCR desactivado o ilegible. Active OCR y use "Reintentar". |
| Respuestas siempre "sin evidencia suficiente" | Umbral de similitud demasiado alto para el modelo de embeddings (cada modelo reparte sus similitudes de forma distinta): calibre con `eval/` (§11) y fije el umbral **del proveedor** en Ajustes, o `RETRIEVAL_MIN_SIMILARITY` para el del entorno. |
| Windows: "Control de aplicaciones bloqueó este archivo" al ejecutar Python/numpy en el host | Política de seguridad del equipo. Ejecute el backend y las pruebas en Docker (Linux). |
| Cambié una variable del `.env` y no tiene efecto | Hay que recrear los contenedores (`docker compose up -d api worker`). Solo llegan al contenedor las variables que el `docker-compose.yml` enumera (ver §10); el frontend necesita `--build` si cambia `PUBLIC_API_URL`. |
| La pantalla muestra «Sin conexión con la API» | La API no responde: `docker compose ps` / `docker compose logs api`. Se reintenta sola y avisa al recuperarse. Un error CORS o `ERR_FAILED` en la consola justo al reiniciar la API es esperable (mientras arranca no hay respuesta que lleve cabeceras CORS). |
| Ajustes: «Clave ilegible» en un proveedor | Cambió la clave maestra (`APP_SECRET_KEY` / `secret.key`) o se restauró una copia en otro entorno. Edite el proveedor y vuelva a introducir la clave (§7). |
| Preguntar devuelve «El proveedor … es externo … no permitido» | El perfil activo es un servicio externo sin el permiso de envío. Márquelo en Ajustes (tras leer qué se envía) o vuelva a un proveedor local. |
| «Probar» en Ajustes: «Credenciales rechazadas» / «Ruta no encontrada» | Clave incorrecta (401/403) o URL base equivocada (404; debe terminar antes de `/chat/completions`). |
| Tras cambiar de modelo de embeddings los documentos siguen «procesando» | Es lo esperado mientras el worker reindexa (`docker compose logs -f worker`); si el proveedor es externo, depende de su velocidad y límites. Si un documento queda `failed`: «Reintentar». Comprobar la clave y la conexión con *Probar* en Ajustes. |
| Activar un proveedor de embeddings devuelve 409 «prefijos…» o «dimensión…» | Ese modelo ya tiene vectores con otros prefijos/dimensión. Restaure los originales, o borre los vectores de ese modelo en «Vectores indexados», o use otro nombre de modelo. |
| Activar devuelve 409 «es externo…» | Marque el permiso de envío del texto en el proveedor (tras leer qué se envía) o use un servicio local. |
| Preguntar o buscar devuelve 503 con «No se pudo conectar…» o «credenciales» | El proveedor de embeddings activo no responde o rechaza la clave. Los documentos no se pierden: arregle el proveedor o vuelva al modelo local (inmediato). La gestión de documentos sigue funcionando. |
| `docker compose up -d` no aplica la migración nueva | La imagen `migrate` es independiente: use `docker compose up -d --build`. |
| Una conversación eliminada reaparece | Se pulsó «Deshacer», o la página se cerró antes de que el borrado diferido llegara a la API (fallo seguro). Vuelva a eliminarla. |
| El PDF se ve cortado en el visor en móvil | Reconstruya el frontend (`docker compose build frontend`): las versiones anteriores no ajustaban el PDF al ancho. |

## 10. Referencia de configuración

Todas se leen de variables de entorno (`.env` en la raíz; el `docker-compose.yml` las pasa a los contenedores con estos mismos valores
por defecto). Tras cambiarlas: `docker compose up -d api worker` (y `frontend --build` para `PUBLIC_API_URL`). Las que afectan al
chunking, OCR o embeddings **solo se aplican a documentos procesados después**; para reaplicarlas a los existentes use «Reintentar»/
reprocesar (el cambio de modelo de embeddings: §4).

| Variable | Defecto | Servicio | Descripción |
|----------|---------|----------|-------------|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `pdfanalyst` / (obligatoria) / `pdfanalyst` | db, todos | Credenciales de PostgreSQL. La contraseña no tiene valor por defecto. |
| `POSTGRES_HOST_PORT` | `5433` | db | Puerto en `127.0.0.1` del host. |
| `MAX_UPLOAD_MB` | `100` | api | Tamaño máximo de PDF. |
| `OCR_ENABLED` / `OCR_LANGS` | `true` / `spa+eng` | worker | OCR con Tesseract. |
| `OCR_MIN_CHARS` / `OCR_DPI` | `30` / `200` | worker | Una página con menos caracteres se considera escaneada; resolución del OCR. |
| `CHUNK_TARGET_CHARS` / `CHUNK_OVERLAP_CHARS` | `1200` / `200` | worker | Tamaño objetivo y solapamiento de fragmentos (nunca cruzan páginas). |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `sentence-transformers` / `intfloat/multilingual-e5-small` | api, worker | Modelo de embeddings (cambiarlo exige reindexar, §4). |
| `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX` | `"query: "` / `"passage: "` | api, worker | Prefijos que exige el modelo (entre comillas, con el espacio). |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_DEVICE` | `32` / `cpu` | worker | Lote y dispositivo (`cuda` solo con una imagen con CUDA). |
| `EMBEDDING_ALLOW_EXTERNAL` | `false` | api, worker | Permiso para un Ollama **remoto** del perfil «Entorno» (envía el texto de los documentos). Los perfiles de Ajustes usan su propia casilla. |
| `EMBEDDING_TIMEOUT_S` | `120` | api, worker | Tiempo máximo por petición a un proveedor de embeddings por HTTP (la primera petición al servicio local carga el modelo: ≈10 s en este equipo). |
| `EMBED_SERVER_MODEL` | `google/embeddinggemma-2` | embedder | Modelo de sentence-transformers que sirve el servicio local (§13). |
| `EMBED_SERVER_TEXT_ONLY` | `true` | embedder | `true` = solo el codificador de texto (≈270M parámetros); `false` = carga también visión/audio (≈740M, mucha más RAM). |
| `EMBED_SERVER_DTYPE` | `float32` | embedder | `float32` o `bfloat16` (menos memoria). **`float16` se rechaza**: EmbeddingGemma 2 no lo admite. |
| `EMBED_SERVER_DEVICE` / `EMBED_SERVER_THREADS` | `cpu` / `0` | embedder | Dispositivo e hilos de PyTorch (`0` = por defecto). Limite los hilos si el equipo se satura al indexar. |
| `EMBED_SERVER_MAX_SEQ_LENGTH` / `EMBED_SERVER_MAX_BATCH` | `2048` / `64` | embedder | Tope de tokens por texto (acota la memoria; el modelo admite 8192) y de textos por petición. |
| `EMBED_SERVER_PRELOAD` / `EMBED_SERVER_API_KEY` | `false` / vacío | embedder | Cargar el modelo al arrancar (si no, en la primera petición) / exigir `Authorization: Bearer` (opcional). |
| `EMBED_SERVER_MEM_LIMIT` | `4608m` | embedder | Tope de memoria del contenedor. Medido (fp32): solo texto ≈2,0 GB · texto+imagen ≈3,0 GB · texto+audio ≈3,7 GB. |
| `EMBED_SERVER_MEDIA_IDLE_S` | `300` | embedder | Segundos sin usar medios tras los cuales el servicio vuelve a solo texto y libera memoria (`0` = nunca). |
| `EMBED_SERVER_IMAGE_TOKENS` / `EMBED_SERVER_VIDEO_TOKENS` | `280` / `140` | embedder | Presupuesto visual (70–1120) por imagen / por fotograma. Medido: 280 ≈ 3,3 s, 140 ≈ 1,5 s, 70 ≈ 0,7 s por imagen en esta CPU. |
| `EMBED_SERVER_MEDIA_MAX_AUDIO_S` | `300` | embedder | Ventana máxima de audio por vector (el modelo admite ≈327 s). |
| `MEDIA_MAX_IMAGE_MB` / `MEDIA_MAX_PIXELS` | `20` / `40000000` | api, worker | Tamaño máximo de una imagen y megapíxeles máximos (protege de bombas de descompresión). |
| `MEDIA_THUMB_PX` | `320` | api, worker | Lado mayor de las miniaturas (JPEG sin EXIF). |
| `VISION_MAX_SIDE` / `VISION_MAX_TOKENS` / `VISION_RETRIES` | `1568` / `500` / `3` | api, worker | Las imágenes se reducen a este lado mayor antes de enviarlas al modelo de visión; longitud máxima de la descripción; reintentos ante 429. |
| `PDF_FIGURES_ENABLED` | `true` | api, worker | Detectar las figuras (imágenes y gráficos) de los PDF. `false` = comportamiento anterior (solo texto). |
| `PDF_FIGURES_MAX` | `60` | api, worker | Tope de figuras por documento (las más grandes primero; se avisa de las omitidas). Cada figura cuesta OCR, vector y, con modelo de visión, una descripción. |
| `PDF_FIGURE_MIN_PX` / `PDF_FIGURE_MIN_AREA` | `100` / `0.02` | api, worker | Una imagen incrustada se ignora si su lado menor es menor que esto o si ocupa menos de esta fracción de la página. |
| `PDF_FIGURE_REPEAT_PAGES` | `3` | api, worker | Una imagen presente en ≥ N páginas (fondo, logo, marca de agua) se ignora. |
| `PDF_VECTOR_MIN_PATHS` / `PDF_VECTOR_MIN_AREA` | `40` / `0.15` | api, worker | Un gráfico vectorial es una región con ≥ N trazos que cubre ≥ esta fracción de la página. |
| `VISION_ANSWER_MAX_IMAGES` | `3` | api | Figuras que se adjuntan al modelo de visión al responder mirándolas. |
| `RETRIEVAL_MIN_SIMILARITY_MEDIA` | `0.65` | api | Umbral coseno para imágenes/audio/vídeo (otra escala que texto→texto). Provisional: medido con imágenes sintéticas (correctas 0,74–0,84; ajenas ≤ 0,62; con una imagen real 0,73–0,74). Calibrar con imágenes reales. |
| `RETRIEVAL_TOP_K` / `RETRIEVAL_CANDIDATES` | `6` / `30` | api | Fragmentos que llegan al LLM / candidatos por rama. |
| `RETRIEVAL_MIN_SIMILARITY` | `0.80` | api | Umbral coseno de relevancia por defecto (calibrado para e5, que comprime las similitudes). Un proveedor de Ajustes puede llevar el suyo (EmbeddingGemma 2: 0,72). **No separa por sí solo** preguntas con y sin respuesta con e5 (README §Estado); la abstención fiable la aporta el LLM. |
| `RETRIEVAL_FTS_MIN_COVERAGE` | `0.5` | api | Fracción de términos distintivos de la pregunta que debe contener un fragmento. |
| `RETRIEVAL_COMMON_TERM_RATIO` | `0.10` | api | Un término presente en más de este % de fragmentos (mín. 2) es «común» y no guía la búsqueda textual. |
| `HYBRID_ENABLED` / `RRF_K` | `true` / `60` | api | Recuperación híbrida y constante de la fusión RRF. |
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `qwen3:8b` | api | Proveedor del perfil «Entorno». Otros proveedores: Ajustes (§7). |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | api | Dirección de Ollama. |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_ALLOW_EXTERNAL` | vacío / vacío / `false` | api | Perfil «Entorno» con proveedor `openai-compatible`. |
| `LLM_DISABLE_THINKING` | `true` | api | Desactiva el razonamiento de modelos qwen3/deepseek-r1. |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT_S` | `0.1` / `700` / `180` | api | Parámetros de generación. |
| `CONTEXT_MAX_TOKENS` | `3500` | api | Presupuesto de contexto (se añaden fragmentos completos hasta agotarlo). |
| `APP_SECRET_KEY` | vacío (usa `/data/secret.key`) | api | Clave maestra Fernet para las claves guardadas desde Ajustes (§7). |
| `JOB_MAX_ATTEMPTS` | `3` | worker | Reintentos de un job con error transitorio. |
| `CORS_ORIGINS` | `http://localhost:3000` | api | Orígenes permitidos (separados por comas). |
| `PUBLIC_API_URL` | `http://localhost:8000` | frontend (build) | URL de la API que usa el navegador. |
| `OLLAMA_MODELS_SOURCE` / `OLLAMA_MODELS_MODE` | vacío | ollama (perfil) | Reutilizar modelos de un Ollama del host. |

## 11. Evaluar la calidad con sus documentos

Procedimiento completo y formato de preguntas: `eval/README.md`; resultados con un documento real: `eval/RESULTS.md`.

```bash
# Recuperación (recall@k, MRR, abstención; sugiere un umbral) — sobre los documentos ya cargados
docker compose cp eval/questions.X.jsonl api:/tmp/q.jsonl
docker compose exec api python -m app.evaluate /tmp/q.jsonl
# Extremo a extremo con el LLM activo (respuesta, citas y abstención) — requiere la API en marcha
python eval/e2e.py eval/questions.X.jsonl eval/_resultado.json
```

Reglas de lectura: las páginas esperadas son «al menos una de estas» (la misma historia puede aparecer en varias); incluya ≥ 20 % de
preguntas sin respuesta, parecidas en estilo a las reales; repita tras cambiar el proveedor/modelo LLM, el modelo de embeddings o los
parámetros de §10, y verifique a mano los fallos antes de concluir (varios «fallos» iniciales eran citas correctas no previstas).

## 12. Probar embeddings por API sin servicios de pago

`scripts/dev_embedding_server.py` es un servidor de **desarrollo** con la API `/v1/embeddings` de OpenAI que sirve el modelo e5 local y registra cuántos textos y caracteres recibe
(sirve para comprobar *qué* saldría del equipo hacia un servicio externo real). Un contenedor desechable en la red del compose:

```bash
docker run --rm -d --name emb-stub --network pdfanalyst_default -e HF_HOME=/data/models -e STUB_API_KEY=clave-de-prueba-123456 \
  -v pdfanalyst_appdata:/data -v "$PWD/scripts":/stub:ro pdfanalyst-api python /stub/dev_embedding_server.py
# Ajustes → Modelo de embeddings → «Otro servicio compatible»: URL base http://emb-stub:9999/v1, clave de arriba, modelo cualquiera (p. ej. e5-stub),
#   prefijos «query: » / «passage: ». Un nombre de host se trata como servicio EXTERNO (pide el permiso de envío), útil para probar el consentimiento.
docker logs -f emb-stub      # qué recibe: nº de textos/caracteres por petición
docker rm -f emb-stub        # al terminar; luego, en Ajustes: volver al modelo local, borrar sus vectores y eliminar el proveedor de prueba
```

Esta prueba reindexa **tus documentos reales** (en tu propia máquina, sin salir de ella): los deja `processing` unos segundos. No es un servicio de producción.

## 13. EmbeddingGemma 2 en local (servicio `embedder`)

El `docker-compose.yml` incluye el servicio **`embedder`** (`backend/app/embedding_server.py`): un único proceso con el modelo `google/embeddinggemma-2` (Apache-2.0) cargado en
**modo solo texto** (≈271M parámetros en lugar de ≈740M: se omiten los codificadores de visión y audio) y expuesto con la API `/v1/embeddings` de OpenAI **solo dentro de la red de Docker**
(sin puerto en el host). La API y el worker lo usan como cualquier otro proveedor por HTTP; así hay una sola copia del modelo en memoria. Ningún texto sale del equipo.

**Puesta en marcha (una vez):**

```bash
docker compose run --rm embedder python -m app.embedding_server --download   # ≈1,5 GB en el volumen appdata (caché de Hugging Face)
docker compose up -d --build                                                 # levanta `embedder`, con torchvision incluido en la imagen
```

Después, en Ajustes → Modelo de embeddings: **Añadir proveedor** → plantilla *EmbeddingGemma 2 (local…)* → *Probar conexión* (la primera vez carga el modelo) → *Usar por defecto*.
La plantilla ya trae la URL `http://embedder:8000/v1`, los prefijos oficiales (`task: search result | query: ` y `title: none | text: `) y el umbral de similitud 0,72.

**Medidas en este equipo** (i7-13620H, sin GPU para embeddings, Docker con 8 GB): carga del modelo ≈9 s; 768 dimensiones; **≈1,7–1,9 GiB de RAM** en el contenedor con el modelo cargado
(la API y el worker, sin el modelo e5 cargado, ocupan ≈100 MB cada uno); indexar 274 fragmentos ≈83 s (≈3 fragmentos/s). Con la memoria justa: `EMBED_SERVER_DTYPE=bfloat16`
reduce los pesos a la mitad (no medido aquí: puede cambiar la velocidad en CPU), y `EMBED_SERVER_MAX_SEQ_LENGTH` acota el pico al indexar. Para liberar la memoria, vuelva al modelo «Entorno»
(`docker compose stop embedder`).

**Evaluación frente a e5** (`eval/questions.babilonia.jsonl`, 22 preguntas; ver `eval/RESULTS.md`): híbrido recall@3 1,00 (e5: 0,93), recall@5 1,00 (1,00), MRR 0,78 (0,72); extremo a extremo 15/15
respuestas con cita correcta y 7/7 abstenciones. Muestra pequeña de un solo documento.

**Solución de problemas del servicio:**

| Síntoma | Causa probable / solución |
|---------|---------------------------|
| *Probar conexión* falla con «No se pudo conectar» | El servicio no está levantado o aún arranca: `docker compose ps embedder` / `docker compose logs embedder`. |
| La primera petición tarda o da *timeout* | Carga del modelo (≈10 s aquí; más con disco lento). Suba `EMBEDDING_TIMEOUT_S` o use `EMBED_SERVER_PRELOAD=true`. |
| `503 El modelo no está disponible` | El modelo no se pudo cargar: `docker compose logs embedder` (falta de memoria → suba `EMBED_SERVER_MEM_LIMIT`, use `bfloat16`; modelo sin descargar y sin red → ejecute `--download`). |
| El contenedor se reinicia (OOM) | Se alcanzó `EMBED_SERVER_MEM_LIMIT` o la memoria de la VM de Docker. Libere memoria, use `bfloat16` o suba el tope. |
| `ImportError … torchvision` al cargar | Imagen antigua: el procesador del modelo importa torchvision aunque se use solo texto. `docker compose build embedder` (el `Dockerfile` ya lo instala). |
| `EMBED_SERVER_DTYPE debe ser float32 o bfloat16` | EmbeddingGemma 2 no admite `float16`; se rechaza a propósito. |
| Quiero usarlo desde Ollama | **No funciona en Windows/Linux con CUDA**: las compilaciones de `embeddinggemma-2` en Ollama exigen MLX (solo Apple Silicon). Por eso se sirve con sentence-transformers. |

## 14. Figuras de PDF y modelo de visión

**Figuras dentro de un PDF (lo principal):** al procesar un PDF se detectan sus imágenes y gráficos, se leen (OCR), se describen (modelo de visión) y se vectorizan; al responder, el modelo de visión puede recibirlas (Ajustes → Modelo de visión → «Responder mirando las figuras»). Para que un PDF ya subido lea sus figuras: botón «Volver a procesar». La lista de documentos muestra «N figuras (M descritas)»; los avisos del documento indican si hubo figuras omitidas por el tope, límites de tasa del proveedor o descripciones que fallaron. Ajuste de la detección: `PDF_FIGURE_*`, `PDF_VECTOR_*`, `PDF_FIGURES_MAX` (§10). Un PDF cuyas páginas son todas imágenes y sin OCR ya no falla: se indexa por sus figuras con un aviso (sin modelo de visión no se podrá responder con su contenido).

| Síntoma | Causa probable / solución |
|---------|---------------------------|
| Un PDF con imágenes no «ve» las figuras | Se subió antes de esta función: «Volver a procesar». Si `figure_count` es 0, la imagen es pequeña, repetida en ≥ 3 páginas o el PDF no tiene imágenes (baje `PDF_FIGURE_MIN_AREA`/`PDF_FIGURE_REPEAT_PAGES` si hace falta). |
| Figuras detectadas pero sin descripción | No hay modelo de visión activo, o el proveedor falló/limitó (ver avisos del documento); reprocese más tarde. |
| Procesar un PDF con muchas figuras tarda | Cada figura hace OCR, una llamada al modelo de visión y un vector (≈3 s en CPU). Con Gemini gratuito (≈10 peticiones/min) 60 figuras tardan ≈6 min. Baje `PDF_FIGURES_MAX`. |
| «Responder mirando» envía datos fuera | Con un modelo de visión externo, la pregunta, los fragmentos y hasta 3 figuras van a ese servicio. Desactive el interruptor para usar solo descripciones y el LLM de texto. |

**Imágenes sueltas:** además se pueden subir imágenes sueltas (el sistema está pensado para PDF). **Subir imágenes:** PNG, JPEG y WebP (la firma del archivo decide, no la extensión; SVG y GIF se rechazan). Límites: `MEDIA_MAX_IMAGE_MB` y `MEDIA_MAX_PIXELS`. Cada imagen genera un fragmento de medio (vector), uno de OCR
(si Tesseract lee texto) y, si hay modelo de visión, uno con la descripción automática.

**Encontrar imágenes por texto (EmbeddingGemma 2):** el perfil de embeddings activo debe ser el servicio local `embedder` con *«admite imágenes, audio y vídeo»* marcado (la plantilla «EmbeddingGemma 2 (local)» lo trae; en un perfil
existente, edítelo y márquelo: no invalida vectores y rige al instante). Con otro modelo (p. ej. e5) las imágenes se aceptan pero solo se buscan por su OCR/descripción. La primera imagen carga el codificador de visión (≈3 GB de
RAM, 20–30 s); a los `EMBED_SERVER_MEDIA_IDLE_S` segundos sin uso el servicio vuelve a solo texto. Con poca RAM libre, cierre otras aplicaciones al indexar imágenes (o suba pocas a la vez).

**Decir qué se ve (modelo de visión):** Ajustes → Modelo de visión. Plantillas: Google Gemini (`https://generativelanguage.googleapis.com/v1beta/openai`, p. ej. `gemini-2.5-flash`), Groq (elija un modelo con visión en
«Cargar modelos»; Groq no indica si está en el plan gratuito) y Ollama local (`ollama pull qwen3-vl:4b`, ≈3,3 GB, o `qwen3-vl:2b`, ≈1,9 GB). Pasos: *Añadir modelo de visión* → clave API → *Cargar modelos* → elegir uno con visión →
marcar *«Permitir el envío de las imágenes»* (servicios externos) → *Probar con una imagen de prueba* → *Usar para describir*. Ya subidas antes de activarlo: «Reintentar»/reprocesar la imagen para generar su descripción.

| Síntoma | Causa probable / solución |
|---------|---------------------------|
| «Probar con una imagen de prueba» dice 400/404 o devuelve un error del modelo | El modelo elegido no admite imágenes: use «Cargar modelos» y elija otro. |
| «El proveedor limitó las peticiones (429)» | Límite de la capa gratuita (por minuto o por día). La ingesta reintenta con espera (`VISION_RETRIES`); si se agota, la imagen queda indexada sin descripción y con un aviso: reprocésela más tarde. |
| La imagen se indexa sin descripción y con el aviso «No se pudo describir la imagen» | Fallo del proveedor o consentimiento retirado; la imagen sigue siendo buscable por su vector y su OCR. |
| Una imagen no aparece en las búsquedas | El perfil de embeddings activo no admite medios (casilla) o aún se está indexando; compruebe `docker compose logs embedder`. |
| El servicio `embedder` se reinicia al indexar imágenes (OOM) | Falta memoria: suba `EMBED_SERVER_MEM_LIMIT`, cierre aplicaciones o baje `EMBED_SERVER_IMAGE_TOKENS`. |
| Gemini/Groq envían las imágenes fuera | Es lo esperado con un servicio externo: solo con el permiso marcado; en la capa gratuita de Gemini el contenido puede usarse para mejorar sus productos. |

