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

Los vectores de modelos distintos **nunca se mezclan**: cada fila de `chunk_embeddings` lleva su `model_id` y cada
consulta filtra por el modelo activo.

1. Edite `EMBEDDING_MODEL` (y, si el modelo lo requiere, `EMBEDDING_QUERY_PREFIX` / `EMBEDDING_PASSAGE_PREFIX`) en `.env`.
   Compruebe la licencia de los pesos del nuevo modelo (docs/LICENSES.md).
2. `docker compose up -d api worker` (recrea los contenedores con la nueva configuración).
3. `docker compose exec api python -m app.cli reindex` — encola la generación de embeddings para todos los documentos.
   Mientras tanto los documentos figuran como `processing` y **no** se consultan con vectores incompatibles.
4. Cuando todos estén `ready`: `python -m app.cli index-create` si usa HNSW y `python -m app.cli purge-model <id_antiguo>`
   para liberar espacio (el modelo activo no se puede purgar).

El sistema se niega a arrancar la indexación si un modelo ya registrado cambia de dimensión o de prefijos.

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
Versiones: `0001` esquema inicial · `0002` ajustes de LLM (`llm_profiles`, `app_settings`).

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
| Respuestas siempre "sin evidencia suficiente" | Umbral `RETRIEVAL_MIN_SIMILARITY` demasiado alto para el modelo de embeddings; calibre con `eval/`. |
| Windows: "Control de aplicaciones bloqueó este archivo" al ejecutar Python/numpy en el host | Política de seguridad del equipo. Ejecute el backend y las pruebas en Docker (Linux). |
| Cambié una variable del `.env` y no tiene efecto | Hay que recrear los contenedores (`docker compose up -d api worker`). Solo llegan al contenedor las variables que el `docker-compose.yml` enumera (ver §10); el frontend necesita `--build` si cambia `PUBLIC_API_URL`. |
| La pantalla muestra «Sin conexión con la API» | La API no responde: `docker compose ps` / `docker compose logs api`. Se reintenta sola y avisa al recuperarse. Un error CORS o `ERR_FAILED` en la consola justo al reiniciar la API es esperable (mientras arranca no hay respuesta que lleve cabeceras CORS). |
| Ajustes: «Clave ilegible» en un proveedor | Cambió la clave maestra (`APP_SECRET_KEY` / `secret.key`) o se restauró una copia en otro entorno. Edite el proveedor y vuelva a introducir la clave (§7). |
| Preguntar devuelve «El proveedor … es externo … no permitido» | El perfil activo es un servicio externo sin el permiso de envío. Márquelo en Ajustes (tras leer qué se envía) o vuelva a un proveedor local. |
| «Probar» en Ajustes: «Credenciales rechazadas» / «Ruta no encontrada» | Clave incorrecta (401/403) o URL base equivocada (404; debe terminar antes de `/chat/completions`). |
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
| `RETRIEVAL_TOP_K` / `RETRIEVAL_CANDIDATES` | `6` / `30` | api | Fragmentos que llegan al LLM / candidatos por rama. |
| `RETRIEVAL_MIN_SIMILARITY` | `0.80` | api | Umbral coseno de relevancia. **No separa por sí solo** preguntas con y sin respuesta (README §Estado); la abstención fiable la aporta el LLM. |
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
