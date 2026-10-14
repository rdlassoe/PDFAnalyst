# PDF Analyst

Plataforma de código abierto para consultar documentos PDF en lenguaje natural, con respuestas respaldadas por
**citas verificadas** (documento + página + fragmento exacto). Funciona 100 % en local: PostgreSQL + pgvector,
embeddings con sentence-transformers y generación con Ollama. Los proveedores externos son opcionales.

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitectura, decisiones técnicas (D1–D17), modelo de datos, pipelines, seguridad, fases, API |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Operación: índices, cambio de modelo, copias de seguridad, proveedor LLM y claves, **referencia de configuración**, evaluación, solución de problemas |
| [docs/FRONTEND.md](docs/FRONTEND.md) | Interfaz: estructura, sistema de notificaciones, pantallas, diseño adaptable, lista de verificación |
| [docs/LICENSES.md](docs/LICENSES.md) | Licencias de dependencias y de los pesos de los modelos |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Historial de cambios y hallazgos |
| [docs/plans/PLAN_EMBEDDINGGEMMA2.md](docs/plans/PLAN_EMBEDDINGGEMMA2.md) | EmbeddingGemma 2 **local**: datos del modelo, decisiones, medidas en este equipo y resultado de la evaluación |
| [eval/README.md](eval/README.md) · [eval/RESULTS.md](eval/RESULTS.md) | Cómo evaluar con sus documentos y resultados con un libro real |

## Estructura del proyecto

```
PDFAnalyst/
├── backend/                 # API FastAPI + worker + pipeline RAG (Python 3.12)
│   ├── app/api/             #   routers: documents, collections, chat, settings, system
│   ├── app/services/        #   ingestion, pdf_extract, chunking, embeddings, retrieval, answering,
│   │                        #   citations, llm_settings, embedding_settings, credentials, net_validation, vector_index, storage
│   ├── app/providers/       #   LLMProvider: ollama, openai_compat (+ ProviderConfig)
│   ├── app/{models,config,db,worker,cli,evaluate}.py
│   ├── alembic/versions/    #   0001 esquema inicial · 0002 ajustes de LLM · 0003 ajustes de embeddings
│   └── tests/               #   240 pruebas (pytest)
├── frontend/                # Next.js 15 + React 19 + Tailwind (app/, components/, lib/)
├── eval/                    # preguntas de evaluación, e2e.py, RESULTS.md
├── scripts/                 # backup / restore (.sh y .ps1)
├── docs/                    # documentación
├── docker-compose.yml       # db, migrate, api, worker, frontend (+ ollama opcional, + test)
└── .env.example             # configuración documentada
```

## Requisitos

- Docker (Engine o Desktop) con Compose v2.
- ~8 GB de RAM libres; GPU NVIDIA opcional (recomendada para el modelo de generación).
- Un modelo de Ollama (por defecto `qwen3:8b`, Apache-2.0).

## Puesta en marcha

```bash
cp .env.example .env            # y edite POSTGRES_PASSWORD (use una contraseña larga y aleatoria)
docker compose up -d --build    # db, migraciones, api, worker, frontend
ollama pull qwen3:8b            # una sola vez (≈5 GB), con Ollama instalado en el host
```

Por defecto se usa el Ollama **del host** (`host.docker.internal:11434`; con GPU usa la del host sin configuración extra).
Alternativa autocontenida: `docker compose --profile ollama up -d` + `OLLAMA_BASE_URL=http://ollama:11434` en `.env`
(con GPU NVIDIA añada `-f docker-compose.gpu.yml`; `OLLAMA_MODELS_SOURCE` reutiliza modelos ya descargados).
En Linux, `host.docker.internal` no alcanza servicios ligados a loopback: use el perfil `ollama`.

| Servicio | URL (solo loopback) |
|----------|---------------------|
| Aplicación web | http://localhost:3000 |
| API + documentación OpenAPI | http://localhost:8000/docs |
| PostgreSQL | 127.0.0.1:5433 |

La primera vez, el modelo de embeddings (`intfloat/multilingual-e5-small`, ≈120 MB) se descarga en el volumen
`appdata` y no vuelve a descargarse.

## Uso

1. Suba uno o varios PDF (el procesamiento ocurre en segundo plano; el panel muestra el estado).
2. Pregunte. Cada afirmación termina con una etiqueta `[C1]`; al pulsarla se abre el PDF en la página exacta con el
   fragmento resaltado.
3. Si los documentos no contienen la respuesta, el sistema lo indica y **no** invoca al modelo.
4. Limite la consulta marcando documentos o eligiendo una colección.
5. **Ajustes** (arriba a la derecha): elija el proveedor de LLM (Ollama por defecto, Groq, Gemini, OpenAI, otro compatible),
   guarde su API y cámbielo cuando quiera (ver la sección siguiente).
6. **Avisos flotantes:** la aplicación notifica subidas, fin del procesamiento de cada documento (con acción «Abrir» o
   «Reintentar»), errores del modelo, caídas y recuperación de la API. Las notificaciones se pausan al pasar el ratón y se
   pueden cerrar. Las acciones destructivas piden confirmación en un diálogo propio; **eliminar una conversación** se puede
   **deshacer** durante 7 s (el borrado real ocurre cuando el aviso desaparece o se cierra la página).
7. En pantallas pequeñas hay pestañas «Documentos / Conversación» y el visor PDF se abre como panel deslizante (Esc lo cierra;
   las flechas ← → cambian de página).

## Proveedor de generación (LLM): Ajustes en la interfaz

La generación de respuestas la hace un LLM **intercambiable**; la búsqueda (embeddings + PostgreSQL) es siempre local. Por
defecto se usa **Ollama** en su equipo. Para cambiar de proveedor abra **Ajustes** (botón arriba a la derecha, o
http://localhost:3000/ajustes):

1. **Añadir proveedor** y elija una plantilla: Ollama (local), **Groq**, **Google Gemini API**, **OpenAI**, LM Studio/llama.cpp/vLLM
   (local), cualquier API **compatible con OpenAI** (indicando su URL base) u Ollama en otra dirección.
2. Pegue la **clave API** y pulse **Cargar modelos**: la aplicación consulta al proveedor, valida URL y clave y le ofrece los
   modelos disponibles (o escriba el nombre exacto).
3. **Guardar** o **Guardar y usar por defecto**. También puede **Probar** cualquier proveedor y pulsar **Usar por defecto** en el que
   quiera. El cambio se aplica a la **siguiente pregunta**, sin reiniciar ni reprocesar documentos. Para volver a local, elija
   «Entorno (.env)» (Ollama) o cualquier perfil local. (El modelo de **embeddings** se elige aparte, en la sección siguiente.)

El proveedor «Entorno (.env)» siempre existe (no editable) y refleja las variables `LLM_*`/`OLLAMA_BASE_URL` del `.env`: es el
predeterminado hasta que usted elija otro, y sirve para despliegues sin interfaz (ver `.env.example`).

| Proveedor | Tipo de API | URL base (plantilla) |
|-----------|-------------|----------------------|
| Ollama (local, por defecto) | Ollama | `http://host.docker.internal:11434` |
| Groq | Compatible con OpenAI | `https://api.groq.com/openai/v1` |
| Google Gemini API | Compatible con OpenAI | `https://generativelanguage.googleapis.com/v1beta/openai` |
| OpenAI | Compatible con OpenAI | `https://api.openai.com/v1` |
| LM Studio / llama.cpp / vLLM | Compatible con OpenAI | p. ej. `http://host.docker.internal:1234/v1` |
| Otro | Compatible con OpenAI | su URL base (la que precede a `/chat/completions`) |

Seguridad y privacidad del módulo:

- **Claves cifradas:** se guardan en PostgreSQL cifradas con Fernet (AES + HMAC). La clave maestra **no** está en la base de datos:
  es `APP_SECRET_KEY` (si la define) o el archivo `secret.key` (permisos 0600) que se crea en el volumen de datos. Un volcado de la
  base de datos o su copia de seguridad no revela las claves. La interfaz **nunca** recibe una clave guardada (solo «Clave guardada
  …1234») y no se escribe en logs. Quien tenga acceso al contenedor sí puede descifrarlas: no protege frente a un servidor comprometido.
- **Consentimiento explícito:** un servicio externo exige marcar *«Permitir el envío de fragmentos a este servicio externo»* tras
  leer qué se envía (la pregunta y los **fragmentos recuperados**, no el PDF completo). Sin esa marca no puede activarse ni usarse,
  y retirarla lo bloquea de inmediato. Cada respuesta indica proveedor, modelo y si el contexto salió de su equipo.
- **URL validadas:** solo `http(s)`, sin usuario/contraseña, y se rechazan direcciones de metadatos de nube/enlace local (anti-SSRF).
  «Probar» y «Cargar modelos» solo listan modelos: no envían contenido de documentos.
- **Un proveedor caído no rompe nada:** subir, borrar y buscar documentos sigue funcionando; solo las preguntas devuelven un aviso.
- Si cambia la clave maestra o restaura la base de datos en otro entorno, las claves guardadas dejan de poder descifrarse: la
  pantalla lo indica («Clave ilegible») y basta con volver a introducirlas.
- No hay autenticación de usuarios: cualquiera con acceso a la aplicación puede cambiar estos ajustes. Manténgala en `127.0.0.1`.
- Un proveedor que **no** sea compatible con la API de OpenAI ni con Ollama requiere un adaptador nuevo (`backend/app/providers/`;
  la interfaz son dos métodos, `chat` y `health`).
- Estado de verificación: Ollama y el adaptador compatible con OpenAI (contra el endpoint `/v1` de Ollama) están probados de extremo
  a extremo; **no** se ha ejecutado contra Groq, Gemini ni OpenAI (no hay claves en el entorno de desarrollo). Las URL de las
  plantillas son las publicadas por cada proveedor: verifíquelas, igual que el nombre exacto del modelo, que cambia con frecuencia.
- La calidad de las citas depende del modelo: evalúe con `eval/` antes de adoptar un modelo pequeño.

## Modelo de embeddings: Ajustes en la interfaz

Los **embeddings** convierten el texto de los documentos y tus preguntas en vectores para buscar. Son independientes del LLM. Por defecto se usa un
modelo **local** (`multilingual-e5-small`, dentro del propio backend). En **Ajustes → Modelo de embeddings** puedes añadir otro proveedor,
probarlo y elegir **uno solo** como predeterminado. El proyecto incluye **EmbeddingGemma 2 en local**: el servicio `embedder` (un contenedor con el modelo en CPU, solo
texto, ≈1,9 GB de RAM) que la API y el worker usan por HTTP; ningún texto sale del equipo. Pasos para activarlo (una sola vez):

```bash
docker compose run --rm embedder python -m app.embedding_server --download   # ≈1,5 GB desde Hugging Face
docker compose up -d                                                         # levanta el servicio «embedder»
```

y en Ajustes → Modelo de embeddings → **Añadir proveedor** → plantilla **«EmbeddingGemma 2 (local…)»** → Guardar → **Usar por defecto** (reindexa tus documentos en segundo plano;
274 fragmentos tardan ≈1,5 min en CPU). Detalles, memoria y solución de problemas: [docs/OPERATIONS.md §13](docs/OPERATIONS.md).

1. **Añadir proveedor de embeddings** → plantilla («EmbeddingGemma 2 (local)», Ollama, OpenAI, otro compatible) → URL base, clave API, modelo,
   prefijos (si el modelo los exige), dimensiones (opcional, solo modelos Matryoshka) y **umbral de similitud** (opcional, ver abajo).
2. **Probar conexión**: embebe una frase fija y muestra la dimensión y la latencia (no envía contenido de documentos).
3. **Usar por defecto**: antes de cambiar, un diálogo muestra la **vista previa**: cuántos documentos se reindexarán, cuántos fragmentos y —si el proveedor es externo—
   **cuánto texto se enviará y a qué host**. Al confirmar, el worker reindexa en segundo plano y te avisa de cada documento al terminar.

Reglas de funcionamiento (decisiones de diseño):

- **Se indexa únicamente con el modelo seleccionado.** No hay indexación paralela con varios modelos ni modo «combinado» (por ahora). Al activar otro modelo, los documentos
  que aún no tengan sus vectores se reencolan y **no se pueden consultar hasta terminar** (no se mezclan espacios vectoriales). Los vectores del modelo anterior se **conservan**:
  volver a él es **inmediato** (sin reindexar). Puedes borrarlos en «Vectores indexados» para liberar espacio (entonces volver exigirá reindexar).
- **Un servicio externo recibe el texto íntegro de todos tus documentos** al indexarlos (todos los fragmentos) y cada pregunta al buscar. Es mucho más que con un LLM, que solo
  recibe los fragmentos relevantes. Por eso exige marcar *«Permitir el envío del texto de los documentos a este servicio externo»*; sin esa marca no se puede activar ni usar, y
  retirarla lo bloquea al instante (la búsqueda y la indexación se detienen sin enviar nada). Un nombre de host distinto de `localhost`/redes privadas/`host.docker.internal` se considera externo.
- Las claves se guardan cifradas (igual que las del LLM) y nunca se devuelven al navegador. Se aplican las mismas validaciones anti-SSRF a la URL.
- Cada proveedor usa **siempre los mismos prefijos** al indexar y al buscar; el sistema rechaza (409) una activación cuyos prefijos o dimensión no coincidan con los vectores ya
  indexados de ese modelo, y no deja cambiar el modelo/prefijos de un proveedor en uso.
- Si el proveedor falla al preguntar, la API responde 503 con el motivo y la gestión de documentos sigue funcionando. Los errores permanentes (clave rechazada, modelo inexistente)
  no se reintentan; los transitorios (429/5xx) sí, con espera.

- **Umbral de similitud por proveedor.** Cada modelo reparte sus similitudes coseno de forma distinta (e5 las comprime hacia 0,8; EmbeddingGemma 2 separa mejor, con las preguntas con
  respuesta ≥ 0,72 y las que no la tienen ≤ 0,71), así que un único umbral global no sirve para todos. Cada perfil puede llevar el suyo (la plantilla de EmbeddingGemma 2 trae 0,72; vacío =
  `RETRIEVAL_MIN_SIMILARITY`). Como solo afecta a la búsqueda, se puede cambiar con el proveedor activo y rige de inmediato, sin reindexar. Calíbralo con `python -m app.evaluate`.

Estado de verificación (2026-10-14): **EmbeddingGemma 2 local está probado de extremo a extremo** en este equipo (CPU, sin GPU) con el libro de 121 páginas: indexación de 274 fragmentos, recuperación
y respuestas con el LLM real. En la evaluación (`eval/RESULTS.md`) el modo híbrido iguala o mejora a e5 (recall@3 1,00 frente a 0,93; recall@5 1,00) y la prueba extremo a extremo da 15/15 respuestas
con cita correcta y 7/7 abstenciones. Es una muestra pequeña de un solo documento: evalúa con los tuyos. El mecanismo de proveedores por API (externos) sigue probado con un servidor de desarrollo
(`scripts/dev_embedding_server.py`, docs/OPERATIONS.md §12), **no** con un servicio externo real. Plan y datos del modelo: [docs/plans/PLAN_EMBEDDINGGEMMA2.md](docs/plans/PLAN_EMBEDDINGGEMMA2.md).

## Configuración

Los parámetros (OCR, tamaño de fragmentos, umbrales de recuperación, parámetros del LLM, límite de subida…) se ajustan con variables
en `.env`; la tabla completa con valores por defecto y qué servicio las usa está en
[docs/OPERATIONS.md §10](docs/OPERATIONS.md). Tras editar: `docker compose up -d api worker`.

## Pruebas

```bash
docker compose --profile test run --rm test                           # suite completa (sin Ollama)
docker compose --profile test run --rm test python -m pytest -m ollama -s   # incluye el flujo real con Ollama
```

Las pruebas usan una base de datos aparte (`<db>_test`), ejecutan las migraciones reales (incluido el downgrade) y el modelo de
embeddings real. Con Ollama en el host: `docker compose --profile test run --rm -e OLLAMA_BASE_URL=http://host.docker.internal:11434 test python -m pytest`.
El **frontend no tiene pruebas automáticas**; su lista de verificación manual está en [docs/FRONTEND.md §7](docs/FRONTEND.md).

## Desarrollo local sin Docker (opcional)

Necesita PostgreSQL con pgvector ≥ 0.8 accesible y Tesseract para OCR. Variables en `.env` (raíz).

```bash
cd backend && python -m venv .venv && .venv/bin/pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload        # API
python -m app.worker                 # worker, en otra terminal
cd ../frontend && npm install && npm run dev
```

## Seguridad y privacidad (resumen)

- Todo local por defecto. Un proveedor de generación externo solo recibe los fragmentos recuperados y la pregunta, y la
  interfaz lo indica en cada respuesta.
- Los puertos se publican únicamente en `127.0.0.1`. Las credenciales de PostgreSQL nunca llegan al navegador.
- Las **claves API** de proveedores guardadas desde Ajustes se almacenan cifradas (Fernet) con una clave maestra fuera de la base
  de datos; la interfaz nunca las recibe. Un servicio externo exige consentimiento explícito. Detalle en la sección «Proveedor de
  generación» y en [docs/ARCHITECTURE.md §6](docs/ARCHITECTURE.md).
- Los PDF se guardan con nombre generado por el servidor; el nombre original solo se muestra. Se valida firma `%PDF-`,
  tamaño y que el archivo se abra antes de aceptarlo.
- El contenido de los documentos se trata como **dato no confiable**: se entrega delimitado al modelo, que tiene
  instrucciones de ignorar órdenes incluidas en él, y no dispone de herramientas con efectos secundarios.
- No hay autenticación: está pensado para uso local de un solo usuario. **No lo exponga a Internet** sin añadir un
  proxy con autenticación (ver "Multiusuario" en docs/OPERATIONS.md).

## Estado (qué está verificado y qué no)

Verificado en esta máquina (Windows 11 + Docker Desktop + RTX 4050), con el stack real y el modelo real:

- 240 pruebas automáticas (esquema/migraciones, extracción, OCR con Tesseract, chunking, embeddings, búsqueda vectorial,
  textual e híbrida, abstención, citas inventadas, inyección en documentos, API, seguridad de rutas, proveedores).
- Flujo completo en la interfaz: subir PDF → procesar → preguntar a `qwen3:8b` → cita → visor PDF en la página con el
  fragmento resaltado. Copia de seguridad y restauración (BD + embeddings + PDF) probadas.
- Módulo de embeddings verificado en el navegador contra el servidor de desarrollo: probar, consentimiento, vista previa (274 fragmentos, ≈238 mil caracteres), reindexado en ~15 s, búsqueda y chat con el modelo nuevo, vuelta inmediata al modelo local, borrado de vectores.
- Interfaz verificada a mano en el navegador: notificaciones flotantes, eliminación de conversaciones con «Deshacer» (comprobada contra
  la API en ambos caminos), subida de duplicado/no-PDF/PDF corrupto, Ajustes (probar, cargar modelos, consentimiento externo), vista móvil.
- Evaluación con un libro real de 121 páginas: 15/15 respuestas con cita correcta y 7/7 abstenciones (`eval/RESULTS.md`).

Pendiente / limitaciones conocidas:

- **Evaluación**: probado con un libro real de 121 páginas (22 preguntas; ver `eval/RESULTS.md`: 15/15 con cita correcta, 7/7
  abstenciones). Es una muestra pequeña de un solo documento. Con e5, el umbral de similitud (0,80) **no** separa por sí solo preguntas con
  y sin respuesta (la abstención fiable la aporta el LLM); con EmbeddingGemma 2 sí las separó (0,72), pero con la misma muestra pequeña. Evalúe con sus propios documentos (`eval/README.md`).
- Rendimiento a escala (>50 000 fragmentos) y recall de HNSW no medidos; procedimiento en docs/OPERATIONS.md.
- Proveedores OpenAI/Groq/Gemini: adaptador y pantalla de Ajustes probados contra un servidor compatible real (Ollama `/v1`), no contra esos servicios.
- **EmbeddingGemma 2** funciona en local, pero consume ≈1,9 GB de RAM mientras está cargado (con poca memoria libre, véase `EMBED_SERVER_DTYPE=bfloat16` y el tope `EMBED_SERVER_MEM_LIMIT` en docs/OPERATIONS.md §13); su ventaja sobre e5 se midió con un solo libro. Los embeddings por API **externa** están probados con un servidor de desarrollo (e5), no con un servicio real.
- Los embeddings por API externa envían **todo el texto de los documentos**: úsese solo con información que pueda salir del equipo.
- Las citas garantizan que el fragmento **existe y fue recuperado**, no que respalde semánticamente cada frase: contraste
  las respuestas críticas con el visor.
- Sin historial en el prompt (cada pregunta es independiente) ni extracción específica de tablas.
- **Frontend sin pruebas automatizadas** (solo verificación manual y chequeo de tipos). El borrado de conversaciones es diferido: si el
  navegador cancela la petición al cerrar la página, la conversación simplemente no se borra.
- Las notificaciones y avisos están en español únicamente (sin internacionalización).
- Monousuario, sin autenticación. En Windows, algunas políticas de "Control de aplicaciones" bloquean DLL de numpy en un
  venv del host; por eso el backend y las pruebas se ejecutan en Docker.

## Licencia

AGPL-3.0-or-later (por la dependencia de PyMuPDF, ver docs/LICENSES.md).
