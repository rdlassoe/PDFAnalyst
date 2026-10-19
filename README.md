# PDF Analyst

Plataforma de código abierto para consultar documentos PDF en lenguaje natural, con respuestas respaldadas por
**citas verificadas** (documento + página + fragmento exacto). Funciona 100 % en local: PostgreSQL + pgvector,
embeddings con sentence-transformers (e5 por defecto, o **EmbeddingGemma 2 en local**) y generación con Ollama; lee también las **imágenes y gráficos de los PDF** con un modelo de visión (Gemini, Groq u Ollama). Los proveedores externos son opcionales.

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitectura, decisiones técnicas (D1–D21), modelo de datos, pipelines, seguridad, fases, API |
| [docs/OPERATIONS.md](docs/OPERATIONS.md) | Operación: índices, cambio de modelo, copias de seguridad, proveedor LLM y claves, **referencia de configuración**, evaluación, solución de problemas |
| [docs/FRONTEND.md](docs/FRONTEND.md) | Interfaz: estructura, sistema de notificaciones, pantallas, diseño adaptable, lista de verificación |
| [docs/LICENSES.md](docs/LICENSES.md) | Licencias de dependencias y de los pesos de los modelos |
| [docs/CHANGELOG.md](docs/CHANGELOG.md) | Historial de cambios y hallazgos |
| [docs/plans/PLAN_EMBEDDINGGEMMA2.md](docs/plans/PLAN_EMBEDDINGGEMMA2.md) | EmbeddingGemma 2 **local**: datos del modelo, decisiones, medidas en este equipo y resultado de la evaluación |
| [docs/plans/PLAN_MULTIMODAL.md](docs/plans/PLAN_MULTIMODAL.md) | Plan multimodal y su desenlace: el sistema **solo recibe PDF**, así que se implementaron las figuras de PDF y el modelo de visión; audio y vídeo quedaron descartados. Medidas de memoria/tiempo de EmbeddingGemma 2 multimodal |
| [eval/README.md](eval/README.md) · [eval/RESULTS.md](eval/RESULTS.md) | Cómo evaluar con sus documentos y resultados con un libro real |

## Estructura del proyecto

```
PDFAnalyst/
├── backend/                 # API FastAPI + worker + pipeline RAG (Python 3.12)
│   ├── app/api/             #   routers: documents, collections, chat, settings (LLM), embeddings_settings, vision_settings, system
│   ├── app/services/        #   ingestion, pdf_extract, pdf_figures, media_ingest, chunking, embeddings, retrieval, answering, vision,
│   │                        #   citations, llm_settings, embedding_settings, vision_settings, credentials, net_validation, vector_index, storage
│   ├── app/providers/       #   LLMProvider: ollama, openai_compat (+ ProviderConfig; mensajes con imágenes para el modelo de visión)
│   ├── app/{models,config,db,worker,cli,evaluate,embedding_server}.py   # embedding_server = servicio local de embeddings
│   ├── alembic/versions/    #   0001 esquema inicial · 0002 ajustes de LLM · 0003 ajustes de embeddings · 0004 umbral por modelo · 0005 modelo de visión · 0006 medios (imágenes)
│   └── tests/               #   359 pruebas (pytest)
├── frontend/                # Next.js 15 + React 19 + Tailwind (app/, components/: Chat, DocumentPanel, PdfViewer, ImageViewer, paneles de Ajustes…, lib/)
├── eval/                    # preguntas de evaluación, e2e.py, RESULTS.md
├── scripts/                 # backup / restore (.sh y .ps1)
├── docs/                    # documentación
├── docker-compose.yml       # db, migrate, api, worker, embedder, frontend (+ ollama opcional, + test)
└── .env.example             # configuración documentada
```

## Requisitos

- Docker (Engine o Desktop) con Compose v2.
- ~8 GB de RAM libres (el servicio `embedder` con EmbeddingGemma 2 añade ≈1,9 GB, y ≈3 GB mientras indexa figuras; es opcional y se puede parar); GPU NVIDIA opcional (recomendada para el modelo de generación).
- Un modelo de Ollama (por defecto `qwen3:8b`, Apache-2.0).
- Opcional, para que el sistema **vea las imágenes de los PDF**: un modelo de visión (Gemini o Groq con su clave API, o un modelo de Ollama con visión). Sin él, las figuras se indexan solo con su vector y su OCR.

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

1. Suba uno o varios PDF (el procesamiento ocurre en segundo plano; el panel muestra el estado y cuántas figuras se detectaron). Si el PDF tiene imágenes o gráficos y hay un modelo de visión activo, también se leen y describen.
2. Pregunte. Cada afirmación termina con una etiqueta `[C1]`; al pulsarla se abre el PDF en la página exacta con el
   fragmento resaltado (o, si es una figura, con la imagen citada). Si la respuesta depende de una figura, el modelo de visión la mira al responder.
3. Si los documentos no contienen la respuesta, el sistema lo indica y **no** invoca al modelo.
4. Limite la consulta marcando documentos o eligiendo una colección.
5. **Ajustes** (arriba a la derecha): elija el proveedor de LLM (Ollama por defecto, Groq, Gemini, OpenAI, otro compatible),
   guarde su API y cámbielo cuando quiera (ver la sección siguiente); el modelo de embeddings; y el **modelo de visión** (Gemini, Groq u Ollama) con el que el sistema describe y mira las imágenes de los PDF.
   Un PDF subido antes de activar el modelo de visión se puede **volver a procesar** (botón en la lista) para que lea sus figuras.
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

## Figuras dentro de los PDF y modelo de visión (Ajustes)

El sistema recibe **PDF**, y un PDF puede contener imágenes: fotos, esquemas, gráficos, diapositivas exportadas como imagen, escaneos. Al procesarlo **detecta esas figuras, las lee y las usa para responder**:

- **Qué se detecta:** imágenes incrustadas (incluida una página entera que es una imagen, como una diapositiva) y **gráficos dibujados con vectores**. Se descartan iconos y viñetas diminutos, líneas, y las imágenes
  **repetidas en 3 o más páginas** (fondos, logos, marcas de agua). Tope por documento: 60 figuras (las más grandes primero; se avisa de las omitidas).
- **Qué se hace con cada figura:** (1) se lee el texto que contiene (OCR); (2) un **modelo de visión** escribe qué se ve en ella (descripción automática, guardada y marcada como tal); (3) EmbeddingGemma 2 la convierte en un
  vector para **encontrarla** con una pregunta en texto. Todo queda ligado a la **página** y la posición de la figura.
- **Al responder, el sistema MIRA las figuras:** si la evidencia de una pregunta incluye figuras, el modelo de visión recibe la pregunta, los fragmentos y las imágenes (hasta 3) y contesta mirándolas, citando [C#] el texto
  del PDF y la figura. La cita de una figura abre el PDF en su página y muestra la imagen. Si el modelo de visión falla o no hay consentimiento, se responde con el LLM de texto y las descripciones guardadas.
- **Modelo de visión (Ajustes → Modelo de visión):** **Google Gemini**, **Groq**, Ollama local u otro compatible con OpenAI. **Desactivado por defecto**. Sin él, las figuras se indexan con su vector y su OCR, pero el sistema no
  puede decir qué muestran. Hay un interruptor «Responder mirando las figuras de los PDF» (activado por defecto cuando hay un modelo de visión).
- **Gemini y Groq son servicios externos: las imágenes salen de su equipo** (reducidas y sin metadatos EXIF), y con «responder mirando» también la pregunta y los fragmentos recuperados. Exigen la clave API (se guarda cifrada)
  y el permiso explícito *«Permitir el envío de las imágenes»*; retirarlo desactiva el modelo al instante. La capa gratuita de Gemini puede usar el contenido para mejorar los productos de Google, según su página de
  precios: no la use con documentos privados. «Probar con una imagen de prueba» envía una imagen sintética (un círculo rojo) para comprobar que el modelo admite imágenes y que la clave funciona.
- **Para que un PDF ya subido lea sus figuras:** botón «Volver a procesar» de la lista de documentos (con un servicio externo vuelve a enviar y describir cada figura).
- **Requisito para encontrar figuras por texto:** perfil «EmbeddingGemma 2 (local)» con la casilla *«admite imágenes, audio y vídeo»* (la plantilla ya la trae). El servicio carga el codificador de imagen solo mientras indexa
  (≈3,0 GB de RAM) y vuelve a solo texto a los 5 minutos sin uso. Los medios **nunca** se envían a un servicio de embeddings externo.
- También se pueden subir imágenes sueltas (PNG, JPEG, WebP; el servidor valida la firma): reciben el mismo tratamiento y un visor con «¿Qué ves en esta imagen?».

Verificado de extremo a extremo (con **Groq `qwen/qwen3.8-27b`** y EmbeddingGemma 2 local, un PDF sintético de 2 páginas con un gráfico de barras y una foto): 2 figuras detectadas y descritas en 33 s; la pregunta «¿qué muestra el
gráfico y qué trimestre tuvo más ventas?» se respondió **mirando la imagen** («Q3, con 420»), con citas al texto del PDF y a la figura; una consulta ajena no devolvió nada. 359 pruebas automáticas. Análisis de PDF reales
(solo lectura): `Analitica Prescriptiva.pdf` son 12 diapositivas que son una imagen por página; `Parcial 3 - Sol Telco.pdf`, `manual-cassandra.pdf` y `book-attachment` tienen figuras y gráficos vectoriales.
**No verificado:** la calidad de las descripciones con tus documentos reales, Gemini real (se probó una clave real: lista modelos pero Google devolvió 403 «Your project has been denied access» al generar, una restricción del proyecto de esa clave; Groq sí funciona), y que los umbrales de detección (tamaño, repetición, trazos vectoriales) sean los óptimos para todos los tipos de PDF
(se pueden ajustar: `PDF_FIGURE_*` y `PDF_VECTOR_*`). Audio y vídeo no se implementan: el sistema solo recibe PDF ([docs/plans/PLAN_MULTIMODAL.md](docs/plans/PLAN_MULTIMODAL.md)).

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
  tamaño y que el archivo se abra antes de aceptarlo. Las figuras extraídas se guardan junto al PDF con nombres generados.
- **Las imágenes de un PDF salen de su equipo solo si activa un modelo de visión externo (Gemini, Groq…) y marca el permiso** «enviar las imágenes» (con «responder mirando», también la pregunta y los fragmentos). Retirar el permiso lo desactiva al instante. Las descripciones son automáticas y se muestran etiquetadas.
- El contenido de los documentos se trata como **dato no confiable**: se entrega delimitado al modelo, que tiene
  instrucciones de ignorar órdenes incluidas en él, y no dispone de herramientas con efectos secundarios.
- No hay autenticación: está pensado para uso local de un solo usuario. **No lo exponga a Internet** sin añadir un
  proxy con autenticación (ver "Multiusuario" en docs/OPERATIONS.md).

## Estado (qué está verificado y qué no)

Verificado en esta máquina (Windows 11 + Docker Desktop + RTX 4050), con el stack real y el modelo real:

- 359 pruebas automáticas (esquema/migraciones, extracción, OCR con Tesseract, chunking, embeddings, búsqueda vectorial,
  textual e híbrida, abstención, citas inventadas, inyección en documentos, API, seguridad de rutas, proveedores).
- Flujo completo en la interfaz: subir PDF → procesar → preguntar a `qwen3:8b` → cita → visor PDF en la página con el
  fragmento resaltado. Copia de seguridad y restauración (BD + embeddings + PDF) probadas.
- **EmbeddingGemma 2 en local** verificado de extremo a extremo (indexación, evaluación frente a e5, 15/15 y 7/7 con el LLM real, interfaz): ver «Modelo de embeddings» y `eval/RESULTS.md`.
- **Figuras de PDF y modelo de visión** verificados de extremo a extremo con **Groq `qwen/qwen3.8-27b`** y EmbeddingGemma 2 local (PDF sintético con un gráfico y una foto): detección, OCR, descripción y vector; respuesta correcta **mirando la imagen**; ver «Figuras dentro de los PDF».
- Módulo de embeddings por API verificado en el navegador contra el servidor de desarrollo: probar, consentimiento, vista previa (274 fragmentos, ≈238 mil caracteres), reindexado en ~15 s, búsqueda y chat con el modelo nuevo, vuelta inmediata al modelo local, borrado de vectores.
- Interfaz verificada a mano en el navegador: notificaciones flotantes, eliminación de conversaciones con «Deshacer» (comprobada contra
  la API en ambos caminos), subida de duplicado/no-PDF/PDF corrupto, Ajustes (probar, cargar modelos, consentimiento externo), vista móvil.
- Evaluación con un libro real de 121 páginas: 15/15 respuestas con cita correcta y 7/7 abstenciones (`eval/RESULTS.md`).

Pendiente / limitaciones conocidas:

- **Evaluación**: probado con un libro real de 121 páginas (22 preguntas; ver `eval/RESULTS.md`: 15/15 con cita correcta, 7/7
  abstenciones). Es una muestra pequeña de un solo documento. Con e5, el umbral de similitud (0,80) **no** separa por sí solo preguntas con
  y sin respuesta (la abstención fiable la aporta el LLM); con EmbeddingGemma 2 sí las separó (0,72), pero con la misma muestra pequeña. Evalúe con sus propios documentos (`eval/README.md`).
- Rendimiento a escala (>50 000 fragmentos) y recall de HNSW no medidos; procedimiento en docs/OPERATIONS.md.
- Proveedores externos: **Groq** está probado para visión (descripción y respuesta con imágenes); Gemini y OpenAI **no** se han probado con su servicio real (sí el adaptador y las pantallas, contra un servidor compatible y transportes simulados).
- **Figuras de PDF:** no se ha medido la calidad de las descripciones con los PDF reales del usuario ni el rendimiento con 60 figuras; los umbrales de detección (`PDF_FIGURE_*`, `PDF_VECTOR_*`) se ajustaron con 5 PDF y pueden necesitar retoques con otros tipos; una descripción automática puede equivocarse.
- **Audio y vídeo no se implementan** (el sistema solo recibe PDF). Se descartaron tras la aclaración del usuario; ver `docs/plans/PLAN_MULTIMODAL.md`.
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
