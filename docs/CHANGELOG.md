# Historial de cambios

Cronología de lo construido y, sobre todo, de **lo que se descubrió al verificar** (varias decisiones de diseño cambiaron por datos
reales). Fecha de referencia: 2026-10-19. Las decisiones numeradas (D1–D18) están en [ARCHITECTURE.md](ARCHITECTURE.md).

## Fase 1–5 — núcleo RAG local

- PostgreSQL 16 + pgvector 0.8 como único almacén; migraciones Alembic; cola de trabajos en PostgreSQL; worker.
- Ingesta de PDF (PyMuPDF, respaldo pypdf, OCR opcional con Tesseract), deduplicación por SHA-256, idempotencia.
- Embeddings locales `multilingual-e5-small` (modelos y dimensiones coexistentes, índices HNSW parciales por modelo, reindexación).
- Recuperación híbrida (vector + FTS con `f_unaccent`, fusión RRF), filtros por documento/colección, abstención.
- Generación con citas verificadas (etiquetas efímeras `[C#]` → `chunk_id`), regeneración controlada, delimitación anti-inyección.
- Copias de seguridad y restauración probadas (BD + embeddings + PDF). 148 pruebas automáticas en Docker.

**Hallazgos que cambiaron el diseño**

| Hallazgo | Cambio |
|----------|--------|
| Chunks de varias páginas daban citas «pp. 1-3» | Los chunks **nunca cruzan páginas** |
| El parser FTS parte `ISO-27001` en `iso` + `-27001` | Tokens originales **y** normalizados en la columna generada |
| Al regenerar tras una cita inválida el modelo respondía a mi instrucción | La regeneración incluye la respuesta fallida del modelo |
| Ollama devolvía 500 transitorios al cargar el modelo | 3 reintentos con espera |
| Una DLL de numpy bloqueada por una política de Windows en el venv del host | Backend y pruebas se ejecutan en Docker |
| El puerto 5432 está reservado en el equipo | `POSTGRES_HOST_PORT` (5433 por defecto) |
| `localhost` tardaba 5 s (IPv6) | Host por defecto `127.0.0.1` |

## Fase 6 — interfaz

- Next.js + PDF.js: documentos, conversación, visor con resaltado del fragmento citado, colecciones.
- Corrección: un `useEffect` devolvía el resultado de `scrollIntoView` (React lo toma por función de limpieza) y la página quedaba en blanco.

## Fase 7 — proveedores LLM

- Interfaz `LLMProvider`; adaptadores Ollama y compatible con OpenAI (Groq, Gemini, OpenAI, LM Studio, llama.cpp…).
- Bloqueo de servicios externos sin consentimiento explícito.
- Corrección de seguridad: `is_local_url` trataba un dominio `10.evil.com` como local; ahora analiza direcciones IP.
- Verificado contra un servidor compatible real (endpoint `/v1` de Ollama); **no** contra Groq/Gemini/OpenAI.

## Evaluación con un documento real (libro de 121 páginas, 22 preguntas)

- Resultado final: 15/15 respuestas con cita en una página con la evidencia y 7/7 abstenciones (`eval/RESULTS.md`).
- Hallazgos: (1) el umbral coseno **no separa** preguntas con y sin respuesta → la abstención fiable la da el LLM (D10);
  (2) palabras muy frecuentes dominaban el ranking textual → ponderación por rareza (D9; recall@1 textual 0,33 → 0,60, recall@5 híbrido 0,87 → 1,00);
  (3) una meta-pregunta filtró el prompt del sistema → regla 6; (4) varios «fallos» eran citas correctas que mi lista de páginas esperadas omitía.
- Límites: un solo libro, 22 preguntas escritas por el desarrollador, una ejecución.

## Módulo Ajustes

- Pantalla `/ajustes`: perfiles de proveedor LLM (Ollama por defecto, Groq, Gemini, OpenAI, compatible, local), probar conexión, cargar modelos
  disponibles, elegir el predeterminado sin reiniciar.
- Claves API cifradas con Fernet (D14 revisada); clave maestra fuera de la BD; la API nunca las devuelve; consentimiento explícito
  para destinos externos; validación anti-SSRF. Migración `0002` (`llm_profiles`, `app_settings`).
- Huecos corregidos al verificar: el compose no pasaba `LLM_*` ni `APP_SECRET_KEY` a los contenedores; `migrate` tiene imagen propia y
  hay que reconstruirla (`up -d --build`).

## Mejoras del frontend (notificaciones y experiencia)

- **Notificaciones flotantes** propias (`components/Toaster.tsx`): tipos, barra de progreso con pausa, acciones, `id` para reemplazo,
  `toast.promise`, accesibilidad. **Diálogo de confirmación** que sustituye a `confirm()`.
- **Eliminación de conversaciones con «Deshacer»** (borrado diferido; verificado contra la API: deshacer conserva, expirar borra).
- Avisos de procesamiento terminado (con «Abrir»/«Reintentar»), API caída/recuperada, modelo LLM caído/recuperado, subidas
  (duplicado, no-PDF, corrupto), colecciones, copiar respuesta, errores del modelo con acceso a Ajustes.
- Diseño adaptable: pestañas en móvil y visor deslizante; PDF ajustado al ancho; flechas ← → en el visor; esqueletos de carga; área de texto
  autoajustable; estados vacíos; respeto de `prefers-reduced-motion`.
- Documentación: `FRONTEND.md`, referencia de configuración y solución de problemas ampliadas.

## Modelo de embeddings seleccionable (API externa o local)

Decisiones del usuario: un modelo predeterminado en Ajustes; **indexar únicamente según el modelo seleccionado**; **sin modo combinado** por ahora; EmbeddingGemma 2 se usará **de forma
externa** (no local) porque el hardware (≈3,5 GB de RAM libres, GPU de 6 GB ocupada por el LLM) lo haría demasiado lento. **Esa decisión fue rectificada: ver la sección «EmbeddingGemma 2 en local» más abajo.**

- Datos verificados del modelo (fuentes oficiales): EmbeddingGemma 2, lanzado el 2026-10-06, Apache-2.0 (metadato), 768 dimensiones (Matryoshka 128–768), contexto 8 192, ≈270M solo texto,
  no usar `float16`, prefijos oficiales `task: search result | query: ` y `title: none | text: `. **No se encontró ningún proveedor alojado confirmado** (solo el EmbeddingGemma original en Cloudflare
  Workers AI); por eso se implementó un adaptador **genérico** (cualquier endpoint `/embeddings` compatible con OpenAI, u Ollama).
- Nuevo módulo (`embedding_settings`, migración `0003`, router `/settings/embeddings`, panel en Ajustes): perfiles con clave cifrada, consentimiento explícito para destinos externos,
  vista previa de lo que se enviaría, activación que comprueba acceso/dimensión/prefijos antes de cambiar, reindexado solo de lo que falta, vuelta inmediata a un modelo ya indexado,
  borrado de vectores de modelos no activos, `/health` con cobertura. Reintentos 429/5xx y errores permanentes sin reintento; el worker resuelve el modelo activo al ejecutar y repite si cambia.
- Refactor: validaciones compartidas en `net_validation`; los adaptadores se construyen desde una configuración explícita (`EmbedderConfig`).
- Defecto corregido al verificar: con la URL base vacía el mensaje decía «servicio externo None» en lugar de «falta la URL base».
- Verificado: 214 pruebas automáticas (37 de integración con el modelo e5 real servido por el protocolo OpenAI + 29 de los adaptadores), y en el navegador contra `scripts/dev_embedding_server.py`:
  probar (solo se enviaron 55 caracteres), bloqueo sin permiso, vista previa, reindexado de 274 fragmentos en ≈15 s (≈240 mil caracteres recibidos, coincide con la vista previa), búsqueda y chat con
  el modelo nuevo, vuelta inmediata al modelo local (sin trabajos ni peticiones externas), borrado de vectores y del proveedor.
- **No verificado:** EmbeddingGemma 2 con un servicio real; calidad de recuperación de ese modelo frente a e5 (requiere la evaluación A/B de `eval/`); los prefijos que aplica cada servicio.

## EmbeddingGemma 2 en local (rectificación de la decisión anterior)

El usuario corrigió la decisión anterior: EmbeddingGemma 2 debe ejecutarse **localmente**, no como servicio externo. Hardware: ≈3,5 GB de RAM libres, GPU ocupada por el LLM, CPU i7-13620H.

- **Ollama descartado, comprobado:** las compilaciones de `embeddinggemma-2` en Ollama exigen MLX («requires MLX support», solo Apple Silicon); se probaron dos etiquetas. Se sirve el modelo de Hugging Face con sentence-transformers.
- **Modo solo texto:** se carga con `config_kwargs={"vision_config": None, "audio_config": None}` → ≈271M parámetros en lugar de ≈740M (carga en ≈6–9 s, 768 dimensiones); los prefijos oficiales del modelo coinciden con los de la plantilla
  (verificado contra los `prompts` del propio modelo). No admite `float16` (la configuración lo rechaza). Su procesador importa `torchvision` aunque se use solo texto: sin él no carga (se añadió a la imagen).
- **Servicio `embedder`** (`backend/app/embedding_server.py`, servicio del compose, sin puerto en el host): API `/v1/embeddings` de OpenAI, `/v1/models`, `/health`; validación de entrada, `dimensions` (Matryoshka) con
  re-normalización, clave opcional, errores sin detalles internos; una sola copia del modelo para la API y el worker. Plantilla «EmbeddingGemma 2 (local)» en Ajustes (la primera); `embedder` entra en la lista de hosts locales.
- **Defecto corregido al verificar:** la primera petición bloqueaba el servicio para siempre (el cerrojo de carga no es reentrante y la propiedad `dimension` volvía a pedirlo): prueba de regresión con 4 hilos simultáneos.
- **Umbral de similitud por perfil** (migración `0004`, `embedding_profiles.min_similarity`, campo en Ajustes): con el modelo real, el umbral global de 0,80 (calibrado para e5) habría descartado por similitud la mitad de las
  preguntas con respuesta (mediana 0,796). Con Gemma las preguntas con respuesta están en ≥ 0,722 y las sin respuesta en ≤ 0,714 (separables); la plantilla trae 0,72. Con él, el modo vectorial pasa de 0,60 a 1,00 en respuestas dadas
  con abstención correcta 1,00. Es editable con el perfil activo (no cambia los vectores).
- **Defecto de interfaz corregido:** la réplica `isLocalUrl` del frontend no incluía `embedder`, por lo que el perfil local habría mostrado el aviso de «servicio externo».
- **Resultados medidos** (este equipo, CPU): carga del modelo ≈9 s; **≈1,7–1,9 GiB de RAM** en el contenedor; reindexar 274 fragmentos ≈83 s; evaluación `eval/` (22 preguntas): híbrido recall@3 **1,00** (e5 0,93), recall@5 1,00 (1,00),
  MRR 0,78 (0,72); extremo a extremo con el LLM real **15/15** respuestas con cita correcta y **7/7** abstenciones.
- **Corrección de las etiquetas de la evaluación** (hecha *después* de ver el resultado, por verificación en el texto): en b2 («quinta manera de llenarse la bolsa») la página 35 contiene el encabezado
  «La quinta manera: Haced que vuestra propiedad sea una inversión rentable» y en b8 («Sharru Nada») el nombre aparece en las páginas 101–104, 110 y 111; se añadieron esas páginas a las esperadas (antes 13/15 citas, ahora 15/15).
  Las páginas esperadas son «al menos una de estas».
- Verificado: 240 pruebas automáticas (incluidas 3 con el modelo real `gemma2`, 21 del servicio y 4 nuevas del umbral por perfil), interfaz en el navegador. Una prueba antigua de pgvector (`test_phase1_pgvector`, plan `EXPLAIN` con HNSW) falló una
  vez de cuatro ejecuciones y pasa al repetir: depende del planificador, no del cambio.
- **No verificado:** que la ventaja sobre e5 se mantenga con otros documentos (un solo libro de 121 páginas); rendimiento de `bfloat16` en CPU; uso con GPU; el tiempo de indexación de colecciones grandes (≈3 fragmentos/s).

## Imágenes y modelo de visión (Fases 0–2 del plan multimodal)

Decisiones del usuario: orden imágenes → audio → vídeo; añadir Whisper; el sistema debe **decir qué ve** en las imágenes; modelo de visión por API con **Gemini y Groq**, configurables desde Ajustes. **Audio, vídeo y Whisper se descartaron después** (el usuario aclaró que el sistema solo recibe PDF: ver la sección siguiente); lo construido aquí es la base de las figuras de PDF. El servicio `embedder` conserva la capacidad de audio, sin uso.

- **Fase 0 (medido en esta máquina, CPU, fp32):** el vector de texto es idéntico con y sin los codificadores de visión/audio (coseno ≥ 0,9999: los vectores de texto existentes siguen valiendo); memoria pico solo texto 2,0 GB · +imagen 3,0 · +audio 3,7 · completo 4,7;
  imagen 3,3 s (280 tokens) / 1,5 s (140) / 0,7 s (70); audio de 30 s en 2,4 s; vídeo 1,5 s por fotograma. **bfloat16 es 3–7× más lento** en esta CPU (sin AVX-512/AMX): se usa float32.
- **Servicio `embedder` multimodal:** modos por modalidad (carga solo los codificadores necesarios, libera el modelo anterior antes de cargar el nuevo, vuelve a solo texto tras 300 s sin uso), `POST /v1/media/embeddings` por referencia a archivos, validación de rutas
  (sin `..`, sin enlaces simbólicos que salgan de la raíz), límites de tamaño/píxeles/audio/fotogramas y errores sin detalles internos. 27 pruebas nuevas (5 con el modelo real).
- **Imágenes de extremo a extremo:** migraciones `0005`/`0006`; subida por firma (PNG, JPEG, WebP; no SVG ni GIF), miniatura sin EXIF, OCR, descripción automática, vector de medio; un fragmento por cada uno; modelos que no admiten medios los omiten sin dejar el documento «pendiente»;
  búsqueda con umbral propio para medios; el LLM de texto solo recibe fragmentos con texto (descripción etiquetada como automática) y las imágenes sin texto salen como «Medios relacionados» sin afirmaciones; citas con modalidad/origen; API de archivo, miniatura y «¿qué ves?».
- **Modelo de visión configurable (Ajustes):** perfiles propios con clave cifrada y consentimiento, `none` por defecto, plantillas Gemini/Groq/Ollama, «Probar con una imagen de prueba», reintentos ante 429 con `Retry-After`. Las imágenes se reducen y se recodifican sin EXIF antes de enviarse.
- **Verificado de extremo a extremo con el modelo real:** una imagen sintética se indexó en 22 s (el servicio pasó a modo imagen, 3,0 GiB); la búsqueda «una imagen con un círculo rojo…» la encontró con similitud 0,74; «gato durmiendo en un sofá» no devolvió nada; las consultas sobre los PDF no cambiaron; el chat devolvió
  la imagen como medio relacionado sin llamar al LLM. 331 pruebas automáticas.
- **No verificado:** Gemini y Groq reales (sin claves; adaptadores probados con transportes simulados), calidad de las descripciones en español, el umbral con imágenes reales variadas (0,65 es provisional), Whisper y todo lo de audio/vídeo, imágenes dentro de PDF.
- Defecto encontrado al integrar: crear la carpeta de miniaturas como efecto secundario del borrado dejaba basura en el almacenamiento de pruebas; el borrado ya no la crea.

## Figuras dentro de los PDF (corrección de prioridad: el sistema solo recibe PDF)

El usuario aclaró que el sistema recibe únicamente PDF y que sus imágenes deben detectarse, verse y usarse para responder; hasta entonces un PDF con imágenes no las leía (el extractor las ignoraba). Se implementó en este orden:

- **Análisis previo con PDF reales (solo lectura):** `Analitica Prescriptiva.pdf` = 12 páginas que son una imagen a página completa (sin texto); `Parcial 3 - Sol Telco.pdf` = 40 imágenes (una de página completa repetida en 10 páginas, es fondo) y 9 páginas con gráficos vectoriales de 67–502 trazos;
  `manual-cassandra.pdf` = 7 figuras y 10 páginas con ≥ 30 trazos; `book-attachment` = 7 figuras relevantes de 29. De ahí los filtros: conservar páginas enteras que son imagen, descartar repetidas en ≥ 3 páginas, tamaño mínimo, y renderizar los gráficos vectoriales.
- **Detector y renderizado** (`pdf_figures.py`), **ingesta** (OCR por figura, descripción con el modelo de visión, vector; fragmentos con página y posición; descripción única de imágenes idénticas; latido del trabajo; corte tras 429 agotado o 3 fallos sin romper la ingesta),
  **respuesta mirando las figuras** (adjunta hasta 3 figuras al modelo de visión, con reintentos ante 429 y respaldo al LLM de texto), **API** (`/chunks/{id}/figure`, contadores por documento), **interfaz** (tarjeta de figura citada, miniaturas, «Volver a procesar», interruptor en Ajustes).
- **Cambio de comportamiento:** un PDF escaneado sin OCR antes fallaba («No se extrajo texto»); ahora se indexa por la figura de su página con un aviso (sin texto ni modelo de visión no se podrá responder con su contenido). Un PDF solo de imágenes (diapositivas) ya funciona.
- **Verificado de extremo a extremo con modelos reales:** PDF sintético de 2 páginas (gráfico de barras y foto): 2 figuras detectadas y descritas por Groq (`qwen/qwen3.8-27b`) en 33 s; «¿qué muestra el gráfico y qué trimestre tuvo más ventas?» respondida **mirando la imagen** («Q3 con 420»), con citas [C1] al texto
  del PDF y [C2] a la figura; el modelo recibió 1 imagen; consulta ajena sin resultados. 359 pruebas automáticas (28 nuevas de figuras, con OCR real y PDF construidos con figuras, logos repetidos, iconos y gráficos vectoriales).
- Defecto de mis pruebas, no del sistema: el helper procesaba el trabajo `embed` de un documento anterior en lugar del `ingest` del actual.
- **No verificado:** calidad con los PDF reales del usuario (no se subieron: enviarían su contenido a Groq sin que lo hubiera pedido), Gemini real, umbrales de detección con más tipos de PDF, rendimiento con 60 figuras (se estima ≈3 min de CPU y ≈1–6 min de API).

## Configuración y documentación

- El `docker-compose.yml` ahora pasa a los contenedores todos los parámetros ajustables (antes solo algunos, de modo que, p. ej.,
  `RETRIEVAL_MIN_SIMILARITY` en `.env` no tenía efecto). Verificado con un override real. Referencia en OPERATIONS §10.
- Documentación reescrita donde estaba obsoleta: modelo de datos, pipelines, abstención, FTS, licencias de modelos, estado por fase.

## Gemini real en visión (2026-10-19): hallazgos

Primera prueba con una clave real de Gemini (perfil del usuario, solo lectura/prueba, sin activarlo): la clave **autentica** (lista 63 modelos) pero **la generación falla**: `gemini-2.5-flash` (el modelo que sugería nuestra plantilla) responde 404 «ya no está disponible para usuarios nuevos» y los modelos actuales (`gemini-3.8-flash`, `3.5-flash`, `3.5-flash-lite`, `3.1-flash-lite`) responden **403 «Your project has been denied access»**: el proyecto de Google de esa clave no tiene permiso para generar (restricción de Google; no se puede arreglar desde el código).
Correcciones nuestras: la plantilla ya sugiere `gemini-3.8-flash` y advierte de ambos casos; los errores del proveedor (401/403/404) muestran ahora **lo que dice el proveedor** (recortado y con la clave enmascarada) en lugar de «credenciales rechazadas» genérico (3 pruebas nuevas, 362 en total). **Gemini sigue sin verificarse de extremo a extremo** (Groq sí).
Aparte: el test con Ollama real falló una vez de forma transitoria (carga del modelo en la GPU) y pasó al repetirlo.

## Documentación: auditoría completa (2026-10-19)

Revisión de toda la documentación tras la ampliación de figuras: decisiones D1–D21 en orden, pipelines de ingesta y consulta con las figuras, modelo de datos y API, seguridad (imágenes a servicios externos), README (PDF como entrada principal, requisitos, estado), OPERATIONS §14 reescrita, FRONTEND (componentes y lista de verificación), LICENSES (modelos de visión como servicios) y el plan multimodal (alcance real). Comprobación mecánica de la configuración: **`RETRIEVAL_MIN_SIMILARITY_MEDIA` estaba documentada pero el compose no la pasaba al contenedor** (corregido), y `WORKER_POLL_SECONDS`/`JOB_STALE_MINUTES` no estaban documentadas ni en el compose (añadidas).

## Pendiente conocido

- **Figuras de PDF:** probar con los PDF reales del usuario y medir la calidad de las descripciones y el tiempo con muchas figuras; ajustar los umbrales de detección con más tipos de PDF; probar Gemini real; evaluar preguntas sobre figuras con un conjunto propio (`eval/`); vector por página completa (tablas y gráficos que el texto no capta).
- Repetir la evaluación A/B de EmbeddingGemma 2 frente a e5 con más documentos y preguntas de usuarios reales; medir `bfloat16` y la indexación de colecciones grandes.
- Evaluar con más documentos reales (técnicos/legales, con tablas) y preguntas de usuarios reales.
- Rendimiento a escala (> 50 000 fragmentos) y recall de HNSW.
- Verificación con servicios reales de Groq, Gemini y OpenAI.
- Pruebas automáticas del frontend; autenticación y multiusuario; historial de conversación en el prompt; extracción de tablas.
