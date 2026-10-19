# Plan multimodal: figuras de PDF (alcance final) — y lo que se descartó

**Aclaración del usuario (2026-10-19, posterior): el sistema solo recibe PDF.** Se priorizó la Fase 2c (figuras dentro de PDF: detectar, leer, describir, vectorizar y responder mirándolas) y está implementada y verificada; audio, vídeo y Whisper dejan de aplicar. Estado anterior: **implementadas la Fase 0, la Fase 1 y la Fase 2 (imágenes de extremo a extremo, con modelo de visión configurable para Gemini/Groq/Ollama); pendientes F2c (imágenes dentro de PDF), audio con Whisper, vídeo y el resto.** Decisiones del usuario: el orden **imágenes → audio → vídeo** es correcto; **se añade Whisper**; **el sistema debe «ver» las imágenes y describirlas** (§3, M3 y M11). Los datos del modelo están verificados en los archivos del propio modelo descargado (`config.json`, `processor_config.json`, `sentence_bert_config.json`, `README.md`).
Lo que son estimaciones se marca como tal; lo que hay que medir antes de comprometerse es la Fase 0.

## 0. Alcance final y estado de implementación (leer primero)

El plan original (abajo, conservado como registro de decisiones y medidas) hablaba de indexar imágenes, audio y vídeo. El usuario aclaró después que **el sistema solo recibe PDF**, que pueden contener imágenes, y que debe **verlas y responder en base a ellas**. El alcance real quedó así:

| Fase | Contenido | Estado |
|------|-----------|--------|
| F0 | Medidas de memoria/tiempo/coherencia del modelo multimodal (§5b) | **Hecha** |
| F1 | Servicio `embedder` multimodal (modos, endpoint de medios por referencia, validación de rutas) | **Hecha** (27 pruebas, 5 con el modelo real) |
| F2 | Imágenes sueltas de extremo a extremo + modelo de visión configurable (Gemini, Groq, Ollama) | **Hecha** (secundaria: el sistema recibe PDF) |
| F2c | **Figuras dentro de PDF**: detectar (imágenes incrustadas y gráficos vectoriales), OCR, describir, vectorizar y **responder mirándolas** | **Hecha y verificada** con Groq real + EmbeddingGemma 2 local (ARCHITECTURE D21) |
| F3 | Audio con Whisper | **Descartada** (el sistema solo recibe PDF) |
| F4 | Vídeo | **Descartada** |
| F5 | Filtros por modalidad y panel unificado | **Parcial**: umbral propio para medios y «Medios relacionados» hechos; filtros por modalidad no (sin sentido con solo PDF) |
| F6 | Vector de la página completa del PDF (tablas y gráficos que el texto no capta) | **Pendiente, opcional** |
| F7 | Consulta con imagen o con voz | **Descartada** |
| F8 | Endurecimiento y documentación | **Documentación al día; pruebas de seguridad de rutas y archivos hechas;** falta probar con PDF reales del usuario y Gemini real |

Consecuencias de la aclaración: la decisión de añadir Whisper (M8) y los límites de audio/vídeo (§9) **ya no aplican**; el servicio `embedder` conserva la capacidad de audio y vídeo sin uso. Las medidas de §5b siguen siendo válidas (el vídeo era el caso caro; no se usa).
Detalle de lo construido: [ARCHITECTURE.md](../ARCHITECTURE.md) D19–D21, [OPERATIONS.md §14](../OPERATIONS.md), [CHANGELOG.md](../CHANGELOG.md).

## 1. Objetivo y alcance (original, anterior a la aclaración)

Hoy PDF Analyst indexa **texto** de PDF (y OCR de páginas escaneadas) y responde con citas verificadas (documento + página + fragmento). El objetivo es que el usuario pueda subir **imágenes, audio y vídeo** y:

1. **Buscarlos con una pregunta en texto** (y, más adelante, con una imagen o con la voz), en el mismo espacio vectorial que los PDF.
2. **Que el sistema describa lo que ve en una imagen** («¿qué ves aquí?», preguntas sobre su contenido) y que esas descripciones sean buscables.
3. **Obtener respuestas con citas** que apunten a la evidencia: imagen, o audio/vídeo **con marca temporal** (reproducir desde el segundo exacto).
4. Mantener lo que ya es principio del proyecto: local primero, citas verificables, abstención cuando no hay evidencia, consentimiento explícito para enviar datos fuera.

Fuera de alcance (primera versión): generar o editar medios, diarización de hablantes, vídeo en directo, ingesta por URL (YouTube, etc.), reconocimiento de caras.

## 2. Qué permite el modelo (verificado) y qué no

| Dato | Valor |
|------|-------|
| Modalidades | texto (incl. código), imagen, vídeo, audio y **combinaciones**, en **un único espacio de 768 dimensiones** (`sentence_bert_config.json`: `text`, `image`, `audio`, `video`, `message`) |
| Tamaño | 740M completo = texto 270M + visión 170M + audio 300M; los codificadores se pueden cargar por separado (`config_kwargs`): solo texto 270M · texto+imagen 440M · texto+audio 570M · completo 740M |
| Contexto | **8 192 tokens compartidos** por todas las modalidades en una misma entrada |
| Coste en tokens | imagen **280** (presupuesto configurable 70–1 120) · vídeo **140 por fotograma** (el procesador muestrea a **1 fps**, máx. **32 fotogramas** por defecto) · audio **25 por segundo** (≈ **327 s** máx.) |
| Formato de entrada | audio **mono a 16 kHz**; imágenes y fotogramas con el procesador del modelo; sentence-transformers **≥ 6.1.0** (ya instalado: 6.1.0) |
| Prefijos | solo para **texto**; imágenes, audio y vídeo se pasan **sin prefijo** |
| Entradas mixtas | una entrada puede intercalar texto y medios con los marcadores `<\|image\|>`, `<\|video\|>`, `<\|audio\|>` y devuelve **un** vector |
| Precisión | `float32` o `bfloat16`; **`float16` produce NaN** (ya prohibido en la configuración) |
| Matryoshka | 768/512/256/128; la ficha avisa de que **128 degrada mucho la calidad multimodal** → mantener 768 |
| Calidad publicada (modelo completo) | imagen MIEB-lite 64,6 · documentos visuales MMEB VisDoc NDCG@5 67,8 · **vídeo MMEB Hit@1 50,7** · **audio MSEB retrieval MRR@10 69,5** (el vídeo es la modalidad más débil) |
| Licencia | Apache-2.0 según los metadatos (ver `docs/LICENSES.md`: confirmar antes de redistribuir) |

**Lo que NO hace** (condiciona todo el diseño):

- Es un modelo de **embeddings**: no genera texto. No transcribe, no describe imágenes, no responde. El LLM de generación actual (`qwen3:8b`) es **solo texto**: no puede «ver» ni «oír».
- Un vector de audio captura el contenido semántico, pero **no garantiza recuperar palabras o nombres exactos**; para citar lo dicho hace falta una **transcripción**.
- La similitud texto→imagen/audio/vídeo vive en otra escala que texto→texto (efecto «brecha entre modalidades»): el umbral de similitud debe calibrarse **por modalidad**.

## 3. Decisiones de diseño

| # | Decisión | Motivo | Revisar cuando |
|---|----------|--------|----------------|
| M1 | **Un solo modelo, un solo espacio**: los vectores de medios se guardan en el mismo `embedding_models` (google/embeddinggemma-2, 768) y el mismo índice HNSW parcial. **Los vectores de texto ya indexados siguen siendo válidos: no hay que reindexar.** | La búsqueda cruzada texto↔medio solo funciona si comparten espacio; el modelo ya lo ofrece. Condición a verificar en F0: el vector de texto con los codificadores cargados es **idéntico** al de solo texto. | Si F0 muestra diferencias. |
| M2 | **Doble representación** de cada medio: (a) **vectores** del propio medio (búsqueda semántica cruzada) y (b) **derivados de texto** (transcripción del audio/vídeo, OCR de imágenes, opcionalmente descripciones) guardados como fragmentos de texto con marca temporal. | Los vectores encuentran; el texto permite búsqueda por palabras exactas (FTS), **citas verificables con fragmento** y respuestas del LLM de solo texto. | Si se adopta un LLM multimodal (M3, nivel 2). |
| M3 | **Las imágenes se «ven» con un modelo de visión generativo (VLM), además de tener vectores.** EmbeddingGemma 2 solo produce vectores: no puede describir. El VLM se usa de dos formas: (a) **al ingerir**, una descripción automática de cada imagen (junto con su OCR) que se guarda como fragmento de texto `source='caption'`, buscable por palabras, citable y utilizable por el LLM de texto; (b) **bajo demanda**, «¿qué ves en esta imagen?» o preguntas sobre las imágenes recuperadas: el VLM recibe la imagen y la pregunta. El audio y el vídeo siguen respondiéndose desde su **transcripción** (el LLM de texto). Los aciertos de medios sin texto utilizable se muestran como **«Medios relacionados»**, sin afirmaciones sobre ellos. | Un LLM de solo texto no puede ver; el VLM sí, y guardar la descripción una vez evita repetir un modelo caro en cada consulta y hace que las imágenes entren en la búsqueda por palabras y en las citas. | Si el VLM resulta lento o impreciso con las imágenes reales del usuario (F0). |
| M4 | **Carga de codificadores por modalidad y bajo demanda en el mismo servicio `embedder`.** Modos: `text` (por defecto, como hoy), `image` (texto+imagen; sirve también para fotogramas de vídeo y figuras de PDF), `audio` (texto+audio) y, solo si hay memoria, `full`. Los trabajos de medios piden el modo que necesitan, procesan en lote y, tras un tiempo inactivo, el servicio **vuelve a `text`**. Todos los modos atienden también consultas de texto (mismos pesos; verificado en F0). | La RAM es el límite: medido en F0 (float32, pico de RSS) texto 2,0 GB · **texto+imagen 3,0 GB** · texto+audio 3,7 GB · completo 4,7 GB. Los codificadores solo hacen falta **al indexar**; las consultas son texto. | Si hay más RAM: dejar un modo ampliado residente. |
| M5 | **Precisión `float32` también en los modos de medios** (se descarta `bfloat16`). | Medido en F0: esta CPU (i7-13620H: AVX2/AVX-VNNI, sin AVX-512 ni AMX) no acelera bf16: el vídeo tardó **7 s por fotograma en bf16 frente a 1,5 s en fp32** y la imagen 10,9 s frente a 3,3 s. (La calidad sí coincide: coseno 0,99995 entre bf16 y fp32 en texto.) | Si se usa otra CPU (AMX/AVX-512) o GPU. |
| M6 | **Fragmentación de medios.** Imagen: 1 vector por imagen (280 tokens; opción de 560 para capturas con texto). Audio: ventanas de **30 s con solape de 5 s** (≈750 tokens, muy por debajo de 327 s). Vídeo: ventanas de 30 s a 1 fps (30 fotogramas × 140 = 4 200 tokens < 8 192) y **la pista de audio como ventanas de audio aparte**. | Ventanas cortas dan marcas temporales útiles para citar y respetan el presupuesto de 8 192 tokens. Embeber vídeo+audio juntos mezcla señales y complica la cita; se podrá comparar en F4. | Tras medir calidad y coste. |
| M7 | **Un vector por fragmento** (se mantiene la clave `(chunk_id, model_id)`). Cada modalidad de una misma ventana es **su propio fragmento**; los derivados de texto también. Al mostrar un acierto se **expanden** los fragmentos de la misma ventana de tiempo (p. ej. acierto de vídeo + su transcripción). | No hay que cambiar la clave de `chunk_embeddings` ni el índice; el RRF actual fusiona ramas sin cambios de fondo. | Si se necesitan varios vectores por fragmento. |
| M8 | ~~**Transcripción local con Whisper vía faster-whisper** (modelo multilingüe, CPU, cuantizado) y **VAD** para silencios; texto con marcas temporales a fragmentos con `source='asr'`. Calidad en español y tamaño del modelo se **miden en F0**.~~ (**descartada: el sistema solo recibe PDF**) | Es open source y local (coherente con el proyecto). Sin transcripción no hay citas textuales ni respuestas desde audio. Whisper alucina con silencio/ruido: filtrar con VAD y umbral de confianza y etiquetar el texto como transcripción automática. | Si F0 muestra tiempos inviables en CPU. |
| M9 | **Umbral de similitud por (perfil, modalidad).** Se extiende `embedding_profiles.min_similarity` (hoy un número) a un valor por modalidad; se calibra con el arnés de evaluación (§7). | Un único número no sirve ni entre modelos (ya se vio con e5 vs Gemma) ni entre modalidades. | — |
| M10 | **Los medios solo se indexan con el servicio local.** El adaptador genérico `/v1/embeddings` de OpenAI **no tiene entrada de medios**; el servicio local expone un endpoint propio y los medios se pasan **por referencia a un archivo** del volumen compartido (no por red). Un servidor externo con soporte de medios exigiría un consentimiento **distinto y reforzado** (voz y caras son datos biométricos) y queda fuera de la primera versión. | Privacidad y simplicidad: sin subir archivos grandes por HTTP ni nuevos destinos de envío. | Si se quiere un servicio externo de embeddings multimodal. |
| M11 | **Modelo de visión configurable, complementario a EmbeddingGemma 2 (no lo sustituye: son dos trabajos distintos; EmbeddingGemma 2 convierte la imagen en un vector para *encontrarla*, el modelo de visión *escribe* qué contiene).** (nuevo bloque en Ajustes, separado del LLM de chat; por defecto Ollama local). Candidatos de la biblioteca de Ollama con imagen de entrada y tamaños verificados: **`qwen3-vl:4b` (3,3 GB)**, `qwen3-vl:2b` (1,9 GB, el más ligero) y `gemma3:4b` (3,4 GB); `qwen3-vl:8b` (6,1 GB) y los de 12B o más **no caben** en la GPU de 6 GB. La elección final se hace en F0 con imágenes reales. La capa de proveedores se amplía para enviar imágenes (`images` en Ollama; `image_url` en los servicios compatibles con OpenAI) y los perfiles llevan la capacidad `vision`. | Cumple el requisito de «decir qué ve», sigue siendo local y reutiliza Ajustes, perfiles cifrados y consentimiento. La GPU de 6 GB la comparte `qwen3:8b`: Ollama descarga uno y carga el otro (unos segundos de espera); por eso se describe **al ingerir** y no en cada consulta. Un servicio **externo** de visión enviaría las imágenes fuera: exige consentimiento propio y reforzado. | Si el modelo elegido se puede usar también como LLM de chat. Las licencias de cada modelo se comprueban con `ollama show --license` (la página de la biblioteca no las indica). |
| M12 | **Modelo de visión por API: Gemini y Groq (decisión del usuario, 2026-10-19), como alternativa opcional al local.** Opciones comprobadas (2026-10-19): **Gemini API** con capa gratuita (varios modelos con imagen de entrada, p. ej. 2.5 Flash, 2.5 Flash-Lite, 3.1 Flash-Lite; límites por proyecto visibles en AI Studio, la prensa reporta ≈10–15 peticiones/min y 1 000–1 500/día, **sin confirmar**); **Groq** (su página de visión lista `qwen/qwen3.8-27b`; **no indica si está en el plan gratuito**; el usuario ya tiene un perfil de Groq); **OpenRouter** con modelos `:free` (50 peticiones/día sin créditos; la política de registro para modelos gratuitos no queda clara). Se usan a través de los perfiles externos ya existentes (compatibles con OpenAI) una vez la capa de proveedores admita imágenes. | Evita la descarga y el cambio de modelo en la GPU. **Pero cada imagen sale del equipo**, y en la capa gratuita de Gemini la página de precios indica que **el contenido se usa para mejorar los productos de Google** (en la de pago, no): no usar con imágenes privadas. Exige el consentimiento reforzado de M11, vista previa de cuántas imágenes se enviarán y respeto de los límites de tasa (cola con reintentos y espera; con ≈10 peticiones/min, 100 imágenes tardan ≈10 min). | Si los límites o las condiciones de la capa gratuita cambian (se comprueban de nuevo en F0). |

## 4. Cambios previstos por área

**Servicio `embedder`** (`backend/app/embedding_server.py`)
- Modos `text` (por defecto) y `media`; `POST /v1/admin/mode` (solo red interna, clave si está definida) y retorno automático a `text` tras `EMBED_SERVER_MEDIA_IDLE_S` (por defecto 300 s). `/health` informa del modo y la memoria.
- `POST /v1/media/embeddings`: `{items: [{modality, path, start_ms?, end_ms?, text?}]}` → vectores de 768 dimensiones (normalizados). Las rutas se **resuelven dentro de un directorio raíz** (`/data/media`), sin `..` ni enlaces simbólicos fuera de la raíz.
- Límites: nº de elementos por petición, duración máxima por ventana, presupuesto de tokens visuales configurable (70–1 120), una inferencia a la vez (como hoy).

**Base de datos** (migración `0005`)
- `documents`: `media_type` (`pdf` por defecto · `image` · `audio` · `video`), `mime`, `duration_ms`, `width`, `height`. `storage_name` conserva la extensión correcta.
- `document_chunks`: `modality` (`text` por defecto · `image` · `audio` · `video` · `page_image`), `source` (`pdf_text` · `ocr` · `asr` · `caption` · `media`), `start_ms`, `end_ms`. Con comprobaciones: audio/vídeo exigen marcas temporales. Para medios sin páginas se usa `page_start = page_end = 1` (compatibilidad); `text` puede ser vacío en los fragmentos de vectores de medios.
- `message_citations`: `modality`, `start_ms`, `end_ms` (el `snapshot` guarda la transcripción/OCR/descripción vigente al responder).
- Los vectores van a `chunk_embeddings` sin cambios de esquema.

**Ingesta** (`services/ingestion.py` y nuevos `media_extract.py`, `transcribe.py`)
- Detección por **firma de archivo** (no por extensión) y nuevos tipos admitidos en `POST /documents`; límites propios por tipo (tamaño, duración, píxeles).
- Imagen: validar/decodificar con límite de píxeles, miniatura sin EXIF, OCR (Tesseract ya instalado) → fragmento de texto, vector de imagen.
- Audio/vídeo: normalizar con **ffmpeg** (mono, 16 kHz) en un subproceso aislado (`-nostdin`, tiempo límite, sin protocolos de red), ventanas, transcripción con marcas, vectores por ventana; vídeo: fotogramas a 1 fps, miniaturas clave.
- Trabajos idempotentes y reanudables como los actuales (`processing_jobs`); progreso por ventana; el worker ordena los trabajos de medios para **cambiar de modo una sola vez** por lote.
- Dependencias a confirmar en F0: `ffmpeg` (binario, como proceso aparte), decodificación de audio/vídeo para el procesador (p. ej. PyAV/soundfile/librosa), `faster-whisper`.

**Recuperación y respuestas** (`retrieval.py`, `answering.py`, `citations.py`)
- Filtro opcional por modalidad; RRF entre ramas (vector de todas las modalidades + FTS de texto/derivados); umbral por modalidad.
- Expansión por ventana de tiempo; el contexto del LLM contiene **solo texto** con la marca temporal en la etiqueta de la cita (p. ej. «[C2] audio.mp3 12:30–13:00»); los aciertos sin texto van a «Medios relacionados» (M3).
- Verificación de citas sin cambios de principio: etiqueta efímera → `chunk_id` recuperado; la cita guarda modalidad y tiempos.

**API**: `GET /documents/{id}/media` con soporte de **Range** (reproducción y salto), `GET /documents/{id}/thumbnail`; `POST /search` y el chat aceptan `modalities`. Todo por **identificador**, nunca por nombre de archivo.

**Interfaz** (`frontend/`)
- Subida: imágenes, audio y vídeo (tipos y límites visibles).
- Visores: imagen con zoom; **reproductor de audio/vídeo que salta al segundo citado** y resalta el segmento de transcripción; el visor de PDF se mantiene.
- Resultados y citas con miniatura/duración/hora; panel «Medios relacionados»; filtros por modalidad.
- Ajustes → Modelo de embeddings: umbral por modalidad y estado del modo `media` (memoria, tiempo de carga).

## 5. Fases (cada una termina con criterios verificables)

**F0 — Verificación técnica y medidas (puerta de decisión). Parte hecha: ver §5b.** En un contenedor desechable y con límite de memoria, sin tocar la instalación en uso:
1. Cargar el modelo en modo solo texto, texto+imagen, texto+audio y completo, en `float32` y `bfloat16`; medir **RAM pico y tiempo de carga**.
2. Medir **segundos por imagen, por ventana de 30 s de audio y de vídeo** en esta CPU (i7-13620H).
3. Comprobar que el vector de texto es **el mismo** con y sin codificadores (coseno ≈ 1) y que bf16 ≈ fp32.
4. Comprobar la **coherencia entre modalidades** con medios sintéticos (imágenes con Pillow; audio/vídeo con ffmpeg; voz con un sintetizador si hay uno disponible sin descargas grandes) y con los medios que aporte el usuario.
5. Identificar las dependencias exactas de decodificación y probar **faster-whisper** en español (tiempo y calidad).
6. Medir el **pico de memoria del equipo** con todo en marcha.
7. **Probar los modelos de visión candidatos** (locales y, si el usuario lo acepta, una API gratuita con imágenes no sensibles) (`qwen3-vl:4b`, `qwen3-vl:2b`, `gemma3:4b` y `gemma4:e2b`, esta última de la misma familia que EmbeddingGemma 2 pero con descarga de 4,6–7,5 GB según la página de Ollama: puede no caber entera en la GPU) con imágenes reales del usuario: calidad de la descripción **en español**, descripciones inventadas, lectura de texto en la imagen, segundos por imagen, VRAM y tiempo de intercambio con `qwen3:8b`; elegir el más pequeño que cumpla.
*Salida:* tabla de medidas y decisión **sí / sí con límites / no**. Criterios propuestos de viabilidad: el modo `media` cabe en RAM (con bf16 si hace falta) sin degradar el equipo, y indexar 1 minuto de audio no cuesta más de ≈2 minutos de CPU. Si no se cumplen, se limitan los medios (p. ej. solo imágenes, o duración máxima menor) o se pospone hasta tener más memoria o GPU.

**F1 — Servicio `embedder` multimodal.** *(Hecha.)* Modos, endpoint de medios por referencia, descarga/unload, límites y pruebas (motor simulado + pruebas con el modelo real marcadas `gemma2_media`). *Salida:* vectores de imagen/audio/vídeo de 768 dimensiones; el servicio vuelve a solo texto por inactividad; ninguna ruta fuera de `/data/media`.

**F2 — Imágenes de extremo a extremo.** *(Hecha.)* (a) Migración `0005`, subida y validación, miniaturas, OCR, vector de imagen, recuperación con modalidad, visor y «Medios relacionados». (b) **Descripción con el modelo de visión:** bloque en Ajustes, `images` en la capa de proveedores, descripción automática al ingerir (fragmento `caption`, con el nombre del modelo en los metadatos), acción «¿Qué ves en esta imagen?» y preguntas sobre las imágenes recuperadas. *Salida:* una consulta en texto encuentra la imagen correcta en el conjunto de evaluación (§7) por encima del umbral calibrado; la descripción de cada imagen de prueba es correcta en lo esencial y **no inventa objetos o texto** (revisada a mano, con tasa de errores medida); el usuario ve la imagen junto a la descripción para poder comprobarla.

**F2c — Imágenes dentro de PDF** *(HECHA y verificada: ver §0)* (se hace tras F2, reutiliza su canal). Hoy el extractor **ignora las imágenes incrustadas** (`pdf_extract.py` salta los bloques de imagen) y solo hace OCR de una página cuando casi no tiene texto: una foto, un gráfico o un esquema dentro de un PDF con texto no se lee. Se añade: (1) extraer las imágenes incrustadas de cada página (PyMuPDF) descartando las **decorativas** (logos, líneas, imágenes diminutas o repetidas en muchas páginas, por tamaño y huella); (2) para **gráficos y tablas dibujados como vectores** (que no son imágenes incrustadas), renderizar la región o la página a imagen solo si la página contiene dibujos significativos; (3) cada imagen pasa por el mismo canal que una imagen suelta (OCR, descripción con el modelo de visión, vector) y queda ligada a **documento + página + posición**; (4) la cita apunta a la página del PDF y el visor resalta la figura. Opcional por documento y con límite de imágenes por PDF (coste de CPU/API). *Salida:* en un PDF de prueba con una figura, la pregunta «¿qué muestra la figura de la página N?» devuelve la descripción con cita a esa página; las imágenes decorativas no se indexan.

**F3 — Audio.** *(Descartada: el sistema solo recibe PDF.)* ffmpeg, ventanas, transcripción con marcas, vectores, reproductor con salto y resaltado, citas con tiempo. *Salida:* una pregunta sobre lo dicho devuelve respuesta con cita «archivo mm:ss» que, al pulsarla, reproduce ese tramo; abstención correcta en preguntas sin respuesta.

**F4 — Vídeo.** *(Descartada.)* Fotogramas, ventanas, pista de audio reutilizando F3, miniaturas. *Salida:* búsqueda por contenido visual y por lo dicho; comparación documentada de «vídeo+audio en un vector» frente a vectores separados.

**F5 — Búsqueda y respuestas multimodales unificadas.** Filtros por modalidad, umbrales por modalidad, expansión temporal, abstención, panel de relacionados; opcional: nivel 2 con LLM de visión. *Salida:* evaluación mixta (texto+imagen+audio+vídeo) sin empeorar la evaluación actual de PDF (recall@5 híbrido 1,00; 15/15 y 7/7).

**F6 — (Opcional) Páginas de PDF como imágenes.** Vector de cada página para encontrar tablas y gráficos que el texto no capta (la ficha da NDCG@5 67,8 en documentos visuales). Opcional por documento por su coste en CPU (121 páginas × 280 tokens).

**F7 — (Opcional) Consulta con imagen o con voz.** *(Descartada.)* «Buscar similares» subiendo una imagen y consulta hablada grabada en el navegador (el modelo admite consultas de audio).

**F8 — Endurecimiento y documentación.** Pruebas de seguridad, límites, copia de seguridad/restauración con medios, `ARCHITECTURE`, `OPERATIONS`, `README`, `LICENSES`.

## 5b. Resultados de la Fase 0 (2026-10-19, esta máquina: i7-13620H, 8 hilos de PyTorch en el contenedor, solo CPU)

Medidos en un contenedor desechable con la imagen del `embedder`, sin tocar la instalación en uso. **Medios sintéticos** (formas, texto y barras dibujados con Pillow; ruido+tono de audio): sirven para tiempos, memoria y comprobar que el canal funciona; **no** miden calidad semántica real (eso lo hará el conjunto de evaluación con medios reales).

| Medida | Resultado |
|--------|-----------|
| **Vector de texto con y sin codificadores** | **Idéntico** (coseno ≥ 0,99996 entre solo texto y completo; 0,99995 entre bf16 y fp32): M1 queda verificado, los vectores de texto ya indexados siguen valiendo |
| Memoria, pico de RSS en fp32 | texto **2,0 GB** · texto+imagen **3,0 GB** · texto+audio **3,7 GB** · completo **4,7 GB** (la prueba en bf16 llegó a 3,2 GB) |
| Imagen, fp32, por imagen | **3,3 s** (280 tokens, por defecto) · **1,5 s** (140) · **0,7 s** (70) |
| Imagen, bf16 | 10,9 s por imagen (≈3× peor que fp32) |
| Audio, ventana de 30 s | **2,4 s en fp32** (≈12× más rápido que el tiempo real); 13,5 s en bf16 |
| Vídeo, por fotograma, fp32 | **1,5 s** (140 tokens) · **0,7 s** (70) → un vídeo de 1 h a 1 fps: ≈ 1,5 h con 140 tokens/≈ 42 min con 70 (en bf16: ≈ 7 h) |
| Búsqueda texto→imagen (sintético) | aciertos 4/4 en los tres presupuestos; similitud de la imagen correcta 0,74–0,84 frente a ≤ 0,62 de las incorrectas y ≤ 0,57 de una consulta sin imagen correspondiente («un gato durmiendo») → umbral inicial orientativo ≈ 0,65 para imágenes (a calibrar con imágenes reales) |
| Formatos de entrada aceptados | imagen: `PIL.Image` (o lista); audio: `numpy` mono 16 kHz; vídeo: lista de fotogramas `PIL.Image` (para muestrear a 1 fps hay que pasar el fps) |

**Conclusiones:** (1) viable en esta máquina **con fp32 y por modos**; (2) imágenes y audio son baratos (segundos); (3) **el vídeo es el caro**: usar presupuesto visual reducido (70–140) y menos de 1 fotograma por segundo, y limitar la duración; (4) la memoria obliga a que el servicio tenga un tope de **≈4–5 GB solo mientras se indexan medios** (el tope actual de 3 GB sirve para el modo `text`); (5) no hay modelo de visión en el Ollama del usuario (solo `qwen3`, `llama3.1`, `DeepSeek-R1`, `qwen2.5-coder`): la descripción de imágenes será por API (Gemini/Groq) hasta que el usuario autorice descargar uno local. **Pendiente de F0:** Whisper (descarga) y la calidad de descripción con imágenes reales.

## 6. Seguridad y privacidad

- **Archivos no confiables:** validar por firma, límites de tamaño/duración/píxeles (contra bombas de descompresión), ffmpeg en subproceso con tiempo límite, sin acceso a red ni protocolos remotos, usuario sin privilegios. Nombres generados por el servidor (como los PDF).
- **Metadatos:** las miniaturas se generan sin EXIF (GPS); no se muestran metadatos personales.
- **Entrega de medios:** solo por identificador, con soporte de Range, tipo de contenido fijo y sin interpretación de HTML/SVG (no admitir SVG en la primera versión).
- **Descripciones del VLM:** son **automáticas y pueden equivocarse o inventar detalles**: se etiquetan como «descripción automática» (con el modelo usado), se muestran junto a la imagen y no se presentan como hecho verificado.
- **Texto derivado no confiable:** transcripciones y OCR pueden contener instrucciones: se delimitan como el texto de los PDF y se añaden pruebas de inyección con ese origen.
- **Datos sensibles:** voz y caras son biométricos. Los medios no salen del equipo; ningún proveedor externo los recibe sin consentimiento específico (§3, M10).
- **Almacenamiento:** los medios ocupan mucho más que los PDF: tope configurable, aviso de espacio y borrado en cascada (archivo, miniaturas, fragmentos, vectores). Las copias de seguridad incluyen los medios (aviso de tamaño).

## 7. Evaluación

Conjunto `eval/media/` (preguntas en JSONL como el actual, con `modality`, `target`, `start_ms`): medios **sintéticos** para el arnés (generados en el repositorio) más los **del usuario** para la calibración real. Métricas por modalidad: recall@1/3/5 y MRR; para audio/vídeo, acierto si el tramo devuelto **solapa** el esperado; negativas difíciles (consultas sin respuesta) para fijar umbral y abstención; consultas en español e inglés. **Línea base obligatoria**: búsqueda solo por transcripción (FTS) frente a vectores de audio, para demostrar que los vectores aportan algo. Criterios de adopción por modalidad: recall@5 mínimo acordado con el usuario tras F0 y **sin regresión** en la evaluación de PDF.

## 8. Riesgos

| Riesgo | Mitigación |
|--------|------------|
| **Memoria:** el modo completo no cabe junto al resto (≈0,7 GB libres; texto solo ya usa ≈1,9 GiB) | Modo `media` bajo demanda con bf16, volver a solo texto por inactividad, límite de memoria del contenedor, cerrar otras aplicaciones al indexar; si F0 lo descarta, empezar solo por imágenes. |
| **Tiempo de CPU** al indexar vídeo/audio largos (no medido) | Medir en F0; límites de duración; cola con progreso y cancelación; indexado nocturno/en segundo plano. |
| **Calidad desigual:** vídeo es la modalidad más débil (Hit@1 50,7) | Umbrales por modalidad, transcripción como respaldo, comunicar la puntuación; no prometer precisión no medida. |
| **Brecha entre modalidades** (escalas de similitud distintas) | Calibración por modalidad (M9) y fusión por rango (RRF), no por puntuación bruta. |
| **Alucinaciones de Whisper** en silencio/ruido | VAD, umbral de confianza, etiqueta «transcripción automática», transcripción editable en una fase posterior. |
| **Descripciones inventadas** del VLM (objetos o texto que no están) y **espera por el cambio de modelo en la GPU** (qwen3:8b ↔ VLM) | Medir la tasa de errores en F0/F2 con imágenes reales; describir al ingerir (no en cada consulta); mostrar siempre la imagen junto a su descripción; elegir el modelo más pequeño que cumpla. |
| **Seguridad de ffmpeg/decodificadores** | Subproceso aislado con límites, binarios de la distribución actualizados, tipos admitidos acotados. |
| **Licencias:** ffmpeg de Debian suele incluir componentes GPL (verificarlo con `ffmpeg -version`); Whisper/faster-whisper (MIT); pesos de Gemma | Usar ffmpeg **como proceso externo** (no enlazado) y documentarlo en `docs/LICENSES.md`; confirmar la licencia efectiva de los pesos y de cada dependencia nueva con sus metadatos. |
| **Crecimiento del almacenamiento y de la base** | Topes, avisos, borrado en cascada, vectores de 768 dimensiones sin truncar (128 degrada la calidad multimodal). |

## 9. Decisiones

**Tomadas por el usuario (2026-10-19):**

0. **El sistema solo recibe PDF** (aclaración posterior que redefine el alcance: ver §0). Audio, vídeo y Whisper se descartan.

1. **Orden de modalidades:** imágenes → audio → vídeo. ✔
2. **Transcripción:** se añade **Whisper** (local, vía faster-whisper). ✔
3. **Imágenes:** el sistema debe **detectarlas y decir qué ve** → modelo de visión (M3 y M11). ✔

4. **Modelo de visión por API:** se integran **Gemini y Groq** (junto al local). ✔
5. **Imágenes dentro de PDF:** deben poder leerse y describirse (F2c). ✔

**Pendientes (con propuesta, se pueden ajustar tras la Fase 0):**

6. **Límites por archivo** (qué se acepta al subir, para que un archivo enorme no deje el sistema colgado ni llene el disco). Propuesta inicial, configurable en `.env`: imagen ≤ 20 MB y ≤ 40 megapíxeles; audio ≤ 200 MB y ≤ 2 h; vídeo ≤ 500 MB y ≤ 1 h; espacio total para medios ≤ 20 GB. Referencia actual: los PDF admiten 100 MB. Los valores definitivos de duración dependen de lo que mida la Fase 0 (cuánta CPU cuesta cada minuto).
7. **PDF visual (F6):** encontrar tablas y gráficos por imagen de página aunque cueste CPU. Propuesta: dejarlo para el final y opcional.
8. **Recursos:** mientras se indexan medios el sistema necesita más memoria; si no se puede liberar, se limita el alcance (p. ej. archivos más cortos).

## 10. Esfuerzo relativo (orientativo)

F0 pequeño · F1 mediano · F2 mediano · F3 grande (transcripción + reproductor) · F4 mediano (reutiliza F3) · F5 mediano · F6/F7 pequeños-medianos opcionales · F8 mediano. La Fase 0 decide si el resto se hace completo, acotado o se pospone.
