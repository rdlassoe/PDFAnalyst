# Historial de cambios

Cronología de lo construido y, sobre todo, de **lo que se descubrió al verificar** (varias decisiones de diseño cambiaron por datos
reales). Fecha de referencia: 2026-10-13. Las decisiones numeradas (D1–D16) están en [ARCHITECTURE.md](ARCHITECTURE.md).

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

## Configuración y documentación

- El `docker-compose.yml` ahora pasa a los contenedores todos los parámetros ajustables (antes solo algunos, de modo que, p. ej.,
  `RETRIEVAL_MIN_SIMILARITY` en `.env` no tenía efecto). Verificado con un override real. Referencia en OPERATIONS §10.
- Documentación reescrita donde estaba obsoleta: modelo de datos, pipelines, abstención, FTS, licencias de modelos, estado por fase.

## Pendiente conocido

- Evaluar con más documentos reales (técnicos/legales, con tablas) y preguntas de usuarios reales.
- Rendimiento a escala (> 50 000 fragmentos) y recall de HNSW.
- Verificación con servicios reales de Groq, Gemini y OpenAI.
- Pruebas automáticas del frontend; autenticación y multiusuario; historial de conversación en el prompt; extracción de tablas.
