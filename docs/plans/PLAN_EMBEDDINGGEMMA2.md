# Plan: EmbeddingGemma 2 como modelo de embeddings (local, seleccionable en Ajustes)

Estado a 2026-10-14: **implementado y verificado en este equipo, en local.** Está activo como modelo de embeddings predeterminado en la instalación de desarrollo.
(Historial de decisiones: la primera versión de este plan proponía Ollama; resultó imposible en este equipo (§2). La segunda lo trató como servicio externo por el hardware; **fue rectificada**:
se ejecuta en local.)

## 1. Decisiones tomadas

| Decisión | Resultado |
|----------|-----------|
| ¿Cómo se elige el modelo? | **Un modelo predeterminado en Ajustes** (no hay selector por consulta). |
| ¿Con qué se indexa? | **Únicamente con el modelo seleccionado.** Los vectores de otros modelos se conservan (volver es inmediato) y pueden borrarse; no se indexa en paralelo. |
| ¿Modo «combinado»? | **Excluido** por ahora. |
| ¿Dónde corre EmbeddingGemma 2? | **En local** (decisión del usuario, que rectificó la anterior de usarlo como servicio externo): servicio `embedder` del `docker-compose.yml`, en CPU, solo texto. |
| ¿Cómo se sirve? | sentence-transformers detrás de una API `/v1/embeddings` de OpenAI **propia** (`backend/app/embedding_server.py`), porque Ollama no sirve (§2). Un único proceso: la API y el worker lo usan por HTTP. |
| ¿Sigue pudiendo usarse un servidor externo? | Sí, con la plantilla genérica; entonces exige consentimiento (envía el texto íntegro de los documentos). No es el camino recomendado. |

## 2. Qué es el modelo (datos verificados)

| Dato | Valor |
|------|-------|
| Identificador HF | `google/embeddinggemma-2` |
| Tamaño | 740M completo (multimodal); **≈271M en modo solo texto** (verificado al cargarlo con `vision_config=None`, `audio_config=None`) |
| Dimensión | 768; Matryoshka 128, 256, 512 o 768 (el servicio acepta `dimensions` y re-normaliza) |
| Contexto / idiomas | 8 192 tokens (el servicio limita a 2 048 para acotar la memoria; tus fragmentos ≈ 300) · más de 100 idiomas |
| Licencia | **Apache-2.0** según los metadatos de la ficha de HF. El repositorio no incluye un archivo `LICENSE` y su README enlaza una página de licencia de Google: léala antes de redistribuir los pesos (ver `docs/LICENSES.md`) |
| Precisión | **`float16` no admitido** (el servicio lo rechaza); `float32` (por defecto) o `bfloat16` |
| Prefijos oficiales | consulta `task: search result \| query: ` · documento `title: none \| text: ` (verificados contra los `prompts` del propio modelo; el servicio no los añade: los añade el perfil) |
| Dependencias | transformers 5.19 + sentence-transformers 6.1; **requiere `torchvision`** incluso en modo solo texto (su procesador lo importa) |
| Ollama | **No sirve aquí:** las compilaciones de `embeddinggemma-2` exigen MLX (solo Apple Silicon); se probaron dos etiquetas |

> **Ampliación posterior:** el mismo servicio `embedder` también vectoriza las **figuras de los PDF** (modo imagen, ≈3,0 GB de RAM mientras indexa; vuelve a solo texto tras 300 s). Medidas, decisiones y alcance final en [PLAN_MULTIMODAL.md](PLAN_MULTIMODAL.md) y ARCHITECTURE D19–D21.

## 3. Qué se implementó

Detalle técnico en [ARCHITECTURE.md](../ARCHITECTURE.md) (D17 y D18), [OPERATIONS.md §13](../OPERATIONS.md) y el README.

- **Servicio `embedder`** (compose, sin puerto en el host, tope de memoria 3 GB, comprobación de salud): `/v1/embeddings`, `/v1/models`, `/health`.
- **Plantilla «EmbeddingGemma 2 (local…)»** en Ajustes (la primera): URL `http://embedder:8000/v1`, prefijos oficiales, sin clave, umbral de similitud 0,72. `embedder` se trata como host local.
- **Umbral de similitud por perfil** (migración `0004`): necesario porque la escala de similitudes de Gemma difiere de la de e5 (§5).
- Todo el módulo de embeddings de D17: perfiles con clave cifrada, *Probar conexión*, vista previa, activación con reindexado solo de lo que falta, tabla de vectores por modelo, vuelta inmediata.

## 4. Puesta en marcha

```bash
docker compose run --rm embedder python -m app.embedding_server --download   # ≈1,5 GB, una vez
docker compose up -d --build
```

Ajustes → Modelo de embeddings → Añadir proveedor → plantilla local → Guardar → *Usar por defecto*. Para volver a e5 basta activar «Entorno (.env)» (inmediato; sus vectores se conservaron).

## 5. Resultados medidos (este equipo: i7-13620H, 15,7 GB de RAM, CPU para embeddings)

| Medida | Valor |
|--------|-------|
| Carga del modelo (primera petición) | ≈9 s |
| Memoria del servicio | **≈1,7–1,9 GiB** (la API y el worker, sin e5 cargado: ≈100 MB cada uno) |
| Indexar 274 fragmentos (libro de 121 págs.) | ≈83 s (≈3 fragmentos/s) |
| Híbrido recall@3 / @5 / MRR | **1,00 / 1,00 / 0,78** (e5: 0,93 / 1,00 / 0,72) |
| Vector puro recall@3 / @5 / MRR | 0,73 / 0,87 / 0,71 (e5: 0,80 / 0,93 / 0,77) — **peor** que e5 |
| Similitud con / sin respuesta | 0,722–0,893 / 0,601–0,714 → separables (e5 no lo era) → umbral 0,72 |
| Extremo a extremo con el LLM real | 15/15 respondidas con cita correcta, 7/7 abstenciones |
| Pruebas automáticas | 240 (3 con el modelo real) |

Criterios de adopción fijados de antemano: recall@5 híbrido no peor que 1,00 ✔; recall@1/MRR no peor en más de 0,05 ✔; e2e 15/15 y 7/7 ✔; RAM libre ≥ ≈2 GB **✘ en este equipo**
(con todo en marcha quedaban ≈0,7 GB libres en Windows; el servicio ocupa ≈1,9 GiB). La adopción se mantiene porque el usuario pidió explícitamente ejecutarlo en local, pero es el punto débil:
véase §6.

Detalle y límites de la evaluación: [eval/RESULTS.md](../../eval/RESULTS.md).

## 6. Riesgos y qué hacer

- **Memoria ajustada.** Si el equipo se queda corto: `EMBED_SERVER_DTYPE=bfloat16` (pesos a la mitad; no medido aquí, puede cambiar la velocidad en CPU), bajar `EMBED_SERVER_MAX_SEQ_LENGTH`, o parar el servicio
  (`docker compose stop embedder`) y volver al modelo «Entorno» (e5, que se carga dentro del proceso de la API/worker).
- **Muestra pequeña.** Un libro, 22 preguntas escritas por quien desarrolló el sistema. La mejora del híbrido es evidencia de que no empeora, no de que sea mejor con tus documentos: evalúa con ellos (`eval/README.md`).
- **El vector puro es peor que e5**; la ganancia depende de la combinación con el texto completo (modo por defecto). Si se desactivara `HYBRID_ENABLED`, e5 sería preferible.
- **Latencia de la primera petición** tras arrancar (carga del modelo): `EMBED_SERVER_PRELOAD=true` la evita a costa de memoria permanente.
- **Licencia:** confirmar la de los pesos si se va a redistribuir (§2).
- **No verificado:** `bfloat16` en CPU, GPU, colecciones grandes (> 50 000 fragmentos), otros documentos.
