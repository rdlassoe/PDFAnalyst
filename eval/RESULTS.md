# Resultados de evaluación con un documento real

**Documento:** *El hombre más rico de Babilonia* (G. S. Clason, PDF descargado por el usuario; 121 páginas, 268 fragmentos,
2 páginas procesadas con OCR). **Preguntas:** `questions.babilonia.jsonl` — 15 con respuesta (paráfrasis, términos exactos,
preguntas en inglés sobre texto en español, datos numéricos) y 7 sin respuesta (5 de ellas "negativas difíciles" del mismo
dominio financiero, 1 inyección/meta-pregunta). Modelos: `multilingual-e5-small` + `qwen3:8b`. Fecha: 2026-10-13.

La evidencia de cada pregunta se localizó buscando el texto en el PDF (no de memoria). En dos ocasiones el sistema citó una
página correcta que mi lista inicial omitía (la misma historia aparece en varias páginas); la lista se corrigió después y
conviene recordar que **las páginas esperadas son "al menos una de estas"**.

## Recuperación (`python -m app.evaluate`) — antes → después de ponderar términos por rareza (IDF)

| Modo | recall@3 | recall@5 | MRR |
|------|----------|----------|-----|
| vector | 0,80 → 0,80 | 0,87 → 0,93* | 0,71 → 0,77* |
| texto completo | 0,73 → 0,87 | 0,73 → 0,87 | 0,56 → 0,74 |
| **híbrido (RRF)** | 0,73 → **0,93** | 0,87 → **1,00** | 0,60 → 0,72 |

\* el vectorial no cambió de código: la mejora viene de haber completado las páginas esperadas de b5 y b10.
El híbrido supera al vectorial en recall@3/@5 (lo que importa: el LLM recibe los 6 mejores fragmentos) pero **no** en recall@1.

## Extremo a extremo con LLM real (`python eval/e2e.py`)

| | Antes | Después |
|--|-------|---------|
| Preguntas con respuesta contestadas | 13/15 (1 error 500 de Ollama, 1 abstención errónea) | **15/15** |
| …con cita en una página que contiene la evidencia | 11/15 (+2 aciertos mal etiquetados) | **15/15** (tras verificar b4 y b5, b10 a mano) |
| Preguntas sin respuesta → abstención | 6/7 (la meta-pregunta filtró el prompt) | **7/7** |

## Hallazgos y cambios

1. **El umbral de similitud coseno no separa preguntas con y sin respuesta**: la peor sin respuesta llega a 0,861 y la peor
   con respuesta es 0,819 (e5-small comprime los scores). La abstención fiable en este corpus la aporta el LLM
   (`NO_ENCONTRADO`), no el umbral. No suba el umbral para "arreglarlo": penalizaría las preguntas parafraseadas o en otro idioma.
2. Palabras muy frecuentes del dominio ("oro", "bolsa", "manera") dominaban el ranking textual → ahora solo guían la consulta
   los términos poco frecuentes (`RETRIEVAL_COMMON_TERM_RATIO`), y un término raro ausente del corpus baja la cobertura.
3. Ollama devolvía 500 transitorios al cargar el modelo → 3 reintentos con espera.
4. Una pregunta sobre "tus instrucciones" hacía que el modelo citara un fragmento irrelevante → regla 6 en el prompt.

## Límites de esta evaluación

Un solo libro, 22 preguntas redactadas por quien desarrolló el sistema, un solo modelo de generación y una sola ejecución
(el LLM no es determinista). Indica que el flujo funciona sobre texto real, **no** estima precisión con significación
estadística. Ampliar con más documentos (técnicos/legales, con tablas) y preguntas escritas por usuarios reales.
