# Evaluación

`python -m app.evaluate eval/questions.jsonl` mide, sobre los documentos **ya cargados** en la aplicación, la calidad de la
recuperación en tres modos (vector / texto completo / híbrido): recall@1/3/k, MRR, tasa de respuesta y de abstención
correcta, y sugiere un umbral `RETRIEVAL_MIN_SIMILARITY` a partir de la separación entre preguntas con y sin respuesta.

Formato (una línea JSON por pregunta):

```json
{"id": "q1", "question": "¿Cuál es el plazo de prescripción?", "document": "ley.pdf", "pages": [12, 13], "answerable": true}
{"id": "q2", "question": "¿Quién ganó el mundial de 1998?", "document": "-", "pages": [], "answerable": false}
```

- `pages` son páginas base 1 donde está la evidencia (basta con acertar una).
- Incluya al menos ~20 % de preguntas `answerable: false`, parecidas en estilo a las reales (esas calibran la abstención).
- Cargue antes los PDF desde la interfaz o la API; el evaluador usa los que estén en estado `ready` y avisa de los que falten.

```bash
docker compose cp eval/questions.jsonl api:/tmp/q.jsonl
docker compose exec api python -m app.evaluate /tmp/q.jsonl --json /tmp/resultado.json
```

`questions.synthetic.jsonl` y `corpus/` contienen un corpus **sintético** (generado en `backend/tests/corpus.py`) solo para
comprobar que el arnés funciona. **No sustituye** la evaluación con documentos reales del usuario, que es lo que calibra el
umbral de abstención de verdad.

Resultados sobre un documento real y análisis: [RESULTS.md](RESULTS.md) (`questions.babilonia.jsonl`, `e2e.py`).
