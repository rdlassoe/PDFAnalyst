"""Operaciones de administración:  python -m app.cli <comando>

  models                 modelos de embeddings registrados y número de vectores de cada uno
  index-status           índices HNSW existentes
  index-create [--m N --ef-construction N]
                         crea el índice HNSW del modelo ACTIVO (recomendado > ~50 000 chunks)
  index-drop             elimina el índice HNSW del modelo activo (vuelve a búsqueda exacta)
  reindex                tras cambiar EMBEDDING_MODEL: encola `embed` para todos los documentos
  purge-model <id>       borra los vectores de un modelo antiguo (y su índice)
"""

from __future__ import annotations

import argparse
import json
import sys

from sqlalchemy import delete

from .db import SessionLocal, engine
from .models import ChunkEmbedding, EmbeddingModel
from .services import vector_index
from .services.embeddings import get_embedder, get_or_create_model_row
from .services.ingestion import embedding_counts, requeue_all_for_active_model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models")
    sub.add_parser("index-status")
    c = sub.add_parser("index-create")
    c.add_argument("--m", type=int, default=16)
    c.add_argument("--ef-construction", type=int, default=64)
    sub.add_parser("index-drop")
    sub.add_parser("reindex")
    p = sub.add_parser("purge-model")
    p.add_argument("model_id", type=int)
    args = ap.parse_args(argv)

    with SessionLocal() as s:
        if args.cmd == "models":
            active = get_or_create_model_row(s, get_embedder(s))
            s.commit()
            for row in embedding_counts(s):
                row["active"] = row["id"] == active.id
                print(json.dumps(row, ensure_ascii=False))
        elif args.cmd == "index-status":
            print(json.dumps(vector_index.index_status(engine), indent=2))
        elif args.cmd == "index-create":
            model = get_or_create_model_row(s, get_embedder(s))
            s.commit()
            print("creado:", vector_index.create_hnsw_index(engine, model, args.m, args.ef_construction))
        elif args.cmd == "index-drop":
            model = get_or_create_model_row(s, get_embedder(s))
            s.commit()
            vector_index.drop_hnsw_index(engine, model.id)
            print("eliminado índice del modelo", model.id)
        elif args.cmd == "reindex":
            n = requeue_all_for_active_model(s)
            print(f"{n} documentos encolados para generar embeddings con el modelo activo")
        elif args.cmd == "purge-model":
            active = get_or_create_model_row(s, get_embedder(s))
            if args.model_id == active.id:
                print("No se puede purgar el modelo activo.", file=sys.stderr)
                return 2
            vector_index.drop_hnsw_index(engine, args.model_id)
            s.execute(delete(ChunkEmbedding).where(ChunkEmbedding.model_id == args.model_id))
            s.execute(delete(EmbeddingModel).where(EmbeddingModel.id == args.model_id))
            s.commit()
            print("modelo purgado:", args.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
