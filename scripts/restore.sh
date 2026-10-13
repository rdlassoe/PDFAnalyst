#!/usr/bin/env bash
# Restaura una copia creada por backup.sh. DESTRUCTIVO: reemplaza la BD y los PDFs actuales.
# Uso: ./scripts/restore.sh backups/<fecha>
set -euo pipefail
cd "$(dirname "$0")/.."
src="${1:?Indique el directorio de la copia}"
[ -f "$src/db.dump" ] && [ -f "$src/storage.tgz" ] || { echo "Faltan db.dump/storage.tgz en $src"; exit 1; }
read -r -p "Esto reemplazará la base de datos y los archivos actuales. Escriba SI para continuar: " ok
[ "$ok" = "SI" ] || { echo "Cancelado"; exit 1; }
set -a; source .env; set +a
docker compose stop api worker >/dev/null 2>&1 || true
docker compose exec -T db psql -U "$POSTGRES_USER" -d postgres -c "DROP DATABASE IF EXISTS \"$POSTGRES_DB\" WITH (FORCE)" -c "CREATE DATABASE \"$POSTGRES_DB\""
docker compose exec -T db pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner < "$src/db.dump"
docker run --rm -v pdfanalyst_appdata:/data -v "$(pwd)/$src":/backup alpine \
  sh -c "rm -rf /data/storage && tar xzf /backup/storage.tgz -C /data && chown -R 1000:1000 /data/storage"
docker compose up -d api worker
echo "Restauración completada."
