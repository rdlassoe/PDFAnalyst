#!/usr/bin/env bash
# Copia de seguridad: volcado de PostgreSQL (formato custom) + PDFs originales.
# Uso: ./scripts/backup.sh [directorio_destino]     (por defecto ./backups/<fecha>)
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
dest="${1:-backups/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$dest"
docker compose exec -T db pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc > "$dest/db.dump"
docker run --rm -v pdfanalyst_appdata:/data:ro -v "$(pwd)/$dest":/backup alpine \
  tar czf /backup/storage.tgz -C /data storage
echo "Copia creada en $dest  (db.dump, storage.tgz)"
