# Copia de seguridad: volcado de PostgreSQL (formato custom) + PDFs originales.
# Uso: .\scripts\backup.ps1 [-Dest backups\<fecha>]
param([string]$Dest = "backups\$(Get-Date -Format 'yyyyMMdd-HHmmss')")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
Get-Content .env | Where-Object { $_ -match '^\s*[A-Z_]+=' } | ForEach-Object {
  $k, $v = $_ -split '=', 2; Set-Item -Path "env:$k" -Value $v.Trim('"')
}
New-Item -ItemType Directory -Force $Dest | Out-Null
$abs = (Resolve-Path $Dest).Path
# pg_dump escribe binario: se hace dentro del contenedor y se copia, para evitar que PowerShell altere los bytes
docker compose exec -T db sh -c "pg_dump -U $env:POSTGRES_USER -d $env:POSTGRES_DB -Fc -f /tmp/db.dump"
docker compose cp db:/tmp/db.dump "$abs\db.dump"
docker run --rm -v pdfanalyst_appdata:/data:ro -v "${abs}:/backup" alpine tar czf /backup/storage.tgz -C /data storage
Write-Host "Copia creada en $Dest (db.dump, storage.tgz)"
