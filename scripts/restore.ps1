# Restaura una copia creada por backup.ps1. DESTRUCTIVO: reemplaza la BD y los PDFs actuales.
# Uso: .\scripts\restore.ps1 -Src backups\<fecha>
param([Parameter(Mandatory)][string]$Src)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$abs = (Resolve-Path $Src).Path
if (-not ((Test-Path "$abs\db.dump") -and (Test-Path "$abs\storage.tgz"))) { throw "Faltan db.dump/storage.tgz en $Src" }
if ((Read-Host "Esto reemplazará la base de datos y los archivos actuales. Escriba SI para continuar") -ne "SI") { throw "Cancelado" }
Get-Content .env | Where-Object { $_ -match '^\s*[A-Z_]+=' } | ForEach-Object {
  $k, $v = $_ -split '=', 2; Set-Item -Path "env:$k" -Value $v.Trim('"')
}
docker compose stop api worker 2>$null
docker compose cp "$abs\db.dump" db:/tmp/db.dump
docker compose exec -T db psql -U $env:POSTGRES_USER -d postgres -c "DROP DATABASE IF EXISTS `"$env:POSTGRES_DB`" WITH (FORCE)" -c "CREATE DATABASE `"$env:POSTGRES_DB`""
docker compose exec -T db pg_restore -U $env:POSTGRES_USER -d $env:POSTGRES_DB --no-owner /tmp/db.dump
docker run --rm -v pdfanalyst_appdata:/data -v "${abs}:/backup" alpine sh -c "rm -rf /data/storage && tar xzf /backup/storage.tgz -C /data && chown -R 1000:1000 /data/storage"
docker compose up -d api worker
Write-Host "Restauración completada."
