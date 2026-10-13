"""Lista la licencia declarada de las dependencias directas (metadatos del paquete instalado)."""
import re, sys, tomllib
from importlib import metadata

deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"] + ["torch", "alembic", "pytest"]
seen = set()
for d in deps:
    name = re.split(r"[\[<>=!~ ]", d, 1)[0]
    if name.lower() in seen:
        continue
    seen.add(name.lower())
    try:
        m = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        print(f"{name:24} (no instalado)"); continue
    lic = m.get("License-Expression") or m.get("License") or ""
    lic = lic.strip().splitlines()[0][:60] if lic.strip() else ""
    cls = [c.split("::")[-1].strip() for c in m.get_all("Classifier") or [] if c.startswith("License ::")]
    print(f"{name:24} {metadata.version(name):12} {lic or '; '.join(cls) or '?'}")
