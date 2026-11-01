"""
Lee config.yaml una sola vez. Si no esta pyyaml o no existe el archivo, devuelve {} y cada
modulo usa sus valores por defecto (asi nada se rompe por la configuracion).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

RUTA_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@lru_cache(maxsize=None)
def cargar_config(ruta: str | None = None) -> dict:
    try:
        import yaml
        with open(ruta or RUTA_CONFIG, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (ImportError, OSError):
        return {}
