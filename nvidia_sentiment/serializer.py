"""Serialización y deserialización del dataset procesado en JSON UTF-8."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_dataset(posts: list[dict], path: Path) -> None:
    """Serializa lista de posts a JSON UTF-8.

    Args:
        posts: Lista de posts como dicts.
        path: Ruta del archivo de destino.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)


def load_dataset(path: Path) -> list[dict]:
    """Deserializa dataset desde JSON.

    Si el archivo no existe, loguea WARNING y retorna [].
    Si el JSON está corrupto, loguea ERROR y retorna [].

    Args:
        path: Ruta del archivo JSON a cargar.

    Returns:
        Lista de posts como dicts, o [] en caso de error.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Dataset no encontrado: %s", path)
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("Dataset corrupto en %s: %s", path, exc)
        return []
