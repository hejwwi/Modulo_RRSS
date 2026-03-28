"""Serialización y deserialización del dataset procesado en CSV UTF-8."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columnas que se guardan en el CSV (orden fijo)
CSV_COLUMNS = [
    "id", "date", "created_utc",
    "title", "selftext",
    "score", "num_comments", "has_image",
    "sent_finbert_label", "sent_finbert_pos", "sent_finbert_neg", "sent_finbert_neu",
    "sent_text_only", "sent_multimodal",
    "image_local_path",
]


def save_dataset(posts: list[dict], path: Path) -> None:
    """Serializa lista de posts a CSV UTF-8.

    Solo guarda las columnas definidas en CSV_COLUMNS.
    Columnas ausentes se rellenan con cadena vacía.
    """
    path = Path(path)
    # Cambiar extensión a .csv si se pasa .json
    if path.suffix == ".json":
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in posts:
        row = {col: p.get(col, "") for col in CSV_COLUMNS}
        rows.append(row)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info("Dataset guardado en %s (%d filas)", path, len(df))


def load_dataset(path: Path) -> list[dict]:
    """Deserializa dataset desde CSV.

    Si el archivo no existe, loguea WARNING y retorna [].
    Acepta tanto .csv como .json (intenta CSV primero).
    """
    path = Path(path)

    # Intentar CSV primero, luego JSON como fallback
    csv_path = path.with_suffix(".csv") if path.suffix == ".json" else path
    if not csv_path.exists() and path.suffix == ".json":
        # fallback: intentar JSON original
        import json
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.error("Error leyendo JSON %s: %s", path, exc)
                return []

    if not csv_path.exists():
        logger.warning("Dataset no encontrado: %s", csv_path)
        return []

    try:
        df = pd.read_csv(csv_path, encoding="utf-8", dtype=str).fillna("")
        return df.to_dict(orient="records")
    except Exception as exc:
        logger.error("Error leyendo CSV %s: %s", csv_path, exc)
        return []


def append_posts(new_posts: list[dict], path: Path) -> int:
    """Añade posts nuevos al CSV existente, deduplicando por 'id'.

    Retorna el número de posts realmente añadidos.
    """
    path = Path(path)
    if path.suffix == ".json":
        path = path.with_suffix(".csv")

    existing = load_dataset(path)
    existing_ids = {p.get("id", "") for p in existing}

    to_add = [p for p in new_posts if p.get("id", "") not in existing_ids]
    if not to_add:
        logger.info("No hay posts nuevos para añadir.")
        return 0

    all_posts = existing + to_add
    save_dataset(all_posts, path)
    logger.info("Añadidos %d posts nuevos. Total: %d", len(to_add), len(all_posts))
    return len(to_add)
