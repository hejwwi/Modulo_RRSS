"""Filtro de posts relevantes para NVIDIA desde el dataset histórico."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NVDA_TERMS: set[str] = {"nvidia", "nvda", "$nvda", "geforce", "rtx", "cuda"}


def filter_nvda_posts(posts: list[dict]) -> list[dict]:
    """Filtra posts que contengan al menos un término NVDA en title o selftext.

    Los términos buscados son: nvidia, nvda, $nvda, geforce, rtx, cuda
    (case-insensitive). Campos ausentes se tratan como cadena vacía.
    Preserva todos los campos originales del post.

    Args:
        posts: Lista de posts como dicts.

    Returns:
        Subconjunto de posts que mencionan NVIDIA.
    """
    result = []
    for post in posts:
        title = (post.get("title") or "").lower()
        selftext = (post.get("selftext") or "").lower()
        combined = title + " " + selftext
        if any(term in combined for term in NVDA_TERMS):
            result.append(post)

    if not result:
        logger.warning("No se encontraron posts relevantes para NVIDIA.")

    return result


def select_sample(posts: list[dict], n: int) -> list[dict]:
    """Retorna los N posts más recientes ordenados por created_utc descendente."""
    sorted_posts = sorted(posts, key=lambda p: p.get("created_utc", 0), reverse=True)
    return sorted_posts[:n]


def select_sample_in_range(
    posts: list[dict],
    n: int,
    date_min: str,
    date_max: str,
) -> list[dict]:
    """Retorna hasta N posts dentro del rango [date_min, date_max] distribuidos uniformemente.

    Args:
        posts: Lista de posts como dicts.
        n: Número máximo de posts a seleccionar.
        date_min: Fecha mínima en formato 'YYYY-MM-DD'.
        date_max: Fecha máxima en formato 'YYYY-MM-DD'.

    Returns:
        Hasta N posts dentro del rango, ordenados por fecha ascendente.
    """
    in_range = [
        p for p in posts
        if date_min <= str(p.get("date", ""))[:10] <= date_max
    ]
    if not in_range:
        logger.warning(
            "No hay posts en el rango %s – %s. Usando select_sample normal.",
            date_min, date_max,
        )
        return select_sample(posts, n)

    # Ordenar por fecha y distribuir uniformemente si hay más de n
    in_range.sort(key=lambda p: str(p.get("date", "")))
    if len(in_range) <= n:
        return in_range

    step = len(in_range) / n
    return [in_range[int(i * step)] for i in range(n)]
