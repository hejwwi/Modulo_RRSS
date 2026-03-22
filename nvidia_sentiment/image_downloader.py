"""Descargador de imágenes para posts filtrados de NVIDIA."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

KNOWN_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _extract_extension(url: str) -> str:
    """Extrae la extensión de una URL. Si no es reconocida, retorna '.jpg'."""
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix in KNOWN_EXTENSIONS else ".jpg"


def download_post_image(post: dict, images_dir: Path, timeout: int = 30) -> dict:
    """Descarga la primera imagen disponible del post.

    Si image_urls está vacío o ausente, marca image_download_status="no_image".
    En éxito: image_download_status="ok", image_local_path=ruta relativa.
    En fallo (RequestException, timeout): image_download_status="failed", log ERROR.

    Args:
        post: Dict del post con campo image_urls.
        images_dir: Directorio donde guardar las imágenes.
        timeout: Segundos máximos de espera para la descarga.

    Returns:
        El post actualizado (modificado in-place y retornado).
    """
    image_urls = post.get("image_urls") or []

    if not image_urls:
        post["image_download_status"] = "no_image"
        post["image_local_path"] = ""
        return post

    url = image_urls[0]
    post_id = post.get("id", "unknown")
    ext = _extract_extension(url)
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{post_id}{ext}"
    dest = images_dir / filename

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        post["image_download_status"] = "ok"
        post["image_local_path"] = str(dest)
    except requests.RequestException as exc:
        logger.error("Error descargando imagen para post_id=%s: %s", post_id, exc)
        post["image_download_status"] = "failed"
        post["image_local_path"] = ""

    return post


def download_all(posts: list[dict], images_dir: Path, timeout: int = 30) -> list[dict]:
    """Descarga imágenes para todos los posts.

    Args:
        posts: Lista de posts como dicts.
        images_dir: Directorio donde guardar las imágenes.
        timeout: Segundos máximos de espera por descarga.

    Returns:
        La lista de posts actualizada.
    """
    for post in posts:
        download_post_image(post, images_dir, timeout)
    return posts
