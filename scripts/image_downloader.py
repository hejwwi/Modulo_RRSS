"""CLI para descargar imágenes de posts filtrados de NVIDIA."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nvidia_sentiment.image_downloader import download_post_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga imágenes de posts filtrados de NVIDIA."
    )
    parser.add_argument("--input", required=True, help="JSON de posts filtrados.")
    parser.add_argument(
        "--output",
        default=None,
        help="Archivo de salida JSON. Por defecto sobreescribe el input.",
    )
    parser.add_argument(
        "--images_dir",
        default="RedditScrapper/Data/images/",
        help="Directorio donde guardar las imágenes (default: RedditScrapper/Data/images/).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout en segundos para cada descarga (default: 30).",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Procesar solo los primeros --sample_size posts con imagen.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="En modo prueba, número de posts con imagen a procesar (default: 100).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Archivo de entrada no encontrado: %s", input_path)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        posts: list[dict] = json.load(f)

    images_dir = Path(args.images_dir)

    if args.test_mode:
        # Seleccionar los primeros sample_size posts que tengan image_urls no vacío
        posts_with_image = [p for p in posts if p.get("image_urls")]
        to_process = posts_with_image[: args.sample_size]
        posts_without_image = [p for p in posts if not p.get("image_urls")]
        # Marcar los que no se procesarán en modo prueba como no_image
        skipped = posts_with_image[args.sample_size :]
        for p in skipped:
            p.setdefault("image_download_status", "no_image")
            p.setdefault("image_local_path", "")
    else:
        to_process = posts
        posts_without_image = []
        skipped = []

    for post in to_process:
        download_post_image(post, images_dir, args.timeout)

    # Marcar posts sin imagen que no se procesaron
    for post in posts_without_image:
        post.setdefault("image_download_status", "no_image")
        post.setdefault("image_local_path", "")

    # Reunir todos los posts en el orden original
    all_posts = posts if not args.test_mode else (to_process + skipped + posts_without_image)

    output_path = Path(args.output) if args.output else input_path
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)

    # Resumen
    statuses = [p.get("image_download_status", "no_image") for p in all_posts]
    total = len(all_posts)
    ok = statuses.count("ok")
    failed = statuses.count("failed")
    no_image = statuses.count("no_image")

    print(f"Total procesados : {total}")
    print(f"  ok             : {ok}")
    print(f"  failed         : {failed}")
    print(f"  no_image       : {no_image}")
    print(f"Guardado en      : {output_path}")


if __name__ == "__main__":
    main()
