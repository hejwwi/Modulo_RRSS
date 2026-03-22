"""CLI para el Filtro_Imágenes del pipeline nvidia-sentiment-prediction.

Uso:
    python image_filter.py --input <ruta> [--output <ruta>] [--ollama_model llama3.2-vision] [--test_mode]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nvidia_sentiment.image_filter import evaluate_image_relevance, is_gif
from nvidia_sentiment.serializer import load_dataset, save_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("image_filter_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtra imágenes no relevantes usando Ollama llama3.2-vision."
    )
    parser.add_argument("--input", required=True, help="Ruta al JSON de entrada.")
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta al JSON de salida. Si no se indica, sobreescribe el input.",
    )
    parser.add_argument(
        "--ollama_model",
        default="llama3.2-vision",
        help="Modelo Ollama a usar (default: llama3.2-vision).",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Procesa solo posts con image_download_status='ok' y que no sean GIF.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    posts = load_dataset(input_path)
    if not posts:
        logger.warning("No se encontraron posts en %s", input_path)
        return

    # En test_mode: procesar solo posts con imagen descargada y no GIF
    if args.test_mode:
        candidates = [
            p for p in posts
            if p.get("image_download_status") == "ok" and not is_gif(p)
        ]
        logger.info(
            "test_mode activo: %d posts candidatos de %d totales",
            len(candidates),
            len(posts),
        )
    else:
        candidates = posts

    total = len(posts)
    relevant = 0
    not_relevant = 0
    errors = 0

    candidate_ids: set[str] = {p.get("id") for p in candidates}

    updated_posts = []
    for post in posts:
        post_id = post.get("id", "")

        if post_id in candidate_ids:
            try:
                updated = evaluate_image_relevance(post, ollama_model=args.ollama_model)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error procesando post_id=%s: %s", post_id, exc)
                updated = dict(post)
                updated["image_relevance"] = False
                errors += 1
            else:
                if updated.get("image_relevance"):
                    relevant += 1
                else:
                    not_relevant += 1
            updated_posts.append(updated)
        else:
            # Posts no candidatos: mantener sin cambios (o marcar False si no tienen imagen)
            if "image_relevance" not in post:
                post = dict(post)
                post["image_relevance"] = False
            updated_posts.append(post)
            not_relevant += 1

    save_dataset(updated_posts, output_path)

    print("\n=== Resumen Filtro_Imágenes ===")
    print(f"  Total posts      : {total}")
    print(f"  Relevantes       : {relevant}")
    print(f"  No relevantes    : {not_relevant}")
    print(f"  Errores          : {errors}")
    print(f"  Guardado en      : {output_path}")


if __name__ == "__main__":
    main()
