"""CLI para el Filtro_NVDA: filtra posts relevantes para NVIDIA."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nvidia_sentiment.nvda_filter import filter_nvda_posts, select_sample
from nvidia_sentiment.serializer import load_dataset, save_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filtra posts de Reddit relevantes para NVIDIA (NVDA)."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Ruta al JSON de entrada (ej. RedditScrapper/Data/final_dataset_clean.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Ruta del JSON de salida. "
            "Por defecto: mismo directorio que --input con nombre nvda_filtered_posts.json"
        ),
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Activa el modo prueba: selecciona solo los N posts más recientes.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Número de posts en modo prueba (default: 200).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_path: Path = (
        args.output
        if args.output is not None
        else input_path.parent / "nvda_filtered_posts.json"
    )

    # Carga
    posts = load_dataset(input_path)
    total_read = len(posts)
    logger.info("Posts leídos: %d", total_read)

    # Filtrado
    filtered = filter_nvda_posts(posts)
    total_filtered = len(filtered)
    logger.info("Posts filtrados (NVDA): %d", total_filtered)

    # Muestra en modo prueba
    if args.test_mode:
        filtered = select_sample(filtered, args.sample_size)
        logger.info("Modo prueba activo — muestra de %d posts más recientes.", len(filtered))

    total_exported = len(filtered)

    # Guardado
    save_dataset(filtered, output_path)
    logger.info("Dataset exportado a: %s", output_path)

    # Resumen
    print(f"\n=== Resumen ===")
    print(f"  Posts leídos    : {total_read}")
    print(f"  Posts filtrados : {total_filtered}")
    print(f"  Posts exportados: {total_exported}")
    print(f"  Archivo salida  : {output_path}")


if __name__ == "__main__":
    main()
