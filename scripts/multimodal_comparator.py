"""CLI del Comparador_Multimodal: fusión texto + imagen y accuracy vs precio real."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nvidia_sentiment.multimodal_comparator import (
    build_comparison_report,
    fuse_sentiment,
    load_price_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comparador multimodal: fusiona señales de texto e imagen y calcula accuracy."
    )
    parser.add_argument("--input", required=True, help="Ruta al JSON de posts analizados.")
    parser.add_argument("--price_data", required=True, help="Ruta al CSV de precios (nvda_top3_backfill.csv).")
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta de salida para comparison_report.csv (default: mismo dir que --input).",
    )
    parser.add_argument("--text_weight", type=float, default=0.75, help="Peso del modelo de texto (default: 0.75).")
    parser.add_argument("--image_weight", type=float, default=0.25, help="Peso del modelo de imagen (default: 0.25).")
    args = parser.parse_args()

    input_path = Path(args.input)
    price_path = Path(args.price_data)

    # Determinar ruta de salida
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "comparison_report.csv"

    # Cargar posts
    if not input_path.exists():
        logger.error("Archivo de entrada no encontrado: %s", input_path)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        try:
            posts: list[dict] = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("JSON de entrada corrupto: %s", exc)
            sys.exit(1)

    logger.info("Posts cargados: %d", len(posts))

    # Fusionar sentimientos
    fused_posts = [
        fuse_sentiment(post, text_weight=args.text_weight, image_weight=args.image_weight)
        for post in posts
    ]

    # Cargar datos de precio
    price_df = load_price_data(price_path)

    # Construir reporte de comparación
    rows = build_comparison_report(fused_posts, price_df)

    # Exportar CSV
    if rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "post_id", "date", "sent_text_only", "sent_multimodal",
            "price_movement_next_day", "correct_text", "correct_multimodal",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Reporte exportado a: %s", output_path)
    else:
        logger.warning("No hay filas comparables; no se exporta CSV.")

    # Resumen
    total = len(rows)
    if total > 0:
        acc_text = sum(1 for r in rows if r["correct_text"]) / total
        acc_multi = sum(1 for r in rows if r["correct_multimodal"]) / total
        print(f"\n=== Resumen ===")
        print(f"Posts comparados:       {total}")
        print(f"Accuracy texto solo:    {acc_text:.2%}")
        print(f"Accuracy multimodal:    {acc_multi:.2%}")
    else:
        print("\n=== Resumen ===")
        print("Posts comparados: 0 (sin datos de precio alineados)")


if __name__ == "__main__":
    main()
