"""CLI para el Analizador_Texto del pipeline nvidia-sentiment-prediction.

Uso:
    python text_analyzer.py --input <ruta_json> [opciones]

Requisitos: 4.5, 7.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("text_analyzer_cli")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analizador de sentimiento de texto (FinBERT / BERT / SocBERT) para posts de Reddit.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Ruta al JSON de entrada (lista de posts).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Ruta al JSON de salida. Por defecto: nvda_analyzed_text.json en el mismo directorio que --input.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["finbert", "bert", "socbert"],
        metavar="MODEL",
        help="Modelos a aplicar: finbert bert socbert (default: los tres).",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Número máximo de tokens para truncar el texto (default: 256).",
    )
    p.add_argument(
        "--test_mode",
        action="store_true",
        help="Modo prueba: procesar solo los primeros --sample_size posts.",
    )
    p.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Número de posts a procesar en modo prueba (default: 200).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("El archivo de entrada no existe: %s", input_path)
        sys.exit(1)

    # Determinar ruta de salida
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "nvda_analyzed_text.json"

    # Cargar posts
    logger.info("Cargando posts desde %s …", input_path)
    try:
        with input_path.open("r", encoding="utf-8") as f:
            posts: list[dict] = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("JSON de entrada corrupto: %s", exc)
        sys.exit(1)

    total_original = len(posts)
    logger.info("Posts cargados: %d", total_original)

    # Modo prueba: primeros N posts
    if args.test_mode:
        posts = posts[: args.sample_size]
        logger.info("Modo prueba activo — procesando los primeros %d posts.", len(posts))

    # Importar aquí para no cargar modelos si hay error de argumentos
    from nvidia_sentiment.text_analyzer import analyze_text_sentiment

    models = [m.lower() for m in args.models]
    logger.info("Modelos: %s | max_length: %d", ", ".join(models), args.max_length)

    analyzed: list[dict] = []
    for i, post in enumerate(posts, start=1):
        if i % 50 == 0 or i == len(posts):
            logger.info("Progreso: %d / %d", i, len(posts))
        result = analyze_text_sentiment(post, models, args.max_length)
        analyzed.append(result)

    # Guardar resultado
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(analyzed, f, ensure_ascii=False, indent=2)

    logger.info("Resultado guardado en: %s", output_path)

    # Resumen
    print("\n=== Resumen ===")
    print(f"Posts de entrada  : {total_original}")
    print(f"Posts procesados  : {len(analyzed)}")
    print(f"Modelos aplicados : {', '.join(models)}")
    print(f"Salida            : {output_path}")

    if analyzed:
        # Distribución de etiquetas por modelo
        for model in models:
            label_key = f"sent_{model}_label"
            if label_key in analyzed[0]:
                from collections import Counter
                counts = Counter(p.get(label_key, "?") for p in analyzed)
                print(f"\n  {model} labels: {dict(counts)}")


if __name__ == "__main__":
    main()
