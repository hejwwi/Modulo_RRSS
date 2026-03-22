"""CLI para el Analizador_Imagen: analiza el impacto de imágenes relevantes en el sentimiento NVDA."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza el impacto de imágenes relevantes en el sentimiento NVDA usando Ollama."
    )
    parser.add_argument("--input", required=True, help="Ruta al JSON de entrada (posts con image_relevance).")
    parser.add_argument(
        "--output",
        default=None,
        help="Ruta al JSON de salida. Por defecto: nvda_analyzed_full.json en el mismo directorio que --input.",
    )
    parser.add_argument(
        "--ollama_model",
        default="llama3.2-vision",
        help="Modelo Ollama a usar (default: llama3.2-vision).",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Modo prueba: procesar solo los primeros --sample_size posts con imagen relevante.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Número de posts con imagen relevante a procesar en modo prueba (default: 50).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Archivo de entrada no encontrado: %s", input_path)
        sys.exit(1)

    # Determinar ruta de salida
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "nvda_analyzed_full.json"

    # Cargar posts
    logger.info("Cargando posts desde %s ...", input_path)
    with input_path.open("r", encoding="utf-8") as f:
        posts: list[dict] = json.load(f)

    total_posts = len(posts)
    logger.info("Total posts cargados: %d", total_posts)

    # Importar aquí para evitar dependencias en el nivel de módulo al parsear args
    from nvidia_sentiment.image_analyzer import analyze_image_sentiment

    # Identificar posts con imagen relevante
    relevant_posts_idx = [
        i for i, p in enumerate(posts)
        if p.get("image_relevance", False) and p.get("image_local_path", "")
    ]

    sin_imagen_relevante = total_posts - len(relevant_posts_idx)

    # Aplicar límite en modo prueba (Requisito 5.5)
    if args.test_mode:
        relevant_posts_idx = relevant_posts_idx[: args.sample_size]
        logger.info(
            "Modo prueba activo: procesando los primeros %d posts con imagen relevante.",
            args.sample_size,
        )

    analizados = 0
    errores = 0

    for idx in relevant_posts_idx:
        post = posts[idx]
        post_id = post.get("id", f"idx={idx}")
        logger.info(
            "[%d/%d] Analizando post_id=%s ...",
            analizados + 1,
            len(relevant_posts_idx),
            post_id,
        )
        updated = analyze_image_sentiment(post, ollama_model=args.ollama_model)
        posts[idx] = updated
        analizados += 1

        image_analysis = updated.get("image_analysis")
        if image_analysis and image_analysis.get("error", False):
            errores += 1

    # Guardar resultado
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    # Resumen
    logger.info("=" * 50)
    logger.info("RESUMEN")
    logger.info("  Total posts:                %d", total_posts)
    logger.info("  Analizados con Ollama:      %d", analizados)
    logger.info("  Errores en análisis:        %d", errores)
    logger.info("  Sin imagen relevante:       %d", sin_imagen_relevante)
    logger.info("  Resultado guardado en:      %s", output_path)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
