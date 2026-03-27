"""Pipeline principal: orquesta todas las fases del análisis de sentimiento NVDA.

Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> list[dict]:
    """Carga un JSON de posts. Retorna [] si no existe o está corrupto."""
    if not path.exists():
        logger.error("[pipeline] Archivo no encontrado: %s", path)
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        logger.error("[pipeline] JSON corrupto en %s: %s", path, exc)
        return []



def _print_sentiment_summary(posts: list[dict]) -> None:
    """Muestra scores de sentimiento promedio (0–1) por modelo."""
    if not posts:
        return

    def avg(key: str) -> float:
        vals = [p[key] for p in posts if isinstance(p.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n--- Sentimientos promedio (0–1) ---")
    print(f"  FinBERT  pos={avg('sent_finbert_pos'):.3f}  neg={avg('sent_finbert_neg'):.3f}  neu={avg('sent_finbert_neu'):.3f}")
    print(f"  BERT     pos={avg('sent_bert_pos'):.3f}  neg={avg('sent_bert_neg'):.3f}")
    print(f"  SocBERT  pos={avg('sent_socbert_pos'):.3f}  neg={avg('sent_socbert_neg'):.3f}")


def _print_test_summary(
    posts: list[dict],
    posts_with_image: int,
    posts_ollama: int,
) -> None:
    """Muestra el resumen final del modo prueba."""
    print("\n=== Resumen Modo Prueba ===")
    print(f"Posts procesados:        {len(posts)}")
    print(f"Posts con imagen:        {posts_with_image}")
    print(f"Posts analizados Ollama: {posts_ollama}")
    _print_sentiment_summary(posts)
    print()


# ---------------------------------------------------------------------------
# Fases del pipeline
# ---------------------------------------------------------------------------

def phase_filter_nvda(posts: list[dict]) -> list[dict]:
    """Fase 1: Filtrado de posts relevantes para NVDA."""
    from nvidia_sentiment.nvda_filter import filter_nvda_posts
    return filter_nvda_posts(posts)


def phase_select_sample(posts: list[dict], sample_size: int) -> list[dict]:
    """Selección de los N posts más recientes (modo prueba, Requisito 7.2)."""
    from nvidia_sentiment.nvda_filter import select_sample
    return select_sample(posts, sample_size)


def phase_download_images(posts: list[dict], images_dir: Path) -> list[dict]:
    """Fase 2: Descarga de imágenes."""
    from nvidia_sentiment.image_downloader import download_all
    return download_all(posts, images_dir)


def phase_filter_images(posts: list[dict], ollama_model: str) -> list[dict]:
    """Fase 3: Filtrado de relevancia de imágenes."""
    from nvidia_sentiment.image_filter import evaluate_image_relevance
    result = []
    for post in posts:
        try:
            result.append(evaluate_image_relevance(post, ollama_model))
        except Exception as exc:
            logger.error(
                "[pipeline] phase=filter_images post_id=%s: %s",
                post.get("id", "?"),
                exc,
            )
            post = dict(post)
            post.setdefault("image_relevance", False)
            result.append(post)
    return result


def phase_analyze_text(
    posts: list[dict],
    models: list[str],
    max_length: int,
) -> list[dict]:
    """Fase 4: Análisis de sentimiento de texto."""
    from nvidia_sentiment.text_analyzer import analyze_batch
    return analyze_batch(posts, models, max_length)


def phase_analyze_images(posts: list[dict], ollama_model: str) -> list[dict]:
    """Fase 5: Análisis de sentimiento de imagen."""
    from nvidia_sentiment.image_analyzer import analyze_image_sentiment
    result = []
    for post in posts:
        try:
            result.append(analyze_image_sentiment(post, ollama_model))
        except Exception as exc:
            logger.error(
                "[pipeline] phase=analyze_images post_id=%s: %s",
                post.get("id", "?"),
                exc,
            )
            result.append(dict(post))
    return result


def phase_fuse_sentiment(
    posts: list[dict],
    text_weight: float,
    image_weight: float,
) -> list[dict]:
    """Fase 6: Fusión multimodal."""
    from nvidia_sentiment.multimodal_comparator import fuse_sentiment
    result = []
    for post in posts:
        try:
            result.append(fuse_sentiment(post, text_weight, image_weight))
        except Exception as exc:
            logger.error(
                "[pipeline] phase=fuse_sentiment post_id=%s: %s",
                post.get("id", "?"),
                exc,
            )
            post = dict(post)
            post.setdefault("sent_text_only", "neutral")
            post.setdefault("sent_multimodal", "neutral")
            result.append(post)
    return result


def phase_fetch_current_posts(symbol: str, subreddits: list[str], limit: int) -> list[dict]:
    """Descarga posts actuales de Reddit via API pública (sin OAuth)."""
    import time
    import requests

    headers = {"User-Agent": "nvda-sentiment-bot/1.0"}
    results: list[dict] = []
    seen_ids: set[str] = set()

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/search.json"
        params = {"q": symbol, "sort": "new", "limit": min(limit, 100), "t": "week", "restrict_sr": 1}
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code != 200:
                logger.warning("[pipeline] Reddit %s devolvió %d", sub, resp.status_code)
                continue
            children = resp.json().get("data", {}).get("children", [])
            for child in children:
                p = child.get("data", {})
                pid = p.get("id", "")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                results.append({
                    "id": pid,
                    "title": p.get("title", ""),
                    "selftext": p.get("selftext", ""),
                    "subreddit": p.get("subreddit", sub),
                    "created_utc": int(p.get("created_utc", 0)),
                    "date": str(pd.to_datetime(p.get("created_utc", 0), unit="s").date()),
                    "score": int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                    "has_image": bool(p.get("url", "").endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))),
                    "image_urls": [p.get("url", "")] if p.get("url", "").startswith("http") else [],
                })
            time.sleep(1)  # respetar rate limit
        except Exception as exc:
            logger.warning("[pipeline] Error scraping r/%s: %s", sub, exc)

    logger.info("[pipeline] Posts actuales descargados de Reddit: %d", len(results))
    return results


def phase_save_dataset(posts: list[dict], output_path: Path) -> None:
    """Fase 7: Serialización del dataset procesado."""
    from nvidia_sentiment.serializer import save_dataset
    save_dataset(posts, output_path)
    logger.info("[pipeline] Dataset guardado en %s (%d posts)", output_path, len(posts))


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Ejecuta el pipeline completo con manejo de errores por fase."""
    input_path = Path(args.input)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "nvda_processed.csv"
    models = args.models.split() if isinstance(args.models, str) else args.models

    # ── Carga inicial ──────────────────────────────────────────────────────
    logger.info("[pipeline] Cargando dataset desde %s", input_path)
    all_posts = _load_json(input_path)
    if not all_posts:
        logger.error("[pipeline] Dataset vacío o no encontrado. Abortando.")
        sys.exit(1)
    logger.info("[pipeline] Posts cargados: %d", len(all_posts))

    # ── Fase 1: Filtrado NVDA ──────────────────────────────────────────────
    logger.info("[pipeline] Fase 1: Filtrado NVDA")
    try:
        posts = phase_filter_nvda(all_posts)
        logger.info("[pipeline] Posts NVDA filtrados: %d", len(posts))
    except Exception as exc:
        logger.error("[pipeline] phase=filter_nvda: %s", exc)
        posts = all_posts

    # ── Modo prueba: selección de N posts ─────────────────────────────────
    if args.test_mode:
        logger.info("[pipeline] Modo prueba activo — seleccionando %d posts", args.sample_size)
        try:
            posts = phase_select_sample(posts, args.sample_size)
            logger.info("[pipeline] Posts seleccionados para prueba: %d", len(posts))
        except Exception as exc:
            logger.error("[pipeline] phase=select_sample: %s", exc)

    # ── Fase 2: Descarga de imágenes ───────────────────────────────────────
    if not args.skip_download:
        logger.info("[pipeline] Fase 2: Descarga de imágenes")
        try:
            posts = phase_download_images(posts, images_dir)
        except Exception as exc:
            logger.error("[pipeline] phase=download_images: %s", exc)
    else:
        logger.info("[pipeline] Fase 2: Descarga de imágenes omitida (--skip_download)")

    # ── Fase 3: Filtrado de imágenes ───────────────────────────────────────
    if not args.skip_image_analysis:
        logger.info("[pipeline] Fase 3: Filtrado de relevancia de imágenes")
        try:
            posts = phase_filter_images(posts, args.ollama_model)
        except Exception as exc:
            logger.error("[pipeline] phase=filter_images: %s", exc)
    else:
        logger.info("[pipeline] Fase 3: Filtrado de imágenes omitido (--skip_image_analysis)")

    # ── Fase 4: Análisis de texto ──────────────────────────────────────────
    logger.info("[pipeline] Fase 4: Análisis de sentimiento de texto (%s)", models)
    try:
        posts = phase_analyze_text(posts, models, args.max_length)
    except Exception as exc:
        logger.error("[pipeline] phase=analyze_text: %s", exc)

    # ── Fase 5: Análisis de imagen ─────────────────────────────────────────
    if not args.skip_image_analysis:
        logger.info("[pipeline] Fase 5: Análisis de sentimiento de imagen")
        try:
            posts = phase_analyze_images(posts, args.ollama_model)
        except Exception as exc:
            logger.error("[pipeline] phase=analyze_images: %s", exc)
    else:
        logger.info("[pipeline] Fase 5: Análisis de imagen omitido (--skip_image_analysis)")

    # ── Fase 6: Fusión multimodal ──────────────────────────────────────────
    logger.info(
        "[pipeline] Fase 6: Fusión multimodal (text_weight=%.2f, image_weight=%.2f)",
        args.text_weight,
        args.image_weight,
    )
    try:
        posts = phase_fuse_sentiment(posts, args.text_weight, args.image_weight)
    except Exception as exc:
        logger.error("[pipeline] phase=fuse_sentiment: %s", exc)

    # ── Fase 6.5: Enriquecer con posts actuales de Reddit ─────────────────
    if args.fetch_current:
        logger.info("[pipeline] Fase 6.5: Descargando posts actuales de Reddit")
        try:
            current_subreddits = ["wallstreetbets", "stocks", "investing", "StockMarket", "nvidia"]
            current_posts = phase_fetch_current_posts("NVDA", current_subreddits, limit=50)
            if current_posts:
                # Analizar sentimiento de los posts actuales
                current_posts = phase_analyze_text(current_posts, models, args.max_length)
                current_posts = phase_fuse_sentiment(current_posts, args.text_weight, args.image_weight)
                # Añadir al CSV existente (deduplicando)
                from nvidia_sentiment.serializer import append_posts
                added = append_posts(current_posts, output_path)
                logger.info("[pipeline] Posts actuales añadidos al CSV: %d", added)
                # Incluir en posts para el resumen
                posts = posts + [p for p in current_posts if p.get("id") not in {x.get("id") for x in posts}]
        except Exception as exc:
            logger.error("[pipeline] phase=fetch_current: %s", exc)

    # ── Fase 7: Serialización ──────────────────────────────────────────────
    logger.info("[pipeline] Fase 7: Guardando dataset procesado en %s", output_path)
    try:
        phase_save_dataset(posts, output_path)
    except Exception as exc:
        logger.error("[pipeline] phase=save_dataset: %s", exc)

    # ── Resumen modo prueba ────────────────────────────────────────────────
    if args.test_mode:
        posts_with_image = sum(1 for p in posts if p.get("image_download_status") == "ok")
        posts_ollama = sum(
            1 for p in posts
            if isinstance(p.get("image_analysis"), dict)
            and not p["image_analysis"].get("error", True)
        )
        _print_test_summary(posts, posts_with_image, posts_ollama)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de análisis de sentimiento multimodal para NVIDIA (NVDA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al archivo final_dataset_clean.json",
    )
    parser.add_argument(
        "--images_dir",
        default="data/images/",
        help="Directorio donde guardar las imágenes descargadas",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Directorio de salida para nvda_processed.json",
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Activar modo prueba sobre un subconjunto reducido de posts",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Número de posts a procesar en modo prueba",
    )
    parser.add_argument(
        "--text_weight",
        type=float,
        default=0.75,
        help="Peso de la señal de texto en la fusión multimodal",
    )
    parser.add_argument(
        "--image_weight",
        type=float,
        default=0.25,
        help="Peso de la señal de imagen en la fusión multimodal",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Omitir la fase de descarga de imágenes",
    )
    parser.add_argument(
        "--skip_image_analysis",
        action="store_true",
        help="Omitir las fases de filtrado y análisis de imagen",
    )
    parser.add_argument(
        "--ollama_model",
        default="llama3.2-vision",
        help="Modelo Ollama para análisis de imagen",
    )
    parser.add_argument(
        "--models",
        default="finbert bert socbert",
        help="Modelos de texto a usar (separados por espacio)",
    )
    parser.add_argument(
        "--fetch_current",
        action="store_true",
        help="Descargar posts actuales de Reddit y añadirlos al CSV de entrenamiento",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Longitud máxima de tokens para el análisis de texto",
    )
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_pipeline(args)
