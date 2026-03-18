# Modulo1_RRSS/main_rrss.py
"""
main_rrss.py
Runner para probar analizadores de sentimiento sobre CSVs de Reddit (texto + imágenes opcional).

Qué hace:
  1) (Opcional) Scrapea/actualiza datos de Reddit (sin OAuth) y descarga imágenes.
  2) Ejecuta FinBERT y/o SocBERT sobre el CSV.
  3) Genera un resumen rápido por subreddit y te deja los CSVs de salida en Data/.

Requisitos (según lo que actives):
  - Scraper: requests pandas python-dotenv
  - FinBERT: transformers torch pandas tqdm
  - SocBERT: transformers torch datasets accelerate pandas tqdm (solo si entrenas)
  - Imágenes: pillow pytesseract (OCR) + tesseract instalado, y/o BLIP (caption) via transformers

Uso:
  python main_rrss.py --symbol NVDA --run finbert
  python main_rrss.py --symbol NVDA --run socbert
  python main_rrss.py --symbol NVDA --run both --include_comments
  python main_rrss.py --symbol NVDA --update --download_images --run finbert --use_images
  python main_rrss.py --symbol NVDA --scrape --limit 100 --subreddits stocks wallstreetbets --run both --use_images
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


from RedditScrapper.reddit_scrapper import (
    load_config,
    search_symbol_posts,
    DEFAULT_SUBREDDITS,
)
from RedditScrapper.reddit_updater import main as updater_main  # opcional, si quieres usar su CLI
from RedditScrapper.sentiments.sentiment_analyzer_finbert import main as finbert_main
from RedditScrapper.sentiments.sentiment_analyzer_socbert import main as socbert_main


def parse_args():
    p = argparse.ArgumentParser(description="Runner RRSS (Reddit) para probar analizadores de sentimiento")

    # Datos
    p.add_argument("--symbol", required=True, help="Ticker/símbolo (e.g., NVDA)")
    p.add_argument("--data_dir", default="RedditScrapper/Data", help="Ruta a Data/ (default RedditScrapper/Data)")

    # Scrape / Update
    p.add_argument("--scrape", action="store_true", help="Descarga posts y crea CSV desde cero (según time_filter/limit)")
    p.add_argument("--update", action="store_true", help="Actualiza el CSV existente (incremental)")
    p.add_argument("--limit", type=int, default=200, help="Posts máximos por subreddit (default 200)")
    p.add_argument("--subreddits", nargs="*", default=DEFAULT_SUBREDDITS, help="Lista de subreddits")
    p.add_argument("--time_filter", choices=["hour", "day", "week", "month", "year", "all"], default="month")
    p.add_argument("--include_comments", action="store_true", help="Incluye comments_text en el texto del post")
    p.add_argument("--download_images", action="store_true", help="Descarga imagen principal a Data/images/<SYMBOL>/")

    # Qué analizadores ejecutar
    p.add_argument("--run", choices=["finbert", "socbert", "both"], default="finbert")

    # Imágenes en analizadores
    p.add_argument("--use_images", action="store_true", help="Activa análisis de imágenes en los analizadores")
    p.add_argument("--image_mode", choices=["ocr", "caption", "both"], default="both")
    p.add_argument("--image_weight", type=float, default=0.25, help="Peso de imagen en el score final (default 0.25)")
    p.add_argument("--min_ocr_chars", type=int, default=25, help="Umbral para decidir OCR vs caption")

    # FinBERT
    p.add_argument("--finbert_batch", type=int, default=16)

    # SocBERT (train opcional)
    p.add_argument("--socbert_train", action="store_true", help="Fuerza fine-tuning de SocBERT (si no existe ya)")
    p.add_argument("--socbert_epochs", type=int, default=1)
    p.add_argument("--socbert_lr", type=float, default=2e-5)
    p.add_argument("--socbert_train_samples", type=int, default=20000)
    p.add_argument("--socbert_eval_samples", type=int, default=4000)
    p.add_argument("--socbert_batch", type=int, default=16)

    # Device para pipeline (FinBERT)
    p.add_argument("--device", type=int, default=-1, help="Device HF pipeline: -1 CPU, 0 GPU0, ...")

    return p.parse_args()


def ensure_dirs(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "images").mkdir(parents=True, exist_ok=True)


def scrape_now(args) -> Path:
    """
    Crea/reescribe el CSV base (SYMBOL_reddit.csv) usando el scraper.
    """
    cfg = load_config()
    data_dir = Path(args.data_dir).resolve()
    ensure_dirs(data_dir)

    images_dir = data_dir / "images"
    df = search_symbol_posts(
        symbol=args.symbol,
        subreddits=args.subreddits,
        limit=args.limit,
        time_filter=args.time_filter,
        include_comments=args.include_comments,
        comments_limit=10 if args.include_comments else 0,
        cfg=cfg,
        download_images=args.download_images,
        images_dir=images_dir,
    )

    out_path = data_dir / f"{args.symbol.upper()}_reddit.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] Scrape guardado: {out_path} (filas={len(df)})")
    return out_path


def quick_summary(csv_path: Path, label_col: str):
    """
    Resumen rápido: distribución global y por subreddit.
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        print(f"[WARN] No existe columna {label_col} en {csv_path.name}")
        return

    print("\n==== Resumen global ====")
    print(df[label_col].value_counts(dropna=False).to_string())

    if "subreddit" in df.columns:
        print("\n==== Resumen por subreddit (top 10) ====")
        tmp = (
            df.groupby(["subreddit", label_col])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        print(tmp.head(10).to_string(index=False))


def run_finbert(args):
    """
    Ejecuta el bert_analyzer.py ya modificado (texto + imágenes opcional).
    Lo llamamos “como si” fuera CLI, parchando sys.argv para reutilizar tu main().
    """
    import sys

    # Construimos argv para bert_analyzer.py
    argv = [
        "bert_analyzer.py",
        "--symbol", args.symbol,
        "--data_dir", Path(args.data_dir).name if "RedditScrapper" in args.data_dir else args.data_dir,
        "--batch_size", str(args.finbert_batch),
        "--device", str(args.device),
    ]
    if args.include_comments:
        argv.append("--include_comments")
    if args.use_images:
        argv += [
            "--use_images",
            "--image_mode", args.image_mode,
            "--image_weight", str(args.image_weight),
            "--min_ocr_chars", str(args.min_ocr_chars),
        ]

    # Ejecuta
    old_argv = sys.argv
    try:
        sys.argv = argv
        finbert_main()
    finally:
        sys.argv = old_argv

    # output esperado:
    suffix = "_reddit_finbert_img.csv" if args.use_images else "_reddit_finbert.csv"
    out_path = Path(args.data_dir).resolve() / f"{args.symbol.upper()}{suffix}"
    if out_path.exists():
        quick_summary(out_path, "final_finbert_label" if args.use_images else "sent_finbert_label")
    else:
        print(f"[WARN] No encuentro salida FinBERT: {out_path}")


def run_socbert(args):
    """
    Ejecuta socbert_analyzer.py ya modificado (texto + imágenes opcional).
    """
    import sys

    argv = [
        "socbert_analyzer.py",
        "--symbol", args.symbol,
        "--data_dir", Path(args.data_dir).name if "RedditScrapper" in args.data_dir else args.data_dir,
        "--batch_size", str(args.socbert_batch),
        "--epochs", str(args.socbert_epochs),
        "--lr", str(args.socbert_lr),
        "--train_samples", str(args.socbert_train_samples),
        "--eval_samples", str(args.socbert_eval_samples),
    ]
    if args.include_comments:
        argv.append("--include_comments")
    if args.socbert_train:
        argv.append("--train")
    if args.use_images:
        argv += [
            "--use_images",
            "--image_mode", args.image_mode,
            "--image_weight", str(args.image_weight),
            "--min_ocr_chars", str(args.min_ocr_chars),
        ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        socbert_main()
    finally:
        sys.argv = old_argv

    suffix = "_reddit_socbert_img.csv" if args.use_images else "_reddit_socbert.csv"
    out_path = Path(args.data_dir).resolve() / f"{args.symbol.upper()}{suffix}"
    if out_path.exists():
        quick_summary(out_path, "final_socbert_label" if args.use_images else "sent_socbert_label")
    else:
        print(f"[WARN] No encuentro salida SocBERT: {out_path}")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    ensure_dirs(data_dir)

    # 1) Preparar datos
    if args.scrape:
        scrape_now(args)
    elif args.update:
        # Llama al updater con su main() (CLI patch)
        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                "reddit_updater.py",
                "--symbol", args.symbol,
                "--limit", str(args.limit),
                "--time_filter", args.time_filter,
                *([] if not args.subreddits else ["--subreddits", *args.subreddits]),
                *([ "--download_images" ] if args.download_images else []),
            ]
            updater_main()
        finally:
            sys.argv = old_argv

    # 2) Ejecutar analizadores
    if args.run in ("finbert", "both"):
        run_finbert(args)

    if args.run in ("socbert", "both"):
        run_socbert(args)

    print("\n[OK] main_rrss terminado.")


if __name__ == "__main__":
    main()
