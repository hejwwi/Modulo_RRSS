# Modulo1_RRSS/RedditScrapper/reddit_updater.py
"""
Actualiza CSV existente de Reddit sin volver a descargar todo.
Versión SIN APP / SIN OAUTH: usa endpoints públicos JSON (requests).

Opcional: descarga imagen principal de cada post nuevo a ./Data/images/<SYMBOL>/
y guarda 'image_path' en el CSV, para análisis posterior (CLIP/BLIP/OCR).

Uso:
  python reddit_updater.py --symbol NVDA
  python reddit_updater.py --symbol NVDA --limit 200 --subreddits stocks wallstreetbets
  python reddit_updater.py --symbol NVDA --download_images
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from RedditScrapper.reddit_scrapper import (
    load_config,
    search_symbol_posts,
    DEFAULT_SUBREDDITS,
)



def parse_args():
    p = argparse.ArgumentParser(description="Actualizador incremental Reddit CSV (sin OAuth)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--subreddits", nargs="*", default=DEFAULT_SUBREDDITS)
    p.add_argument("--time_filter", choices=["hour", "day", "week", "month", "year", "all"], default="month")
    p.add_argument(
        "--download_images",
        action="store_true",
        help="Descarga la imagen principal de los posts nuevos a Data/images/<SYMBOL>/",
    )
    return p.parse_args()


def _count_downloaded_images(df: pd.DataFrame) -> int:
    if df is None or df.empty or "image_path" not in df.columns:
        return 0
    s = df["image_path"].fillna("").astype(str)
    return int((s.str.len() > 0).sum())


def main():
    args = parse_args()

    script_path = Path(__file__).resolve()
    data_dir = script_path.parent / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)

    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"{args.symbol.upper()}_reddit.csv"
    cfg = load_config()

    # 1) Si no existe CSV, primera carga (y opcionalmente baja imágenes)
    if not csv_path.exists():
        print("[INFO] No existe CSV previo, ejecutando scraper completo (sin OAuth)...")
        df = search_symbol_posts(
            symbol=args.symbol,
            subreddits=args.subreddits,
            limit=args.limit,
            time_filter=args.time_filter,
            include_comments=False,
            comments_limit=0,
            cfg=cfg,
            download_images=args.download_images,
            images_dir=images_dir,
        )
        df.to_csv(csv_path, index=False, encoding="utf-8")

        print(f"[OK] Guardado: {csv_path}")
        print(f"[OK] Filas: {len(df)}")
        if args.download_images:
            print(f"[OK] Imágenes descargadas: {_count_downloaded_images(df)}")
        return

    # 2) Cargar CSV existente
    old_df = pd.read_csv(csv_path)

    if old_df.empty:
        last_timestamp = 0.0
        print("[INFO] CSV vacío, se tratará como primera carga.")
    else:
        # Asegurar dtype
        if "created_utc" in old_df.columns:
            old_df["created_utc"] = old_df["created_utc"].astype(float)
            last_timestamp = float(old_df["created_utc"].max())
        else:
            # por si el CSV antiguo no tenía created_utc (raro), fuerzo a 0
            last_timestamp = 0.0

        last_date = datetime.fromtimestamp(last_timestamp, tz=timezone.utc)
        print(f"[INFO] Último post guardado: {last_date}")

    # 3) Descargar candidatos nuevos (opcional: también imágenes)
    new_df = search_symbol_posts(
        symbol=args.symbol,
        subreddits=args.subreddits,
        limit=args.limit,
        time_filter=args.time_filter,
        include_comments=False,
        comments_limit=0,
        cfg=cfg,
        download_images=args.download_images,
        images_dir=images_dir,
    )

    if new_df.empty:
        print("[OK] No hay nuevos posts (o no devolvió resultados).")
        return

    # 4) Filtrar solo los más recientes
    new_df["created_utc"] = new_df["created_utc"].astype(float)
    new_df = new_df[new_df["created_utc"] > last_timestamp]

    if new_df.empty:
        print("[OK] No hay posts más recientes que los existentes.")
        return

    # 5) Merge + dedupe + sort
    final_df = pd.concat([old_df, new_df], ignore_index=True)

    # Si el CSV viejo no tenía columnas de imagen y el nuevo sí, pandas las crea con NaN en las antiguas (perfecto)
    final_df = final_df.drop_duplicates(subset=["id"]).sort_values(by="created_utc", ascending=False)

    final_df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"[OK] CSV actualizado: {csv_path}")
    print(f"[OK] Nuevos posts añadidos: {len(new_df)}")
    if args.download_images:
        print(f"[OK] Imágenes descargadas en esta actualización: {_count_downloaded_images(new_df)}")
    print(f"[OK] Total posts: {len(final_df)}")


if __name__ == "__main__":
    main()
