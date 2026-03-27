#!/usr/bin/env python3
"""Actualizador incremental del dataset NVDA.

Descarga los posts más relevantes de los últimos N días de Reddit,
filtra los que ya están en el CSV y añade solo los nuevos con sentimiento analizado.

Uso:
    python scripts/update_dataset.py
    python scripts/update_dataset.py --days 7 --min_score 10
    python scripts/update_dataset.py --csv data/nvda_processed.csv --days 14 --dry_run
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "StockMarket",
    "nvidia",
    "pennystocks",
    "wallstreetbetsnews",
    "wallstreetbets2",
]

NVDA_TERMS = {"nvidia", "nvda", "geforce", "rtx", "cuda"}
HEADERS = {"User-Agent": "nvda-sentiment-updater/1.0"}


def _is_nvda(title: str, selftext: str) -> bool:
    combined = (title + " " + (selftext or "")).lower()
    return any(t in combined for t in NVDA_TERMS)


def _fetch_subreddit(
    subreddit: str,
    since_ts: float,
    min_score: int,
    existing_ids: set[str],
) -> list[dict]:
    """Descarga posts nuevos de un subreddit desde `since_ts` (unix timestamp)."""
    results: list[dict] = []
    after: str | None = None
    url = f"https://www.reddit.com/r/{subreddit}/search.json"
    page = 0

    while True:
        params: dict = {
            "q": "NVDA OR nvidia OR GeForce OR RTX",
            "sort": "top",
            "limit": 100,
            "t": "month",   # ventana amplia; filtramos por fecha después
            "restrict_sr": 1,
        }
        if after:
            params["after"] = after

        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if resp.status_code == 429:
                logger.warning("Rate limit en r/%s — esperando 60s", subreddit)
                time.sleep(60)
                continue
            if resp.status_code != 200:
                logger.warning("r/%s devolvió HTTP %d", subreddit, resp.status_code)
                break

            data = resp.json().get("data", {})
            children = data.get("children", [])
            if not children:
                break

            stop = False
            for child in children:
                p = child.get("data", {})
                pid = p.get("id", "")
                created = float(p.get("created_utc", 0))

                # Si el post es más antiguo que nuestro umbral, parar paginación
                if created < since_ts:
                    stop = True
                    break

                if not pid or pid in existing_ids:
                    continue

                title = p.get("title", "")
                selftext = p.get("selftext", "") or ""
                score = int(p.get("score", 0))

                if score < min_score:
                    continue
                if not _is_nvda(title, selftext):
                    continue

                dt = datetime.datetime.fromtimestamp(created, tz=datetime.timezone.utc)
                results.append({
                    "id": pid,
                    "title": title,
                    "selftext": selftext,
                    "subreddit": p.get("subreddit", subreddit),
                    "created_utc": int(created),
                    "date": str(dt.date()),
                    "score": score,
                    "num_comments": int(p.get("num_comments", 0)),
                    "has_image": str(p.get("url", "")).endswith(
                        (".jpg", ".jpeg", ".png", ".gif", ".webp")
                    ),
                    "image_urls": (
                        [p.get("url", "")]
                        if str(p.get("url", "")).startswith("http")
                        else []
                    ),
                })

            page += 1
            after = data.get("after")

            if stop or not after or page >= 5:
                break

            time.sleep(1.2)

        except Exception as exc:
            logger.error("Error en r/%s: %s", subreddit, exc)
            break

    return results


def fetch_recent_posts(
    days: int,
    min_score: int,
    existing_ids: set[str],
) -> list[dict]:
    """Descarga posts relevantes de los últimos `days` días no presentes en el CSV."""
    since = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(days=days)
    since_ts = since.timestamp()
    logger.info("Buscando posts desde %s (últimos %d días)", since.date(), days)

    all_posts: list[dict] = []
    for sub in SUBREDDITS:
        posts = _fetch_subreddit(sub, since_ts, min_score, existing_ids)
        if posts:
            logger.info("  r/%-24s → %d posts nuevos", sub, len(posts))
            all_posts.extend(posts)
        time.sleep(0.5)

    # Deduplicar entre subreddits
    seen: set[str] = set()
    unique: list[dict] = []
    for p in all_posts:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)

    # Ordenar por score descendente (más relevantes primero)
    unique.sort(key=lambda p: p["score"], reverse=True)
    return unique


def analyze_and_save(
    new_posts: list[dict],
    csv_path: Path,
    models: list[str],
    dry_run: bool,
) -> int:
    """Analiza sentimiento y añade al CSV. Retorna número de posts añadidos."""
    from nvidia_sentiment.text_analyzer import analyze_batch
    from nvidia_sentiment.multimodal_comparator import fuse_sentiment
    from nvidia_sentiment.serializer import append_posts

    logger.info("Analizando sentimiento de %d posts nuevos...", len(new_posts))
    new_posts = analyze_batch(new_posts, models, max_length=256)
    new_posts = [fuse_sentiment(p) for p in new_posts]

    if dry_run:
        logger.info("[dry_run] Se añadirían %d posts (no se escribe nada)", len(new_posts))
        return len(new_posts)

    added = append_posts(new_posts, csv_path)
    return added


def print_summary(new_posts: list[dict], added: int, csv_path: Path) -> None:
    from collections import Counter

    print(f"\n{'='*50}")
    print(f"Posts nuevos encontrados:  {len(new_posts)}")
    print(f"Posts añadidos al CSV:     {added}")

    if new_posts:
        subs = Counter(p["subreddit"] for p in new_posts)
        dates = Counter(p["date"] for p in new_posts)
        print("\nPor subreddit:")
        for s, c in subs.most_common():
            print(f"  {s:<28} {c:>3}")
        print("\nPor fecha:")
        for d, c in sorted(dates.items(), reverse=True)[:7]:
            print(f"  {d}  {c:>3}")
        print("\nTop 5 por score:")
        for p in new_posts[:5]:
            print(f"  [{p['date']}] score={p['score']:>6}  r/{p['subreddit']:<20}  {p['title'][:50]}")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\nCSV total: {len(df)} posts")
        years = df["date"].str[:4].value_counts().sort_index()
        for yr, cnt in years.items():
            print(f"  {yr}  {cnt:>5}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Actualiza el CSV con los posts más relevantes de los últimos días."
    )
    parser.add_argument(
        "--csv", default="data/nvda_processed.csv",
        help="Ruta al CSV del dataset (default: data/nvda_processed.csv)"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Días hacia atrás a buscar (default: 7)"
    )
    parser.add_argument(
        "--min_score", type=int, default=10,
        help="Upvotes mínimos para incluir un post (default: 10)"
    )
    parser.add_argument(
        "--models", default="finbert bert socbert",
        help="Modelos de sentimiento a usar"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Mostrar qué se añadiría sin escribir nada"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    models = args.models.split()

    # Cargar IDs existentes para deduplicar
    existing_ids: set[str] = set()
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path, usecols=["id"], dtype=str).fillna("")
        existing_ids = set(df_existing["id"].tolist())
        logger.info("CSV existente: %d posts (IDs cargados para deduplicar)", len(existing_ids))
    else:
        logger.info("CSV no existe, se creará nuevo en %s", csv_path)

    # Descargar posts nuevos
    new_posts = fetch_recent_posts(args.days, args.min_score, existing_ids)

    if not new_posts:
        logger.info("No hay posts nuevos relevantes en los últimos %d días.", args.days)
        print_summary([], 0, csv_path)
        return

    # Analizar y guardar
    added = analyze_and_save(new_posts, csv_path, models, args.dry_run)
    print_summary(new_posts, added, csv_path)


if __name__ == "__main__":
    main()
