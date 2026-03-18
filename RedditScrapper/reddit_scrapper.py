from __future__ import annotations

import argparse
import html
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

# ----------------------------
# CONFIG
# ----------------------------

BASE_PRIMARY = "https://www.reddit.com"
BASE_FALLBACK = "https://old.reddit.com"

DEFAULT_UA = "TFG-NVDA-Backfill/1.0 (no-oauth; contact: u/your_user)"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

TOP3_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]

SEARCH_QUERIES = [
    r'(NVDA OR "$NVDA" OR NVIDIA)',
    r'("Jensen Huang" OR Huang OR Jensen)',
    r'(Blackwell OR Hopper OR "H100" OR "B200" OR "GB200")',
    r'(CUDA OR "TensorRT" OR "DGX" OR "Grace")',
    r'(GeForce OR RTX OR "RTX 4090" OR "RTX 5090" OR "AI chips")',
    r'(earnings OR guidance OR revenue OR margin) (NVDA OR NVIDIA)',
]

# conservador para evitar 429
SLEEP_BETWEEN_PAGES = 2.2
SLEEP_BETWEEN_IMAGES = 0.9
MAX_RETRIES = 10
TIMEOUT = 25

# ----------------------------
# HELPERS
# ----------------------------

@dataclass
class RedditConfig:
    user_agent: str


def load_config() -> RedditConfig:
    load_dotenv()
    ua = os.getenv("REDDIT_USER_AGENT", "").strip() or DEFAULT_UA
    return RedditConfig(user_agent=ua)


def unix_to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def clean_url(u: str) -> str:
    return html.unescape(u or "").strip()


def guess_ext(url: str) -> str:
    try:
        ext = Path(urlparse(url).path).suffix.lower()
        if ext in IMAGE_EXTS:
            return ext
    except Exception:
        pass
    return ".jpg"


def ensure_dirs(script_path: Path) -> Tuple[Path, Path, Path]:
    data_dir = script_path.parent / "Data"
    images_dir = data_dir / "images"
    state_dir = data_dir / "_state"
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, images_dir, state_dir


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(path: Path, state: Dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def request_json(base_url: str, endpoint: str, params: Optional[Dict], cfg: RedditConfig) -> Optional[Dict]:
    if params is None:
        params = {}
    params.setdefault("raw_json", 1)

    headers = {"User-Agent": cfg.user_agent, "Accept": "application/json,text/plain,*/*"}
    bases = [base_url]
    if base_url != BASE_FALLBACK:
        bases.append(BASE_FALLBACK)

    for base in bases:
        url = f"{base}{endpoint}"

        for i in range(MAX_RETRIES):
            try:
                r = requests.get(url, params=params, headers=headers, timeout=TIMEOUT, allow_redirects=True)

                if r.status_code == 200:
                    ct = (r.headers.get("Content-Type") or "").lower()
                    if "json" not in ct and not r.text.strip().startswith("{"):
                        wait = min(20, (1.6 ** i) + 1) + random.uniform(0.0, 1.5)
                        time.sleep(wait)
                        continue
                    return r.json()

                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    wait = (int(ra) + 2) if (ra and ra.isdigit()) else (2.0 ** i) + 5
                    wait += random.uniform(0.0, 2.0)
                    print(f"[WARN] HTTP 429 | wait {wait:.1f}s")
                    time.sleep(wait)
                    continue

                if r.status_code == 403:
                    wait = min(25, (1.8 ** i) + 2) + random.uniform(0.0, 1.5)
                    print(f"[WARN] HTTP 403 | wait {wait:.1f}s")
                    time.sleep(wait)
                    continue

                wait = min(15, (1.5 ** i) + 1) + random.uniform(0.0, 1.5)
                print(f"[WARN] HTTP {r.status_code} | wait {wait:.1f}s")
                time.sleep(wait)

            except Exception:
                wait = min(15, (1.6 ** i) + 1) + random.uniform(0.0, 1.5)
                time.sleep(wait)

    return None


def pick_main_image_url(p: Dict) -> Optional[str]:
    post_hint = (p.get("post_hint") or "").lower()
    url = clean_url(p.get("url_overridden_by_dest") or p.get("url") or "")

    if post_hint == "image" and url:
        return url

    preview = p.get("preview") or {}
    images = preview.get("images") or []
    if images:
        src = (images[0].get("source") or {}).get("url")
        src = clean_url(src)
        if src:
            return src

    if p.get("is_gallery") and isinstance(p.get("media_metadata"), dict):
        mm = p["media_metadata"]
        for _, meta in mm.items():
            if not isinstance(meta, dict):
                continue
            s = meta.get("s") or {}
            u = clean_url(s.get("u") or "")
            if u:
                return u

    thumb = clean_url(p.get("thumbnail") or "")
    if thumb.startswith("http"):
        return thumb

    return None


def download_image(url: str, out_path: Path, cfg: RedditConfig) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    headers = {"User-Agent": cfg.user_agent}
    for i in range(7):
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
            if r.status_code == 200 and r.content:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(r.content)
                return True

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = (int(ra) + 2) if (ra and ra.isdigit()) else (2.0 ** i) + 4
                wait += random.uniform(0.0, 1.5)
                time.sleep(wait)
                continue

            time.sleep(min(12, (1.6 ** i) + 1))

        except Exception:
            time.sleep(min(12, (1.6 ** i) + 1))

    return False


def search_window(subreddit: str, query: str, start_ts: int, end_ts: int, cfg: RedditConfig, limit: int) -> List[Dict]:
    endpoint = f"/r/{subreddit}/search.json"
    after = None
    collected: List[Dict] = []
    per_page = 100

    q = f'({query}) AND timestamp:{start_ts}..{end_ts}'

    while len(collected) < limit:
        params = {
            "q": q,
            "restrict_sr": 1,
            "sort": "new",
            "syntax": "lucene",
            "limit": per_page,
            "after": after,
            "raw_json": 1,
        }

        data = request_json(BASE_PRIMARY, endpoint, params=params, cfg=cfg)
        if not data:
            break

        listing = data.get("data", {}) if isinstance(data, dict) else {}
        children = listing.get("children", [])
        if not children:
            break

        for child in children:
            d = child.get("data", {}) if isinstance(child, dict) else {}
            if d:
                collected.append(d)
                if len(collected) >= limit:
                    break

        after = listing.get("after")
        if not after:
            break

        time.sleep(SLEEP_BETWEEN_PAGES)

    return collected


def build_row(p: Dict, subreddit: str, query_used: str, start_ts: int, end_ts: int) -> Dict:
    created_utc = float(p.get("created_utc") or 0.0)
    permalink = p.get("permalink", "")
    title = (p.get("title") or "").strip()
    selftext = (p.get("selftext") or "").strip()

    image_url = pick_main_image_url(p)
    has_image = bool(image_url)

    return {
        "subreddit": subreddit,
        "id": str(p.get("id") or "").strip(),
        "title": title,
        "selftext": selftext,
        "created_utc": created_utc,
        "created_iso": unix_to_iso(int(created_utc)) if created_utc else "",
        "score": int(p.get("score") or 0),
        "num_comments": int(p.get("num_comments") or 0),
        "author": str(p.get("author") or ""),
        "permalink": f"{BASE_PRIMARY}{permalink}" if permalink else "",
        "url": p.get("url", ""),
        "has_image": has_image,
        "image_url": image_url or "",
        "image_path": "",
        "query_used": query_used,
        "window_start": start_ts,
        "window_end": end_ts,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Backfill NVDA year by running multiple distinct windows")
    p.add_argument("--window_days", type=int, default=7, help="tamaño de ventana hacia atrás (default 7 días)")
    p.add_argument("--limit_per_query", type=int, default=1200, help="máx por query en una ventana (default 1200)")
    p.add_argument("--out_csv", default="nvda_top3_backfill.csv", help="CSV acumulado en Data/")
    p.add_argument("--reset", action="store_true", help="reinicia cursor (empieza desde ahora)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    script_path = Path(__file__).resolve()
    data_dir, images_dir, state_dir = ensure_dirs(script_path)

    out_csv = data_dir / args.out_csv
    state_path = state_dir / "nvda_backfill_state.json"
    state = {} if args.reset else load_state(state_path)

    now_ts = int(datetime.now(timezone.utc).timestamp())
    year_ago_ts = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp())

    # cursor_end_ts: hasta dónde llega el backfill en esta ejecución
    cursor_end_ts = int(state.get("cursor_end_ts", now_ts))

    # siguiente ventana (hacia atrás)
    window_seconds = args.window_days * 24 * 3600
    start_ts = max(year_ago_ts, cursor_end_ts - window_seconds)
    end_ts = cursor_end_ts

    if end_ts <= year_ago_ts:
        print("[OK] Ya has completado el último año.")
        return

    print(f"[INFO] Ventana backfill: {unix_to_iso(start_ts)} -> {unix_to_iso(end_ts)}")

    rows: List[Dict] = []

    for sub in TOP3_SUBREDDITS:
        print(f"\n[INFO] r/{sub}")
        for q in SEARCH_QUERIES:
            posts = search_window(sub, q, start_ts, end_ts, cfg, limit=args.limit_per_query)

            for p in posts:
                row = build_row(p, sub, q, start_ts, end_ts)
                if not row["has_image"]:
                    continue

                ext = guess_ext(row["image_url"])
                out_path = images_dir / sub / f"{row['id']}{ext}"
                if download_image(row["image_url"], out_path, cfg):
                    try:
                        row["image_path"] = str(out_path.relative_to(data_dir))
                    except Exception:
                        row["image_path"] = str(out_path)

                time.sleep(SLEEP_BETWEEN_IMAGES)
                rows.append(row)

    new_df = pd.DataFrame(rows)

    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        final_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    if not final_df.empty and "created_utc" in final_df.columns:
        final_df["created_utc"] = final_df["created_utc"].astype(float)
        final_df = final_df.drop_duplicates(subset=["id"]).sort_values(by="created_utc", ascending=False)

    final_df.to_csv(out_csv, index=False, encoding="utf-8")

    # mover cursor hacia atrás y guardar estado
    state["cursor_end_ts"] = start_ts
    save_state(state_path, state)

    print(f"\n[OK] Añadidos (sin dedupe): {len(new_df)}")
    print(f"[OK] Total acumulado (dedupe): {len(final_df)}")
    print(f"[OK] Siguiente ejecución cubrirá: {unix_to_iso(max(year_ago_ts, start_ts - window_seconds))[:19]} -> {unix_to_iso(start_ts)[:19]}")
    print(f"[OK] CSV: {out_csv}")


if __name__ == "__main__":
    main()