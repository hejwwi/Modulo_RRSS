#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

INPUT_FILE = Path("merged_sorted_reddit.json")
OUTPUT_FILE = Path("final_dataset_clean.json")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return value.strip()


def normalize_url(url: Any) -> Optional[str]:
    if url is None:
        return None
    if not isinstance(url, str):
        url = str(url)

    url = html.unescape(url.strip())

    if not url:
        return None

    return url


def is_probably_image_url(url: Optional[str]) -> bool:
    if not url:
        return False

    u = url.lower()

    if any(ext in u for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
        return True

    if any(host in u for host in ["i.redd.it", "preview.redd.it", "i.imgur.com"]):
        return True

    return False


def deduplicate_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extrae URLs desde selftext/combined_text.
    Detecta también URLs con &amp; y las normaliza.
    """
    if not text:
        return []

    text = html.unescape(text)

    # URLs normales hasta espacio/salto de línea
    matches = re.findall(r'https?://[^\s)>\]"]+', text, flags=re.IGNORECASE)

    urls = []
    for m in matches:
        url = normalize_url(m)
        if is_probably_image_url(url):
            urls.append(url)

    return deduplicate_keep_order(urls)


def extract_image_urls(post: Dict[str, Any]) -> List[str]:
    urls: List[str] = []

    # 1. preview.images[].source.url y resolutions[]
    preview = post.get("preview")
    if isinstance(preview, dict):
        images = preview.get("images", [])
        if isinstance(images, list):
            for img in images:
                if not isinstance(img, dict):
                    continue

                source = img.get("source", {})
                if isinstance(source, dict):
                    url = normalize_url(source.get("url"))
                    if url:
                        urls.append(url)

                resolutions = img.get("resolutions", [])
                if isinstance(resolutions, list):
                    for r in resolutions:
                        if not isinstance(r, dict):
                            continue
                        url = normalize_url(r.get("url"))
                        if url:
                            urls.append(url)

    # 2. url_overridden_by_dest
    url = normalize_url(post.get("url_overridden_by_dest"))
    if is_probably_image_url(url):
        urls.append(url)

    # 3. url
    url = normalize_url(post.get("url"))
    if is_probably_image_url(url):
        urls.append(url)

    # 4. galerías
    media_metadata = post.get("media_metadata")
    if isinstance(media_metadata, dict):
        for _, item in media_metadata.items():
            if not isinstance(item, dict):
                continue

            s = item.get("s", {})
            if isinstance(s, dict):
                url = normalize_url(s.get("u"))
                if url:
                    urls.append(url)

            p_list = item.get("p", [])
            if isinstance(p_list, list):
                for p in p_list:
                    if not isinstance(p, dict):
                        continue
                    url = normalize_url(p.get("u"))
                    if url:
                        urls.append(url)

    # 5. selftext
    selftext = normalize_text(post.get("selftext", ""))
    urls.extend(extract_urls_from_text(selftext))

    # 6. combined_text
    combined_text = normalize_text(post.get("combined_text", ""))
    urls.extend(extract_urls_from_text(combined_text))

    return deduplicate_keep_order(urls)


def build_combined_text(post: Dict[str, Any]) -> str:
    title = normalize_text(post.get("title", ""))
    selftext = normalize_text(post.get("selftext", ""))

    if selftext and selftext.lower() not in {"[deleted]", "[removed]"}:
        return f"{title}\n\n{selftext}".strip()

    return title


def clean_post(post: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}

    clean["id"] = post.get("id")
    clean["title"] = normalize_text(post.get("title", ""))
    clean["selftext"] = normalize_text(post.get("selftext", ""))
    clean["combined_text"] = build_combined_text(post)

    clean["subreddit"] = post.get("subreddit")
    clean["created_utc"] = post.get("created_utc")
    clean["date"] = post.get("date")
    clean["score"] = post.get("score", 0)
    clean["num_comments"] = post.get("num_comments", 0)

    image_urls = extract_image_urls(post)
    clean["image_urls"] = image_urls
    clean["has_image"] = len(image_urls) > 0

    clean["image_analysis"] = post.get("image_analysis", None)

    return clean


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El archivo de entrada debe contener una lista JSON.")

    cleaned: List[Dict[str, Any]] = []
    total = len(data)
    posts_with_images = 0

    for i, post in enumerate(data, start=1):
        if not isinstance(post, dict):
            continue

        clean = clean_post(post)
        cleaned.append(clean)

        if clean["has_image"]:
            posts_with_images += 1

        if i % 10000 == 0 or i == total:
            print(f"Procesados: {i}/{total}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("\n===== RESUMEN =====")
    print(f"Total posts:         {len(cleaned)}")
    print(f"Posts con imagen:    {posts_with_images}")
    print(f"Posts sin imagen:    {len(cleaned) - posts_with_images}")
    print(f"Guardado en:         {OUTPUT_FILE}")
    print("===================\n")

    # debug rápido
    print("Ejemplo de posts con imagen detectada:")
    shown = 0
    for post in cleaned:
        if post["has_image"]:
            print(f"- {post['id']} -> {post['image_urls'][:1]}")
            shown += 1
            if shown == 10:
                break


if __name__ == "__main__":
    main()