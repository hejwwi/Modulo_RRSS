#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime

# 📂 Tus datasets
INPUT_FILES = [
    ("wsb_clean.json", "wallstreetbets"),
    ("Stocks_clean.json", "stocks"),
    ("investing_clean.json", "investing"),
    ("EducatedInvesting_clean.json", "EducatedInvesting"),
    ("StockMarket_clean.json", "StockMarket"),
    ("pennystocks_clean.json", "pennystocks"),
    ("wsbNEWS_clean.json", "wallstreetbetsnews"),
    ("wallstreetbets2_clean.json", "wallstreetbets2"),
    
]

OUTPUT_FILE = "merged_sorted_reddit.json"


def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    merged = []
    seen_ids = set()

    total = 0
    duplicates = 0

    for file, subreddit_name in INPUT_FILES:
        print(f"Cargando {file}...")

        data = load_json(file)

        for post in data:
            total += 1

            post_id = post.get("id")

            # ❌ eliminar duplicados globales
            if post_id and post_id in seen_ids:
                duplicates += 1
                continue

            if post_id:
                seen_ids.add(post_id)

            # ✅ guardar subreddit origen
            post["subreddit"] = subreddit_name

            # ⚠️ asegurar timestamp
            created = post.get("created_utc", 0)

            # si viene como string lo convertimos
            try:
                created = int(created)
            except:
                created = 0

            post["created_utc"] = created

            # ✅ fecha legible (MUY útil para análisis)
            post["date"] = datetime.utcfromtimestamp(created).strftime("%Y-%m-%d %H:%M:%S")

            merged.append(post)

    # 🔥 ORDENAR POR TIEMPO
    merged.sort(key=lambda x: x["created_utc"])

    print("\n===== FUSIÓN =====")
    print(f"Total leídos:   {total}")
    print(f"Duplicados:     {duplicates}")
    print(f"Final:          {len(merged)}")
    print("==================\n")

    # 💾 guardar
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()