#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Limpieza básica de dataset Reddit")
    parser.add_argument("--input", required=True, help="Archivo de entrada")
    parser.add_argument("--output", required=True, help="Archivo de salida")
    return parser.parse_args()


def is_deleted(post):
    # 1. eliminado por moderador
    if "removal_type" in post:
        return True

    # 2. título eliminado
    if post.get("title") in ["[deleted]", "[removed]"]:
        return True

    # 3. texto eliminado
    if post.get("selftext") in ["[deleted]", "[removed]"]:
        return True

    return False


def detect_format(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                raise ValueError("Archivo vacío")
            if not ch.isspace():
                return "json_array" if ch == "[" else "jsonl"


def load_posts(filepath):
    fmt = detect_format(filepath)

    if fmt == "json_array":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            for post in data:
                yield post
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except:
                        continue


def main():
    args = parse_args()

    seen_ids = set()
    clean_posts = []

    total = 0
    removed = 0
    duplicates = 0

    for post in load_posts(args.input):
        total += 1

        # eliminar eliminados
        if is_deleted(post):
            removed += 1
            continue

        # eliminar duplicados
        post_id = post.get("id")
        if post_id:
            if post_id in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(post_id)

        clean_posts.append(post)

    print("\n===== LIMPIEZA =====")
    print(f"Total:        {total}")
    print(f"Eliminados:   {removed}")
    print(f"Duplicados:   {duplicates}")
    print(f"Final:        {len(clean_posts)}")
    print("====================\n")

    # guardar como JSON (lista)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(clean_posts, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()