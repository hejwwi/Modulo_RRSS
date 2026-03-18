"""
bert_general_analyzer.py
Analiza sentimiento en CSV de Reddit usando un BERT general (SST-2).
Salida: Data/{SYMBOL}_reddit_bert_general.csv

Requisitos:
  pip install transformers torch pandas tqdm

Modelo recomendado (rápido y fiable):
  distilbert-base-uncased-finetuned-sst-2-english
(Es "BERT-like" y va genial para comparar vs FinBERT/SocBERT)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import pipeline


def build_text(row: pd.Series, include_comments: bool) -> str:
    title = str(row.get("title", "") or "")
    selftext = str(row.get("selftext", "") or "")
    text = (title + " " + selftext).strip()
    if include_comments:
        comments = str(row.get("comments_text", "") or "")
        if comments.strip():
            text = (text + " " + comments).strip()
    return text if text else " "


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sentiment Analyzer (BERT general) para Reddit CSV")
    p.add_argument("--symbol", required=True, help="Ticker, e.g. NVDA")
    p.add_argument("--data_dir", default="Data", help="Carpeta Data dentro de RedditScrapper (default: Data)")
    p.add_argument("--include_comments", action="store_true", help="Incluye comments_text si existe")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size para inferencia (default 32)")
    p.add_argument(
        "--model",
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Modelo HF para sentimiento general (default: distilbert SST-2)",
    )
    p.add_argument("--max_length", type=int, default=256, help="Longitud máxima de tokens (default 256)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve()
    in_path = data_dir / f"{args.symbol.upper()}_reddit.csv"

    if not in_path.exists():
        raise SystemExit(f"No existe el CSV de entrada: {in_path}")

    df = pd.read_csv(in_path)

    clf = pipeline(
        task="sentiment-analysis",
        model=args.model,
        tokenizer=args.model,
        truncation=True,
        max_length=args.max_length,
    )

    texts = [build_text(df.iloc[i], args.include_comments) for i in range(len(df))]

    out_label = []
    out_score = []
    out_pos = []
    out_neg = []

    for i in tqdm(range(0, len(texts), args.batch_size), desc="BERT general"):
        batch = texts[i : i + args.batch_size]
        preds = clf(batch)  # [{'label': 'POSITIVE'/'NEGATIVE', 'score': ...}, ...]

        for p in preds:
            label = str(p["label"]).lower()  # positive/negative
            score = float(p["score"])

            # Convertimos a columnas comparables
            if "pos" in label:
                pos, neg = score, 1.0 - score
                norm_label = "positive"
            else:
                neg, pos = score, 1.0 - score
                norm_label = "negative"

            out_label.append(norm_label)
            out_score.append(score)
            out_pos.append(pos)
            out_neg.append(neg)

    df_out = df.copy()
    df_out["sent_bert_label"] = out_label
    df_out["sent_bert_score"] = out_score
    df_out["sent_bert_pos"] = out_pos
    df_out["sent_bert_neg"] = out_neg

    out_path = data_dir / f"{args.symbol.upper()}_reddit_bert_general.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Guardado: {out_path}")
    print(df_out[["created_iso", "subreddit", "sent_bert_label", "sent_bert_score"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
