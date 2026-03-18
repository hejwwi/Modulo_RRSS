"""
bert_analyzer.py
Analiza sentimiento en CSV de Reddit usando FinBERT (sentimiento financiero).
Incluye (opcional) sentimiento de la imagen del post (image_path) convirtiéndola a texto
con OCR y/o caption (BLIP), y fusiona ambos.

Requisitos base:
  pip install transformers torch pandas tqdm

Para imágenes:
  pip install pillow pytesseract
  (y tener Tesseract instalado en el sistema)  -> solo si usas OCR

Para captions:
  pip install transformers torch pillow  -> (ya lo tienes si usas transformers+torch)

Modelo sentimiento:
  ProsusAI/finbert (3 clases: positive/negative/neutral)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

# --- imports opcionales (solo si usas imágenes) ---
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception:
    torch = None
    BlipProcessor = None
    BlipForConditionalGeneration = None


def clean_text(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_post_text(row: pd.Series, include_comments: bool) -> str:
    title = str(row.get("title", "") or "")
    selftext = str(row.get("selftext", "") or "")
    text = (title + " " + selftext).strip()
    if include_comments:
        comments = str(row.get("comments_text", "") or "")
        if comments.strip():
            text = (text + " " + comments).strip()
    return clean_text(text)


def ocr_image(img_path: Path) -> str:
    if Image is None or pytesseract is None:
        return ""
    try:
        img = Image.open(img_path).convert("RGB")
        txt = pytesseract.image_to_string(img)
        return clean_text(txt)
    except Exception:
        return ""


def blip_caption(img_path: Path, processor, model, device: str) -> str:
    if Image is None:
        return ""
    try:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=40)
        cap = processor.decode(out[0], skip_special_tokens=True)
        return clean_text(cap)
    except Exception:
        return ""


def scores_from_finbert_output(out_list: List[dict]) -> Tuple[float, float, float, str]:
    """
    out_list: lista de dict(label, score) (porque usamos top_k=None)
    devuelve pos, neg, neu y etiqueta argmax
    """
    score_map = {d["label"].lower(): float(d["score"]) for d in out_list}
    pos = score_map.get("positive", 0.0)
    neg = score_map.get("negative", 0.0)
    neu = score_map.get("neutral", 0.0)
    best = max([("positive", pos), ("negative", neg), ("neutral", neu)], key=lambda x: x[1])[0]
    return pos, neg, neu, best


def parse_args():
    p = argparse.ArgumentParser(description="Sentiment Analyzer (FinBERT) para Reddit CSV (+ imágenes opcional)")
    p.add_argument("--symbol", required=True, help="Ticker, e.g. NVDA")
    p.add_argument("--data_dir", default="Data", help="Carpeta Data dentro de RedditScrapper (default: Data)")
    p.add_argument("--include_comments", action="store_true", help="Incluye comments_text si existe")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size para inferencia (default 16)")

    # ---- imágenes ----
    p.add_argument("--use_images", action="store_true", help="Analiza imágenes usando image_path (si existe)")
    p.add_argument("--image_mode", choices=["ocr", "caption", "both"], default="both",
                   help="Cómo convertir imagen a texto: ocr/caption/both (default both)")
    p.add_argument("--min_ocr_chars", type=int, default=25,
                   help="Si OCR tiene < min_ocr_chars, usa caption como fallback (default 25)")
    p.add_argument("--image_weight", type=float, default=0.25,
                   help="Peso de imagen en la fusión final (default 0.25). Texto pesa 1-w.")
    p.add_argument("--caption_model", default="Salesforce/blip-image-captioning-base",
                   help="Modelo de caption (default BLIP base)")

    # device para pipeline (transformers pipeline usa: -1 cpu, >=0 gpu id)
    p.add_argument("--device", type=int, default=-1, help="Device para pipeline: -1 CPU, 0 GPU0, 1 GPU1, ...")
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve()
    in_path = data_dir / f"{args.symbol.upper()}_reddit.csv"

    if not in_path.exists():
        raise SystemExit(f"No existe el CSV de entrada: {in_path}")

    df = pd.read_csv(in_path)

    # Pipeline de sentimiento (FinBERT)
    clf = pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,
        truncation=True,
        max_length=256,
        device=args.device,
    )

    # --------- 1) TEXT SENTIMENT ----------
    post_texts = [build_post_text(df.iloc[i], args.include_comments) for i in range(len(df))]
    post_texts = [t if t.strip() else " " for t in post_texts]

    post_labels, post_pos, post_neg, post_neu = [], [], [], []

    for i in tqdm(range(0, len(post_texts), args.batch_size), desc="FinBERT (post text)"):
        batch = post_texts[i:i + args.batch_size]
        outputs = clf(batch)
        for out in outputs:
            pos, neg, neu, best = scores_from_finbert_output(out)
            post_labels.append(best)
            post_pos.append(pos)
            post_neg.append(neg)
            post_neu.append(neu)

    df_out = df.copy()
    df_out["sent_finbert_label"] = post_labels
    df_out["sent_finbert_pos"] = post_pos
    df_out["sent_finbert_neg"] = post_neg
    df_out["sent_finbert_neu"] = post_neu

    # --------- 2) IMAGE SENTIMENT (optional) ----------
    if args.use_images:
        if "image_path" not in df_out.columns:
            raise SystemExit("Tu CSV no tiene 'image_path'. Ejecuta scraper/updater con --download_images.")

        # Preparar BLIP si hace falta
        blip_processor = None
        blip_model = None
        blip_device = "cpu"
        if args.image_mode in ("caption", "both"):
            if torch is None or BlipProcessor is None:
                raise SystemExit("Falta torch/transformers para captions. Instala: pip install torch transformers")
            blip_device = "cuda" if (torch.cuda.is_available() and args.device >= 0) else "cpu"
            blip_processor = BlipProcessor.from_pretrained(args.caption_model)
            blip_model = BlipForConditionalGeneration.from_pretrained(args.caption_model).to(blip_device)

        img_texts = []
        img_ocr_texts = []
        img_caps = []

        for _, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Extract image text"):
            rel = str(row.get("image_path") or "").strip()
            if not rel:
                img_ocr_texts.append("")
                img_caps.append("")
                img_texts.append(" ")
                continue

            img_path = data_dir / rel  # Data/ + images/...
            if not img_path.exists():
                img_ocr_texts.append("")
                img_caps.append("")
                img_texts.append(" ")
                continue

            ocr_txt = ""
            cap_txt = ""

            if args.image_mode in ("ocr", "both"):
                ocr_txt = ocr_image(img_path)

            if args.image_mode in ("caption", "both"):
                # si both: caption solo si OCR flojo
                if args.image_mode == "caption" or len(ocr_txt) < args.min_ocr_chars:
                    cap_txt = blip_caption(img_path, blip_processor, blip_model, blip_device)

            chosen = ocr_txt if len(ocr_txt) >= args.min_ocr_chars else cap_txt
            chosen = clean_text(chosen)
            img_ocr_texts.append(ocr_txt)
            img_caps.append(cap_txt)
            img_texts.append(chosen if chosen else " ")

        # pasar texto de imagen por FinBERT
        img_labels, img_pos, img_neg, img_neu = [], [], [], []

        for i in tqdm(range(0, len(img_texts), args.batch_size), desc="FinBERT (image text)"):
            batch = img_texts[i:i + args.batch_size]
            outputs = clf(batch)
            for out in outputs:
                pos, neg, neu, best = scores_from_finbert_output(out)
                img_labels.append(best)
                img_pos.append(pos)
                img_neg.append(neg)
                img_neu.append(neu)

        df_out["image_ocr_text"] = img_ocr_texts
        df_out["image_caption"] = img_caps
        df_out["image_text_for_bert"] = img_texts
        df_out["img_finbert_label"] = img_labels
        df_out["img_finbert_pos"] = img_pos
        df_out["img_finbert_neg"] = img_neg
        df_out["img_finbert_neu"] = img_neu

        # --------- 3) FUSIÓN ----------
        w_img = float(args.image_weight)
        w_img = max(0.0, min(1.0, w_img))
        w_txt = 1.0 - w_img

        # Si no hay imagen_text real (solo " "), ponemos w_img = 0 para esa fila
        has_img_text = pd.Series(img_texts).apply(lambda x: bool(x.strip()) and x.strip() != "")
        has_img_text = has_img_text & df_out["image_path"].fillna("").astype(str).str.len().gt(0)

        # final_* = w_txt*text + w_img*img (si no hay img, solo text)
        df_out["final_finbert_pos"] = df_out["sent_finbert_pos"]
        df_out["final_finbert_neg"] = df_out["sent_finbert_neg"]
        df_out["final_finbert_neu"] = df_out["sent_finbert_neu"]

        df_out.loc[has_img_text, "final_finbert_pos"] = (
            w_txt * df_out.loc[has_img_text, "sent_finbert_pos"] + w_img * df_out.loc[has_img_text, "img_finbert_pos"]
        )
        df_out.loc[has_img_text, "final_finbert_neg"] = (
            w_txt * df_out.loc[has_img_text, "sent_finbert_neg"] + w_img * df_out.loc[has_img_text, "img_finbert_neg"]
        )
        df_out.loc[has_img_text, "final_finbert_neu"] = (
            w_txt * df_out.loc[has_img_text, "sent_finbert_neu"] + w_img * df_out.loc[has_img_text, "img_finbert_neu"]
        )

        # label final = argmax de los 3
        def _final_label(row):
            pos, neg, neu = row["final_finbert_pos"], row["final_finbert_neg"], row["final_finbert_neu"]
            return max([("positive", pos), ("negative", neg), ("neutral", neu)], key=lambda x: x[1])[0]

        df_out["final_finbert_label"] = df_out.apply(_final_label, axis=1)

    # --------- Guardado ----------
    suffix = "_reddit_finbert_img.csv" if args.use_images else "_reddit_finbert.csv"
    out_path = data_dir / f"{args.symbol.upper()}{suffix}"
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Guardado: {out_path}")
    cols_show = ["created_iso", "subreddit", "sent_finbert_label"]
    if args.use_images:
        cols_show += ["img_finbert_label", "final_finbert_label"]
    print(df_out[cols_show].head(10).to_string(index=False))


if __name__ == "__main__":
    main()


