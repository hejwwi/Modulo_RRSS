"""
socbert_analyzer.py
Analiza sentimiento con SocBERT como backbone.

Como SocBERT-base es MLM (no viene como clasificador), este script permite:
  - --train: fine-tune rápido sobre Sentiment140 (pos/neg)
  - inferencia en tu CSV de Reddit usando el modelo entrenado local

Además (opcional):
  - --use_images: usa 'image_path' (si existe) para extraer texto de la imagen (OCR y/o caption BLIP)
    y pasar ese texto por el clasificador SocBERT, fusionándolo con el sentimiento del texto del post.

Requisitos:
  pip install transformers torch datasets accelerate pandas tqdm

Para OCR (opcional):
  pip install pillow pytesseract
  (y tener Tesseract instalado en el sistema)

Para captions (opcional):
  pip install pillow
  (torch+transformers ya están arriba)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


# --- imports opcionales para imagen ---
try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception:
    BlipProcessor = None
    BlipForConditionalGeneration = None


SOCBERT_BACKBONE = "sarkerlab/SocBERT-base"         # pretrained MLM backbone
LOCAL_MODEL_DIR = "models/socbert_sentiment"        # donde guardamos el fine-tuned


def clean_text(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def build_text(row: pd.Series, include_comments: bool) -> str:
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


def parse_args():
    p = argparse.ArgumentParser(description="Sentiment Analyzer (SocBERT) para Reddit CSV (+ imágenes opcional)")
    p.add_argument("--symbol", required=True, help="Ticker, e.g. NVDA")
    p.add_argument("--data_dir", default="Data", help="Carpeta Data dentro de RedditScrapper (default: Data)")
    p.add_argument("--include_comments", action="store_true", help="Incluye comments_text si existe")

    # Entrenamiento
    p.add_argument("--train", action="store_true", help="Fine-tunea SocBERT con Sentiment140 (pos/neg)")
    p.add_argument("--train_samples", type=int, default=20000, help="Muestras de train a usar (default 20000)")
    p.add_argument("--eval_samples", type=int, default=4000, help="Muestras de eval a usar (default 4000)")
    p.add_argument("--epochs", type=int, default=1, help="Épocas (default 1)")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size (default 16)")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default 2e-5)")

    # Imágenes
    p.add_argument("--use_images", action="store_true", help="Analiza imágenes usando image_path (si existe)")
    p.add_argument("--image_mode", choices=["ocr", "caption", "both"], default="both",
                   help="Cómo convertir imagen a texto: ocr/caption/both (default both)")
    p.add_argument("--min_ocr_chars", type=int, default=25,
                   help="Si OCR tiene < min_ocr_chars, usa caption como fallback (default 25)")
    p.add_argument("--image_weight", type=float, default=0.25,
                   help="Peso de imagen en la fusión final (default 0.25). Texto pesa 1-w.")
    p.add_argument("--caption_model", default="Salesforce/blip-image-captioning-base",
                   help="Modelo de caption (default BLIP base)")

    return p.parse_args()


def finetune_socbert(save_dir: Path, train_samples: int, eval_samples: int, epochs: int, batch_size: int, lr: float):
    """
    Fine-tuning ligero para tener un clasificador pos/neg.
    """
    print("[INFO] Cargando dataset Sentiment140…")
    ds = load_dataset("stanfordnlp/sentiment140")  # requiere internet la primera vez

    # Sentiment140 labels típicas: 0=negative, 4=positive
    def preprocess(example):
        text = example["text"]
        label_raw = int(example["sentiment"])
        label = 0 if label_raw == 0 else 1  # 0 neg, 1 pos
        return {"text": text, "label": label}

    ds = ds.map(preprocess, remove_columns=ds["train"].column_names)

    # Submuestreo para hacerlo rápido
    train_ds = ds["train"].shuffle(seed=42).select(range(min(train_samples, len(ds["train"]))))
    eval_ds = ds["train"].shuffle(seed=123).select(range(min(eval_samples, len(ds["train"]))))

    tokenizer = AutoTokenizer.from_pretrained(SOCBERT_BACKBONE, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_tok = train_ds.map(tokenize, batched=True)
    eval_tok = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(SOCBERT_BACKBONE, num_labels=2)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(save_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=False,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("[INFO] Entrenando…")
    trainer.train()

    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    print(f"[OK] Modelo guardado en: {save_dir}")


@torch.inference_mode()
def _infer_socbert_probs(
    tokenizer,
    model,
    texts: List[str],
    batch_size: int,
) -> Tuple[List[str], List[float], List[float]]:
    """
    Devuelve labels, score_neg, score_pos para cada texto.
    """
    labels: List[str] = []
    score_neg: List[float] = []
    score_pos: List[float] = []

    device = next(model.parameters()).device

    for i in tqdm(range(0, len(texts), batch_size), desc="SocBERT"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for p in probs:
            neg = float(p[0])
            pos = float(p[1])
            lab = "positive" if pos >= neg else "negative"
            labels.append(lab)
            score_neg.append(neg)
            score_pos.append(pos)

    return labels, score_neg, score_pos


@torch.inference_mode()
def run_inference_with_images(
    model_dir: Path,
    df: pd.DataFrame,
    data_dir: Path,
    include_comments: bool,
    batch_size: int,
    use_images: bool,
    image_mode: str,
    min_ocr_chars: int,
    image_weight: float,
    caption_model: str,
) -> pd.DataFrame:
    """
    Infiere sentimiento del texto del post, opcionalmente del texto derivado de la imagen,
    y fusiona ambos.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 1) Texto del post ---
    post_texts = [build_text(df.iloc[i], include_comments) for i in range(len(df))]
    post_texts = [t if t.strip() else " " for t in post_texts]

    post_labels, post_neg, post_pos = _infer_socbert_probs(tokenizer, model, post_texts, batch_size)

    out = df.copy()
    out["sent_socbert_label"] = post_labels
    out["sent_socbert_neg"] = post_neg
    out["sent_socbert_pos"] = post_pos

    # --- 2) Imagen (opcional) ---
    if not use_images:
        return out

    if "image_path" not in out.columns:
        raise SystemExit("Tu CSV no tiene 'image_path'. Ejecuta scraper/updater con --download_images.")

    # Preparar BLIP si hace falta
    blip_processor = None
    blip_model = None
    blip_device = "cpu"
    if image_mode in ("caption", "both"):
        if BlipProcessor is None or BlipForConditionalGeneration is None:
            raise SystemExit("Falta BLIP en transformers. Actualiza transformers o instala bien transformers.")
        blip_device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor = BlipProcessor.from_pretrained(caption_model)
        blip_model = BlipForConditionalGeneration.from_pretrained(caption_model).to(blip_device)

    img_texts: List[str] = []
    img_ocr_texts: List[str] = []
    img_caps: List[str] = []

    for _, row in tqdm(out.iterrows(), total=len(out), desc="Extract image text"):
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

        if image_mode in ("ocr", "both"):
            ocr_txt = ocr_image(img_path)

        if image_mode in ("caption", "both"):
            if image_mode == "caption" or len(ocr_txt) < min_ocr_chars:
                cap_txt = blip_caption(img_path, blip_processor, blip_model, blip_device)

        chosen = ocr_txt if len(ocr_txt) >= min_ocr_chars else cap_txt
        chosen = clean_text(chosen)

        img_ocr_texts.append(ocr_txt)
        img_caps.append(cap_txt)
        img_texts.append(chosen if chosen else " ")

    img_labels, img_neg, img_pos = _infer_socbert_probs(tokenizer, model, img_texts, batch_size)

    out["image_ocr_text"] = img_ocr_texts
    out["image_caption"] = img_caps
    out["image_text_for_socbert"] = img_texts
    out["img_socbert_label"] = img_labels
    out["img_socbert_neg"] = img_neg
    out["img_socbert_pos"] = img_pos

    # --- 3) Fusión ---
    w_img = max(0.0, min(1.0, float(image_weight)))
    w_txt = 1.0 - w_img

    # si no hay imagen real, no aplicamos peso de imagen
    has_img = out["image_path"].fillna("").astype(str).str.len().gt(0) & pd.Series(img_texts).apply(lambda x: x.strip() != "")

    out["final_socbert_neg"] = out["sent_socbert_neg"]
    out["final_socbert_pos"] = out["sent_socbert_pos"]

    out.loc[has_img, "final_socbert_neg"] = (
        w_txt * out.loc[has_img, "sent_socbert_neg"] + w_img * out.loc[has_img, "img_socbert_neg"]
    )
    out.loc[has_img, "final_socbert_pos"] = (
        w_txt * out.loc[has_img, "sent_socbert_pos"] + w_img * out.loc[has_img, "img_socbert_pos"]
    )

    out["final_socbert_label"] = out.apply(
        lambda r: "positive" if float(r["final_socbert_pos"]) >= float(r["final_socbert_neg"]) else "negative",
        axis=1,
    )

    return out


def main():
    args = parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = (base_dir / args.data_dir).resolve()
    in_path = data_dir / f"{args.symbol.upper()}_reddit.csv"

    if not in_path.exists():
        raise SystemExit(f"No existe el CSV de entrada: {in_path}")

    model_dir = (base_dir / LOCAL_MODEL_DIR).resolve()

    if args.train or not model_dir.exists():
        print("[INFO] No hay modelo fine-tuned local o has pedido --train.")
        print(f"[INFO] Backbone: {SOCBERT_BACKBONE}")
        finetune_socbert(
            save_dir=model_dir,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )

    df = pd.read_csv(in_path)

    df_out = run_inference_with_images(
        model_dir=model_dir,
        df=df,
        data_dir=data_dir,
        include_comments=args.include_comments,
        batch_size=args.batch_size,
        use_images=args.use_images,
        image_mode=args.image_mode,
        min_ocr_chars=args.min_ocr_chars,
        image_weight=args.image_weight,
        caption_model=args.caption_model,
    )

    suffix = "_reddit_socbert_img.csv" if args.use_images else "_reddit_socbert.csv"
    out_path = data_dir / f"{args.symbol.upper()}{suffix}"
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Guardado: {out_path}")
    cols = ["created_iso", "subreddit", "sent_socbert_label"]
    if args.use_images:
        cols += ["img_socbert_label", "final_socbert_label"]
    print(df_out[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
