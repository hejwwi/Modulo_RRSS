"""Comparador multimodal: fusión de señales de texto e imagen para NVDA."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Mapeo de etiqueta → índice para argmax
_LABEL_IDX = {"positive": 0, "negative": 1, "neutral": 2}
_IDX_LABEL = {0: "positive", 1: "negative", 2: "neutral"}


def _text_probs(post: dict) -> tuple[float, float, float]:
    """Extrae (pos, neg, neu) del modelo de texto preferido del post."""
    if post.get("sent_finbert_label") not in (None, ""):
        pos = float(post.get("sent_finbert_pos", 0.0))
        neg = float(post.get("sent_finbert_neg", 0.0))
        neu = float(post.get("sent_finbert_neu", 1.0))
        return pos, neg, neu
    if post.get("sent_bert_label") not in (None, ""):
        pos = float(post.get("sent_bert_pos", 0.0))
        neg = float(post.get("sent_bert_neg", 0.0))
        neu = max(0.0, 1.0 - pos - neg)
        return pos, neg, neu
    return 0.0, 0.0, 1.0


def _text_label(post: dict) -> str:
    """Devuelve la etiqueta de texto preferida del post."""
    if post.get("sent_finbert_label") not in (None, ""):
        return post["sent_finbert_label"]
    if post.get("sent_bert_label") not in (None, ""):
        return post["sent_bert_label"]
    return "neutral"


def _image_probs(post: dict) -> tuple[float, float, float]:
    """Normaliza el score de imagen Ollama a (pos_img, neg_img, neu_img)."""
    image_analysis = post.get("image_analysis")
    if not image_analysis:
        return 0.0, 0.0, 1.0
    score = float(image_analysis.get("score", 0.0) if isinstance(image_analysis, dict)
                  else getattr(image_analysis, "score", 0.0))
    if score > 0:
        pos_img = min(score / 10.0, 1.0)
        neg_img = 0.0
        neu_img = max(0.0, 1.0 - pos_img)
    elif score < 0:
        neg_img = min(abs(score) / 10.0, 1.0)
        pos_img = 0.0
        neu_img = max(0.0, 1.0 - neg_img)
    else:
        pos_img = 0.0
        neg_img = 0.0
        neu_img = 1.0
    return pos_img, neg_img, neu_img


def fuse_sentiment(
    post: dict,
    text_weight: float = 0.75,
    image_weight: float = 0.25,
) -> dict:
    """Calcula sent_text_only y sent_multimodal para el post.

    sent_text_only: etiqueta basada solo en el modelo de texto con mayor
        confianza (FinBERT por defecto, luego BERT, luego 'neutral').
    sent_multimodal: fusión lineal de score_texto * text_weight +
        score_imagen * image_weight.

    Si no hay imagen relevante (image_relevance=False o sin image_analysis):
        sent_multimodal == sent_text_only.

    La etiqueta final se determina por argmax de las probabilidades fusionadas.
    Retorna el post actualizado.
    """
    post = dict(post)

    # --- sent_text_only ---
    text_label = _text_label(post)
    post["sent_text_only"] = text_label

    # --- sent_multimodal ---
    has_relevant_image = bool(post.get("image_relevance", False))

    if not has_relevant_image:
        post["sent_multimodal"] = text_label
        return post

    pos_text, neg_text, neu_text = _text_probs(post)
    pos_img, neg_img, neu_img = _image_probs(post)

    pos_final = text_weight * pos_text + image_weight * pos_img
    neg_final = text_weight * neg_text + image_weight * neg_img
    neu_final = text_weight * neu_text + image_weight * neu_img

    scores = [pos_final, neg_final, neu_final]
    best_idx = scores.index(max(scores))
    post["sent_multimodal"] = _IDX_LABEL[best_idx]

    return post


def label_price_movement(price_today: float, price_tomorrow: float) -> int:
    """Retorna 1 si price_tomorrow > price_today, 0 en caso contrario."""
    return 1 if price_tomorrow > price_today else 0


def _sentiment_to_movement(label: str) -> int:
    """Convierte etiqueta de sentimiento a predicción de movimiento (1/0)."""
    return 1 if label == "positive" else 0


def build_comparison_report(
    posts: list[dict],
    price_df: pd.DataFrame,
) -> list[dict]:
    """Alinea posts con datos de precio por fecha y calcula price_movement_next_day.

    Args:
        posts: Lista de posts ya procesados con sent_text_only y sent_multimodal.
        price_df: DataFrame con columnas 'date' (YYYY-MM-DD) y 'close'.

    Returns:
        Lista de ComparisonRow como dicts.
    """
    if price_df is None or price_df.empty:
        logger.warning("DataFrame de precios vacío; no se puede calcular accuracy.")
        return []

    # Normalizar índice de precios
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")
    price_map: dict[str, float] = dict(zip(price_df["date"], price_df["close"].astype(float)))

    # Obtener fechas ordenadas para buscar el día siguiente
    sorted_dates = sorted(price_map.keys())
    next_day: dict[str, str] = {}
    for i, d in enumerate(sorted_dates[:-1]):
        next_day[d] = sorted_dates[i + 1]

    rows: list[dict] = []
    for post in posts:
        raw_date = post.get("date", "")
        # Tomar solo la parte YYYY-MM-DD
        post_date = str(raw_date)[:10] if raw_date else ""

        if post_date not in price_map:
            continue
        if post_date not in next_day:
            continue

        tomorrow = next_day[post_date]
        price_today = price_map[post_date]
        price_tomorrow = price_map[tomorrow]
        movement = label_price_movement(price_today, price_tomorrow)

        sent_text = post.get("sent_text_only", "neutral")
        sent_multi = post.get("sent_multimodal", "neutral")

        correct_text = _sentiment_to_movement(sent_text) == movement
        correct_multi = _sentiment_to_movement(sent_multi) == movement

        rows.append({
            "post_id": post.get("id", ""),
            "date": post_date,
            "sent_text_only": sent_text,
            "sent_multimodal": sent_multi,
            "price_movement_next_day": movement,
            "correct_text": correct_text,
            "correct_multimodal": correct_multi,
        })

    return rows


def load_price_data(price_path: Path) -> pd.DataFrame:
    """Carga el CSV de precios. Retorna DataFrame vacío si no existe o está vacío."""
    price_path = Path(price_path)
    if not price_path.exists():
        logger.warning("Archivo de precios no encontrado: %s", price_path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(price_path)
        if df.empty or "date" not in df.columns or "close" not in df.columns:
            logger.warning("CSV de precios vacío o sin columnas requeridas: %s", price_path)
            return pd.DataFrame()
        return df
    except Exception as exc:
        logger.warning("Error al leer CSV de precios %s: %s", price_path, exc)
        return pd.DataFrame()
