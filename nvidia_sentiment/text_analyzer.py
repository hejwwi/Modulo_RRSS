"""Analizador de sentimiento de texto para posts de Reddit (FinBERT, BERT, SocBERT).

Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

MODEL_IDS = {
    "finbert": "ProsusAI/finbert",
    "bert": "distilbert-base-uncased-finetuned-sst-2-english",
    "socbert": "sarkerlab/SocBERT-base",
}

# ---------------------------------------------------------------------------
# Lazy-loaded pipeline/model cache (cargados una sola vez por proceso)
# ---------------------------------------------------------------------------
_FINBERT_PIPELINE: Any = None
_BERT_PIPELINE: Any = None
_SOCBERT_TOKENIZER: Any = None
_SOCBERT_MODEL: Any = None


def _get_finbert_pipeline():
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        from transformers import pipeline
        _FINBERT_PIPELINE = pipeline(
            "text-classification",
            model=MODEL_IDS["finbert"],
            tokenizer=MODEL_IDS["finbert"],
            top_k=None,
            truncation=True,
            max_length=512,  # truncación real se hace antes
        )
    return _FINBERT_PIPELINE


def _get_bert_pipeline():
    global _BERT_PIPELINE
    if _BERT_PIPELINE is None:
        from transformers import pipeline
        _BERT_PIPELINE = pipeline(
            "sentiment-analysis",
            model=MODEL_IDS["bert"],
            tokenizer=MODEL_IDS["bert"],
            truncation=True,
            max_length=512,
        )
    return _BERT_PIPELINE


def _get_socbert():
    global _SOCBERT_TOKENIZER, _SOCBERT_MODEL
    if _SOCBERT_TOKENIZER is None or _SOCBERT_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _SOCBERT_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_IDS["socbert"], use_fast=True)
        _SOCBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_IDS["socbert"])
        _SOCBERT_MODEL.eval()
    return _SOCBERT_TOKENIZER, _SOCBERT_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text(post: dict) -> str:
    """Concatena title + selftext; campos ausentes → ''."""
    title = str(post.get("title") or "")
    selftext = str(post.get("selftext") or "")
    return (title + " " + selftext).strip()


def _apply_finbert(text: str, max_length: int) -> dict:
    """Devuelve campos sent_finbert_* para un texto no vacío."""
    clf = _get_finbert_pipeline()
    # Truncamos el texto a max_length tokens usando el tokenizer del pipeline
    tokenizer = clf.tokenizer
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    truncated = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

    result = clf(truncated)[0]  # lista de {label, score}
    score_map = {d["label"].lower(): float(d["score"]) for d in result}
    pos = score_map.get("positive", 0.0)
    neg = score_map.get("negative", 0.0)
    neu = score_map.get("neutral", 0.0)
    label = max([("positive", pos), ("negative", neg), ("neutral", neu)], key=lambda x: x[1])[0]
    return {
        "sent_finbert_label": label,
        "sent_finbert_pos": pos,
        "sent_finbert_neg": neg,
        "sent_finbert_neu": neu,
    }


def _apply_bert(text: str, max_length: int) -> dict:
    """Devuelve campos sent_bert_* para un texto no vacío."""
    clf = _get_bert_pipeline()
    tokenizer = clf.tokenizer
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    truncated = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

    pred = clf(truncated)[0]  # {label, score}
    raw_label = str(pred["label"]).lower()
    score = float(pred["score"])

    if "pos" in raw_label:
        pos, neg = score, 1.0 - score
        label = "positive"
    else:
        neg, pos = score, 1.0 - score
        label = "negative"

    return {
        "sent_bert_label": label,
        "sent_bert_pos": pos,
        "sent_bert_neg": neg,
    }


@torch.inference_mode()
def _apply_socbert(text: str, max_length: int) -> dict:
    """Devuelve campos sent_socbert_* para un texto no vacío."""
    tokenizer, model = _get_socbert()
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True,
    )
    logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    neg = float(probs[0])
    pos = float(probs[1])
    label = "positive" if pos >= neg else "negative"
    return {
        "sent_socbert_label": label,
        "sent_socbert_pos": pos,
        "sent_socbert_neg": neg,
    }


# ---------------------------------------------------------------------------
# Valores por defecto para texto vacío
# ---------------------------------------------------------------------------

_EMPTY_DEFAULTS = {
    "finbert": {
        "sent_finbert_label": "neutral",
        "sent_finbert_pos": 0.0,
        "sent_finbert_neg": 0.0,
        "sent_finbert_neu": 1.0,
    },
    "bert": {
        "sent_bert_label": "neutral",
        "sent_bert_pos": 0.5,
        "sent_bert_neg": 0.5,
    },
    "socbert": {
        "sent_socbert_label": "neutral",
        "sent_socbert_pos": 0.5,
        "sent_socbert_neg": 0.5,
    },
}


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def analyze_text_sentiment(post: dict, models: list[str], max_length: int = 256) -> dict:
    """Analiza el sentimiento del texto (title + selftext) del post con los modelos indicados.

    - Concatena title + selftext (campos ausentes = "")
    - Texto vacío → neutral con scores por defecto
    - Añade campos sent_{model}_label, sent_{model}_pos, sent_{model}_neg
      (y sent_finbert_neu para finbert)
    - Retorna el post actualizado (copia superficial con campos añadidos)

    Args:
        post: Dict con al menos los campos 'title' y 'selftext'.
        models: Lista de modelos a aplicar. Valores válidos: 'finbert', 'bert', 'socbert'.
        max_length: Número máximo de tokens para truncar el texto.

    Returns:
        Post actualizado con los campos de sentimiento añadidos.
    """
    result = dict(post)
    text = _build_text(post)

    for model in models:
        model = model.lower()
        if not text:
            result.update(_EMPTY_DEFAULTS.get(model, {}))
            continue

        try:
            if model == "finbert":
                result.update(_apply_finbert(text, max_length))
            elif model == "bert":
                result.update(_apply_bert(text, max_length))
            elif model == "socbert":
                result.update(_apply_socbert(text, max_length))
            else:
                logger.warning("Modelo desconocido: %s", model)
        except Exception as exc:
            logger.error(
                "[text_analyzer] post_id=%s model=%s: %s",
                post.get("id", "?"),
                model,
                exc,
            )
            result.update(_EMPTY_DEFAULTS.get(model, {}))

    return result


def analyze_batch(posts: list[dict], models: list[str], max_length: int = 256) -> list[dict]:
    """Analiza una lista de posts en batch para mayor eficiencia.

    Args:
        posts: Lista de posts como dicts.
        models: Lista de modelos a aplicar.
        max_length: Número máximo de tokens.

    Returns:
        Lista de posts actualizados con campos de sentimiento.
    """
    return [analyze_text_sentiment(post, models, max_length) for post in posts]
