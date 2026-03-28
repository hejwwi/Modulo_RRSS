"""Comparador multimodal: fusión de señales de texto e imagen para NVDA."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_IDX_LABEL = {0: "positive", 1: "negative", 2: "neutral"}


def _text_probs(post: dict) -> tuple[float, float, float]:
    # FinBERT es el modelo principal
    if post.get("sent_finbert_label") not in (None, ""):
        return (
            float(post.get("sent_finbert_pos", 0.0)),
            float(post.get("sent_finbert_neg", 0.0)),
            float(post.get("sent_finbert_neu", 1.0)),
        )
    return 0.0, 0.0, 1.0


def _text_label(post: dict) -> str:
    if post.get("sent_finbert_label") not in (None, ""):
        return post["sent_finbert_label"]
    return "neutral"


def _image_probs(post: dict) -> tuple[float, float, float]:
    image_analysis = post.get("image_analysis")
    if not image_analysis:
        return 0.0, 0.0, 1.0
    score = float(
        image_analysis.get("score", 0.0) if isinstance(image_analysis, dict)
        else getattr(image_analysis, "score", 0.0)
    )
    if score > 0:
        pos_img = min(score / 10.0, 1.0)
        return pos_img, 0.0, max(0.0, 1.0 - pos_img)
    if score < 0:
        neg_img = min(abs(score) / 10.0, 1.0)
        return 0.0, neg_img, max(0.0, 1.0 - neg_img)
    return 0.0, 0.0, 1.0


def fuse_sentiment(
    post: dict,
    text_weight: float = 0.75,
    image_weight: float = 0.25,
) -> dict:
    """Calcula sent_text_only y sent_multimodal para el post.

    Si no hay imagen relevante, sent_multimodal == sent_text_only.
    """
    post = dict(post)
    text_label = _text_label(post)
    post["sent_text_only"] = text_label

    if not bool(post.get("image_relevance", False)):
        post["sent_multimodal"] = text_label
        return post

    pos_t, neg_t, neu_t = _text_probs(post)
    pos_i, neg_i, neu_i = _image_probs(post)

    scores = [
        text_weight * pos_t + image_weight * pos_i,
        text_weight * neg_t + image_weight * neg_i,
        text_weight * neu_t + image_weight * neu_i,
    ]
    post["sent_multimodal"] = _IDX_LABEL[scores.index(max(scores))]
    return post
