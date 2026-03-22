"""Modelos de datos base para el pipeline nvidia-sentiment-prediction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ImageAnalysis:
    """Resultado del análisis de imagen con Ollama llama3.2-vision."""

    score: float = 0.0       # >0 alcista, <0 bajista, 0 neutro
    analisis: str = ""       # explicación textual
    error: bool = False      # True si Ollama falló o JSON inválido


@dataclass
class Post:
    """Post de Reddit con todos los campos del pipeline."""

    # Campos originales del dataset
    id: str = ""
    title: str = ""                    # "" si ausente
    selftext: str = ""                 # "" si ausente
    created_utc: int = 0
    date: str = ""                     # ISO 8601
    subreddit: str = ""
    image_urls: list[str] = field(default_factory=list)

    # Añadidos por Descargador_Imágenes
    image_local_path: str = ""         # ruta relativa o ""
    image_download_status: str = "no_image"  # "ok" | "failed" | "no_image"

    # Añadido por Filtro_Imágenes
    image_relevance: bool = False

    # Añadidos por Analizador_Texto — FinBERT
    sent_finbert_label: str = "neutral"
    sent_finbert_pos: float = 0.0
    sent_finbert_neg: float = 0.0
    sent_finbert_neu: float = 1.0

    # Añadidos por Analizador_Texto — BERT general
    sent_bert_label: str = "neutral"
    sent_bert_pos: float = 0.0
    sent_bert_neg: float = 0.0

    # Añadidos por Analizador_Texto — SocBERT
    sent_socbert_label: str = "neutral"
    sent_socbert_pos: float = 0.0
    sent_socbert_neg: float = 0.0

    # Añadido por Analizador_Imagen
    image_analysis: ImageAnalysis | None = field(default=None)

    # Añadidos por Comparador_Multimodal
    sent_text_only: str = ""
    sent_multimodal: str = ""


@dataclass
class ModelMetrics:
    """Métricas de evaluación de un modelo de clasificación."""

    model_name: str = ""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class ComparisonRow:
    """Fila del CSV de comparación multimodal."""

    post_id: str = ""
    date: str = ""
    sent_text_only: str = ""
    sent_multimodal: str = ""
    price_movement_next_day: int = 0   # 1 sube, 0 baja/mantiene
    correct_text: bool = False
    correct_multimodal: bool = False
