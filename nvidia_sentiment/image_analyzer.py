"""Analizador de imagen: evalúa el impacto de imágenes relevantes en el sentimiento NVDA."""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """
Analiza esta imagen y evalúa su posible impacto en el precio del stock de NVIDIA.
Devuelve únicamente un JSON con la siguiente estructura:
{
  "score": número,      # positivo = alcista, negativo = bajista, 0 = neutro
  "analisis": "breve explicación justificando el score"
}
No añadas texto adicional ni comentarios.
"""


def _parse_ollama_response(content: str) -> dict:
    """Parsea la respuesta JSON de Ollama. Intenta regex como fallback."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Fallback: buscar JSON con regex
    match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"score": 0.0, "analisis": "", "error": True}


def analyze_image_sentiment(post: dict, ollama_model: str = "llama3.2-vision") -> dict:
    """
    Analiza el impacto de la imagen del post en el sentimiento de NVDA usando Ollama.

    - Solo procesa posts con image_relevance=True e image_local_path válido y existente
    - Si image_relevance=False o sin imagen → retorna post sin image_analysis
    - Envía imagen a Ollama con prompt de análisis de impacto en precio NVDA
    - Parsea respuesta JSON extrayendo score (float) y analisis (str)
    - Fallback: regex para extraer JSON; si falla → score=0.0, error=True
    - Almacena resultado en post["image_analysis"] = {"score": float, "analisis": str, "error": bool}
    - Retorna el post actualizado
    """
    post = dict(post)
    post_id = post.get("id", "<unknown>")

    # Solo procesar posts con image_relevance=True (Requisito 5.1)
    if not post.get("image_relevance", False):
        return post

    local_path = post.get("image_local_path", "")
    if not local_path:
        return post

    image_path = Path(local_path)
    if not image_path.exists():
        logger.warning(
            "[image_analyzer] post_id=%s: archivo no encontrado: %s", post_id, local_path
        )
        return post

    # Llamar a Ollama con la imagen (Requisitos 5.1, 5.2)
    try:
        with image_path.open("rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = ollama.chat(
            model=ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": _ANALYSIS_PROMPT,
                    "images": [image_b64],
                }
            ],
        )

        content = response["message"]["content"]
        parsed = _parse_ollama_response(content)

        score = float(parsed.get("score", 0.0))
        analisis = str(parsed.get("analisis", ""))
        error = bool(parsed.get("error", False))

        post["image_analysis"] = {
            "score": score,
            "analisis": analisis,
            "error": error,
        }

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[image_analyzer] post_id=%s: Ollama falló (%s) → score=0.0, error=True",
            post_id,
            exc,
        )
        post["image_analysis"] = {
            "score": 0.0,
            "analisis": "",
            "error": True,
        }

    return post
