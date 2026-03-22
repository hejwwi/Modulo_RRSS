"""Filtro de imágenes: descarta GIFs y evalúa relevancia con Ollama llama3.2-vision."""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import ollama

logger = logging.getLogger(__name__)

# Palabras clave que indican no relevancia en el campo 'analisis' de Ollama
_NO_RELEVANCE_KEYWORDS = [
    "no relevante",
    "irrelevante",
    "no aporta",
    "no contiene",
    "no muestra",
]

_RELEVANCE_PROMPT = """
Analiza esta imagen y determina si es relevante para el análisis de sentimiento financiero de NVIDIA (NVDA).
Una imagen es relevante si contiene: gráficos de precio, noticias financieras, logos de NVIDIA en contexto financiero,
datos de mercado, resultados de earnings, o cualquier contenido que aporte información sobre el sentimiento del mercado hacia NVDA.
Una imagen NO es relevante si es: un meme genérico, un GIF animado, un icono, una imagen decorativa sin contenido financiero.

Devuelve únicamente un JSON con la siguiente estructura:
{
  "score": número,
  "analisis": "breve explicación"
}

Donde score=0 significa que la imagen no aporta información financiera relevante para NVDA.
No añadas texto adicional ni comentarios.
"""


def is_gif(post: dict) -> bool:
    """Retorna True si la URL o image_local_path del post apunta a un GIF."""
    url = ""
    image_urls = post.get("image_urls") or []
    if image_urls:
        url = image_urls[0] if isinstance(image_urls, list) else str(image_urls)

    local_path = post.get("image_local_path") or ""

    return url.lower().endswith(".gif") or local_path.lower().endswith(".gif")


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


def _is_irrelevant_by_analysis(score: float, analisis: str) -> bool:
    """Retorna True si score==0 y el análisis indica no relevancia."""
    if score != 0:
        return False
    analisis_lower = analisis.lower()
    return any(kw in analisis_lower for kw in _NO_RELEVANCE_KEYWORDS)


def evaluate_image_relevance(post: dict, ollama_model: str = "llama3.2-vision") -> dict:
    """
    Evalúa si la imagen del post es relevante para el análisis de sentimiento NVDA.

    - Si es GIF → image_relevance=False sin llamar a Ollama
    - Si image_download_status != "ok" → image_relevance=False
    - Llama a Ollama con la imagen en base64
    - Parsea respuesta JSON: si score==0 y analisis indica no relevancia → image_relevance=False
    - Si Ollama falla → image_relevance=False + log ERROR
    - Retorna el post actualizado
    """
    post = dict(post)
    post_id = post.get("id", "<unknown>")

    # Descarte automático de GIFs (Requisito 3.1)
    if is_gif(post):
        logger.debug("[image_filter] post_id=%s: GIF detectado → image_relevance=False", post_id)
        post["image_relevance"] = False
        return post

    # Sin imagen descargada correctamente (Requisito 3.2)
    if post.get("image_download_status") != "ok":
        post["image_relevance"] = False
        return post

    local_path = post.get("image_local_path", "")
    if not local_path:
        post["image_relevance"] = False
        return post

    image_path = Path(local_path)
    if not image_path.exists():
        logger.warning(
            "[image_filter] post_id=%s: archivo no encontrado: %s", post_id, local_path
        )
        post["image_relevance"] = False
        return post

    # Llamar a Ollama (Requisitos 3.2, 3.3, 3.4)
    try:
        with image_path.open("rb") as f:
            image_bytes = f.read()

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        response = ollama.chat(
            model=ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": _RELEVANCE_PROMPT,
                    "images": [image_b64],
                }
            ],
        )

        content = response["message"]["content"]
        parsed = _parse_ollama_response(content)

        score = float(parsed.get("score", 0.0))
        analisis = str(parsed.get("analisis", ""))

        if _is_irrelevant_by_analysis(score, analisis):
            post["image_relevance"] = False
        else:
            post["image_relevance"] = True

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[image_filter] post_id=%s: Ollama falló (%s) → image_relevance=False",
            post_id,
            exc,
        )
        post["image_relevance"] = False

    return post
