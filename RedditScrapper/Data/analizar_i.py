#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import ollama
import pathlib
import tempfile
import urllib.request
import urllib.parse
import re
import os
from typing import Any, Dict, Optional

INPUT_FILE = pathlib.Path("final_dataset_clean.json")
OUTPUT_FILE = pathlib.Path("final_dataset_with_image_analysis.json")

MODEL_NAME = "llama3.2-vision"

PROMPT = """
Analiza esta imagen y evalúa su posible impacto en el precio del stock de NVIDIA.
Devuelve únicamente un JSON con la siguiente estructura:

{
  "score": número,
  "analisis": "breve explicación justificando el score"
}

Reglas:
- score > 0 = posible impacto alcista
- score < 0 = posible impacto bajista
- score = 0 = neutro o sin señal clara

No añadas texto adicional ni comentarios.
"""


def extraer_json(texto: str) -> Optional[Dict[str, Any]]:
    texto = texto.strip()

    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        bloque = match.group(0)
        try:
            return json.loads(bloque)
        except json.JSONDecodeError:
            return None

    return None


def normalizar_resultado(data: Dict[str, Any]) -> Dict[str, Any]:
    score = data.get("score", 0)
    analisis = data.get("analisis", "")

    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0

    if analisis is None:
        analisis = ""
    analisis = str(analisis).strip()

    return {
        "score": score,
        "analisis": analisis
    }


def guess_extension_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()

    for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"]:
        if path.endswith(ext):
            return ext

    return ".jpg"


def descargar_imagen(url: str) -> str:
    ext = guess_extension_from_url(url)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    request = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_file.write(data)
    temp_file.close()

    return temp_file.name


def analizar_imagen_local(img_path: str) -> Dict[str, Any]:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": PROMPT,
                "images": [img_path]
            }
        ]
    )

    contenido = response["message"]["content"]
    data = extraer_json(contenido)

    if data is None:
        return {
            "score": 0.0,
            "analisis": "No se pudo parsear la respuesta JSON del modelo.",
            "raw_response": contenido,
            "error": True
        }

    resultado = normalizar_resultado(data)
    resultado["raw_response"] = contenido
    resultado["error"] = False
    return resultado


def analizar_url_imagen(url: str) -> Dict[str, Any]:
    temp_path = None

    try:
        temp_path = descargar_imagen(url)
        return analizar_imagen_local(temp_path)
    except Exception as e:
        return {
            "score": 0.0,
            "analisis": f"Error analizando imagen: {e}",
            "raw_response": "",
            "error": True
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El archivo de entrada debe contener una lista JSON.")

    total = len(data)
    analizados = 0
    con_imagen = 0

    for i, post in enumerate(data, start=1):
        print(f"{i}/{total}")

        image_urls = post.get("image_urls", [])
        has_image = post.get("has_image", False)

        if not has_image or not image_urls:
            post["image_analysis"] = None
            continue

        con_imagen += 1

        # Analizamos solo la primera imagen del post
        image_url = image_urls[0]

        print(f"  Analizando imagen de post {post.get('id')}...")
        print(f"  URL: {image_url}")

        resultado = analizar_url_imagen(image_url)
        post["image_analysis"] = resultado
        analizados += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\n===== RESUMEN =====")
    print(f"Total posts:            {total}")
    print(f"Posts con imagen:       {con_imagen}")
    print(f"Posts analizados:       {analizados}")
    print(f"Salida guardada en:     {OUTPUT_FILE}")
    print("===================\n")


if __name__ == "__main__":
    main()