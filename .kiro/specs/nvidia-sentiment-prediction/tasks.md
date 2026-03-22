# Plan de Implementación: nvidia-sentiment-prediction

## Visión General

Pipeline de análisis de sentimiento multimodal para NVIDIA (NVDA) que procesa posts históricos de Reddit (2023–2025). La implementación sigue el orden del flujo de datos: filtrado → descarga de imágenes → filtrado de imágenes → análisis de texto → análisis de imagen → comparación multimodal → predicción → serialización → pipeline orquestador.

## Tareas

- [x] 1. Crear estructura del proyecto y modelos de datos base
  - Crear directorio `nvidia_sentiment/` con `__init__.py`
  - Definir dataclasses `Post`, `ImageAnalysis`, `ModelMetrics`, `ComparisonRow` en `nvidia_sentiment/models.py`
  - Crear `tests/unit/` y `tests/property/` con sus `__init__.py`
  - _Requisitos: 9.1, 9.2_

- [x] 2. Implementar Serializador (`nvidia_sentiment/serializer.py`)
  - [x] 2.1 Implementar `save_dataset` y `load_dataset` con JSON UTF-8
    - Manejar archivo inexistente o corrupto con log de error
    - _Requisitos: 9.1, 9.2, 9.4_

  - [ ]* 2.2 Escribir test de propiedad: round-trip de serialización
    - **Propiedad 14: Round-trip de serialización JSON**
    - **Valida: Requisitos 9.1, 9.3**

  - [ ]* 2.3 Escribir tests unitarios para el serializador
    - Caso: archivo corrupto → log de error
    - Caso: lista vacía → round-trip correcto
    - _Requisitos: 9.3, 9.4_

- [x] 3. Implementar Filtro_NVDA (`nvidia_sentiment/nvda_filter.py`)
  - [x] 3.1 Implementar `filter_nvda_posts(posts)` con los términos NVDA
    - Términos: `nvidia`, `nvda`, `$nvda`, `geforce`, `rtx`, `cuda` (case-insensitive)
    - Campos ausentes tratados como cadena vacía
    - Log de advertencia si resultado vacío
    - _Requisitos: 1.1, 1.2, 1.3, 1.4_

  - [x] 3.2 Implementar CLI (`nvda_filter.py`) con `--input`, `--output`, `--test_mode`, `--sample_size`
    - Exportar resultado a `nvda_filtered_posts.json`
    - _Requisitos: 1.5, 7.1_

  - [ ]* 3.3 Escribir test de propiedad: corrección del filtro NVDA
    - **Propiedad 1: Corrección del filtro NVDA**
    - **Valida: Requisitos 1.1, 1.2**

  - [ ]* 3.4 Escribir test de propiedad: preservación de campos en el filtro
    - **Propiedad 2: Preservación de campos en el filtro**
    - **Valida: Requisito 1.3**

  - [ ]* 3.5 Escribir tests unitarios para el filtro NVDA
    - Caso: post con "NVDA" en título → incluido
    - Caso: post sin términos NVDA → excluido
    - Caso: post sin campo `title` → no lanza excepción
    - _Requisitos: 1.1, 1.2_

- [x] 4. Checkpoint — Asegurarse de que todos los tests pasan hasta aquí
  - Ejecutar `pytest tests/ -x`. Preguntar al usuario si hay dudas.

- [x] 5. Implementar Descargador_Imágenes (`nvidia_sentiment/image_downloader.py`)
  - [x] 5.1 Implementar `download_post_image(post, images_dir, timeout)` 
    - Descargar primera URL de `image_urls`, nombrar `{post_id}.{ext}`
    - Actualizar `image_local_path` e `image_download_status`
    - Timeout de 30 s, log de error en fallo, continuar con siguiente post
    - _Requisitos: 2.1, 2.2, 2.3, 2.4_

  - [x] 5.2 Implementar CLI (`image_downloader.py`) con `--input`, `--images_dir`, `--timeout`, `--test_mode`, `--sample_size`
    - En modo prueba, procesar solo los primeros 100 posts con imagen
    - _Requisitos: 2.5, 7.1_

  - [ ]* 5.3 Escribir test de propiedad: descarga con nombre correcto
    - **Propiedad 3: Descarga de imagen con nombre correcto**
    - **Valida: Requisitos 2.1, 2.2**

  - [ ]* 5.4 Escribir test de propiedad: límite de posts en modo prueba
    - **Propiedad 4: Límite de posts en modo prueba**
    - **Valida: Requisitos 2.5, 4.5, 5.5**

  - [ ]* 5.5 Escribir tests unitarios para el descargador
    - Caso: URL válida → `image_download_status="ok"`
    - Caso: timeout → `image_download_status="failed"`, log de error
    - Caso: post sin `image_urls` → `image_download_status="no_image"`
    - _Requisitos: 2.3, 2.4_

- [x] 6. Implementar Filtro_Imágenes (`nvidia_sentiment/image_filter.py`)
  - [x] 6.1 Implementar descarte automático de GIFs sin llamar a Ollama
    - Detectar `.gif` en URL o extensión de archivo → `image_relevance=False`
    - _Requisitos: 3.1_

  - [x] 6.2 Implementar `evaluate_image_relevance(post, ollama_model)` con Ollama
    - Llamar a `llama3.2-vision`; si score==0 y análisis indica no relevancia → `image_relevance=False`
    - Si Ollama falla → `image_relevance=False` + log de error
    - _Requisitos: 3.2, 3.3, 3.4_

  - [x] 6.3 Implementar CLI (`image_filter.py`) con `--input`, `--output`, `--ollama_model`, `--test_mode`
    - _Requisitos: 3.5_

  - [ ]* 6.4 Escribir test de propiedad: GIFs siempre marcados como no relevantes
    - **Propiedad 5: GIFs siempre marcados como no relevantes**
    - **Valida: Requisito 3.1**

  - [ ]* 6.5 Escribir tests unitarios para el filtro de imágenes
    - Caso: URL con `.gif` → `image_relevance=False` sin llamar a Ollama
    - Caso: Ollama falla → `image_relevance=False`, log de error
    - _Requisitos: 3.1, 3.4_

- [x] 7. Implementar Analizador_Texto (`nvidia_sentiment/text_analyzer.py`)
  - [x] 7.1 Implementar `analyze_text_sentiment(post, models, max_length)` con FinBERT, BERT y SocBERT
    - Concatenar `title + selftext`, truncar a `max_length` tokens
    - Texto vacío → etiqueta `"neutral"`, scores por defecto
    - Añadir campos `sent_{model}_label`, `sent_{model}_pos`, `sent_{model}_neg`, `sent_{model}_neu`
    - _Requisitos: 4.1, 4.2, 4.3, 4.4_

  - [x] 7.2 Implementar CLI (`text_analyzer.py`) con `--input`, `--output`, `--models`, `--max_length`, `--test_mode`, `--sample_size`
    - En modo prueba, procesar solo los primeros 200 posts
    - _Requisitos: 4.5, 7.1_

  - [ ]* 7.3 Escribir test de propiedad: completitud de campos de sentimiento de texto
    - **Propiedad 7: Completitud de campos de sentimiento de texto**
    - **Valida: Requisitos 4.1, 4.3**

  - [ ]* 7.4 Escribir test de propiedad: límite de posts en modo prueba (Analizador_Texto)
    - **Propiedad 4: Límite de posts en modo prueba**
    - **Valida: Requisito 4.5**

  - [ ]* 7.5 Escribir tests unitarios para el analizador de texto
    - Caso: texto vacío → etiqueta `"neutral"`, scores por defecto
    - Caso: texto con "NVDA is going to the moon" → etiqueta `"positive"`
    - _Requisitos: 4.3, 4.4_

- [x] 8. Checkpoint — Asegurarse de que todos los tests pasan hasta aquí
  - Ejecutar `pytest tests/ -x`. Preguntar al usuario si hay dudas.

- [x] 9. Implementar Analizador_Imagen (`nvidia_sentiment/image_analyzer.py`)
  - [x] 9.1 Implementar `analyze_image_sentiment(post, ollama_model)` con Ollama `llama3.2-vision`
    - Solo procesar posts con `image_relevance=True` e `image_local_path` válido
    - Parsear JSON de respuesta; fallback regex; si falla → `score=0.0, error=True`
    - Almacenar resultado en `image_analysis: {score, analisis, error}`
    - _Requisitos: 5.1, 5.2, 5.3, 5.4_

  - [x] 9.2 Implementar CLI (`image_analyzer.py`) con `--input`, `--output`, `--ollama_model`, `--test_mode`, `--sample_size`
    - En modo prueba, procesar solo los primeros 50 posts con imagen relevante
    - _Requisitos: 5.5, 7.1_

  - [ ]* 9.3 Escribir test de propiedad: posts no relevantes excluidos del análisis de imagen
    - **Propiedad 6: Posts no relevantes excluidos del análisis de imagen**
    - **Valida: Requisito 3.5**

  - [ ]* 9.4 Escribir test de propiedad: completitud de image_analysis para posts con imagen relevante
    - **Propiedad 8: Completitud de image_analysis para posts con imagen relevante**
    - **Valida: Requisitos 5.1, 5.3**

  - [ ]* 9.5 Escribir tests unitarios para el analizador de imagen
    - Caso: respuesta Ollama con JSON válido → campos correctos
    - Caso: respuesta Ollama sin JSON → `score=0.0, error=True`
    - Caso: `image_relevance=False` → `image_analysis` no generado
    - _Requisitos: 5.3, 5.4_

- [x] 10. Implementar Comparador_Multimodal (`nvidia_sentiment/multimodal_comparator.py`)
  - [x] 10.1 Implementar `fuse_sentiment(post, text_weight, image_weight)` con fusión lineal
    - Si no hay imagen relevante → `sent_multimodal == sent_text_only`
    - _Requisitos: 6.1, 6.2, 6.3_

  - [x] 10.2 Implementar cálculo de accuracy contra movimiento real de precio
    - Leer `nvda_top3_backfill.csv`, alinear por fecha, calcular `price_movement_next_day`
    - Exportar `comparison_report.csv` con columnas requeridas
    - _Requisitos: 6.4, 6.5_

  - [x] 10.3 Implementar CLI (`multimodal_comparator.py`) con `--input`, `--price_data`, `--output`, `--text_weight`, `--image_weight`
    - _Requisitos: 6.5_

  - [ ]* 10.4 Escribir test de propiedad: fusión aritmética correcta texto + imagen
    - **Propiedad 9: Fusión aritmética correcta texto + imagen**
    - **Valida: Requisito 6.1**

  - [ ]* 10.5 Escribir test de propiedad: sin imagen relevante implica señales iguales
    - **Propiedad 10: Sin imagen relevante implica señales iguales**
    - **Valida: Requisito 6.3**

  - [ ]* 10.6 Escribir test de propiedad: etiquetado correcto del movimiento de precio
    - **Propiedad 12: Etiquetado correcto del movimiento de precio**
    - **Valida: Requisito 8.3**

  - [ ]* 10.7 Escribir tests unitarios para el comparador multimodal
    - Caso: post sin imagen → `sent_text_only == sent_multimodal`
    - Caso: fusión con pesos 0.75/0.25 → resultado aritmético correcto
    - _Requisitos: 6.1, 6.3_

- [x] 11. Implementar Predictor (`nvidia_sentiment/predictor.py`)
  - [x] 11.1 Implementar `train_and_evaluate(X_train, y_train, X_test, y_test)` con LR, RF, GBM y MLP
    - División temporal sin shuffle (80/20), desempate por F1
    - Calcular accuracy, precision, recall, f1 por modelo
    - _Requisitos: 8.1, 8.2, 8.4, 8.5, 8.7_

  - [x] 11.2 Implementar función de etiquetado de precio `label_price_movement(precio_hoy, precio_mañana)`
    - Retornar `1` si `precio_mañana > precio_hoy`, `0` en caso contrario
    - _Requisito: 8.3_

  - [x] 11.3 Implementar exportación de `model_comparison.csv` y gráfico de barras de accuracy
    - _Requisitos: 8.6_

  - [x] 11.4 Implementar CLI (`predictor.py`) con `--input`, `--price_data`, `--output`, `--test_size`
    - _Requisitos: 8.1_

  - [ ]* 11.5 Escribir test de propiedad: etiquetado correcto del movimiento de precio
    - **Propiedad 12: Etiquetado correcto del movimiento de precio**
    - **Valida: Requisito 8.3**

  - [ ]* 11.6 Escribir test de propiedad: división temporal sin data leakage
    - **Propiedad 13: División temporal sin data leakage**
    - **Valida: Requisito 8.4**

  - [ ]* 11.7 Escribir tests unitarios para el predictor
    - Caso: dos modelos con mismo accuracy → desempate por F1
    - Caso: dataset ordenado temporalmente → split correcto
    - _Requisitos: 8.4, 8.7_

- [x] 12. Checkpoint — Asegurarse de que todos los tests pasan hasta aquí
  - Ejecutar `pytest tests/ -x`. Preguntar al usuario si hay dudas.

- [x] 13. Implementar Pipeline Principal (`pipeline.py`)
  - [x] 13.1 Orquestar todos los módulos en secuencia con manejo de errores por fase
    - Flags: `--input`, `--test_mode`, `--sample_size`, `--text_weight`, `--image_weight`, `--skip_download`, `--skip_image_analysis`
    - En modo prueba: seleccionar los N posts más recientes por `created_utc`
    - Mostrar resumen final al terminar modo prueba
    - _Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 13.2 Escribir test de propiedad: selección de N posts más recientes en modo prueba
    - **Propiedad 11: Selección de N posts más recientes en modo prueba**
    - **Valida: Requisito 7.2**

  - [ ]* 13.3 Escribir tests unitarios para el pipeline
    - Caso: error en una fase → continúa con la siguiente, log con contexto
    - Caso: `--skip_download` → módulo de descarga no se ejecuta
    - _Requisitos: 7.3, 7.5_

- [x] 14. Checkpoint final — Asegurarse de que todos los tests pasan
  - Ejecutar `pytest tests/ -v`. Preguntar al usuario si hay dudas antes de cerrar.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia los requisitos específicos para trazabilidad
- Los tests de propiedad usan `hypothesis` con `@settings(max_examples=100)`
- Los tests unitarios usan `pytest` con mocks para Ollama y llamadas de red
- La división temporal del predictor usa `sorted` por `created_utc` sin shuffle para evitar data leakage
