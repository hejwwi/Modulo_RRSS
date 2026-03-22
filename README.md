# NVDA Reddit Sentiment Prediction

Pipeline completo de análisis de sentimiento multimodal sobre posts de Reddit relacionados con NVIDIA (NVDA), con predicción a 1 día vista usando modelos de ML clásicos y LSTM.

---

## Descripción

El proyecto analiza el sentimiento de la comunidad inversora en Reddit (wallstreetbets, stocks, investing, etc.) sobre NVIDIA entre 2023 y 2025. Combina análisis de texto con tres modelos NLP (FinBERT, BERT, SocBERT) y análisis de imagen con Ollama para predecir:

- **Movimiento del precio de NVDA al día siguiente** (con datos de bolsa)
- **Sentimiento de la comunidad al día siguiente** (sin datos de bolsa)

---

## Estructura del proyecto

```
proyecto/
├── nvidia_sentiment/           # Módulo principal — lógica de negocio
│   ├── models.py               # Dataclasses: Post, ImageAnalysis, ModelMetrics, ComparisonRow
│   ├── serializer.py           # save_dataset / load_dataset (JSON UTF-8)
│   ├── nvda_filter.py          # Filtrado de posts relevantes para NVDA
│   ├── image_downloader.py     # Descarga de imágenes de posts
│   ├── image_filter.py         # Filtrado de relevancia de imágenes (Ollama)
│   ├── text_analyzer.py        # Análisis de sentimiento: FinBERT, BERT, SocBERT
│   ├── image_analyzer.py       # Análisis de sentimiento de imagen (Ollama llama3.2-vision)
│   ├── multimodal_comparator.py# Fusión texto + imagen, comparación vs precio
│   └── predictor.py            # Entrenamiento LR, RF, GBM, MLP con datos de bolsa
│
├── scripts/                    # CLIs individuales por módulo
│   ├── nvda_filter.py
│   ├── image_downloader.py
│   ├── image_filter.py
│   ├── text_analyzer.py
│   ├── image_analyzer.py
│   ├── multimodal_comparator.py
│   ├── predictor.py            # Predictor con precio de bolsa
│   └── sentiment_predictor.py  # Predictor de sentimiento SIN precio
│
├── RedditScrapper/
│   ├── Data/
│   │   ├── final_dataset_clean.json   # Dataset principal de entrada (276k posts)
│   │   └── nvda_top3_backfill.csv     # Precios históricos NVDA (2023-2025)
│   ├── reddit_scrapper.py
│   └── reddit_updater.py
│
├── data/                       # Salidas generadas (en .gitignore)
│   ├── images/                 # Imágenes descargadas
│   ├── nvda_processed.json     # Posts procesados con sentimientos
│   ├── model_comparison.csv    # Accuracy modelos (con precio)
│   ├── sent_model_comparison.csv # Accuracy modelos (sin precio)
│   └── accuracy_comparison.png # Gráfico de barras
│
├── pipeline.py                 # Orquestador principal del pipeline
├── CHANGES.md                  # Historial de cambios de reorganización
└── README.md
```

---

## Requisitos

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib yfinance lightgbm requests
```

Ollama (para análisis de imagen):
```bash
ollama pull llama3.2-vision
```

---

## Pipeline principal

El pipeline ejecuta 7 fases secuenciales con manejo de errores por fase:

```
Dataset JSON
    │
    ▼
[Fase 1] Filtrado NVDA
    │  Términos: nvidia, nvda, $nvda, geforce, rtx, cuda
    │  276.445 posts → 13.326 posts NVDA
    ▼
[Fase 2] Descarga de imágenes
    │  Descarga imágenes de los posts que las tienen
    ▼
[Fase 3] Filtrado de imágenes (Ollama)
    │  Descarta GIFs y imágenes no relevantes para el sentimiento
    ▼
[Fase 4] Análisis de texto
    │  FinBERT (financiero) + BERT (general) + SocBERT (redes sociales)
    │  Scores de 0 a 1 por clase
    ▼
[Fase 5] Análisis de imagen (Ollama llama3.2-vision)
    │  Solo posts con imagen relevante
    ▼
[Fase 6] Fusión multimodal
    │  text_weight=0.75, image_weight=0.25
    ▼
[Fase 7] Serialización
    └─ data/nvda_processed.json
```

### Uso básico

```bash
# Modo prueba (rápido, sin imágenes)
python pipeline.py \
  --input RedditScrapper/Data/final_dataset_clean.json \
  --test_mode \
  --sample_size 500 \
  --skip_image_analysis

# Pipeline completo con predicción de precio
python pipeline.py \
  --input RedditScrapper/Data/final_dataset_clean.json \
  --price_data RedditScrapper/Data/nvda_top3_backfill.csv \
  --output_dir data

# Pipeline completo con análisis de imagen
python pipeline.py \
  --input RedditScrapper/Data/final_dataset_clean.json \
  --price_data RedditScrapper/Data/nvda_top3_backfill.csv \
  --output_dir data \
  --ollama_model llama3.2-vision
```

### Parámetros del pipeline

| Parámetro | Default | Descripción |
|---|---|---|
| `--input` | requerido | Ruta al JSON de posts |
| `--price_data` | None | CSV de precios (date, close) para accuracy |
| `--images_dir` | `data/images/` | Directorio de imágenes descargadas |
| `--output_dir` | `data` | Directorio de salida |
| `--test_mode` | False | Activa modo prueba con subconjunto |
| `--sample_size` | 200 | Posts a procesar en modo prueba |
| `--skip_download` | False | Omite descarga de imágenes |
| `--skip_image_analysis` | False | Omite fases 3 y 5 (Ollama) |
| `--models` | `finbert bert socbert` | Modelos de texto a usar |
| `--text_weight` | 0.75 | Peso del texto en fusión multimodal |
| `--image_weight` | 0.25 | Peso de la imagen en fusión multimodal |
| `--ollama_model` | `llama3.2-vision` | Modelo Ollama para imágenes |

---

## Modelos de análisis de texto

Cada post recibe scores de probabilidad (0–1) de los tres modelos:

| Campo | Modelo | Descripción |
|---|---|---|
| `sent_finbert_pos/neg/neu` | [FinBERT](https://huggingface.co/ProsusAI/finbert) | Sentimiento financiero (3 clases) |
| `sent_bert_pos/neg` | [DistilBERT SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) | Sentimiento general (2 clases) |
| `sent_socbert_pos/neg` | [SocBERT](https://huggingface.co/sarkerlab/SocBERT-base) | Sentimiento en redes sociales (2 clases) |
| `image_analysis.score` | Ollama llama3.2-vision | Relevancia de imagen para NVDA (0–1) |

Resultados observados con 1000 posts NVDA (2023-2025):

```
FinBERT  pos=0.171  neg=0.178  neu=0.652   → tono mayormente neutro
BERT     pos=0.234  neg=0.766              → más negativo (modelo general)
SocBERT  pos=0.411  neg=0.589             → ligeramente negativo
```

---

## Predictor con precio de bolsa

Predice si el precio de NVDA sube o baja al día siguiente usando los scores de sentimiento como features.

```bash
python scripts/predictor.py \
  --input data/nvda_processed.json \
  --price_data RedditScrapper/Data/nvda_top3_backfill.csv \
  --output data/model_comparison.csv
```

### Resultados (1000 posts, 809 muestras etiquetadas)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| GradientBoosting | 0.5185 | 0.5041 | 0.7848 | 0.6139 |
| LogisticRegression | 0.4938 | 0.4907 | 1.0000 | 0.6583 |
| MLP | 0.4877 | 0.4877 | 1.0000 | 0.6556 |
| RandomForest | 0.4877 | 0.4818 | 0.6709 | 0.5608 |

> El accuracy ronda el 50% porque predecir el precio de bolsa a partir de sentimiento de Reddit es inherentemente ruidoso. El baseline de "siempre sube" es ~55% dado la tendencia alcista de NVDA en 2023-2024.

Genera `data/model_comparison.csv` y `data/accuracy_comparison.png`.

---

## Predictor de sentimiento (sin precio de bolsa)

Predice si el sentimiento de la comunidad será más positivo mañana que hoy, usando únicamente señales de Reddit.

```bash
python scripts/sentiment_predictor.py \
  --input data/nvda_processed.json \
  --output data/sent_model_comparison.csv \
  --window 1 \
  --seq_len 7
```

### Features utilizadas

| Feature | Descripción |
|---|---|
| `sent_finbert_pos/neg/neu` | Scores FinBERT promedio del día |
| `sent_bert_pos/neg` | Scores BERT promedio del día |
| `sent_socbert_pos/neg` | Scores SocBERT promedio del día |
| `sent_*_roll3/roll7` | Rolling mean 3 y 7 días (momentum) |
| `sent_*_delta` | Cambio respecto al día anterior |
| `sent_*_momentum` | Diferencia roll3 - roll7 (tendencia corta vs larga) |
| `finbert_vol7` | Volatilidad del sentimiento (std 7 días) |
| `finbert_confidence` | max(pos, neg, neu) — certeza del modelo |
| `sentiment_agreement` | 1 si los 3 modelos coinciden |
| `finbert_bert_diff` | Diferencia entre FinBERT y BERT |
| `score_norm` | Upvotes normalizados |
| `num_comments_norm` | Comentarios normalizados |
| `n_posts` | Volumen de posts del día |
| `sub_*` | Fracción de posts por subreddit (one-hot) |

### Resultados (1000 posts, 541 días)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **LogisticRegression** | **0.7156** | 0.7037 | 0.7170 | 0.7103 |
| MLP | 0.6881 | 0.7111 | 0.6038 | 0.6531 |
| RandomForest | 0.6789 | 0.6957 | 0.6038 | 0.6465 |
| GradientBoosting | 0.6422 | 0.6591 | 0.5472 | 0.5979 |
| LSTM (seq=7) | 0.5327 | 0.5106 | 0.4706 | 0.4898 |
| Baseline | 0.4862 | — | — | — |

> El sentimiento de Reddit tiene inercia: si hoy es positivo, mañana tiende a serlo. La Regresión Logística captura bien esta autocorrelación con 71.5% de accuracy.

### Features más importantes (RandomForest)

```
sent_finbert_pos          0.1574  ████████
sent_finbert_pos_delta    0.1057  █████       ← cambio diario es clave
finbert_bert_diff         0.0685  ███
sent_finbert_pos_roll3    0.0462  ██
sent_finbert_neg          0.0432  ██
```

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `--window` | 1 | Días vista para la etiqueta (1=mañana, 3=promedio 3 días) |
| `--seq_len` | 7 | Longitud de secuencia para LSTM |

---

## Arquitectura LSTM

Para capturar patrones temporales, el predictor incluye un LSTM bidireccional:

```
Secuencia de seq_len días
        │
    [LSTM × 2 capas, hidden=64]
        │
    [Dropout 0.3]
        │
    [último timestep]
        │
    [Linear 64→32 → ReLU → Dropout → Linear 32→1]
        │
    [BCEWithLogitsLoss]
```

- Early stopping con paciencia de 15 epochs
- Gradient clipping (max_norm=1.0)
- ReduceLROnPlateau scheduler
- División temporal estricta (sin shuffle)

> Con 541 días el LSTM tiene datos limitados. Con los 13.326 posts NVDA completos (~700+ días únicos) el rendimiento mejora significativamente.

---

## Cómo mejorar la accuracy

Para mejorar la predicción de sentimiento a 1 día vista:

1. **Más datos** — procesar todos los 13.326 posts NVDA filtrados en lugar de 1000
2. **Análisis de imagen** — activar Ollama para los posts con imagen (actualmente `image_score=0`)
3. **Ventana de 3 días** — usar `--window 3` reduce el ruido de días individuales
4. **Secuencia más larga** — `--seq_len 14` o `--seq_len 30` para el LSTM con más datos
5. **Features externas** — volumen de trading, VIX, noticias de NVDA del día
6. **Fine-tuning de FinBERT** — entrenar FinBERT específicamente sobre posts de NVDA

---

## Dataset

El dataset `final_dataset_clean.json` contiene 276.445 posts de Reddit (2023-2025) con los campos:

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | str | ID único del post |
| `title` | str | Título del post |
| `selftext` | str | Cuerpo del post |
| `subreddit` | str | Subreddit de origen |
| `created_utc` | int | Timestamp Unix |
| `date` | str | Fecha ISO 8601 |
| `score` | int | Upvotes netos |
| `num_comments` | int | Número de comentarios |
| `image_urls` | list | URLs de imágenes adjuntas |
| `has_image` | bool | Si el post tiene imagen |

Tras el pipeline se añaden los campos de sentimiento (`sent_*`) y análisis de imagen (`image_analysis`).

---

## Scraper de Reddit

Para actualizar el dataset con posts nuevos:

```bash
python RedditScrapper/reddit_scrapper.py   # scraping inicial
python RedditScrapper/reddit_updater.py    # actualización incremental
```

---

## Tests

```bash
pytest tests/
```

La carpeta `tests/` incluye tests unitarios y de propiedades (property-based testing con `hypothesis`).
