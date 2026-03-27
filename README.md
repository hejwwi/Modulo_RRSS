# NVDA Reddit Sentiment Prediction

Pipeline de análisis de sentimiento multimodal sobre posts de Reddit relacionados con NVIDIA (NVDA), con predicción del sentimiento de la comunidad a 1 día vista usando histórico de posts.

---

## Descripción

Analiza el sentimiento de la comunidad inversora en Reddit (wallstreetbets, stocks, investing, etc.) sobre NVIDIA entre 2023 y 2025. Combina análisis de texto con tres modelos NLP (FinBERT, BERT, SocBERT) y análisis de imagen con Ollama para predecir si el sentimiento del día siguiente será más positivo o negativo que el actual.

**No usa datos de bolsa.** La predicción se basa exclusivamente en el histórico de posts de Reddit.

---

## Estructura del proyecto

```
proyecto/
├── nvidia_sentiment/              # Módulo principal — lógica de negocio
│   ├── models.py                  # Dataclasses: Post, ImageAnalysis, ModelMetrics
│   ├── serializer.py              # save_dataset / load_dataset (JSON UTF-8)
│   ├── nvda_filter.py             # Filtrado de posts relevantes para NVDA
│   ├── image_downloader.py        # Descarga de imágenes de posts
│   ├── image_filter.py            # Filtrado de relevancia de imágenes (Ollama)
│   ├── text_analyzer.py           # Análisis de sentimiento: FinBERT, BERT, SocBERT
│   ├── image_analyzer.py          # Análisis de sentimiento de imagen (Ollama)
│   └── multimodal_comparator.py   # Fusión texto + imagen
│
├── scripts/                       # CLIs individuales
│   ├── nvda_filter.py
│   ├── image_downloader.py
│   ├── image_filter.py
│   ├── text_analyzer.py
│   ├── image_analyzer.py
│   └── sentiment_predictor.py     # Predictor de sentimiento a 1 día vista
│
├── RedditScrapper/
│   ├── Data/
│   │   └── final_dataset_clean.json   # Dataset de entrada (276k posts, 2023-2025)
│   ├── reddit_scrapper.py
│   └── reddit_updater.py
│
├── data/                          # Salidas generadas (en .gitignore)
│   ├── images/                    # Imágenes descargadas
│   ├── nvda_processed.json        # Posts procesados con sentimientos
│   └── sent_model_comparison.csv  # Resultados de los modelos de predicción
│
├── pipeline.py                    # Orquestador principal
└── README.md
```

---

## Requisitos

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib requests
```

Ollama (para análisis de imagen, opcional):
```bash
ollama pull llama3.2-vision
```

---

## Pipeline principal

7 fases secuenciales con manejo de errores por fase:

```
Dataset JSON (276k posts)
    │
    ▼
[Fase 1] Filtrado NVDA
    │  Términos: nvidia, nvda, $nvda, geforce, rtx, cuda
    │  276.445 posts → ~13.326 posts NVDA
    ▼
[Fase 2] Descarga de imágenes
    │  Descarga imágenes de los posts que las tienen
    ▼
[Fase 3] Filtrado de imágenes (Ollama)
    │  Descarta GIFs e imágenes no relevantes para el sentimiento
    ▼
[Fase 4] Análisis de texto
    │  FinBERT + BERT + SocBERT → scores 0–1 por clase
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

### Uso

```bash
# Modo prueba rápido (sin imágenes)
python pipeline.py \
  --input RedditScrapper/Data/final_dataset_clean.json \
  --test_mode \
  --sample_size 500 \
  --skip_image_analysis

# Pipeline completo
python pipeline.py \
  --input RedditScrapper/Data/final_dataset_clean.json \
  --output_dir data
```

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `--input` | requerido | Ruta al JSON de posts |
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

Cada post recibe scores de probabilidad (0–1):

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

## Predictor de sentimiento a 1 día vista

Predice si el sentimiento de la comunidad será más positivo mañana que hoy, usando únicamente el histórico de posts de Reddit.

```bash
python scripts/sentiment_predictor.py \
  --input data/nvda_processed.json \
  --output data/sent_model_comparison.csv \
  --window 1 \
  --seq_len 7
```

### Cómo funciona

1. Agrega los posts por día (promedio de scores de sentimiento)
2. Calcula features de momentum (rolling 3d, 7d, delta, volatilidad)
3. Etiqueta: ¿el sentimiento positivo de mañana > hoy? (1=sí, 0=no)
4. Entrena 5 modelos con división temporal estricta (80% train / 20% test)
5. Incluye un LSTM que aprende la secuencia de días

### Features utilizadas

| Feature | Descripción |
|---|---|
| `sent_finbert/bert/socbert_*` | Scores de sentimiento promedio del día |
| `sent_*_roll3 / roll7` | Media móvil 3 y 7 días (momentum) |
| `sent_*_delta` | Cambio respecto al día anterior |
| `sent_*_momentum` | Diferencia roll3 - roll7 (tendencia corta vs larga) |
| `finbert_vol7` | Volatilidad del sentimiento (std 7 días) |
| `finbert_confidence` | Certeza del modelo (max de las 3 clases) |
| `sentiment_agreement` | 1 si los 3 modelos coinciden |
| `finbert_bert_diff` | Discrepancia entre FinBERT y BERT |
| `score_norm` | Upvotes normalizados |
| `num_comments_norm` | Comentarios normalizados |
| `n_posts` | Volumen de posts del día |
| `sub_*` | Fracción de posts por subreddit (one-hot top-5) |

### Resultados (1000 posts, 541 días)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **LogisticRegression** | **0.7156** | 0.7037 | 0.7170 | 0.7103 |
| MLP | 0.6881 | 0.7111 | 0.6038 | 0.6531 |
| RandomForest | 0.6789 | 0.6957 | 0.6038 | 0.6465 |
| GradientBoosting | 0.6422 | 0.6591 | 0.5472 | 0.5979 |
| LSTM (seq=7) | 0.5327 | 0.5106 | 0.4706 | 0.4898 |
| Baseline | 0.4862 | — | — | — |

El sentimiento de Reddit tiene inercia: si hoy es positivo, mañana tiende a serlo. La Regresión Logística captura bien esta autocorrelación con **71.5% de accuracy**.

### Features más importantes (RandomForest)

```
sent_finbert_pos          0.1574  ← score positivo del día
sent_finbert_pos_delta    0.1057  ← cambio diario (momentum)
finbert_bert_diff         0.0685  ← discrepancia entre modelos
sent_finbert_pos_roll3    0.0462  ← tendencia de 3 días
sent_finbert_neg          0.0432
```

### Parámetros

| Parámetro | Default | Descripción |
|---|---|---|
| `--window` | 1 | Días vista para la etiqueta (1=mañana, 3=promedio 3 días) |
| `--seq_len` | 7 | Longitud de secuencia para LSTM |

### Arquitectura LSTM

```
Secuencia de seq_len días
        │
    [LSTM × 2 capas, hidden=64, dropout=0.3]
        │
    [último timestep → Linear 64→32 → ReLU → Linear 32→1]
        │
    [BCEWithLogitsLoss + early stopping (paciencia=15)]
```

> Con 541 días el LSTM tiene datos limitados. Procesando los 13.326 posts NVDA completos se obtienen ~700 días únicos y el rendimiento mejora.

---

## Cómo mejorar la accuracy

1. **Más datos** — procesar todos los 13.326 posts NVDA filtrados (actualmente se usan 1000)
2. **Ventana de 3 días** — `--window 3` reduce el ruido de días individuales
3. **Secuencia más larga** — `--seq_len 14` o `--seq_len 30` con más datos para el LSTM
4. **Análisis de imagen** — activar Ollama para posts con imagen (actualmente `image_score=0`)
5. **Subreddit como señal** — wallstreetbets vs investing tienen tonos muy distintos (ya incluido)

---

## Dataset

`final_dataset_clean.json` — 276.445 posts de Reddit (2023-2025):

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | str | ID único del post |
| `title` | str | Título |
| `selftext` | str | Cuerpo del post |
| `subreddit` | str | Subreddit de origen |
| `created_utc` | int | Timestamp Unix |
| `date` | str | Fecha ISO 8601 |
| `score` | int | Upvotes netos |
| `num_comments` | int | Número de comentarios |
| `image_urls` | list | URLs de imágenes adjuntas |
| `has_image` | bool | Si el post tiene imagen |

Tras el pipeline se añaden los campos `sent_finbert_*`, `sent_bert_*`, `sent_socbert_*`, `sent_text_only`, `sent_multimodal` e `image_analysis`.

---

## Scraper de Reddit

Para actualizar el dataset con posts nuevos:

```bash
python RedditScrapper/reddit_scrapper.py    # scraping inicial
python RedditScrapper/reddit_updater.py     # actualización incremental
```
