# NVDA Reddit Sentiment Prediction

Pipeline de análisis de sentimiento sobre posts de Reddit relacionados con NVIDIA (NVDA), con predicción del sentimiento de la comunidad a N días vista usando histórico 2023-2026.

**No usa datos de bolsa.** La predicción se basa exclusivamente en el histórico de posts de Reddit.

---

## Estructura del proyecto

```
proyecto/
├── nvidia_sentiment/              # Módulo principal — lógica de negocio
│   ├── models.py                  # Dataclasses: Post, ImageAnalysis, ModelMetrics
│   ├── serializer.py              # save/load dataset en CSV UTF-8
│   ├── nvda_filter.py             # Filtrado de posts relevantes para NVDA
│   ├── image_downloader.py        # Descarga de imágenes de posts
│   ├── image_filter.py            # Filtrado de relevancia de imágenes (Ollama)
│   ├── text_analyzer.py           # Análisis de sentimiento: FinBERT, BERT, SocBERT
│   ├── image_analyzer.py          # Análisis de sentimiento de imagen (Ollama)
│   └── multimodal_comparator.py   # Fusión texto + imagen
│
├── scripts/
│   ├── sentiment_predictor.py     # Predictor principal (Optuna + LSTM + modelos clásicos)
│   └── update_dataset.py          # Actualizador incremental de posts nuevos
│
├── RedditScrapper/
│   ├── Data/
│   │   └── final_dataset_clean.json   # Dataset de entrada (276k posts, 2023-2025)
│   ├── reddit_scrapper.py
│   └── reddit_updater.py
│
├── data/                          # Salidas generadas (en .gitignore)
│   ├── images/                    # Imágenes descargadas
│   ├── nvda_processed.csv         # Posts procesados con sentimiento FinBERT
│   └── sent_model_comparison.csv  # Resultados de los modelos de predicción
│
└── pipeline.py                    # Orquestador principal
```

---

## Requisitos

```bash
pip install torch transformers pandas numpy scikit-learn optuna requests
```

Ollama (para análisis de imagen, opcional):
```bash
ollama pull llama3.2-vision
```

---

## Pipeline principal

```bash
# Procesar todos los posts NVDA con FinBERT (tarda ~40 min)
python pipeline.py --input RedditScrapper/Data/final_dataset_clean.json \
  --skip_image_analysis --skip_download --output_dir data

# Modo prueba rápido
python pipeline.py --input RedditScrapper/Data/final_dataset_clean.json \
  --test_mode --sample_size 200 --skip_image_analysis
```

## Actualizar con posts nuevos

```bash
# Añade posts de los últimos 7 días (deduplicando automáticamente)
python scripts/update_dataset.py

# Personalizado
python scripts/update_dataset.py --days 14 --min_score 20 --dry_run
```

## Entrenar los modelos de predicción

```bash
python scripts/sentiment_predictor.py \
  --input data/nvda_processed.csv \
  --windows 1 3 5 10 21 \
  --n_trials 30 \
  --seq_len 30
```

---

## Dataset

13.326 posts NVDA filtrados de 276.445 posts totales (2023-2025), más posts de 2026 añadidos via API de Reddit. Distribuidos en 1.080 días únicos con 12.3 posts/día de promedio.

| Año | Posts |
|---|---|
| 2023 | 2.697 |
| 2024 | 6.903 |
| 2025 | 3.726 |

---

## Por qué FinBERT y no SocBERT

Se probaron tres modelos de análisis de sentimiento de texto:

| Modelo | Dominio | std diario | Adecuado |
|---|---|---|---|
| [FinBERT](https://huggingface.co/ProsusAI/finbert) | Noticias financieras | **0.24** | Sí |
| [DistilBERT SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) | Reseñas generales | 0.18 | Parcial |
| [SocBERT](https://huggingface.co/sarkerlab/SocBERT-base) | Redes sociales | 0.027 | No |

**SocBERT fue descartado** porque sus scores tienen una varianza 10 veces menor que FinBERT. Todos los posts de NVDA en Reddit reciben scores entre 0.45 y 0.55, sin apenas diferenciación. Esto hace que cualquier etiqueta derivada de SocBERT sea predecible trivialmente por mean-reversion estadística (cuando el score es 0.47 casi siempre sube a 0.51 al día siguiente, y viceversa), no por aprendizaje real. El accuracy del 72% que mostraba con SocBERT era un artefacto matemático, no señal predictiva.

**FinBERT fue elegido** porque está entrenado específicamente en noticias y análisis financieros, el mismo dominio que los posts de inversión en Reddit. Sus scores tienen std=0.24, lo que significa que diferencia claramente entre posts muy positivos (earnings beat, nuevo producto) y muy negativos (caída de mercado, competencia). Esta varianza es la que permite a los modelos de ML aprender patrones reales.

---

## Por qué se eliminó el subreddit como feature

Se realizó un test de ablación comparando accuracy con y sin las features de subreddit (wallstreetbets, stocks, investing, StockMarket, nvidia, pennystocks):

| Ventana | Con subreddit | Sin subreddit | Diferencia |
|---|---|---|---|
| +1 día | 68.3% | 68.7% | -0.4% (ruido) |
| +3 días | 53.3% | 54.6% | -1.3% (ruido) |
| +5 días | 52.0% | 48.5% | +3.5% (útil) |
| +10 días | 50.7% | 47.1% | +3.5% (útil) |

A corto plazo (1-3 días) el subreddit de origen es ruido puro — añadirlo empeora el modelo. Esto tiene sentido: el tono del día a día en wallstreetbets vs investing es diferente, pero esa diferencia no predice el sentimiento del día siguiente porque los posts de distintos subreddits se mezclan en el agregado diario.

A partir de 5 días el subreddit aporta algo (+3.5%), probablemente porque distintas comunidades tienen ciclos de hype/miedo con duraciones diferentes. Sin embargo, la ganancia es pequeña y añade complejidad al modelo, por lo que se eliminó para mantener el pipeline simple y robusto.

---

## Por qué unos modelos son mejores que otros

### Resultados con FinBERT (13.326 posts, split 60/20/20, Optuna 30 trials)

| Ventana | Mejor modelo | Accuracy | LSTM accuracy |
|---|---|---|---|
| +1 día | RandomForest | **61.1%** | 52.2% |
| +3 días | GradientBoosting | 59.7% | 51.6% |
| +5 días | MLP | 54.9% | 51.4% |
| +10 días | LogisticRegression | 55.1% | 51.6% |
| +21 días | RandomForest | 55.7% | 52.2% |

**Por qué RandomForest y GradientBoosting ganan a LogisticRegression:**
Los modelos de árbol capturan interacciones no lineales entre features. Por ejemplo, "FinBERT positivo + alto volumen de posts + momentum alcista" es una combinación que predice mejor que cada feature por separado. La regresión logística solo puede aprender relaciones lineales.

**Por qué MLP (red neuronal) es competitivo:**
El MLP con early stopping aprende representaciones intermedias de las features, similar a los árboles pero de forma continua. Con Optuna ajustando el tamaño de capas y learning rate, encuentra configuraciones que capturan patrones que los árboles no ven.

**Por qué el LSTM no supera a los modelos clásicos:**
El LSTM necesita señal temporal clara y suficientes datos para aprender secuencias. Con 1.080 días y split 60/20/20, el conjunto de entrenamiento tiene ~648 secuencias de 30 días. El LSTM hace early stopping en epoch 22-25 con val_loss ~0.69 (entropía máxima para clasificación binaria), lo que indica que no aprende patrones secuenciales útiles. Los modelos clásicos con features de momentum (rolling 3d, 7d, delta) ya capturan la información temporal de forma más eficiente con menos datos.

**Por qué el accuracy es moderado (~55-61%):**
El sentimiento de Reddit predice el sentimiento futuro de Reddit, no el precio de la acción. La señal real existe (supera al baseline de ~50%) pero es débil porque:
1. Los posts de Reddit son ruidosos — muchos son memes, no análisis
2. El sentimiento tiene inercia de 1-2 días, no más
3. Sin datos externos (precio, volumen de trading, noticias) el techo es ~65%

---

## Metodología anti-overfitting

Para garantizar resultados honestos:

- **Split temporal 60/20/20** — train (2023-2024) / validation (2024-2025) / test (2025-2026). El test nunca se toca durante el desarrollo
- **Optuna sobre validation** — los hiperparámetros se buscan sobre el conjunto de validación, nunca sobre el test
- **Permutation importance** — en lugar de impurity-based importance (sesgada hacia features con muchos valores), se usa permutation importance que mide el impacto real en el test set
- **Etiqueta robusta** — la etiqueta usa cruce de EMAs (EMA3 > EMA10 en el futuro) en lugar de comparar futuro vs presente, evitando la correlación matemática trivial por mean-reversion
- **Shuffle test** — se verificó que el accuracy real supera al obtenido con fechas aleatorizadas, confirmando que hay señal temporal genuina
