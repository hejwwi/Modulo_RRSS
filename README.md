# NVDA Reddit Sentiment Prediction

Pipeline de análisis de sentimiento sobre posts de Reddit relacionados con NVIDIA (NVDA), con predicción del sentimiento de la comunidad a N días vista usando histórico 2021-2026.

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
│   ├── nvda_processed.csv         # Posts procesados con sentimiento FinBERT (15.285 posts)
│   ├── nvda_processed_clean.csv   # Dataset filtrado sin ruido (6.260 posts, score>=5)
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
# Con dataset completo (15k posts) — máximo accuracy
python scripts/sentiment_predictor.py \
  --input data/nvda_processed.csv \
  --windows 1 3 5 10 21 \
  --n_trials 30 \
  --seq_len 30

# Con dataset limpio (6k posts, sin ruido) — mejor F1 para LSTM
python scripts/sentiment_predictor.py \
  --input data/nvda_processed_clean.csv \
  --windows 1 3 5 \
  --n_trials 20 \
  --seq_len 30
```

---

## Dataset

**15.285 posts** con sentimiento FinBERT analizado, distribuidos en **1.624 días únicos** (9.4 posts/día de promedio). El dataset se construyó en varias fases:

- **2023-2025**: 13.326 posts extraídos del dataset histórico `final_dataset_clean.json` (276k posts totales filtrados por términos NVDA)
- **2026**: 98 posts descargados via API pública de Reddit (posts más recientes)
- **2022**: 1.034 posts descargados via [Arctic Shift](https://arctic-shift.photon-reddit.com) (archivo histórico completo de Reddit)
- **2021**: 925 posts descargados via Arctic Shift

| Año | Posts | Fuente |
|---|---|---|
| 2021 | 925 | Arctic Shift |
| 2022 | 1.034 | Arctic Shift |
| 2023 | 2.697 | Dataset histórico |
| 2024 | 6.903 | Dataset histórico |
| 2025 | 3.726 | Dataset histórico |
| 2026 | 98 | API Reddit |
| **Total** | **15.285** | |

Subreddits incluidos: wallstreetbets, stocks, investing, StockMarket, nvidia, pennystocks, wallstreetbetsnews, wallstreetbets2.

### Por qué se añadieron 2021 y 2022

Con solo los datos de 2023-2026 el LSTM tenía 648 secuencias de entrenamiento (seq_len=30), insuficiente para aprender patrones temporales. Añadir 2021 y 2022 aumentó a **943 secuencias**, lo que permitió al LSTM con atención superar a los modelos clásicos en F1 para ventanas de +3 y +5 días.

Sin embargo, la curva de aprendizaje muestra que el modelo se satura alrededor de 400 días de train — añadir más años históricos (2019, 2020) no mejoraría el accuracy porque la señal de sentimiento de Reddit sobre NVDA tiene un techo natural sin datos externos.

---

## Análisis de ruido en el dataset

El 60% de los posts originales son ruido que distorsiona los promedios diarios de sentimiento:

| Tipo de ruido | Posts afectados | Impacto |
|---|---|---|
| Score < 5 (nadie lo votó) | 7.640 (50%) | Promedio diario puede variar hasta 0.75 puntos |
| Título < 15 chars | 1.563 (10%) | FinBERT analiza texto insuficiente |
| Selftext [deleted]/[removed] | ~500 (3%) | Texto eliminado, solo queda título |
| Sentimiento plano (max(pos,neg) < 0.05) | ~2.000 (13%) | No aporta señal |

Se creó `nvda_processed_clean.csv` aplicando estos filtros: `score >= 5`, `len(title) >= 15`, `selftext != [deleted]`, `max(sent_finbert_pos, sent_finbert_neg) > 0.05`. Resultado: **6.260 posts de calidad** (4.5 posts/día).

---

## Qué modelo usar según el objetivo

La elección depende de qué significa "acertar" en tu caso de uso:

| Objetivo | Modelo recomendado | Dataset | Accuracy | Precisión | Recall |
|---|---|---|---|---|---|
| Máxima exactitud general | **MLP** | original (15k) | **64.6%** | 64.8% | 52.3% |
| No perderse ningún día positivo | **LSTM+Attn** | limpio (6k) | 48.6% | 48.6% | **100%** |

**MLP + dataset original** es la mejor opción si quieres predecir con la mayor exactitud posible. Cuando predice "mañana el sentimiento será positivo", acierta el 64.8% de las veces.

**LSTM+Attn + dataset limpio** es útil si el coste de perderse un día positivo es alto. Detecta el 100% de los días positivos pero genera muchas falsas alarmas (precisión 48.6%).

### Qué es el F1 y por qué importa

El F1 es la media armónica entre precisión y recall:

- **Precisión**: de todas las veces que el modelo predijo "positivo", ¿cuántas acertó?
- **Recall**: de todos los días realmente positivos, ¿cuántos detectó el modelo?
- **F1**: combina ambos — penaliza tanto los falsos positivos como los falsos negativos

El accuracy puede ser engañoso cuando las clases están desbalanceadas (43% positivos vs 57% negativos). Un modelo que prediga siempre "negativo" tendría 57% de accuracy sin aprender nada. El F1 detecta ese comportamiento.

### Resultados completos por ventana (dataset original, 15k posts)

**+1 día**

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|
| **MLP** | **64.6%** | 64.8% | 52.3% | 0.579 |
| LogisticRegression | 63.4% | 64.8% | 46.4% | 0.541 |
| RandomForest | 60.9% | 60.3% | 46.4% | 0.524 |
| GradientBoosting | 60.3% | 60.4% | 42.4% | 0.498 |
| LSTM+Attn (seq=30) | 57.3% | 53.6% | 59.1% | 0.562 |

**+3 días**

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|
| **RandomForest** | **59.7%** | 71.7% | 21.9% | 0.335 |
| MLP | 59.4% | 66.1% | 25.8% | 0.371 |
| LSTM+Attn (seq=30) | 49.8% | 47.8% | 86.1% | **0.615** |

**+5 días**

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|
| **RandomForest** | **58.3%** | 63.3% | 25.2% | 0.360 |
| MLP | 54.9% | 57.1% | 13.2% | 0.215 |
| LSTM+Attn (seq=30) | 46.6% | 46.6% | 100% | **0.636** |

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

### Resultados con FinBERT (15.285 posts, 2021-2026, split 60/20/20, Optuna 20 trials)

Ver tabla completa en la sección "Qué modelo usar según el objetivo".

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

- **Split temporal 60/20/20** — train (2021-2024) / validation (2024-2025) / test (2025-2026). El test nunca se toca durante el desarrollo
- **Optuna sobre validation** — los hiperparámetros se buscan sobre el conjunto de validación, nunca sobre el test
- **Permutation importance** — en lugar de impurity-based importance (sesgada hacia features con muchos valores), se usa permutation importance que mide el impacto real en el test set
- **Etiqueta robusta** — la etiqueta usa cruce de EMAs (EMA3 > EMA10 en el futuro) en lugar de comparar futuro vs presente, evitando la correlación matemática trivial por mean-reversion
- **Shuffle test** — se verificó que el accuracy real supera al obtenido con fechas aleatorizadas, confirmando que hay señal temporal genuina
