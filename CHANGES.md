# Cambios de Reorganización del Proyecto

## Estructura nueva vs antigua

### Antes
```
proyecto/
├── RedditScrapper/
│   ├── Data/
│   │   ├── *.json (datasets grandes)
│   │   ├── *.py   (scripts de limpieza ad-hoc)
│   │   └── images/
│   └── sentiments/
│       ├── sentiment_analyzer_finbert.py
│       ├── sentiment_analyzer_socbert.py
│       ├── sentiments_analyzer_bert.py
│       └── Ollama/
│           └── analizador_imagen.py
├── nvda_filter.py        ← CLI en raíz
├── image_downloader.py   ← CLI en raíz
├── image_filter.py       ← CLI en raíz
├── text_analyzer.py      ← CLI en raíz
├── image_analyzer.py     ← CLI en raíz
├── multimodal_comparator.py ← CLI en raíz
├── predictor.py          ← CLI en raíz
└── pipeline.py
```

### Después
```
proyecto/
├── nvidia_sentiment/       ← Módulo principal (lógica de negocio)
│   ├── models.py
│   ├── serializer.py
│   ├── nvda_filter.py
│   ├── image_downloader.py
│   ├── image_filter.py
│   ├── text_analyzer.py
│   ├── image_analyzer.py
│   ├── multimodal_comparator.py
│   └── predictor.py
├── scripts/                ← CLIs individuales por módulo
│   ├── nvda_filter.py
│   ├── image_downloader.py
│   ├── image_filter.py
│   ├── text_analyzer.py
│   ├── image_analyzer.py
│   ├── multimodal_comparator.py
│   └── predictor.py
├── data/                   ← Salidas generadas (en .gitignore)
│   └── images/
├── RedditScrapper/
│   └── Data/
│       ├── final_dataset_clean.json  ← Dataset de entrada (versionado)
│       └── nvda_top3_backfill.csv    ← Precios históricos (versionado)
└── pipeline.py             ← Orquestador principal
```

---

## Archivos eliminados

### JSONs grandes (no escalables en git)
- `RedditScrapper/Data/*_2023_2025.json`
- `RedditScrapper/Data/*_clean.json` (excepto `final_dataset_clean.json`)
- `RedditScrapper/Data/merged_sorted_reddit.json`
- `RedditScrapper/Data/nvda_processed.json`

### Scripts ad-hoc obsoletos
- `RedditScrapper/Data/analizar_i.py`
- `RedditScrapper/Data/clean_data_from_merge.py`
- `RedditScrapper/Data/detectar_img_post.py`
- `RedditScrapper/Data/fusion_dataset.py`
- `RedditScrapper/Data/historico_post_wallstreetbeats.py`
- `RedditScrapper/Data/limpieza_de_dataset.py`

### Analizadores duplicados
- `RedditScrapper/sentiments/sentiment_analyzer_finbert.py`
- `RedditScrapper/sentiments/sentiment_analyzer_socbert.py`
- `RedditScrapper/sentiments/sentiments_analyzer_bert.py`
- `RedditScrapper/sentiments/Ollama/analizador_imagen.py`

---

## Sentimientos (scores de 0 a 1)

Cada post analizado incluye los siguientes campos de probabilidad:

| Campo | Modelo | Descripción |
|---|---|---|
| `sent_finbert_pos` | FinBERT | Probabilidad de sentimiento positivo (financiero) |
| `sent_finbert_neg` | FinBERT | Probabilidad de sentimiento negativo (financiero) |
| `sent_finbert_neu` | FinBERT | Probabilidad de sentimiento neutro (financiero) |
| `sent_bert_pos` | DistilBERT SST-2 | Probabilidad de sentimiento positivo (general) |
| `sent_bert_neg` | DistilBERT SST-2 | Probabilidad de sentimiento negativo (general) |
| `sent_socbert_pos` | SocBERT | Probabilidad de sentimiento positivo (social media) |
| `sent_socbert_neg` | SocBERT | Probabilidad de sentimiento negativo (social media) |
| `image_analysis.score` | Ollama llama3.2-vision | Relevancia de la imagen para NVDA (0.0–1.0) |

Todos los scores son probabilidades entre 0 y 1. La suma de pos+neg+neu = 1.0 para FinBERT.

---

## Modelos de predicción y accuracy

El predictor entrena 4 modelos con división temporal (sin shuffle) para predecir el movimiento del precio de NVDA al día siguiente (subida=1, bajada=0):

| Modelo | Descripción |
|---|---|
| LogisticRegression | Regresión logística, baseline lineal |
| RandomForest | Ensemble de árboles de decisión (100 estimadores) |
| GradientBoosting | LightGBM si disponible, sino sklearn GBM |
| MLP | Red neuronal (capas 64→32), max 500 iteraciones |

Las métricas reportadas son:

| Métrica | Descripción |
|---|---|
| Accuracy | % de predicciones correctas sobre el conjunto de prueba |
| Precision | De las predicciones "subida", cuántas fueron correctas |
| Recall | De las subidas reales, cuántas fueron detectadas |
| F1 | Media armónica de precision y recall |

Los resultados se ordenan por accuracy descendente. El CSV `data/model_comparison.csv` y el gráfico `data/accuracy_comparison.png` se generan automáticamente al ejecutar el predictor.

---

## Uso del pipeline

```bash
# Modo prueba (5 posts, sin análisis de imagen)
python pipeline.py \
  --input RedditScrapper/Data/final_dataset_clean.json \
  --test_mode \
  --sample_size 5 \
  --skip_image_analysis

# Pipeline completo con predicción
python pipeline.py \
  --input RedditScrapper/Data/final_dataset_clean.json \
  --price_data RedditScrapper/Data/nvda_top3_backfill.csv \
  --output_dir data

# CLI individual del predictor
python scripts/predictor.py \
  --input data/nvda_processed.json \
  --price_data RedditScrapper/Data/nvda_top3_backfill.csv
```
