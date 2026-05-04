# Predicción de Sentimiento de la Comunidad Inversora sobre NVIDIA en Reddit mediante Modelos de Aprendizaje Automático

**Trabajo de Fin de Grado — Ingeniería Informática / Ciencia de Datos**

---

## Índice

1. [Introducción y motivación](#1-introducción-y-motivación)
2. [Módulo de adquisición de datos](#2-módulo-de-adquisición-de-datos)
3. [Módulo de análisis de sentimiento](#3-módulo-de-análisis-de-sentimiento)
4. [Módulo de ingeniería de características](#4-módulo-de-ingeniería-de-características)
5. [Módulo de predicción](#5-módulo-de-predicción)
6. [Módulo de validación y evaluación](#6-módulo-de-validación-y-evaluación)
7. [Análisis de ruido y calidad del dataset](#7-análisis-de-ruido-y-calidad-del-dataset)
8. [Resultados y comparativa de modelos](#8-resultados-y-comparativa-de-modelos)
9. [Referencias en formato APA 7](#9-referencias-en-formato-apa-7)

---

## 1. Introducción y motivación

El análisis de sentimiento en redes sociales ha emergido como una herramienta relevante para comprender el comportamiento colectivo de comunidades inversoras. Plataformas como Reddit concentran millones de publicaciones diarias sobre activos financieros, generando un flujo continuo de opinión no estructurada que puede contener señales predictivas sobre la evolución del sentimiento del mercado.

Este trabajo desarrolla un pipeline completo de extracción, análisis y predicción del sentimiento de la comunidad inversora de Reddit sobre NVIDIA Corporation (ticker: NVDA) para el período 2021-2026. El sistema predice si el sentimiento colectivo del día siguiente será más positivo o negativo que el actual, utilizando exclusivamente datos de publicaciones de Reddit sin recurrir a datos de cotización bursátil.

**Objetivo principal:** Construir un sistema reproducible y escalable que prediga el sentimiento de la comunidad inversora sobre NVDA a 1, 3 y 5 días vista con una exactitud superior al azar (baseline ~53%).

---

## 2. Módulo de adquisición de datos

### 2.1 Fuentes de datos

Se utilizaron tres fuentes complementarias para construir el dataset histórico:

**Dataset histórico (2023-2025)**
El archivo `final_dataset_clean.json` contiene 276.445 publicaciones de Reddit recopiladas mediante scraping previo. Se filtraron las publicaciones relevantes para NVDA mediante búsqueda de términos clave en título y cuerpo del post.

Términos de filtrado: `nvidia`, `nvda`, `geforce`, `rtx`, `cuda` (búsqueda case-insensitive).

**API pública de Reddit (2026)**
Para los posts más recientes se utilizó la API JSON pública de Reddit sin autenticación OAuth:

```
GET https://www.reddit.com/r/{subreddit}/search.json
Parámetros: q, sort, limit, t, restrict_sr
```

**Arctic Shift — Archivo histórico de Reddit (2021-2022)**
Para ampliar el dataset con datos anteriores a 2023 se utilizó Arctic Shift, sucesor del servicio Pushshift, que mantiene un archivo completo de Reddit desde 2005:

```
GET https://arctic-shift.photon-reddit.com/api/posts/search
Parámetros: title, subreddit, after (unix timestamp), before, limit, sort
```

*Nota: el parámetro `title` requiere obligatoriamente `subreddit` o `author`. Las búsquedas con operadores booleanos (OR) no están soportadas; se realizaron peticiones separadas por término.*

### 2.2 Subreddits monitorizados

| Subreddit | Tipo de comunidad | Posts NVDA |
|---|---|---|
| r/wallstreetbets | Inversión especulativa / retail | 5.651 |
| r/stocks | Análisis de acciones | 3.304 |
| r/investing | Inversión a largo plazo | 1.751 |
| r/StockMarket | Mercados generales | 1.499 |
| r/nvidia | Comunidad de producto | 1.388 |
| r/pennystocks | Acciones de bajo precio | 580 |
| r/wallstreetbetsnews | Noticias WSB | 370 |
| r/wallstreetbets2 | Comunidad alternativa WSB | 127 |

### 2.3 Estructura del dataset

Cada publicación almacena los siguientes campos en `nvda_processed.csv`:

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | str | Identificador único del post en Reddit |
| `date` | str | Fecha y hora de publicación (ISO 8601) |
| `created_utc` | int | Timestamp Unix de creación |
| `title` | str | Título de la publicación |
| `selftext` | str | Cuerpo del texto (puede estar vacío) |
| `score` | int | Upvotes netos recibidos |
| `num_comments` | int | Número de comentarios |
| `has_image` | bool | Indica si el post incluye imagen |
| `sent_finbert_label` | str | Etiqueta FinBERT: positive / negative / neutral |
| `sent_finbert_pos` | float | Probabilidad de sentimiento positivo [0, 1] |
| `sent_finbert_neg` | float | Probabilidad de sentimiento negativo [0, 1] |
| `sent_finbert_neu` | float | Probabilidad de sentimiento neutro [0, 1] |
| `sent_text_only` | str | Etiqueta final basada en texto |
| `sent_multimodal` | str | Etiqueta fusionada texto + imagen |
| `image_local_path` | str | Ruta local de imagen descargada |

### 2.4 Composición final del dataset

| Año | Posts | Fuente |
|---|---|---|
| 2021 | 925 | Arctic Shift |
| 2022 | 1.034 | Arctic Shift |
| 2023 | 2.697 | Dataset histórico |
| 2024 | 6.903 | Dataset histórico |
| 2025 | 3.726 | Dataset histórico |
| 2026 | 98 | API Reddit |
| **Total** | **15.285** | |

**Días únicos con publicaciones:** 1.624  
**Media de posts por día:** 9.4

### 2.5 Actualización incremental

El módulo `scripts/update_dataset.py` permite actualizar el dataset con publicaciones recientes sin duplicar entradas. Utiliza la API pública de Reddit con paginación y deduplicación por `id`:

```bash
python scripts/update_dataset.py --days 7 --min_score 10
python scripts/update_dataset.py --days 14 --dry_run
```

---

## 3. Módulo de análisis de sentimiento

### 3.1 Selección del modelo NLP

Se evaluaron tres modelos de clasificación de sentimiento basados en la arquitectura Transformer:

| Modelo | Identificador HuggingFace | Dominio | std diario |
|---|---|---|---|
| FinBERT | `ProsusAI/finbert` | Noticias financieras | **0.24** |
| DistilBERT SST-2 | `distilbert-base-uncased-finetuned-sst-2-english` | Reseñas generales | 0.18 |
| SocBERT | `sarkerlab/SocBERT-base` | Redes sociales | 0.027 |

**Criterio de selección:** La desviación estándar del score diario promedio es el indicador clave. Un modelo con baja varianza (SocBERT, std=0.027) asigna scores similares a todos los posts, haciendo que cualquier etiqueta derivada sea predecible por mean-reversion estadística en lugar de por aprendizaje real. FinBERT, con std=0.24, diferencia claramente entre posts muy positivos y muy negativos, proporcionando señal real para los modelos de predicción.

**Modelo seleccionado: FinBERT** (Araci, 2019), entrenado sobre 10.000 artículos financieros de Reuters y The Financial Times.

### 3.2 Proceso de análisis

Para cada publicación se concatena `title + selftext` y se trunca a 256 tokens. FinBERT devuelve tres probabilidades que suman 1.0:

- `sent_finbert_pos`: probabilidad de sentimiento positivo
- `sent_finbert_neg`: probabilidad de sentimiento negativo  
- `sent_finbert_neu`: probabilidad de sentimiento neutro

Los modelos se cargan de forma lazy (una sola vez por proceso) para optimizar el rendimiento en procesamiento por lotes.

### 3.3 Fusión multimodal

Para publicaciones con imagen relevante se aplica un modelo de visión (Ollama `llama3.2-vision`) que genera un score de relevancia visual. La fusión se realiza mediante combinación lineal ponderada:

```
sent_multimodal = 0.75 × sent_texto + 0.25 × sent_imagen
```

En el dataset actual las imágenes no han sido analizadas (pipeline ejecutado con `--skip_image_analysis`), por lo que `sent_multimodal == sent_text_only` para todos los posts.

---

## 4. Módulo de ingeniería de características

### 4.1 Agregación diaria

Los modelos de predicción no operan sobre posts individuales sino sobre series temporales diarias. Para cada día se calcula el promedio de los scores FinBERT de todos los posts de ese día:

```
sent_finbert_pos[t] = mean({sent_finbert_pos[i] : post i publicado el día t})
```

### 4.2 Features derivadas

A partir de las series diarias se calculan las siguientes características:

| Feature | Fórmula | Descripción |
|---|---|---|
| `sent_finbert_pos_roll3` | rolling(3).mean() | Media móvil 3 días |
| `sent_finbert_pos_roll7` | rolling(7).mean() | Media móvil 7 días |
| `sent_finbert_pos_delta` | diff(1) | Cambio respecto al día anterior |
| `sent_finbert_pos_momentum` | roll3 - roll7 | Tendencia corta vs larga |
| `finbert_vol7` | rolling(7).std() | Volatilidad del sentimiento |
| `finbert_confidence` | max(pos, neg, neu) | Certeza del modelo |
| `n_posts` | count | Volumen de publicaciones del día |
| `n_posts_roll3` | rolling(3).mean() | Tendencia del volumen |
| `n_posts_delta` | diff(1) | Cambio en volumen |
| `score_norm` | score / max_score | Upvotes normalizados |
| `num_comments_norm` | comments / max_comments | Comentarios normalizados |

Las mismas derivadas se calculan para `sent_finbert_neg` y `sent_finbert_neu`, resultando en **23 features** totales.

### 4.3 Definición de la etiqueta

La variable objetivo es el **régimen de sentimiento futuro** basado en cruce de medias exponenciales:

```
EMA_corta[t] = EMA(sent_finbert_pos, span=3)[t]
EMA_larga[t] = EMA(sent_finbert_pos, span=10)[t]
régimen[t]   = 1 si EMA_corta[t] > EMA_larga[t], 0 en caso contrario
etiqueta[t]  = régimen[t + window]
```

Esta definición evita la correlación matemática trivial que surge al comparar el valor futuro directamente con el valor presente (mean-reversion estadística). Se verificó mediante shuffle test que el accuracy real supera al obtenido con fechas aleatorizadas, confirmando señal temporal genuina.

---

## 5. Módulo de predicción

### 5.1 Modelos clásicos

Se entrenaron cuatro modelos de clasificación supervisada con búsqueda de hiperparámetros mediante Optuna (Akiba et al., 2019):

**Regresión Logística** (`sklearn.linear_model.LogisticRegression`)
- Hiperparámetro optimizado: `C` ∈ [0.01, 10.0] (log-uniform)
- Ventaja: interpretable, robusto con pocos datos
- Limitación: solo captura relaciones lineales entre features

**Random Forest** (`sklearn.ensemble.RandomForestClassifier`)
- Hiperparámetros: `n_estimators` ∈ [50, 400], `max_depth` ∈ [2, 10], `max_features` ∈ [0.3, 1.0]
- Ventaja: captura interacciones no lineales, resistente al overfitting
- Limitación: no captura dependencias temporales secuenciales

**Gradient Boosting** (LightGBM si disponible, `sklearn.ensemble.GradientBoostingClassifier` como fallback)
- Hiperparámetros: `learning_rate`, `n_estimators`, `max_depth`, `subsample`
- Ventaja: alta capacidad predictiva con regularización implícita

**MLP — Perceptrón Multicapa** (`sklearn.neural_network.MLPClassifier`)
- Hiperparámetros: tamaño de capas ocultas (h1, h2), `learning_rate_init`, `alpha`
- Arquitectura: [input → h1 → h2 → 1], con early stopping sobre fracción de validación
- Ventaja: aprende representaciones no lineales complejas

### 5.2 LSTM con mecanismo de atención

Para capturar dependencias temporales secuenciales se implementó una red LSTM con atención temporal:

**Arquitectura:**
```
Entrada: secuencia de seq_len=30 días × 23 features
    ↓
LSTM bidireccional (hidden=128, layers=2, dropout=0.2)
    ↓
Mecanismo de atención temporal:
    scores = Linear(H → 1)
    weights = softmax(scores)
    context = Σ(weights × hidden_states)
    ↓
BatchNorm1d(128)
    ↓
Linear(128 → 64) → ReLU → Dropout(0.2)
Linear(64 → 32) → ReLU
Linear(32 → 1) → BCEWithLogitsLoss
```

**Detalles de entrenamiento:**
- Optimizador: AdamW (lr=5×10⁻⁴, weight_decay=10⁻³)
- Scheduler: CosineAnnealingLR (T_max=100)
- Función de pérdida: BCEWithLogitsLoss con peso de clase positiva para manejar desbalance
- Early stopping: paciencia=25 épocas sobre F1 de validación
- Gradient clipping: max_norm=0.5
- Umbral de decisión: optimizado sobre validación en rango [0.30, 0.70]

**Construcción de secuencias:**
Para cada día t con t ≥ seq_len, la secuencia de entrada es la matriz de features de los días [t-seq_len, t-1]. Esto garantiza que no hay data leakage temporal.

---

## 6. Módulo de validación y evaluación

### 6.1 División temporal 60/20/20

Se utiliza una división temporal estricta sin shuffle para respetar la naturaleza secuencial de los datos:

| Conjunto | Proporción | Período aproximado | Días |
|---|---|---|---|
| Train | 60% | 2021 → mediados 2024 | 974 |
| Validation | 20% | mediados 2024 → 2025 | 325 |
| Test | 20% | 2025 → 2026 | 325 |

**Principio fundamental:** el conjunto de test nunca se utiliza durante el desarrollo ni la búsqueda de hiperparámetros. Optuna optimiza exclusivamente sobre el conjunto de validación.

### 6.2 Búsqueda de hiperparámetros con Optuna

Optuna implementa el algoritmo TPE (Tree-structured Parzen Estimator) para la búsqueda bayesiana de hiperparámetros. Para cada modelo se ejecutan 20-30 trials, maximizando el F1-score sobre el conjunto de validación.

Una vez identificados los mejores hiperparámetros, el modelo final se entrena sobre train+validation antes de evaluar en test.

### 6.3 Métricas de evaluación

**Accuracy:** proporción de predicciones correctas sobre el total.

**Precisión:** de todas las predicciones positivas, fracción que son verdaderos positivos.
```
Precisión = VP / (VP + FP)
```

**Recall:** de todos los casos positivos reales, fracción detectada correctamente.
```
Recall = VP / (VP + FN)
```

**F1-score:** media armónica de precisión y recall.
```
F1 = 2 × (Precisión × Recall) / (Precisión + Recall)
```

El F1 es la métrica principal porque el dataset está desbalanceado (43% positivos, 57% negativos). Un modelo que prediga siempre la clase mayoritaria obtendría 57% de accuracy sin aprender nada.

### 6.4 Permutation Importance

Para evaluar la importancia real de cada feature se utiliza permutation importance (Breiman, 2001) en lugar de impurity-based importance. El método permuta aleatoriamente los valores de cada feature en el conjunto de test y mide la caída en accuracy:

```
importancia(f) = accuracy_original - accuracy_con_f_permutada
```

Esto evita el sesgo de las métricas basadas en impureza hacia features con alta cardinalidad.

### 6.5 Shuffle test

Para confirmar que el accuracy proviene de señal temporal real y no de artefactos estadísticos, se ejecutó un shuffle test: se aleatorizaron las fechas del dataset y se reentrenó el modelo. Si el accuracy con fechas reales supera significativamente al obtenido con fechas aleatorias, existe señal temporal genuina.

Resultado: accuracy real (64.6%) > accuracy con shuffle (media ~52%), confirmando señal real.

### 6.6 Ablación de subreddit

Se realizó un experimento de ablación comparando el rendimiento con y sin las features de subreddit (one-hot encoding de los 8 subreddits monitorizados):

| Ventana | Con subreddit | Sin subreddit | Conclusión |
|---|---|---|---|
| +1 día | 68.3% | 68.7% | Ruido (-0.4%) |
| +3 días | 53.3% | 54.6% | Ruido (-1.3%) |
| +5 días | 52.0% | 48.5% | Útil (+3.5%) |

A corto plazo el subreddit es ruido; se eliminó del modelo final para mantener simplicidad y robustez.

---

## 7. Análisis de ruido y calidad del dataset

### 7.1 Tipos de ruido identificados

| Tipo | Posts afectados | Impacto en promedio diario |
|---|---|---|
| Score < 5 (sin engagement) | 7.640 (50%) | Hasta ±0.75 en sent_finbert_pos |
| Título < 15 caracteres | 1.563 (10%) | FinBERT analiza texto insuficiente |
| Selftext [deleted]/[removed] | ~500 (3%) | Solo título disponible |
| Sentimiento plano (max(pos,neg) < 0.05) | ~2.000 (13%) | No aporta señal diferenciadora |

### 7.2 Dataset limpio

Se generó `nvda_processed_clean.csv` aplicando los filtros: `score ≥ 5`, `len(title) ≥ 15`, `selftext ∉ {[deleted], [removed]}`, `max(sent_finbert_pos, sent_finbert_neg) > 0.05`.

Resultado: **6.260 posts** (4.5 posts/día), eliminando el 59% del dataset original.

### 7.3 Impacto del filtrado en los modelos

El filtrado mejora el F1 del LSTM pero reduce el accuracy de los modelos clásicos, porque los posts de bajo score actuaban como "relleno" que estabilizaba los promedios diarios. Con menos posts por día los promedios son más volátiles y más difíciles de predecir con features estáticas.

### 7.4 Curva de aprendizaje

Se analizó el impacto de añadir más datos históricos mediante una curva de aprendizaje con test fijo (últimos 325 días):

| Días de train | Accuracy | F1 |
|---|---|---|
| 200 | 61.9% | 0.475 |
| 400 | **64.9%** | **0.565** |
| 600 | 64.3% | 0.540 |
| 800 | 62.8% | 0.502 |
| 974 | 64.6% | 0.531 |

El modelo se satura alrededor de 400 días de train. Añadir datos anteriores a 2021 no mejoraría el accuracy porque la señal de sentimiento de Reddit sobre NVDA tiene un techo natural sin datos externos.

---

## 8. Resultados y comparativa de modelos

### 8.1 Resultados principales (dataset original, 15.285 posts)

**Ventana +1 día**

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|
| **MLP** | **64.6%** | 64.8% | 52.3% | 0.579 |
| LogisticRegression | 63.4% | 64.8% | 46.4% | 0.541 |
| RandomForest | 60.9% | 60.3% | 46.4% | 0.524 |
| GradientBoosting | 60.3% | 60.4% | 42.4% | 0.498 |
| LSTM+Attn (seq=30) | 57.3% | 53.6% | 59.1% | 0.562 |
| Baseline (mayoría) | 53.5% | — | — | — |

**Ventana +3 días**

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|
| **RandomForest** | **59.7%** | 71.7% | 21.9% | 0.335 |
| MLP | 59.4% | 66.1% | 25.8% | 0.371 |
| LSTM+Attn (seq=30) | 49.8% | 47.8% | 86.1% | **0.615** |

**Ventana +5 días**

| Modelo | Accuracy | Precisión | Recall | F1 |
|---|---|---|---|---|
| **RandomForest** | **58.3%** | 63.3% | 25.2% | 0.360 |
| MLP | 54.9% | 57.1% | 13.2% | 0.215 |
| LSTM+Attn (seq=30) | 46.6% | 46.6% | 100% | **0.636** |

### 8.2 Recomendación de uso

| Objetivo | Modelo | Dataset | Accuracy | Precisión |
|---|---|---|---|---|
| Máxima exactitud | MLP | original (15k) | **64.6%** | 64.8% |
| No perderse días positivos | LSTM+Attn | limpio (6k) | 48.6% | 48.6% |

### 8.3 Features más importantes (permutation importance, ventana +1 día)

| Posición | Feature | Importancia |
|---|---|---|
| 1 | `sent_finbert_pos` | +0.127 |
| 2 | `sent_finbert_neg` | +0.020 |
| 3 | `sent_finbert_pos_roll7` | +0.014 |
| 4 | `sent_finbert_pos_delta` | +0.012 |
| 5 | `sent_finbert_neg_roll7` | +0.012 |

### 8.4 Limitaciones

1. **Techo de accuracy (~65%):** sin datos externos (precio, volumen, noticias) la señal de Reddit tiene autocorrelación baja (lag=1: 0.10) que limita la capacidad predictiva.
2. **LSTM con datos limitados:** con 943 secuencias de entrenamiento el LSTM no converge a soluciones óptimas; necesitaría al menos 3.000-5.000 secuencias.
3. **Imágenes no analizadas:** 5.694 posts tienen imagen pero no se procesaron con el modelo de visión, perdiendo potencial señal visual.
4. **Régimen de mercado:** el modelo entrenado en 2021-2024 puede no generalizar bien a regímenes de mercado muy diferentes (crisis, burbujas).

---

## 9. Referencias en formato APA 7

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623–2631. https://doi.org/10.1145/3292500.3330701

Araci, D. (2019). *FinBERT: Financial sentiment analysis with pre-trained language models* [Preprint]. arXiv. https://arxiv.org/abs/1908.10063

Breiman, L. (2001). Random forests. *Machine Learning*, *45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186. https://doi.org/10.18653/v1/N19-1423

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, *30*, 3146–3154.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR 2019)*. https://arxiv.org/abs/1711.05101

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesneau, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

Pushshift.io / Arctic Shift. (2023). *Reddit data archive* [Dataset]. https://arctic-shift.photon-reddit.com

Reddit Inc. (2024). *Reddit JSON API* [API]. https://www.reddit.com/dev/api

Sarker, A. (2021). *SocBERT: A pretrained language model for social media text* [Preprint]. arXiv. https://arxiv.org/abs/2108.13898

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, *30*, 5998–6008.

Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., … Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6
