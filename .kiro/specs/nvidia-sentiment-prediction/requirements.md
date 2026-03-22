# Documento de Requisitos

## Introducción

Este sistema analiza el sentimiento de la comunidad de Reddit en torno a NVIDIA (ticker: NVDA) utilizando datos históricos de posts (2023–2025) almacenados en `final_dataset_clean.json`. El pipeline cubre cuatro grandes áreas:

1. **Filtrado de datos**: selección de posts relevantes para NVIDIA desde el dataset histórico.
2. **Descarga y filtrado de imágenes**: obtención de imágenes de los posts y descarte de las no relevantes (GIFs, imágenes sin valor informativo).
3. **Análisis de sentimiento multimodal**: análisis de texto (título + cuerpo del post) y de imagen (vía Ollama con `llama3.2-vision`) para determinar si la imagen aporta valor al análisis.
4. **Predicción de sentimiento a 1 día vista**: entrenamiento y comparación de múltiples modelos de clasificación para predecir el movimiento de sentimiento del día siguiente.

El sistema incluye un modo de prueba sobre un subconjunto reducido de posts antes de ejecutar el pipeline completo, dado el volumen de datos.

---

## Glosario

- **Sistema**: el pipeline completo de predicción de sentimiento sobre NVIDIA.
- **Filtro_NVDA**: módulo que selecciona posts relevantes para NVIDIA desde el dataset histórico.
- **Descargador_Imágenes**: módulo que descarga imágenes de los posts filtrados.
- **Filtro_Imágenes**: módulo que descarta imágenes no relevantes (GIFs, imágenes sin contenido informativo).
- **Analizador_Texto**: módulo que aplica modelos NLP (FinBERT, BERT general, SocBERT) al título y cuerpo del post.
- **Analizador_Imagen**: módulo que usa Ollama (`llama3.2-vision`) para evaluar el impacto de una imagen en el sentimiento sobre NVDA.
- **Comparador_Multimodal**: módulo que evalúa si incluir la señal de imagen mejora la precisión del análisis respecto al análisis solo de texto.
- **Predictor**: módulo que entrena y evalúa modelos de clasificación para predecir el sentimiento a 1 día vista.
- **Modo_Prueba**: ejecución del pipeline sobre un subconjunto reducido de posts para validar el flujo antes del procesamiento completo.
- **Dataset_Histórico**: archivo `final_dataset_clean.json` con posts de Reddit de 2023 a 2025.
- **Score_Imagen**: valor numérico devuelto por Ollama indicando impacto alcista (>0), bajista (<0) o neutro (=0) de una imagen.
- **Etiqueta_Sentimiento**: clasificación final del sentimiento de un post: `positive`, `negative` o `neutral`.
- **Accuracy**: métrica de precisión global de un modelo de clasificación.

---

## Requisitos

### Requisito 1: Filtrado de posts relevantes para NVIDIA

**User Story:** Como analista, quiero filtrar el dataset histórico para obtener únicamente los posts relacionados con NVIDIA, de modo que el análisis de sentimiento sea específico y no contaminado por ruido.

#### Criterios de Aceptación

1. THE Filtro_NVDA SHALL leer el archivo `final_dataset_clean.json` y producir un subconjunto de posts que contengan referencias explícitas a NVIDIA (términos: "nvidia", "nvda", "$nvda", "geforce", "rtx", "cuda") en el título o en el cuerpo del post, sin distinción de mayúsculas/minúsculas.
2. WHEN el Dataset_Histórico contiene posts sin campo `title` o sin campo `selftext`, THE Filtro_NVDA SHALL tratar esos campos como cadena vacía y aplicar el filtro sobre los campos disponibles.
3. THE Filtro_NVDA SHALL preservar todos los campos originales del post en el subconjunto filtrado.
4. WHEN el subconjunto filtrado contiene 0 posts, THE Filtro_NVDA SHALL registrar un mensaje de advertencia indicando que no se encontraron posts relevantes para NVIDIA.
5. THE Filtro_NVDA SHALL exportar el subconjunto filtrado a un archivo JSON (`nvda_filtered_posts.json`) en la misma carpeta que el dataset de entrada.

---

### Requisito 2: Descarga de imágenes de los posts filtrados

**User Story:** Como analista, quiero descargar las imágenes asociadas a los posts filtrados de NVIDIA, de modo que pueda incluirlas en el análisis multimodal de sentimiento.

#### Criterios de Aceptación

1. WHEN un post filtrado contiene el campo `image_urls` con al menos una URL, THE Descargador_Imágenes SHALL descargar la primera imagen disponible y almacenarla localmente en `RedditScrapper/Data/images/`.
2. THE Descargador_Imágenes SHALL nombrar cada imagen descargada usando el `id` del post como nombre de archivo, preservando la extensión original (`.jpg`, `.jpeg`, `.png`, `.webp`).
3. IF la descarga de una imagen falla por error de red o timeout (>30 segundos), THEN THE Descargador_Imágenes SHALL registrar el error en un log, marcar el campo `image_download_status` del post como `"failed"` y continuar con el siguiente post.
4. THE Descargador_Imágenes SHALL actualizar el campo `image_local_path` de cada post con la ruta relativa al archivo descargado.
5. WHILE el Modo_Prueba está activo, THE Descargador_Imágenes SHALL procesar únicamente los primeros 100 posts filtrados con imagen.

---

### Requisito 3: Filtrado de imágenes no relevantes

**User Story:** Como analista, quiero descartar imágenes que no aporten información de sentimiento (GIFs, memes sin texto, iconos), de modo que el análisis de imagen sea de mayor calidad.

#### Criterios de Aceptación

1. THE Filtro_Imágenes SHALL descartar automáticamente cualquier imagen cuya URL o extensión de archivo sea `.gif`.
2. THE Filtro_Imágenes SHALL usar Ollama (`llama3.2-vision`) para evaluar si una imagen es relevante para el análisis de sentimiento de NVDA, asignando un campo `image_relevance: true/false` a cada post.
3. WHEN el Score_Imagen devuelto por Ollama es `0` y el campo `analisis` indica explícitamente que la imagen no aporta información financiera, THE Filtro_Imágenes SHALL marcar la imagen como `image_relevance: false`.
4. IF Ollama no puede procesar una imagen (error de modelo o imagen corrupta), THEN THE Filtro_Imágenes SHALL marcar `image_relevance: false` y registrar el error.
5. THE Filtro_Imágenes SHALL preservar las imágenes marcadas como `image_relevance: false` en disco pero excluirlas del análisis multimodal posterior.

---

### Requisito 4: Análisis de sentimiento de texto

**User Story:** Como analista, quiero analizar el sentimiento del título y cuerpo de cada post usando múltiples modelos NLP, de modo que pueda comparar su rendimiento y seleccionar el más adecuado.

#### Criterios de Aceptación

1. THE Analizador_Texto SHALL aplicar los modelos FinBERT (`ProsusAI/finbert`), BERT general (`distilbert-base-uncased-finetuned-sst-2-english`) y SocBERT (`sarkerlab/SocBERT-base`) a la concatenación de `title` y `selftext` de cada post.
2. THE Analizador_Texto SHALL truncar el texto de entrada a un máximo de 256 tokens antes de pasarlo a cada modelo.
3. WHEN el texto de un post está vacío tras la limpieza, THE Analizador_Texto SHALL asignar la Etiqueta_Sentimiento `"neutral"` con scores `pos=0.0`, `neg=0.0`, `neu=1.0` para FinBERT, y `"neutral"` con `score=0.5` para BERT general.
4. THE Analizador_Texto SHALL guardar para cada post y cada modelo: la etiqueta predicha (`positive`/`negative`/`neutral`) y las probabilidades individuales de cada clase.
5. WHILE el Modo_Prueba está activo, THE Analizador_Texto SHALL procesar únicamente los primeros 200 posts filtrados.

---

### Requisito 5: Análisis de sentimiento de imagen con Ollama

**User Story:** Como analista, quiero analizar el impacto de las imágenes de los posts usando Ollama, de modo que pueda incorporar la señal visual al análisis de sentimiento.

#### Criterios de Aceptación

1. WHEN un post tiene `image_relevance: true` y `image_local_path` válido, THE Analizador_Imagen SHALL enviar la imagen al modelo `llama3.2-vision` de Ollama con el prompt estándar de análisis de impacto en el precio de NVDA.
2. THE Analizador_Imagen SHALL parsear la respuesta JSON de Ollama extrayendo los campos `score` (numérico) y `analisis` (texto).
3. IF la respuesta de Ollama no contiene un JSON válido, THEN THE Analizador_Imagen SHALL intentar extraer el JSON mediante expresión regular y, si falla, asignar `score: 0.0` y marcar `error: true`.
4. THE Analizador_Imagen SHALL almacenar el resultado del análisis en el campo `image_analysis` de cada post con la estructura `{score, analisis, error}`.
5. WHILE el Modo_Prueba está activo, THE Analizador_Imagen SHALL procesar únicamente los primeros 50 posts con imagen relevante.

---

### Requisito 6: Comparación multimodal (texto vs. texto + imagen)

**User Story:** Como analista, quiero comparar la precisión del análisis de sentimiento usando solo texto frente a texto más imagen, de modo que pueda decidir si la señal visual mejora la predicción.

#### Criterios de Aceptación

1. THE Comparador_Multimodal SHALL calcular una señal de sentimiento fusionada para cada post combinando el score de texto (peso configurable, por defecto 0.75) y el Score_Imagen normalizado (peso por defecto 0.25).
2. THE Comparador_Multimodal SHALL generar dos conjuntos de etiquetas por post: `sent_text_only` (solo texto) y `sent_multimodal` (texto + imagen).
3. WHEN un post no tiene imagen relevante, THE Comparador_Multimodal SHALL usar únicamente la señal de texto para ambos conjuntos de etiquetas.
4. THE Comparador_Multimodal SHALL calcular y reportar el Accuracy de cada conjunto de etiquetas contra una etiqueta de referencia derivada del movimiento real del precio de NVDA al día siguiente.
5. THE Comparador_Multimodal SHALL exportar un informe comparativo en formato CSV con columnas: `post_id`, `date`, `sent_text_only`, `sent_multimodal`, `price_movement_next_day`, `correct_text`, `correct_multimodal`.

---

### Requisito 7: Modo de prueba sobre subconjunto reducido

**User Story:** Como desarrollador, quiero ejecutar el pipeline completo sobre un subconjunto pequeño de posts antes del procesamiento masivo, de modo que pueda validar el flujo sin consumir recursos excesivos.

#### Criterios de Aceptación

1. THE Sistema SHALL aceptar un parámetro `--test_mode` (o `--sample_size N`) que active el Modo_Prueba con un subconjunto de N posts (por defecto N=200).
2. WHEN el Modo_Prueba está activo, THE Sistema SHALL seleccionar los N posts más recientes del subconjunto filtrado de NVIDIA.
3. THE Sistema SHALL ejecutar todas las fases del pipeline (filtrado, descarga, análisis de texto, análisis de imagen, comparación) sobre el subconjunto reducido.
4. WHEN el Modo_Prueba finaliza, THE Sistema SHALL mostrar un resumen con: total de posts procesados, posts con imagen, posts analizados con Ollama, y Accuracy preliminar de cada modelo.
5. IF el Modo_Prueba detecta un error en cualquier fase, THEN THE Sistema SHALL detener la ejecución de esa fase, registrar el error con contexto suficiente para depuración, y continuar con la siguiente fase.

---

### Requisito 8: Entrenamiento y comparación de modelos de predicción

**User Story:** Como analista, quiero entrenar múltiples modelos de clasificación sobre las señales de sentimiento extraídas y comparar su Accuracy, de modo que pueda identificar el modelo más preciso para predecir el sentimiento a 1 día vista.

#### Criterios de Aceptación

1. THE Predictor SHALL usar como features de entrada las probabilidades de sentimiento generadas por FinBERT, BERT general y SocBERT (texto), junto con el Score_Imagen de Ollama cuando esté disponible.
2. THE Predictor SHALL entrenar al menos los siguientes modelos de clasificación: Regresión Logística, Random Forest, Gradient Boosting (XGBoost o LightGBM) y una red neuronal simple (MLP).
3. THE Predictor SHALL usar como etiqueta objetivo el movimiento del precio de cierre de NVDA al día siguiente: `1` si el precio sube respecto al día actual, `0` si baja o se mantiene.
4. THE Predictor SHALL dividir el dataset en conjuntos de entrenamiento (80%) y prueba (20%) respetando el orden temporal (sin shuffle aleatorio) para evitar data leakage.
5. THE Predictor SHALL calcular y reportar para cada modelo: Accuracy, Precision, Recall y F1-score sobre el conjunto de prueba.
6. THE Predictor SHALL exportar una tabla comparativa de métricas en formato CSV (`model_comparison.csv`) y generar una visualización (gráfico de barras) del Accuracy por modelo.
7. WHEN dos modelos tienen el mismo Accuracy, THE Predictor SHALL desempatar usando el F1-score como criterio secundario.

---

### Requisito 9: Serialización y carga del dataset procesado

**User Story:** Como desarrollador, quiero que el dataset procesado pueda guardarse y recargarse de forma consistente, de modo que no sea necesario reprocesar los datos en cada ejecución.

#### Criterios de Aceptación

1. THE Sistema SHALL serializar el dataset procesado (posts filtrados con sentimientos y análisis de imagen) en formato JSON con codificación UTF-8.
2. THE Sistema SHALL deserializar el dataset desde el archivo JSON produciendo un objeto equivalente al original.
3. FOR ALL datasets procesados válidos, serializar y luego deserializar SHALL producir un objeto con los mismos campos y valores que el original (propiedad de round-trip).
4. IF el archivo de dataset procesado no existe o está corrupto al intentar cargarlo, THEN THE Sistema SHALL registrar el error y ofrecer la opción de reprocesar desde el Dataset_Histórico original.
