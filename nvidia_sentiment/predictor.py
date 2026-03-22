"""Predictor: entrenamiento y evaluación de modelos de clasificación para NVDA."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "sent_finbert_pos", "sent_finbert_neg", "sent_finbert_neu",
    "sent_bert_pos", "sent_bert_neg",
    "sent_socbert_pos", "sent_socbert_neg",
    "image_score",  # score de Ollama (0.0 si no hay imagen)
]


# ---------------------------------------------------------------------------
# 11.2 – Etiquetado de movimiento de precio
# ---------------------------------------------------------------------------

def label_price_movement(price_today: float, price_tomorrow: float) -> int:
    """Retorna 1 si price_tomorrow > price_today, 0 en caso contrario."""
    return 1 if price_tomorrow > price_today else 0


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(
    posts: list[dict],
    price_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construye el DataFrame de features y la serie de etiquetas objetivo.

    - Alinea posts con precios por fecha.
    - Extrae FEATURE_COLS de cada post (0.0 si ausente).
    - image_score: post["image_analysis"]["score"] si existe, sino 0.0.
    - Etiqueta: label_price_movement(precio_hoy, precio_mañana).
    - División temporal: ordenar por created_utc antes de retornar.
    """
    if price_df is None or price_df.empty:
        logger.warning("DataFrame de precios vacío; no se pueden construir features.")
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"]).dt.strftime("%Y-%m-%d")
    price_map: dict[str, float] = dict(
        zip(price_df["date"], price_df["close"].astype(float))
    )

    sorted_dates = sorted(price_map.keys())
    next_day: dict[str, str] = {
        d: sorted_dates[i + 1] for i, d in enumerate(sorted_dates[:-1])
    }

    rows: list[dict] = []
    for post in posts:
        raw_date = post.get("date", "")
        post_date = str(raw_date)[:10] if raw_date else ""

        if post_date not in price_map or post_date not in next_day:
            continue

        tomorrow = next_day[post_date]
        label = label_price_movement(price_map[post_date], price_map[tomorrow])

        image_analysis = post.get("image_analysis")
        if isinstance(image_analysis, dict):
            image_score = float(image_analysis.get("score", 0.0))
        elif hasattr(image_analysis, "score"):
            image_score = float(image_analysis.score)
        else:
            image_score = 0.0

        row: dict = {col: float(post.get(col, 0.0)) for col in FEATURE_COLS}
        row["image_score"] = image_score
        row["_created_utc"] = int(post.get("created_utc", 0))
        row["_label"] = label
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS), pd.Series(dtype=int)

    df = pd.DataFrame(rows).sort_values("_created_utc").reset_index(drop=True)
    y = df.pop("_label").astype(int)
    df.pop("_created_utc")

    return df[FEATURE_COLS], y


# ---------------------------------------------------------------------------
# 11.1 – División temporal y entrenamiento
# ---------------------------------------------------------------------------

def temporal_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """División temporal sin shuffle: los últimos test_size% son el conjunto de prueba."""
    n = len(X)
    split_idx = int(n * (1 - test_size))
    X_train = X.iloc[:split_idx].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx:].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def _build_models() -> list[tuple[str, object]]:
    """Construye la lista de modelos a entrenar."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    models: list[tuple[str, object]] = [
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=42)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ]

    try:
        import lightgbm as lgb  # noqa: F401
        models.append(("GradientBoosting", lgb.LGBMClassifier(random_state=42)))
    except ImportError:
        models.append((
            "GradientBoosting",
            GradientBoostingClassifier(random_state=42),
        ))

    models.append(("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)))
    return models


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    """Entrena LR, RF, GBM (LightGBM si disponible, sino GradientBoosting), MLP.

    Retorna lista de ModelMetrics como dicts ordenada por accuracy desc
    (desempate por F1 desc).
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    results: list[dict] = []
    for model_name, clf in _build_models():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results.append({
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            })
        except Exception as exc:
            logger.error("Error entrenando %s: %s", model_name, exc)

    # Ordenar por accuracy desc, desempate por f1 desc (Requisito 8.7)
    results.sort(key=lambda m: (m["accuracy"], m["f1"]), reverse=True)
    return results


# ---------------------------------------------------------------------------
# 11.3 – Exportación de model_comparison.csv y gráfico de barras
# ---------------------------------------------------------------------------

def export_comparison(metrics: list[dict], output_path: Path) -> None:
    """Exporta model_comparison.csv y genera gráfico de barras de accuracy."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(metrics)
    df.to_csv(output_path, index=False)
    logger.info("Tabla comparativa exportada a %s", output_path)

    # Gráfico de barras de accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df["model_name"], df["accuracy"], color="steelblue")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparación de Accuracy por Modelo")
    ax.set_ylim(0, 1)
    for i, val in enumerate(df["accuracy"]):
        ax.text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    chart_path = output_path.parent / "accuracy_comparison.png"
    fig.savefig(chart_path)
    plt.close(fig)
    logger.info("Gráfico de accuracy guardado en %s", chart_path)
