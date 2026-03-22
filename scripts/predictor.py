#!/usr/bin/env python3
"""CLI del Predictor: entrena y evalúa modelos de clasificación para NVDA.

Uso:
    python predictor.py \
        --input nvda_analyzed_full.json \
        --price_data nvda_top3_backfill.csv \
        [--output model_comparison.csv] \
        [--test_size 0.2]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_posts(input_path: Path) -> list[dict]:
    if not input_path.exists():
        logger.error("Archivo de entrada no encontrado: %s", input_path)
        sys.exit(1)
    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_prices(price_path: Path) -> pd.DataFrame:
    if not price_path.exists():
        logger.error("Archivo de precios no encontrado: %s", price_path)
        sys.exit(1)
    df = pd.read_csv(price_path)
    if "date" not in df.columns or "close" not in df.columns:
        logger.error("El CSV de precios debe tener columnas 'date' y 'close'.")
        sys.exit(1)
    return df


def _print_metrics_table(metrics: list[dict]) -> None:
    header = f"{'Modelo':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print("\n" + header)
    print("-" * len(header))
    for m in metrics:
        print(
            f"{m['model_name']:<22} "
            f"{m['accuracy']:>9.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['f1']:>8.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa modelos de clasificación para predicción de movimiento de NVDA."
    )
    parser.add_argument("--input", required=True, help="JSON de posts analizados")
    parser.add_argument("--price_data", required=True, help="CSV de precios (columnas: date, close)")
    parser.add_argument("--output", default=None, help="Ruta de salida para model_comparison.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fracción del dataset para prueba (default: 0.2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    price_path = Path(args.price_data)

    # Determinar ruta de salida
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "model_comparison.csv"

    # Importar módulo del predictor
    from nvidia_sentiment.predictor import build_features, temporal_split, train_and_evaluate, export_comparison

    logger.info("Cargando posts desde %s ...", input_path)
    posts = _load_posts(input_path)
    logger.info("Posts cargados: %d", len(posts))

    logger.info("Cargando precios desde %s ...", price_path)
    price_df = _load_prices(price_path)

    logger.info("Construyendo features ...")
    X, y = build_features(posts, price_df)

    if X.empty:
        logger.error("No se pudieron construir features. Verifica que los posts tengan fechas alineadas con el CSV de precios.")
        sys.exit(1)

    logger.info("Total de muestras con etiqueta: %d", len(y))

    logger.info("Dividiendo dataset temporalmente (test_size=%.2f) ...", args.test_size)
    X_train, y_train, X_test, y_test = temporal_split(X, y, test_size=args.test_size)
    logger.info("Train: %d muestras | Test: %d muestras", len(y_train), len(y_test))

    if len(y_test) == 0:
        logger.error("El conjunto de prueba está vacío. Reduce --test_size o usa más datos.")
        sys.exit(1)

    logger.info("Entrenando y evaluando modelos ...")
    metrics = train_and_evaluate(X_train, y_train, X_test, y_test)

    if not metrics:
        logger.error("No se pudo entrenar ningún modelo.")
        sys.exit(1)

    _print_metrics_table(metrics)

    best = metrics[0]
    print(f"Mejor modelo: {best['model_name']}  (accuracy={best['accuracy']:.4f}, f1={best['f1']:.4f})")

    logger.info("Exportando resultados a %s ...", output_path)
    export_comparison(metrics, output_path)
    logger.info("Listo.")


if __name__ == "__main__":
    main()
