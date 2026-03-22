#!/usr/bin/env python3
"""Predictor de sentimiento NVDA a 1 dia vista sin datos de bolsa.

Mejoras:
  1. Momentum: rolling mean 3d y 7d + delta
  2. Volatilidad del sentimiento (std 7d)
  3. Volumen de posts como feature
  4. Subreddit one-hot (top-5)
  5. Ventana de etiqueta configurable (1 o 3 dias)
  6. LSTM sobre secuencia temporal de dias
  7. Early stopping y StandardScaler

Uso:
    python scripts/sentiment_predictor.py --input data/nvda_processed.json
    python scripts/sentiment_predictor.py --input data/nvda_processed.json --window 3 --seq_len 7
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TOP_SUBREDDITS = ["wallstreetbets", "stocks", "StockMarket", "investing", "nvidia"]

BASE_FEATURES = [
    "sent_finbert_pos", "sent_finbert_neg", "sent_finbert_neu",
    "sent_bert_pos", "sent_bert_neg",
    "sent_socbert_pos", "sent_socbert_neg",
    "score_norm", "num_comments_norm", "has_image",
    "finbert_confidence", "sentiment_agreement", "finbert_bert_diff",
    "n_posts",
]


def _aggregate_day(day_posts: list[dict]) -> dict:
    def avg(key: str) -> float:
        vals = [float(p[key]) for p in day_posts if isinstance(p.get(key), (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    fp  = avg("sent_finbert_pos")
    fn  = avg("sent_finbert_neg")
    fne = avg("sent_finbert_neu")
    bp  = avg("sent_bert_pos")
    sp  = avg("sent_socbert_pos")

    finbert_confidence = max(fp, fn, fne)
    fl = 1 if fp > fn and fp > fne else 0
    bl = 1 if bp > 0.5 else 0
    sl = 1 if sp > 0.5 else 0
    sentiment_agreement = 1.0 if fl == bl == sl else 0.0

    sub_counts: dict[str, int] = defaultdict(int)
    for p in day_posts:
        sub_counts[p.get("subreddit", "")] += 1
    n = len(day_posts)
    sub_feats = {f"sub_{s}": sub_counts.get(s, 0) / n for s in TOP_SUBREDDITS}

    return {
        "sent_finbert_pos": fp,
        "sent_finbert_neg": fn,
        "sent_finbert_neu": fne,
        "sent_bert_pos": bp,
        "sent_bert_neg": avg("sent_bert_neg"),
        "sent_socbert_pos": sp,
        "sent_socbert_neg": avg("sent_socbert_neg"),
        "score_norm": avg("score_norm"),
        "num_comments_norm": avg("num_comments_norm"),
        "has_image": avg("has_image"),
        "finbert_confidence": finbert_confidence,
        "sentiment_agreement": sentiment_agreement,
        "finbert_bert_diff": fp - bp,
        "n_posts": float(n),
        **sub_feats,
    }


def build_daily_df(posts: list[dict]) -> pd.DataFrame:
    scores   = [float(p.get("score", 0) or 0) for p in posts]
    comments = [float(p.get("num_comments", 0) or 0) for p in posts]
    max_score    = max(scores)    if scores    else 1.0
    max_comments = max(comments)  if comments  else 1.0

    for p in posts:
        p["score_norm"]        = float(p.get("score", 0) or 0) / max_score
        p["num_comments_norm"] = float(p.get("num_comments", 0) or 0) / max_comments
        p["has_image"]         = 1.0 if p.get("has_image") else 0.0

    by_date: dict[str, list[dict]] = defaultdict(list)
    for p in posts:
        date = str(p.get("date", ""))[:10]
        if date and len(date) == 10:
            by_date[date].append(p)

    sorted_dates = sorted(by_date.keys())
    logger.info("Dias unicos: %d | Posts totales: %d", len(sorted_dates), len(posts))

    rows = []
    for date in sorted_dates:
        row = _aggregate_day(by_date[date])
        row["date"] = date
        rows.append(row)

    return pd.DataFrame(rows).set_index("date").sort_index()


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sent_finbert_pos", "sent_finbert_neg", "sent_bert_pos", "sent_socbert_pos"]:
        if col not in df.columns:
            continue
        df[f"{col}_roll3"]    = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll7"]    = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_delta"]    = df[col].diff().fillna(0.0)
        df[f"{col}_momentum"] = df[f"{col}_roll3"] - df[f"{col}_roll7"]

    df["finbert_vol7"]   = df["sent_finbert_pos"].rolling(7, min_periods=2).std().fillna(0.0)
    df["n_posts_roll3"]  = df["n_posts"].rolling(3, min_periods=1).mean()
    df["n_posts_delta"]  = df["n_posts"].diff().fillna(0.0)
    return df


def build_labels(df: pd.DataFrame, window: int = 1) -> pd.Series:
    fp = df["sent_finbert_pos"]
    future = fp.shift(-1) if window == 1 else fp.rolling(window).mean().shift(-window)
    return (future > fp).astype(float)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith("_")]


def train_classical(X_tr, y_tr, X_te, y_te, feature_names: list[str]) -> list[dict]:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.dummy import DummyClassifier

    models = [
        ("Baseline",           DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
        ("RandomForest",       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ("MLP",                MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                             random_state=42, early_stopping=True,
                                             validation_fraction=0.1)),
    ]
    try:
        import lightgbm as lgb
        models.append(("GradientBoosting", lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, verbose=-1)))
    except ImportError:
        models.append(("GradientBoosting", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)))

    results = []
    trained_rf = None
    for name, clf in models:
        try:
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            results.append({
                "model":     name,
                "accuracy":  float(accuracy_score(y_te, y_pred)),
                "precision": float(precision_score(y_te, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_te, y_pred, zero_division=0)),
                "type": "classical",
            })
            if name == "RandomForest":
                trained_rf = clf
        except Exception as exc:
            logger.error("Error en %s: %s", name, exc)

    if trained_rf is not None:
        importances = sorted(
            zip(feature_names, trained_rf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        print("\n--- Importancia de features (RandomForest, top 15) ---")
        for feat, imp in importances[:15]:
            bar = "X" * int(imp * 50)
            print(f"  {feat:<32} {imp:.4f}  {bar}")

    return results


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(X_tr_seq, y_tr, X_te_seq, y_te, seq_len: int, n_features: int):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    except ImportError:
        logger.warning("PyTorch no disponible, saltando LSTM.")
        return None

    class SentimentLSTM(nn.Module):
        def __init__(self, input_size: int, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden, 32), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("LSTM en: %s | seq_len=%d | features=%d", device, seq_len, n_features)

    X_tr_t = torch.tensor(X_tr_seq).to(device)
    y_tr_t = torch.tensor(y_tr).to(device)
    X_te_t = torch.tensor(X_te_seq).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=False)

    model = SentimentLSTM(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, wei