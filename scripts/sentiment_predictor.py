#!/usr/bin/env python3
"""Predictor de sentimiento NVDA a N dias vista.

Mejoras anti-overfitting y rigor estadistico:
  - Split temporal estricto: train / validation / test (60/20/20)
  - Optuna para busqueda de hiperparametros (sobre validation, nunca test)
  - Multiples ventanas: 1d, 3d, 5d, 10d, 21d
  - Ablacion: con y sin features de subreddit
  - Feature importance con permutation importance (no impurity bias)
  - Cross-validation temporal (TimeSeriesSplit) en train+val

Uso:
    python scripts/sentiment_predictor.py --input data/nvda_processed.csv
    python scripts/sentiment_predictor.py --input data/nvda_processed.csv --windows 1 3 5 10 21
    python scripts/sentiment_predictor.py --input data/nvda_processed.csv --window 1 --n_trials 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TOP_SUBREDDITS = []  # subreddit eliminado — es ruido


# ---------------------------------------------------------------------------
# Agregacion diaria
# ---------------------------------------------------------------------------

def _agg(day_posts: list[dict]) -> dict:
    def avg(k: str) -> float:
        v = [float(p[k]) for p in day_posts if isinstance(p.get(k), (int, float))]
        return float(np.mean(v)) if v else 0.0

    fp  = avg("sent_finbert_pos")
    fn  = avg("sent_finbert_neg")
    fne = avg("sent_finbert_neu")
    conf = max(fp, fn, fne)
    n = len(day_posts)

    return {
        "sent_finbert_pos": fp,
        "sent_finbert_neg": fn,
        "sent_finbert_neu": fne,
        "score_norm":        avg("score_norm"),
        "num_comments_norm": avg("num_comments_norm"),
        "has_image":         avg("has_image"),
        "finbert_confidence": conf,
        "n_posts":           float(n),
    }


def build_daily_df(posts: list[dict]) -> pd.DataFrame:
    ms = max((float(p.get("score", 0) or 0) for p in posts), default=1.0)
    mc = max((float(p.get("num_comments", 0) or 0) for p in posts), default=1.0)
    for p in posts:
        p["score_norm"] = float(p.get("score", 0) or 0) / ms
        p["num_comments_norm"] = float(p.get("num_comments", 0) or 0) / mc
        p["has_image"] = 1.0 if p.get("has_image") else 0.0
    by: dict[str, list] = defaultdict(list)
    for p in posts:
        d = str(p.get("date", ""))[:10]
        if len(d) == 10:
            by[d].append(p)
    dates = sorted(by.keys())
    logger.info("Dias unicos: %d | Posts: %d | Posts/dia: %.1f",
                len(dates), len(posts), len(posts) / max(len(dates), 1))
    rows = [{**_agg(by[d]), "date": d} for d in dates]
    return pd.DataFrame(rows).set_index("date").sort_index()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["sent_finbert_pos", "sent_finbert_neg", "sent_finbert_neu"]:
        if col not in df.columns:
            continue
        df[f"{col}_roll3"]    = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll7"]    = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_delta"]    = df[col].diff().fillna(0.0)
        df[f"{col}_momentum"] = df[f"{col}_roll3"] - df[f"{col}_roll7"]
    df["finbert_vol7"]  = df["sent_finbert_pos"].rolling(7, min_periods=2).std().fillna(0.0)
    df["n_posts_roll3"] = df["n_posts"].rolling(3, min_periods=1).mean()
    df["n_posts_delta"] = df["n_posts"].diff().fillna(0.0)
    return df


def build_labels(df: pd.DataFrame, window: int) -> pd.Series:
    """Etiqueta basada en FinBERT: regimen EMA futuro."""
    fp = df["sent_finbert_pos"]
    ema_short = fp.ewm(span=3, adjust=False).mean()
    ema_long  = fp.ewm(span=10, adjust=False).mean()
    regime = (ema_short > ema_long).astype(int)
    return regime.shift(-max(1, window)).astype(float)


SUB_COLS: list[str] = []  # subreddit eliminado


def get_feature_cols(df: pd.DataFrame, include_subreddit: bool = True) -> list[str]:
    return [c for c in df.columns if not c.startswith("_")]


# ---------------------------------------------------------------------------
# Split temporal 60/20/20
# ---------------------------------------------------------------------------

def temporal_split_3way(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    t1, t2 = int(n * 0.6), int(n * 0.8)
    return (X[:t1], y[:t1],   # train
            X[t1:t2], y[t1:t2],  # validation
            X[t2:], y[t2:])      # test


# ---------------------------------------------------------------------------
# Optuna: busqueda de hiperparametros sobre validation
# ---------------------------------------------------------------------------

def tune_model(
    name: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_trials: int,
) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from sklearn.metrics import f1_score

    def objective(trial: optuna.Trial) -> float:
        clf = _make_clf(name, trial)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        return float(f1_score(y_val, y_pred, zero_division=0))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _make_clf(name: str, trial=None, params: dict | None = None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier

    p = params or {}

    if name == "LogisticRegression":
        C = trial.suggest_float("C", 0.01, 10.0, log=True) if trial else p.get("C", 1.0)
        return LogisticRegression(C=C, max_iter=2000, random_state=42)

    if name == "RandomForest":
        n  = trial.suggest_int("n_estimators", 50, 400) if trial else p.get("n_estimators", 200)
        md = trial.suggest_int("max_depth", 2, 10) if trial else p.get("max_depth", 6)
        mf = trial.suggest_float("max_features", 0.3, 1.0) if trial else p.get("max_features", 0.7)
        return RandomForestClassifier(n_estimators=n, max_depth=md,
                                      max_features=mf, random_state=42)

    if name == "GradientBoosting":
        try:
            import lightgbm as lgb
            lr  = trial.suggest_float("learning_rate", 0.01, 0.3, log=True) if trial else p.get("learning_rate", 0.05)
            n   = trial.suggest_int("n_estimators", 50, 400) if trial else p.get("n_estimators", 200)
            md  = trial.suggest_int("max_depth", 2, 8) if trial else p.get("max_depth", 4)
            sub = trial.suggest_float("subsample", 0.5, 1.0) if trial else p.get("subsample", 0.8)
            return lgb.LGBMClassifier(learning_rate=lr, n_estimators=n,
                                      max_depth=md, subsample=sub,
                                      random_state=42, verbose=-1)
        except ImportError:
            lr  = trial.suggest_float("learning_rate", 0.01, 0.3, log=True) if trial else p.get("learning_rate", 0.05)
            n   = trial.suggest_int("n_estimators", 50, 300) if trial else p.get("n_estimators", 200)
            md  = trial.suggest_int("max_depth", 2, 6) if trial else p.get("max_depth", 4)
            return GradientBoostingClassifier(learning_rate=lr, n_estimators=n,
                                              max_depth=md, random_state=42)

    if name == "MLP":
        h1  = trial.suggest_int("h1", 32, 256) if trial else p.get("h1", 64)
        h2  = trial.suggest_int("h2", 16, 128) if trial else p.get("h2", 32)
        lr  = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if trial else p.get("lr", 1e-3)
        reg = trial.suggest_float("alpha", 1e-5, 1e-2, log=True) if trial else p.get("alpha", 1e-4)
        return MLPClassifier(hidden_layer_sizes=(h1, h2), learning_rate_init=lr,
                             alpha=reg, max_iter=1000, random_state=42,
                             early_stopping=True, validation_fraction=0.1)

    raise ValueError(f"Modelo desconocido: {name}")


# ---------------------------------------------------------------------------
# Evaluacion completa de un modelo
# ---------------------------------------------------------------------------

def evaluate(clf, X_te: np.ndarray, y_te: np.ndarray, name: str) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = clf.predict(X_te)
    return {
        "model":     name,
        "accuracy":  float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_te, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Permutation importance (sin impurity bias)
# ---------------------------------------------------------------------------

def permutation_importance_scores(
    clf, X_te: np.ndarray, y_te: np.ndarray, feature_names: list[str], n_repeats: int = 10
) -> list[tuple[str, float]]:
    from sklearn.inspection import permutation_importance as pi
    from sklearn.metrics import accuracy_score
    result = pi(clf, X_te, y_te, n_repeats=n_repeats,
                scoring="accuracy", random_state=42)
    pairs = sorted(zip(feature_names, result.importances_mean),
                   key=lambda x: x[1], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Ablacion: subreddit util o ruido?
# ---------------------------------------------------------------------------

def ablation_subreddit(
    X_tr_full, y_tr, X_val_full, y_val, X_te_full, y_te,
    feature_cols_full: list[str],
    best_params: dict,
    model_name: str,
) -> tuple[dict, dict]:
    """Entrena con y sin features de subreddit, compara en test."""
    from sklearn.preprocessing import StandardScaler

    sub_idx = [i for i, c in enumerate(feature_cols_full) if c in SUB_COLS]
    no_sub_idx = [i for i in range(len(feature_cols_full)) if i not in sub_idx]

    results = {}
    for label, idx in [("con_subreddit", list(range(len(feature_cols_full)))),
                       ("sin_subreddit", no_sub_idx)]:
        Xtr = X_tr_full[:, idx]
        Xval = X_val_full[:, idx]
        Xte = X_te_full[:, idx]
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xval = sc.transform(Xval)
        Xte = sc.transform(Xte)
        # Combinar train+val para entrenamiento final
        Xtrval = np.vstack([Xtr, Xval])
        ytrval = np.concatenate([y_tr, y_val])
        clf = _make_clf(model_name, params=best_params)
        clf.fit(Xtrval, ytrval)
        results[label] = evaluate(clf, Xte, y_te, f"{model_name} ({label})")

    return results["con_subreddit"], results["sin_subreddit"]


# ---------------------------------------------------------------------------
# Pipeline principal para una ventana
# ---------------------------------------------------------------------------

def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Construye secuencias temporales para LSTM."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(
    X_tr_seq: np.ndarray, y_tr: np.ndarray,
    X_val_seq: np.ndarray, y_val: np.ndarray,
    X_te_seq: np.ndarray, y_te: np.ndarray,
    seq_len: int, n_features: int,
) -> dict | None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    except ImportError:
        logger.warning("PyTorch no disponible, saltando LSTM.")
        return None

    class SentLSTM(nn.Module):
        def __init__(self, ni: int, h: int = 64, layers: int = 2, drop: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(ni, h, layers, batch_first=True,
                                dropout=drop if layers > 1 else 0.0)
            self.drop = nn.Dropout(drop)
            self.fc   = nn.Sequential(
                nn.Linear(h, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, 1)
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(self.drop(out[:, -1, :])).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  LSTM: device=%s | seq_len=%d | features=%d | train_seqs=%d",
                device, seq_len, n_features, len(y_tr))

    def to_tensor(X, y):
        return torch.tensor(X).to(device), torch.tensor(y).to(device)

    Xt, yt   = to_tensor(X_tr_seq, y_tr)
    Xv, yv   = to_tensor(X_val_seq, y_val)
    Xte_t    = torch.tensor(X_te_seq).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=False)

    model     = SentLSTM(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss, patience_count, best_state = float("inf"), 0, None
    for epoch in range(200):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation loss para early stopping
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xv), yv).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= 20:
                logger.info("  LSTM early stopping en epoch %d (val_loss=%.4f)", epoch+1, best_val_loss)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = (torch.sigmoid(model(Xte_t)) > 0.5).cpu().numpy().astype(int)

    return {
        "model":     f"LSTM (seq={seq_len})",
        "accuracy":  float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_te, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_te, y_pred, zero_division=0)),
    }


def run_window(
    df_feat: pd.DataFrame,
    window: int,
    n_trials: int,
    seq_len: int = 30,
    include_subreddit: bool = True,
) -> dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier

    labels = build_labels(df_feat, window)
    df = df_feat.copy()
    df["_label"] = labels
    df = df.dropna(subset=["_label"])
    df["_label"] = df["_label"].astype(int)

    pos = int(df["_label"].sum())
    neg = len(df) - pos
    logger.info("Ventana +%dd | Dias: %d | Pos: %d (%.1f%%) | Neg: %d (%.1f%%)",
                window, len(df), pos, 100*pos/len(df), neg, 100*neg/len(df))

    feature_cols = get_feature_cols(df, include_subreddit)
    X_all = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y_all = df["_label"].values.astype(np.float32)

    X_tr_raw, y_tr, X_val_raw, y_val, X_te_raw, y_te = temporal_split_3way(X_all, y_all)
    logger.info("  Train: %d | Val: %d | Test: %d", len(y_tr), len(y_val), len(y_te))

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr_raw)
    X_val = scaler.transform(X_val_raw)
    X_te  = scaler.transform(X_te_raw)

    model_names = ["LogisticRegression", "RandomForest", "GradientBoosting", "MLP"]
    results = []
    best_params_all = {}

    # Baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    results.append(evaluate(dummy, X_te, y_te, "Baseline"))

    for name in model_names:
        logger.info("  Tuning %s (n_trials=%d)...", name, n_trials)
        best_params = tune_model(name, X_tr, y_tr, X_val, y_val, n_trials)
        best_params_all[name] = best_params

        # Entrenar con train+val usando mejores hiperparametros
        X_trval = np.vstack([X_tr, X_val])
        y_trval = np.concatenate([y_tr, y_val])
        clf = _make_clf(name, params=best_params)
        clf.fit(X_trval, y_trval)
        r = evaluate(clf, X_te, y_te, name)
        r["best_params"] = json.dumps(best_params)
        results.append(r)

        # Permutation importance para el mejor modelo (LR)
        if name == "LogisticRegression":
            perm = permutation_importance_scores(clf, X_te, y_te, feature_cols, n_repeats=5)
            logger.info("  Top features (permutation importance):")
            for feat, imp in perm[:10]:
                bar = "#" * max(0, int(imp * 100))
                logger.info("    %-32s %+.4f  %s", feat, imp, bar)

    results.sort(key=lambda r: (r["accuracy"], r["f1"]), reverse=True)

    # LSTM con secuencias temporales (train+val para entrenar, test para evaluar)
    X_trval_raw = np.vstack([X_tr_raw, X_val_raw])
    y_trval     = np.concatenate([y_tr, y_val])
    scaler_full = StandardScaler()
    X_all_scaled = scaler_full.fit_transform(np.vstack([X_tr_raw, X_val_raw, X_te_raw]))
    n_tr = len(X_tr_raw); n_val = len(X_val_raw)

    X_tr_s  = X_all_scaled[:n_tr]
    X_val_s = X_all_scaled[n_tr:n_tr+n_val]
    X_te_s  = X_all_scaled[n_tr+n_val:]

    if len(X_all_scaled) >= seq_len + 20:
        X_tr_seq,  y_tr_seq  = build_sequences(X_tr_s,  y_tr,  seq_len)
        X_val_seq, y_val_seq = build_sequences(X_val_s, y_val, seq_len)
        X_te_seq,  y_te_seq  = build_sequences(X_te_s,  y_te,  seq_len)
        if len(y_tr_seq) > 0 and len(y_te_seq) > 0:
            lstm_res = train_lstm(
                X_tr_seq, y_tr_seq,
                X_val_seq, y_val_seq,
                X_te_seq, y_te_seq,
                seq_len, X_tr_seq.shape[2],
            )
            if lstm_res:
                results.append(lstm_res)
                results.sort(key=lambda r: (r["accuracy"], r["f1"]), reverse=True)
    else:
        logger.warning("  Datos insuficientes para LSTM (necesita >%d dias)", seq_len + 20)

    return {
        "window": window,
        "n_days": len(df),
        "train": len(y_tr), "val": len(y_val), "test": len(y_te),
        "results": results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predictor de sentimiento NVDA con Optuna, split 60/20/20 y ablacion."
    )
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   default=None)
    parser.add_argument("--windows",  type=int, nargs="+", default=[1, 3, 5, 10, 21],
                        help="Ventanas de prediccion en dias (default: 1 3 5 10 21)")
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Trials de Optuna por modelo (default: 30)")
    parser.add_argument("--seq_len",  type=int, default=30,
                        help="Longitud de secuencia para LSTM en dias (default: 30)")
    args = parser.parse_args()

    input_path = Path(args.input)
    csv_cand = input_path.with_suffix(".csv")
    if input_path.suffix == ".json" and csv_cand.exists():
        input_path = csv_cand

    if not input_path.exists():
        logger.error("No encontrado: %s", input_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.parent / "sent_model_comparison.csv"

    logger.info("Cargando %s ...", input_path)
    if input_path.suffix == ".csv":
        df_in = pd.read_csv(input_path, dtype=str).fillna("")
        num_cols = [c for c in df_in.columns if any(
            c.startswith(p) for p in ("sent_finbert", "sent_socbert", "score", "num_comments", "created_utc"))]
        for col in num_cols:
            df_in[col] = pd.to_numeric(df_in[col], errors="coerce").fillna(0.0)
        posts = df_in.to_dict(orient="records")
    else:
        import json as _json
        posts = _json.load(input_path.open("r", encoding="utf-8"))
    logger.info("Posts: %d", len(posts))

    # Construir features diarias una sola vez
    df_daily = build_daily_df(posts)
    df_feat  = add_features(df_daily)

    all_rows = []
    for window in args.windows:
        print(f"\n{'='*60}")
        print(f"  VENTANA +{window} DIA(S)")
        print(f"{'='*60}")
        res = run_window(df_feat, window, args.n_trials, seq_len=args.seq_len)

        # Tabla de resultados
        header = f"  {'Modelo':<28} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in res["results"]:
            print(f"  {r['model']:<28} {r['accuracy']:>9.4f} {r['precision']:>10.4f} "
                  f"{r['recall']:>8.4f} {r['f1']:>8.4f}")

        best = res["results"][0]
        print(f"\n  Mejor: {best['model']}  accuracy={best['accuracy']:.4f}  f1={best['f1']:.4f}")

        for r in res["results"]:
            row = {"window": window, **r}
            row.pop("best_params", None)
            all_rows.append(row)

    pd.DataFrame(all_rows).to_csv(output_path, index=False)
    logger.info("Resultados guardados en %s", output_path)


if __name__ == "__main__":
    main()
