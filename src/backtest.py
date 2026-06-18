"""Backtest walk-forward del modelo Dixon-Coles.

Validación honesta, al estilo de una estrategia cuantitativa: nunca se usa
informacion del futuro. Se entrena con los partidos ANTERIORES a cada fecha y
se predice el partido siguiente; el modelo se reajusta a medida que avanza la
temporada. Se evalua con metricas propias de probabilidades (log loss y Brier),
no con accuracy a secas, y se compara contra un baseline ingenuo (las tasas
base de local/empate/visitante).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import model_dixon_coles as dc


def _outcome(hs: int, as_: int) -> str:
    return "H" if hs > as_ else ("A" if hs < as_ else "D")


def _multiclass_log_loss(probs: np.ndarray, y_idx: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def _multiclass_brier(probs: np.ndarray, y_idx: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def walk_forward(matches: pd.DataFrame, xi: float = 0.0,
                 initial_train_frac: float = 0.5, refit_every: int = 10,
                 max_goals: int = 10) -> dict:
    """Devuelve {'predictions': DataFrame, 'metrics': dict}."""
    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    split = int(len(df) * initial_train_frac)
    classes = ["H", "D", "A"]
    cidx = {c: i for i, c in enumerate(classes)}

    rows = []
    model = None
    last_fit = -10 ** 9
    for i in range(split, len(df)):
        if model is None or (i - last_fit) >= refit_every:
            model = dc.fit(df.iloc[:i], xi=xi, ref_date=str(df.iloc[i]["match_date"].date()))
            last_fit = i
        r = df.iloc[i]
        try:
            p = model.predict(r["home_team"], r["away_team"], max_goals)
        except KeyError:
            continue
        actual = _outcome(r["home_score"], r["away_score"])
        total = int(r["home_score"] + r["away_score"])
        rows.append({
            "match_date": r["match_date"].date(),
            "home_team": r["home_team"], "away_team": r["away_team"],
            "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
            "p_over_2.5": p["p_over_2.5"], "xg_home": p["xg_home"], "xg_away": p["xg_away"],
            "actual_home": int(r["home_score"]), "actual_away": int(r["away_score"]),
            "actual_outcome": actual, "actual_total": total,
            "actual_over_2.5": int(total > 2),
        })

    preds = pd.DataFrame(rows)

    # --- Metricas 1X2 ---
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    y = preds["actual_outcome"].map(cidx).to_numpy()
    pred_class = probs.argmax(axis=1)

    # Baseline ingenuo: tasas base del set de entrenamiento inicial
    train0 = df.iloc[:split]
    base = (train0.apply(lambda r: _outcome(r["home_score"], r["away_score"]), axis=1)
            .value_counts(normalize=True).reindex(classes).fillna(0).to_numpy())
    base_probs = np.tile(base, (len(preds), 1))

    # --- Metricas over/under 2.5 ---
    po = preds["p_over_2.5"].to_numpy()
    yo = preds["actual_over_2.5"].to_numpy()

    metrics = {
        "n_test": int(len(preds)),
        "log_loss_1x2": round(_multiclass_log_loss(probs, y), 4),
        "log_loss_baseline": round(_multiclass_log_loss(base_probs, y), 4),
        "brier_1x2": round(_multiclass_brier(probs, y), 4),
        "brier_baseline": round(_multiclass_brier(base_probs, y), 4),
        "accuracy_1x2": round(float((pred_class == y).mean()), 4),
        "accuracy_baseline": round(float((base.argmax() == y).mean()), 4),
        "over25_log_loss": round(float(-np.mean(
            yo * np.log(np.clip(po, 1e-12, 1)) + (1 - yo) * np.log(np.clip(1 - po, 1e-12, 1)))), 4),
        "over25_brier": round(float(np.mean((po - yo) ** 2)), 4),
        "over25_accuracy": round(float(((po > 0.5).astype(int) == yo).mean()), 4),
    }
    return {"predictions": preds, "metrics": metrics}
