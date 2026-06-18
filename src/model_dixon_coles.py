"""Modelo Dixon-Coles para goles de fútbol.

Es el caballo de batalla del análisis de fútbol: un Poisson bivariado con dos
ajustes sobre el Poisson independiente clásico (Dixon & Coles, 1997):

  1. Corrección tau para marcadores bajos (0-0, 1-0, 0-1, 1-1), donde el
     Poisson independiente predice mal por la dependencia entre goles.
  2. Ponderación por decaimiento temporal: los partidos recientes pesan más.

Cada equipo tiene una fuerza de ATAQUE (alpha) y de DEFENSA (delta). Para un
partido local i vs visitante j:

    goles_local    ~ Poisson(exp(alpha_i + delta_j + gamma))   # gamma = ventaja local
    goles_visitante~ Poisson(exp(alpha_j + delta_i))

Se estiman por máxima verosimilitud. De los parámetros se deriva la matriz de
marcadores y, de ahí, CUALQUIER probabilidad: 1X2, over/under, BTTS, exacto.
Ese es el enfoque riguroso: modelar la distribución y derivar los mercados,
en vez de entrenar un clasificador por cada umbral.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


def _tau_vector(x, y, lh, la, rho):
    """Corrección Dixon-Coles para marcadores bajos (vectorizada)."""
    t = np.ones_like(lh, dtype=float)
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    t[m00] = 1.0 - lh[m00] * la[m00] * rho
    t[m01] = 1.0 + lh[m01] * rho
    t[m10] = 1.0 + la[m10] * rho
    t[m11] = 1.0 - rho
    return t


@dataclass
class DixonColesModel:
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    home_adv: float
    rho: float
    xi: float = 0.0
    n_matches: int = 0
    fit_meta: dict = field(default_factory=dict)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """xG implícito (lambda) para local y visitante."""
        lh = np.exp(self.attack[home] + self.defense[away] + self.home_adv)
        la = np.exp(self.attack[away] + self.defense[home])
        return float(lh), float(la)

    def score_matrix(self, home: str, away: str, max_goals: int = 10) -> np.ndarray:
        """Matriz P(goles_local=x, goles_visitante=y), ya normalizada."""
        lh, la = self.expected_goals(home, away)
        gh = np.arange(max_goals + 1)
        # Poisson pmf para cada marginal
        ph = np.exp(gh * np.log(lh) - lh - gammaln(gh + 1))
        pa = np.exp(gh * np.log(la) - la - gammaln(gh + 1))
        mat = np.outer(ph, pa)
        # Corrección tau en las cuatro celdas bajas
        mat[0, 0] *= 1.0 - lh * la * self.rho
        mat[0, 1] *= 1.0 + lh * self.rho
        mat[1, 0] *= 1.0 + la * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.clip(mat, 0, None)
        return mat / mat.sum()

    def predict(self, home: str, away: str, max_goals: int = 10) -> dict:
        """Probabilidades de los principales mercados para un partido."""
        if home not in self.attack or away not in self.attack:
            raise KeyError(f"Equipo no presente en el ajuste: {home!r} o {away!r}")
        mat = self.score_matrix(home, away, max_goals)
        idx = np.arange(max_goals + 1)
        home_grid, away_grid = np.meshgrid(idx, idx, indexing="ij")
        total = home_grid + away_grid

        lh, la = self.expected_goals(home, away)
        out = {
            "home": home,
            "away": away,
            "xg_home": round(lh, 3),
            "xg_away": round(la, 3),
            "p_home": float(mat[home_grid > away_grid].sum()),
            "p_draw": float(mat[home_grid == away_grid].sum()),
            "p_away": float(mat[home_grid < away_grid].sum()),
            "p_btts": float(mat[(home_grid >= 1) & (away_grid >= 1)].sum()),
        }
        for line in (1.5, 2.5, 3.5):
            out[f"p_over_{line}"] = float(mat[total > line].sum())
            out[f"p_under_{line}"] = float(mat[total < line].sum())
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def fit(matches: pd.DataFrame, xi: float = 0.0, ref_date: str | None = None) -> DixonColesModel:
    """Ajusta el modelo por máxima verosimilitud sobre una tabla de partidos.

    matches: DataFrame con columnas home_team, away_team, home_score,
             away_score y match_date.
    xi:      tasa de decaimiento temporal por día (0 = sin decaimiento).
             Valores tipicos 0.001-0.005; mayor = los partidos viejos pesan menos.
    ref_date: fecha de referencia para el decaimiento (por defecto, el último
              partido). Los pesos son exp(-xi * dias_hasta_ref).
    """
    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)
    t_index = {t: i for i, t in enumerate(teams)}

    hi = df["home_team"].map(t_index).to_numpy()
    ai = df["away_team"].map(t_index).to_numpy()
    x = df["home_score"].to_numpy(dtype=int)
    y = df["away_score"].to_numpy(dtype=int)

    ref = pd.to_datetime(ref_date) if ref_date else df["match_date"].max()
    days = (ref - df["match_date"]).dt.days.to_numpy().astype(float)
    weights = np.exp(-xi * np.clip(days, 0, None))

    lx = gammaln(x + 1)
    ly = gammaln(y + 1)

    def neg_log_lik(params):
        alpha = params[:n]
        delta = params[n:2 * n]
        gamma = params[2 * n]
        rho = params[2 * n + 1]
        lh = np.exp(alpha[hi] + delta[ai] + gamma)
        la = np.exp(alpha[ai] + delta[hi])
        base = x * np.log(lh) - lh - lx + y * np.log(la) - la - ly
        tau = _tau_vector(x, y, lh, la, rho)
        tau = np.clip(tau, 1e-10, None)
        return -np.sum(weights * (base + np.log(tau)))

    p0 = np.zeros(2 * n + 2)
    p0[2 * n] = 0.25   # ventaja local inicial (~exp(0.25)=1.28)
    p0[2 * n + 1] = -0.05  # rho inicial
    # Identificabilidad: la suma de los ataques debe ser 0
    cons = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]
    bounds = [(None, None)] * (2 * n) + [(None, None), (-0.2, 0.2)]

    res = minimize(neg_log_lik, p0, method="SLSQP", constraints=cons,
                   bounds=bounds, options={"maxiter": 300, "ftol": 1e-7})

    alpha = res.x[:n]
    delta = res.x[n:2 * n]
    return DixonColesModel(
        teams=teams,
        attack={t: float(alpha[i]) for t, i in t_index.items()},
        defense={t: float(delta[i]) for t, i in t_index.items()},
        home_adv=float(res.x[2 * n]),
        rho=float(res.x[2 * n + 1]),
        xi=xi,
        n_matches=len(df),
        fit_meta={"converged": bool(res.success), "neg_log_lik": float(res.fun)},
    )


def team_ratings(model: DixonColesModel) -> pd.DataFrame:
    """Tabla ordenada de fuerzas de ataque y defensa (interpretable)."""
    rows = [{"team": t,
             "attack": round(np.exp(model.attack[t]), 3),
             "defense": round(np.exp(model.defense[t]), 3)} for t in model.teams]
    df = pd.DataFrame(rows)
    # attack alto = marca mas; defense bajo = encaja menos
    return df.sort_values("attack", ascending=False).reset_index(drop=True)
