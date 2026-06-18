"""Capa de visualización (Streamlit).

Lee las tablas que deja `run.py` en DuckDB y las presenta en cuatro vistas:
calendario del Mundial, predicción de partido, histórico vs predicción y
ratings de equipo. Incluye un botón "Actualizar datos" que ejecuta run.py por
detrás, para no tener que tocar la terminal una vez montado.

Arranque:  streamlit run app.py
"""
import subprocess
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import storage
import model_dixon_coles as dc

ACCENT = "#1d9e75"  # verde campo, único acento del tema

st.set_page_config(page_title="futbol-predict", page_icon="⚽", layout="wide")


# ---------------------------------------------------------------- helpers puros
def pick_league_table(tables: list[str]) -> str | None:
    """Elige la tabla de partidos de liga más grande (la usada para el modelo)."""
    candidates = [t for t in tables if t.startswith("sb_matches_")]
    if not candidates:
        return None
    sizes = {t: len(storage.query(f'SELECT 1 FROM "{t}"')) for t in candidates}
    return max(sizes, key=sizes.get)


def compute_metrics(preds: pd.DataFrame) -> dict:
    """Métricas de calibración sobre la tabla de predicciones del backtest."""
    classes = ["H", "D", "A"]
    cidx = {c: i for i, c in enumerate(classes)}
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    y = preds["actual_outcome"].map(cidx).to_numpy()
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    pick = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return {
        "Partidos evaluados": int(len(preds)),
        "Log loss (1X2)": round(float(-np.mean(np.log(pick))), 3),
        "Brier (1X2)": round(float(np.mean(np.sum((probs - onehot) ** 2, axis=1))), 3),
        "Acierto (1X2)": f"{(probs.argmax(1) == y).mean():.1%}",
    }


def reliability_table(preds: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Curva de fiabilidad para P(victoria local): predicho vs observado."""
    df = preds.copy()
    df["bin"] = pd.cut(df["p_home"], np.linspace(0, 1, bins + 1), include_lowest=True)
    grp = df.groupby("bin", observed=True)
    out = pd.DataFrame({
        "prob_predicha": grp["p_home"].mean(),
        "frec_observada": grp["actual_outcome"].apply(lambda s: (s == "H").mean()),
        "n": grp.size(),
    }).dropna().reset_index(drop=True)
    return out


# --------------------------------------------------------------- carga de datos
@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    return storage.query(f'SELECT * FROM "{name}"')


@st.cache_resource(show_spinner=True)
def get_model(league_table: str) -> dc.DixonColesModel:
    matches = storage.query(f'SELECT * FROM "{league_table}"')
    return dc.fit(matches, xi=0.0)


def available_tables() -> list[str]:
    return storage.list_tables()


# ------------------------------------------------------------------------ barra
st.sidebar.title("⚽ futbol-predict")
tables = available_tables()

if st.sidebar.button("Actualizar datos", width='stretch', type="primary"):
    with st.spinner("Descargando datos y reentrenando el modelo..."):
        res = subprocess.run([sys.executable, str(ROOT / "run.py")],
                             capture_output=True, text=True)
    if res.returncode == 0:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("Datos actualizados")
        st.rerun()
    else:
        st.sidebar.error("Falló la actualización. Revisa la conexión.")
        st.sidebar.code(res.stderr[-800:] or res.stdout[-800:])

st.sidebar.caption("Tablas disponibles:")
st.sidebar.write(", ".join(tables) if tables else "ninguna todavía")

if not tables:
    st.title("No hay datos todavía")
    st.info("Pulsa **Actualizar datos** en la barra lateral para descargar los "
            "datos y entrenar el modelo por primera vez.")
    st.stop()

league_table = pick_league_table(tables)

# ------------------------------------------------------------------------ vistas
st.title("Análisis predictivo de fútbol")
tab_cal, tab_pred, tab_hist, tab_rat = st.tabs(
    ["Calendario", "Predicción de partido", "Histórico vs predicción", "Ratings"])

# --- Calendario ---------------------------------------------------------------
with tab_cal:
    st.subheader("Calendario de partidos")
    if "fd_matches_WC" in tables:
        cal = load_table("fd_matches_WC")
        cols = [c for c in ["utcDate", "stage", "group", "status",
                            "homeTeam.name", "awayTeam.name",
                            "score.fullTime.home", "score.fullTime.away"]
                if c in cal.columns]
        view = cal[cols].copy() if cols else cal
        stages = sorted(s for s in view.get("stage", pd.Series()).dropna().unique())
        if stages:
            chosen = st.multiselect("Fase", stages, default=stages)
            view = view[view["stage"].isin(chosen)]
        st.dataframe(view, width='stretch', hide_index=True)
    else:
        st.warning("Aún no hay calendario del Mundial desde football-data.org "
                   "(requiere conexión al actualizar). Mostrando el Mundial 2022 "
                   "de StatsBomb como referencia.")
        wc = [t for t in tables if t.startswith("sb_matches_43_")]
        if wc:
            m = load_table(wc[0])
            cols = [c for c in ["match_date", "home_team", "away_team",
                                "home_score", "away_score", "competition_stage"]
                    if c in m.columns]
            st.dataframe(m[cols].sort_values("match_date"),
                         width='stretch', hide_index=True)

# --- Predicción de partido ----------------------------------------------------
with tab_pred:
    st.subheader("Probabilidades por partido")
    if not league_table:
        st.info("No hay liga cargada para el modelo.")
    else:
        model = get_model(league_table)
        st.caption(f"Modelo Dixon-Coles ajustado sobre {model.n_matches} partidos. "
                   f"Ventaja local exp(γ)={np.exp(model.home_adv):.2f}.")
        c1, c2 = st.columns(2)
        home = c1.selectbox("Local", model.teams, index=0)
        away = c2.selectbox("Visitante", model.teams, index=1)
        if home == away:
            st.info("Elige dos equipos distintos.")
        else:
            p = model.predict(home, away)
            m1, mX, m2 = st.columns(3)
            m1.metric(f"Gana {home}", f"{p['p_home']:.0%}")
            mX.metric("Empate", f"{p['p_draw']:.0%}")
            m2.metric(f"Gana {away}", f"{p['p_away']:.0%}")
            o1, o2, o3 = st.columns(3)
            o1.metric("Over 2.5 goles", f"{p['p_over_2.5']:.0%}")
            o2.metric("Ambos marcan (BTTS)", f"{p['p_btts']:.0%}")
            o3.metric("xG estimado", f"{p['xg_home']:.2f} – {p['xg_away']:.2f}")

            # Elemento característico: matriz de marcadores del Poisson bivariado
            st.markdown("**Distribución de marcadores** (probabilidad de cada resultado)")
            mat = model.score_matrix(home, away, max_goals=5)
            grid = pd.DataFrame(
                [(int(i), int(j), float(mat[i, j]))
                 for i in range(mat.shape[0]) for j in range(6)],
                columns=["local", "visitante", "prob"])
            heat = (alt.Chart(grid).mark_rect().encode(
                        x=alt.X("visitante:O", title=f"Goles {away}"),
                        y=alt.Y("local:O", title=f"Goles {home}", sort="descending"),
                        color=alt.Color("prob:Q", scale=alt.Scale(scheme="greens"),
                                        legend=None),
                        tooltip=[alt.Tooltip("prob:Q", format=".1%")])
                    .properties(height=260))
            st.altair_chart(heat, width='stretch')

# --- Histórico vs predicción --------------------------------------------------
with tab_hist:
    st.subheader("Histórico vs predicción (backtest sin fuga de información)")
    if "predicciones" not in tables:
        st.info("Aún no hay predicciones. Pulsa Actualizar datos.")
    else:
        preds = load_table("predicciones")
        met = compute_metrics(preds)
        cols = st.columns(len(met))
        for col, (k, v) in zip(cols, met.items()):
            col.metric(k, v)

        st.markdown("**Curva de fiabilidad** — si el modelo está bien calibrado, "
                    "los puntos caen sobre la diagonal.")
        rel = reliability_table(preds)
        diag = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        base = alt.Chart(rel).mark_circle(size=90, color=ACCENT).encode(
            x=alt.X("prob_predicha:Q", title="Probabilidad predicha (victoria local)",
                    scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("frec_observada:Q", title="Frecuencia observada",
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("prob_predicha:Q", format=".2f"),
                     alt.Tooltip("frec_observada:Q", format=".2f"),
                     alt.Tooltip("n:Q", title="partidos")])
        line = alt.Chart(diag).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
        st.altair_chart((line + base).properties(height=320), width='stretch')

        st.markdown("**Detalle por partido**")
        show = preds[["match_date", "home_team", "away_team", "p_home", "p_draw",
                      "p_away", "actual_home", "actual_away", "actual_outcome"]]
        st.dataframe(show, width='stretch', hide_index=True)

# --- Ratings ------------------------------------------------------------------
with tab_rat:
    st.subheader("Fuerza de los equipos")
    if "ratings" not in tables:
        st.info("Aún no hay ratings. Pulsa Actualizar datos.")
    else:
        ratings = load_table("ratings")
        st.caption("Ataque alto = marca más. Defensa baja = encaja menos.")
        chart = (alt.Chart(ratings).mark_bar(color=ACCENT).encode(
                    x=alt.X("attack:Q", title="Fuerza de ataque"),
                    y=alt.Y("team:N", sort="-x", title=None),
                    tooltip=["team", "attack", "defense"])
                 .properties(height=26 * len(ratings)))
        st.altair_chart(chart, width='stretch')
        st.dataframe(ratings, width='stretch', hide_index=True)
