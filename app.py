"""Capa de visualización (Streamlit) — calendario multi-liga + predicción.

Lee las tablas que deja el robot de GitHub Actions (o run.py en local):
  - fixtures: partidos de la temporada en curso de varias competiciones.
  - predicciones_upcoming: probabilidades de los próximos partidos.
  - ratings_by_comp / coverage / predicciones (validación histórica).

Vista principal: eliges un día y las ligas que quieras, y ves los partidos de
ese día con sus probabilidades del modelo.
"""
import datetime as dt
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

ACCENT = "#1d9e75"
st.set_page_config(page_title="futbol-predict", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    return storage.query(f'SELECT * FROM "{name}"')


def has_table(name: str) -> bool:
    return name in storage.list_tables()


@st.cache_resource(show_spinner=True)
def model_for(code: str) -> dc.DixonColesModel | None:
    """Ajusta (cacheado) el Dixon-Coles de una competición con sus FINISHED."""
    fx = load_table("fixtures")
    fin = fx[(fx["code"] == code) & (fx["status"] == "FINISHED")].dropna(
        subset=["home_score", "away_score"]).copy()
    if len(fin) < 40:
        return None
    fin["home_score"] = fin["home_score"].astype(int)
    fin["away_score"] = fin["away_score"].astype(int)
    return dc.fit(fin, xi=0.003)


def calendar_rows(fixtures: pd.DataFrame, preds: pd.DataFrame,
                  day: str, comps: list[str]) -> pd.DataFrame:
    sel = fixtures[(fixtures["match_date"] == day) & (fixtures["competition"].isin(comps))]
    out = []
    for _, m in sel.iterrows():
        row = {"Competición": m["competition"], "Local": m["home_team"],
               "Visitante": m["away_team"], "1": "—", "X": "—", "2": "—",
               "Over 2.5": "—", "Resultado": "—"}
        if m["status"] == "FINISHED" and pd.notna(m["home_score"]):
            row["Resultado"] = f"{int(m['home_score'])}–{int(m['away_score'])}"
        elif not preds.empty:
            p = preds[(preds["code"] == m["code"]) & (preds["home_team"] == m["home_team"])
                      & (preds["away_team"] == m["away_team"]) & (preds["match_date"] == day)]
            if len(p):
                p = p.iloc[0]
                row["1"] = f"{p['p_home']:.0%}"
                row["X"] = f"{p['p_draw']:.0%}"
                row["2"] = f"{p['p_away']:.0%}"
                row["Over 2.5"] = f"{p['p_over_2.5']:.0%}"
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ barra
st.sidebar.title("⚽ futbol-predict")
if st.sidebar.button("Actualizar datos (local)", width="stretch"):
    with st.spinner("Descargando resultados y reentrenando..."):
        res = subprocess.run([sys.executable, str(ROOT / "run.py")],
                             capture_output=True, text=True)
    st.cache_data.clear(); st.cache_resource.clear()
    (st.sidebar.success if res.returncode == 0 else st.sidebar.error)(
        "Listo" if res.returncode == 0 else "Falló (¿clave/conexión?)")
    st.rerun()
st.sidebar.caption("En la nube, los datos se refrescan solos cada día.")

st.title("Análisis predictivo de fútbol")
tabs = st.tabs(["Calendario", "Predicción de partido", "Validación del modelo", "Ratings"])

# --- Calendario ---------------------------------------------------------
with tabs[0]:
    if not has_table("fixtures"):
        st.info("Aún no hay partidos cargados. En la nube se llenará con el primer "
                "refresco automático; en local, pulsa **Actualizar datos** "
                "(necesita la clave de football-data.org).")
    else:
        fixtures = load_table("fixtures")
        preds = load_table("predicciones_upcoming") if has_table("predicciones_upcoming") else pd.DataFrame()
        comps_all = sorted(fixtures["competition"].unique())
        c1, c2 = st.columns([1, 2])
        day = c1.date_input("Día", value=dt.date.today()).isoformat()
        comps = c2.multiselect("Competiciones", comps_all, default=comps_all)
        rows = calendar_rows(fixtures, preds, day, comps)
        if rows.empty:
            st.caption("No hay partidos ese día en las competiciones elegidas.")
        else:
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption("1 / X / 2 = probabilidad de victoria local, empate y "
                       "visitante según el modelo. Resultado = partido ya jugado.")

# --- Predicción de partido ----------------------------------------------
with tabs[1]:
    if not has_table("coverage"):
        st.info("Sin modelos todavía. Actualiza los datos primero.")
    else:
        cov = load_table("coverage")
        active = cov[cov["n_finished"] >= 40]
        if active.empty:
            st.info("Ninguna competición tiene aún suficientes partidos para un modelo fiable.")
        else:
            comp = st.selectbox("Competición", active["competition"].tolist())
            code = active[active["competition"] == comp]["code"].iloc[0]
            model = model_for(code)
            if model is None:
                st.info("Modelo no disponible para esa competición.")
            else:
                col1, col2 = st.columns(2)
                home = col1.selectbox("Local", model.teams, index=0)
                away = col2.selectbox("Visitante", model.teams, index=1)
                if home != away:
                    p = model.predict(home, away)
                    a, b, c = st.columns(3)
                    a.metric(f"Gana {home}", f"{p['p_home']:.0%}")
                    b.metric("Empate", f"{p['p_draw']:.0%}")
                    c.metric(f"Gana {away}", f"{p['p_away']:.0%}")
                    d, e, f = st.columns(3)
                    d.metric("Over 2.5", f"{p['p_over_2.5']:.0%}")
                    e.metric("Ambos marcan", f"{p['p_btts']:.0%}")
                    f.metric("xG estimado", f"{p['xg_home']:.2f} – {p['xg_away']:.2f}")
                    st.markdown("**Distribución de marcadores**")
                    mat = model.score_matrix(home, away, max_goals=5)
                    grid = pd.DataFrame([(i, j, float(mat[i, j])) for i in range(6)
                                         for j in range(6)], columns=["local", "visitante", "prob"])
                    st.altair_chart(alt.Chart(grid).mark_rect().encode(
                        x=alt.X("visitante:O", title=f"Goles {away}"),
                        y=alt.Y("local:O", title=f"Goles {home}", sort="descending"),
                        color=alt.Color("prob:Q", scale=alt.Scale(scheme="greens"), legend=None),
                        tooltip=[alt.Tooltip("prob:Q", format=".1%")]).properties(height=260),
                        width="stretch")

# --- Validación del modelo ----------------------------------------------
with tabs[2]:
    st.subheader("¿Funciona el modelo? Backtest sin fuga de información")
    if not has_table("predicciones"):
        st.info("Sin validación todavía.")
    else:
        preds = load_table("predicciones")
        classes = ["H", "D", "A"]; cidx = {c: i for i, c in enumerate(classes)}
        probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
        y = preds["actual_outcome"].map(cidx).to_numpy()
        pick = np.clip(probs[np.arange(len(y)), y], 1e-12, 1)
        a, b, c = st.columns(3)
        a.metric("Partidos evaluados", len(preds))
        b.metric("Log loss (1X2)", f"{-np.mean(np.log(pick)):.3f}")
        c.metric("Acierto (1X2)", f"{(probs.argmax(1) == y).mean():.1%}")
        st.caption("Validado sobre la Premier 2015/16. Demuestra que el motor que "
                   "alimenta el calendario está calibrado y bate a un baseline ingenuo.")
        st.dataframe(preds[["match_date", "home_team", "away_team", "p_home",
                            "p_draw", "p_away", "actual_outcome"]],
                     width="stretch", hide_index=True)

# --- Ratings ------------------------------------------------------------
with tabs[3]:
    if not has_table("ratings_by_comp"):
        st.info("Sin ratings todavía.")
    else:
        ratings = load_table("ratings_by_comp")
        comp = st.selectbox("Competición ", sorted(ratings["competition"].unique()))
        r = ratings[ratings["competition"] == comp]
        st.caption("Ataque alto = marca más. Defensa baja = encaja menos.")
        st.altair_chart(alt.Chart(r).mark_bar(color=ACCENT).encode(
            x=alt.X("attack:Q", title="Fuerza de ataque"),
            y=alt.Y("team:N", sort="-x", title=None),
            tooltip=["team", "attack", "defense"]).properties(height=26 * len(r)),
            width="stretch")
