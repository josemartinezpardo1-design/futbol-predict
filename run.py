"""Punto de entrada único. Ejecuta toda la tubería con un solo comando:

    python run.py

Fase 0 (datos):
  - StatsBomb open data: Mundial 2022 (eventos con xG). Gratis, sin clave.
  - football-data.org: calendario y clasificaciones del Mundial (clave en .env).

Fase 1 (modelo de goles):
  - Ajusta un Dixon-Coles sobre una liga completa (Premier League 2015/16) y
    lo valida con un backtest walk-forward honesto (log loss, Brier, acierto).
  - Guarda las predicciones (histórico vs predicción) y los ratings de equipo
    en DuckDB, listos para la futura capa de visualización.

Si una fuente falla (p. ej. sin conexión), el resto continúa igual.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

import config
import storage
import ingest_statsbomb as sbi
import ingest_footballdata as fdi
import model_dixon_coles as dc
import backtest as bt

# Liga completa usada para ajustar/validar el modelo (20 equipos, 380 partidos)
LEAGUE = {"name": "Premier League 2015/16", "competition_id": 2, "season_id": 27}


def ingest_statsbomb_pilot() -> None:
    print("== StatsBomb open data (Mundial 2022, gratis y sin clave) ==")
    comps = sbi.list_competitions()
    wc = comps[comps["competition_name"].eq("FIFA World Cup")].copy()
    wc["season_year"] = wc["season_name"].astype(int)
    row = wc.sort_values("season_year", ascending=False).iloc[0]
    cid, sid = int(row["competition_id"]), int(row["season_id"])
    print(f"   Piloto: {row['competition_name']} {row['season_name']}")
    matches = sbi.ingest_matches(cid, sid)
    print(f"   {len(matches)} partidos ingestados")
    shots = sbi.ingest_shots(matches["match_id"].head(5).tolist())
    print(f"   {len(shots)} disparos (con xG) de los primeros 5 partidos")


def ingest_world_cup_calendar() -> None:
    print("\n== football-data.org (calendario del Mundial, clave desde .env) ==")
    if not config.FOOTBALL_DATA_TOKEN:
        print("   Omitido: no hay clave en .env")
        return
    try:
        matches = fdi.ingest_matches("WC")
        print(f"   {len(matches)} partidos del Mundial en el calendario")
        try:
            standings = fdi.ingest_standings("WC")
            print(f"   {len(standings)} filas de clasificación por grupo")
        except Exception as exc:
            print(f"   Clasificaciones no disponibles todavía ({type(exc).__name__})")
    except Exception as exc:
        print(f"   No se pudo contactar con football-data.org ({type(exc).__name__}).")
        print("   Reintenta con conexión; la clave y el código ya están listos.")


def model_and_backtest() -> None:
    print(f"\n== Modelo Dixon-Coles (ajuste y backtest sobre {LEAGUE['name']}) ==")
    matches = sbi.ingest_matches(LEAGUE["competition_id"], LEAGUE["season_id"])

    # Ajuste sobre toda la temporada (para ratings y predicciones a demanda)
    model = dc.fit(matches, xi=0.0)
    ratings = dc.team_ratings(model)
    storage.save_table(ratings, "ratings")
    print(f"   Ajustado: {model.n_matches} partidos, ventaja local "
          f"={model.home_adv:.3f}, rho={model.rho:.3f}")
    print("   Top 3 ataque:", ", ".join(ratings.head(3)["team"].tolist()))

    # Backtest walk-forward (validación sin fuga de información)
    res = bt.walk_forward(matches, xi=0.0, initial_train_frac=0.5, refit_every=10)
    m = res["metrics"]
    storage.save_table(res["predictions"], "predicciones")
    print(f"\n   Backtest sobre {m['n_test']} partidos (mitad final de la liga):")
    print(f"     log loss 1X2 : {m['log_loss_1x2']}  (baseline {m['log_loss_baseline']})")
    print(f"     Brier 1X2    : {m['brier_1x2']}  (baseline {m['brier_baseline']})")
    print(f"     acierto 1X2  : {m['accuracy_1x2']}  (baseline {m['accuracy_baseline']})")
    print(f"     over 2.5     : log loss {m['over25_log_loss']}, "
          f"Brier {m['over25_brier']}, acierto {m['over25_accuracy']}")

    # Ejemplos de predicción a demanda
    print("\n   Ejemplos de predicción:")
    for home, away in [("Leicester City", "Chelsea"), ("Arsenal", "Tottenham Hotspur")]:
        p = model.predict(home, away)
        print(f"     {home} vs {away}:  "
              f"1={p['p_home']:.0%}  X={p['p_draw']:.0%}  2={p['p_away']:.0%}  | "
              f"over2.5={p['p_over_2.5']:.0%}  BTTS={p['p_btts']:.0%}  | "
              f"xG {p['xg_home']}-{p['xg_away']}")


def main() -> None:
    ingest_statsbomb_pilot()
    ingest_world_cup_calendar()
    model_and_backtest()
    print("\nTablas en DuckDB:", ", ".join(storage.list_tables()))
    print("Listo. Datos y predicciones en data/tables/*.parquet, consultables vía DuckDB.")


if __name__ == "__main__":
    main()
