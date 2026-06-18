"""Tubería de datos + modelo. La ejecuta el robot de GitHub Actions cada día
(y puedes ejecutarla en local con: python run.py).

1. Ingesta multi-liga desde football-data.org (temporada en curso) -> 'fixtures'.
   Requiere la clave en la variable de entorno FOOTBALL_DATA_TOKEN.
2. Ajusta un Dixon-Coles por competición sobre los resultados hasta la fecha
   (con decaimiento temporal) y predice los próximos partidos.
3. Valida el motor con un backtest sobre una liga histórica (Premier 2015/16).

Si no hay clave, se salta el paso 1-2 y solo corre la validación histórica.
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
import predict_competitions as pc
import model_dixon_coles as dc
import backtest as bt

VALIDATION_LEAGUE = {"name": "Premier League 2015/16", "competition_id": 2, "season_id": 27}


def live_multiliga() -> None:
    print("== football-data.org: ingesta multi-liga (temporada en curso) ==")
    if not config.FOOTBALL_DATA_TOKEN:
        print("   Omitido: no hay clave en FOOTBALL_DATA_TOKEN.")
        return
    fixtures = fdi.ingest_all()
    if fixtures.empty:
        print("   Sin datos (revisa la clave/conexión).")
        return
    print(f"   {len(fixtures)} partidos en total")
    print("\n== Ajuste por competición y predicción de próximos partidos ==")
    res = pc.fit_and_predict(fixtures)
    cov = res["coverage"].sort_values("n_finished", ascending=False)
    for _, c in cov.iterrows():
        estado = "modelo activo" if c["n_finished"] >= 40 else "muestra insuficiente"
        print(f"   {c['competition']:<18} {c['n_finished']:>4} jugados  ({estado})")
    print(f"\n   {len(res['predictions'])} próximos partidos con predicción")


def validacion_historica() -> None:
    print("\n== Validación del motor (backtest sobre Premier 2015/16) ==")
    matches = sbi.ingest_matches(VALIDATION_LEAGUE["competition_id"],
                                 VALIDATION_LEAGUE["season_id"])
    res = bt.walk_forward(matches, xi=0.0, initial_train_frac=0.5, refit_every=10)
    storage.save_table(res["predictions"], "predicciones")
    m = res["metrics"]
    print(f"   {m['n_test']} partidos | log loss {m['log_loss_1x2']} "
          f"(baseline {m['log_loss_baseline']}) | acierto {m['accuracy_1x2']}")


def main() -> None:
    live_multiliga()
    validacion_historica()
    print("\nTablas en DuckDB:", ", ".join(storage.list_tables()))
    print("Listo.")


if __name__ == "__main__":
    main()
