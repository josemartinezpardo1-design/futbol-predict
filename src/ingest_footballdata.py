"""Ingesta de football-data.org (tier gratuito).

Cubre calendario, resultados y clasificaciones de 12 competiciones, incluida
la Copa del Mundo (código "WC"). Esta es la fuente de la capa de visualización:
el calendario filtrable de partidos.

Límite del tier gratuito: 10 peticiones/minuto. El cliente espera ~6,5 s entre
llamadas para no superarlo. Necesitas una clave gratuita en .env.
"""
import time

import httpx
import pandas as pd

from config import FOOTBALL_DATA_TOKEN, FOOTBALL_DATA_BASE
from storage import save_table

_MIN_INTERVAL = 6.5  # segundos entre peticiones (~10 req/min)


def _get(path: str, params: dict | None = None) -> dict:
    if not FOOTBALL_DATA_TOKEN:
        raise RuntimeError(
            "Falta FOOTBALL_DATA_TOKEN. Copia .env.example a .env y añade tu "
            "clave gratuita de https://www.football-data.org/client/register"
        )
    headers = {"X-Auth-Token": FOOTBALL_DATA_TOKEN}
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{FOOTBALL_DATA_BASE}{path}", headers=headers, params=params)
        resp.raise_for_status()
        time.sleep(_MIN_INTERVAL)  # respeta el límite de 10 req/min
        return resp.json()


def ingest_competitions() -> pd.DataFrame:
    """Competiciones accesibles con tu clave (para descubrir códigos/IDs)."""
    data = _get("/competitions")
    df = pd.json_normalize(data["competitions"])
    save_table(df, "fd_competitions")
    return df


def ingest_matches(competition: str = "WC") -> pd.DataFrame:
    """Partidos de una competición. 'WC' = Copa del Mundo."""
    data = _get(f"/competitions/{competition}/matches")
    df = pd.json_normalize(data["matches"])
    save_table(df, f"fd_matches_{competition}")
    return df


def ingest_standings(competition: str = "WC") -> pd.DataFrame:
    """Clasificaciones (aplanadas: una fila por equipo, con su grupo)."""
    data = _get(f"/competitions/{competition}/standings")
    rows = []
    for group in data.get("standings", []):
        for entry in group.get("table", []):
            entry["group"] = group.get("group")
            entry["stage"] = group.get("stage")
            rows.append(entry)
    df = pd.json_normalize(rows)
    save_table(df, f"fd_standings_{competition}")
    return df
