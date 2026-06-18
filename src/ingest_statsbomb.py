"""Ingesta de StatsBomb open data (gratuito, vía statsbombpy).

Esta es tu mina para ML: datos de eventos de alta granularidad (disparos con
xG, pases, presiones...) de competiciones publicadas, incluidos Mundiales.
No es en vivo: es histórico, justo lo que necesitas para entrenar y validar.

Flujo típico:
    comps   = list_competitions()                     # qué hay disponible
    matches = ingest_matches(competition_id, season_id)
    shots   = ingest_shots(matches["match_id"].tolist())
"""
import pandas as pd
from statsbombpy import sb

from storage import save_table


def list_competitions() -> pd.DataFrame:
    """Catálogo de competición-temporadas disponibles en open data."""
    comps = sb.competitions()
    save_table(comps, "sb_competitions")
    return comps


def ingest_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Partidos de una competición/temporada concretas."""
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    save_table(matches, f"sb_matches_{competition_id}_{season_id}")
    return matches


def ingest_shots(match_ids: list[int]) -> pd.DataFrame:
    """Eventos de disparo (con xG de StatsBomb) de una lista de partidos.

    Devuelve un único DataFrame y lo persiste como tabla 'sb_shots'.
    El xG de cada disparo viene en la columna 'shot_statsbomb_xg'; te servirá
    de baseline y de etiqueta para entrenar tu propio modelo de xG en Fase 2.
    """
    frames = []
    for mid in match_ids:
        events = sb.events(match_id=mid)
        shots = events[events["type"] == "Shot"].copy()
        shots["match_id"] = mid
        frames.append(shots)
    all_shots = pd.concat(frames, ignore_index=True)
    save_table(all_shots, "sb_shots")
    return all_shots
