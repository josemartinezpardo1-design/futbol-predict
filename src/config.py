"""Configuración central del proyecto: rutas de datos y credenciales.

Todo lo sensible (claves) se lee de variables de entorno / fichero .env,
nunca se hardcodea. Copia .env.example a .env y rellena tu clave gratuita.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Raíz del proyecto (este fichero vive en src/, por eso parents[1])
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"        # respuestas crudas (JSON), por si quieres re-procesar
TABLES_DIR = DATA_DIR / "tables"  # tablas tidy en Parquet, consultables por DuckDB
DB_PATH = DATA_DIR / "futbol.duckdb"

for _d in (RAW_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# football-data.org: clave gratuita en https://www.football-data.org/client/register
# El tier gratuito cubre 12 competiciones (incluida la Copa del Mundo, código "WC")
# con un límite de 10 peticiones/minuto.
FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "")
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
