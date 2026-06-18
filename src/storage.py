"""Capa de almacenamiento.

Patrón sencillo y sin servidor: cada conjunto de datos se guarda como un
fichero Parquet en data/tables/. DuckDB monta una VISTA por cada Parquet,
así puedes escribir SQL analítico sobre todo el histórico sin cargar nada
en memoria a mano. Es como tener un pandas que escala, con sintaxis SQL.
"""
from pathlib import Path

import duckdb
import pandas as pd

from config import TABLES_DIR


def save_table(df: pd.DataFrame, name: str) -> Path:
    """Guarda un DataFrame como data/tables/{name}.parquet y devuelve la ruta.

    Si alguna columna tiene tipos mixtos (p. ej. ids de entrenador combinados),
    se coacciona a texto automáticamente para que Parquet nunca falle.
    """
    path = TABLES_DIR / f"{name}.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        clean = df.copy()
        for col in clean.columns:
            if clean[col].dtype == object:
                clean[col] = clean[col].map(_coerce_scalar)
        clean.to_parquet(path, index=False)
    return path


def _coerce_scalar(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value if isinstance(value, str) else str(value)


def connect() -> duckdb.DuckDBPyConnection:
    """Conexión DuckDB con una vista por cada Parquet en data/tables/.

    Usa una conexión en memoria y monta las vistas sobre los Parquet del
    repositorio, así no hace falta ningún fichero de base de datos persistente
    (ideal para desplegar en la nube).
    """
    con = duckdb.connect()  # en memoria
    for pq in TABLES_DIR.glob("*.parquet"):
        con.execute(
            f'CREATE OR REPLACE VIEW "{pq.stem}" AS '
            f"SELECT * FROM read_parquet('{pq.as_posix()}')"
        )
    return con


def query(sql: str) -> pd.DataFrame:
    """Ejecuta SQL contra todas las tablas y devuelve un DataFrame."""
    con = connect()
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def list_tables() -> list[str]:
    """Nombres de las tablas disponibles (un Parquet = una tabla)."""
    return sorted(p.stem for p in TABLES_DIR.glob("*.parquet"))
