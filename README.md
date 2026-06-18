# futbol-predict — Fases 0 y 1 (datos + modelo de goles)

Herramienta de análisis predictivo de fútbol. Cubre la **capa de datos**
(ingesta y almacenamiento) y la primera **capa analítica**: un modelo
Dixon-Coles de goles que produce probabilidades de 1X2, over/under y BTTS,
validado con un backtest honesto. Todo con tier gratuito.

## Qué hace
Fase 0 — datos:
- Ingesta de **StatsBomb open data** (eventos con xG; gratis, sin clave).
- Ingesta de **football-data.org** (calendario del Mundial; gratis con clave).
- Almacenamiento en **Parquet + DuckDB**, consultable por SQL.

Fase 1 — modelo de goles:
- **Dixon-Coles** (Poisson bivariado con corrección de marcadores bajos y
  decaimiento temporal) ajustado sobre una liga completa.
- De la distribución de marcadores se derivan 1X2, over/under (1.5/2.5/3.5) y
  BTTS — el enfoque riguroso: modelar la distribución, no un clasificador por
  umbral.
- **Backtest walk-forward** sin fuga de información, evaluado con log loss y
  Brier (no solo accuracy) y comparado contra un baseline ingenuo.

## Arranque rápido
```bash
pip install -r requirements.txt
streamlit run app.py
```
Se abrirá la app en el navegador. La **primera vez**, pulsa el botón
**Actualizar datos** (barra lateral): descarga los datos, entrena el modelo y
guarda todo. A partir de ahí ese botón es lo único que necesitas tocar —cero
terminal—. Usa la clave que **ya está en `.env`** (no hay que editar nada).

La app tiene cuatro pestañas: Calendario (partidos del Mundial), Predicción de
partido (elige dos equipos y ve 1X2, over/under, BTTS, xG y la matriz de
marcadores), Histórico vs predicción (backtest y calibración) y Ratings.

Si prefieres solo la tubería de datos sin abrir la app: `python run.py`.

## Estructura
```
futbol-predict/
├── app.py                # interfaz Streamlit: streamlit run app.py
├── run.py                # tubería de datos + modelo (lo usa el botón Actualizar)
├── requirements.txt
├── .env                  # tu clave (ya configurada; ignorado por git)
├── .env.example
├── .gitignore
├── .streamlit/
│   └── config.toml       # tema de la app
└── src/
    ├── config.py             # rutas y credenciales
    ├── storage.py            # Parquet + DuckDB (save_table, query, connect)
    ├── ingest_statsbomb.py   # eventos/xG de StatsBomb open data
    ├── ingest_footballdata.py# calendario/resultados de football-data.org
    ├── model_dixon_coles.py  # modelo de goles (fit, predict, ratings)
    └── backtest.py           # backtest walk-forward + métricas
```
Los datos se guardan solos en `data/` (se crea al primer uso; no se sube a git).

## Tablas que deja en DuckDB
- `predicciones` — backtest: una fila por partido con probabilidades 1X2,
  over 2.5, xG estimado y el resultado real (datos para "histórico vs
  predicción" en la futura capa de visualización).
- `ratings` — fuerza de ataque y defensa por equipo (interpretable).
- `sb_matches_*`, `sb_shots`, `sb_competitions` — datos crudos de StatsBomb.
- `fd_matches_WC`, `fd_standings_WC` — calendario del Mundial (al conectar).

## Consultar los datos
```python
import sys; sys.path.insert(0, "src")
import storage

storage.list_tables()
storage.query("SELECT COUNT(*) FROM sb_shots WHERE shot_statsbomb_xg > 0.3")
```

## Siguiente paso (Fase 2 y visualización)
- Modelos de conteo (Poisson/binomial negativa) para córners, tarjetas y
  disparos, reusando la misma mecánica de "modelar la distribución".
- Tu propio modelo de **xG** sobre los eventos de disparo de StatsBomb
  (`sb_shots`, columna `shot_statsbomb_xg` como etiqueta).
- Capa **Streamlit**: calendario filtrable, dashboards de probabilidad por
  variable y la vista "histórico vs predicción" leyendo la tabla `predicciones`.
  Un botón "Actualizar datos" disparará `run.py` por detrás (cero terminal).

## Desplegar en la nube (gratis, sin tener el PC encendido)
La app está lista para Streamlit Community Cloud. Resumen:
1. Sube esta carpeta a un repositorio de GitHub (público). **No subas `.env`**
   (ya está en `.gitignore`); el repo no necesita la clave para funcionar.
2. Entra en https://share.streamlit.io, conecta tu cuenta de GitHub y pulsa
   "Create app".
3. Elige el repositorio, rama `main` y, en "Main file path", pon `app.py`.
   Deploy. En unos minutos tendrás una URL pública `*.streamlit.app`.

Los datos históricos ya vienen incluidos en `data/tables/`, así que la web
carga al instante sin descargar nada. (El botón "Actualizar datos" sigue
disponible para refrescar dentro de una sesión.)

## Notas de seguridad
- El fichero `.env` con la clave **nunca** debe subirse a GitHub. Está en
  `.gitignore`, pero si subes por la web arrastrando ficheros, asegúrate de no
  incluirlo. Para datos en vivo en la nube se usarían los "Secrets" de
  Streamlit, no un fichero en el repo.
- El Mundial tiene poca muestra para ajustar fuerzas de equipo, por eso el
  modelo se valida sobre una liga completa. Para el Mundial 2026 se usará
  pooling bayesiano (Fase 3); las probabilidades irán con su incertidumbre.
- StatsBomb open data es histórico, no en vivo: ideal para desarrollo.
- Cita la fuente como StatsBomb si publicas análisis basados en sus datos.
