# futbol-predict — calendario multi-liga + modelo que aprende

Herramienta de análisis predictivo de fútbol. Eliges un día y las ligas que
quieras (Premier, LaLiga, Serie A, Bundesliga, Ligue 1, Champions, Mundial...)
y ves los partidos de ese día con sus probabilidades (1X2, over/under, BTTS).
El modelo se reentrena con los resultados hasta la fecha —dando más peso a los
recientes—, así que **aprende conforme pasan las jornadas**.

## Cómo funciona
- Datos de la temporada en curso desde **football-data.org** (tier gratuito):
  fixtures y resultados de las 12 competiciones del plan gratuito.
- Un modelo **Dixon-Coles por competición** (Poisson bivariado con corrección
  de marcadores bajos y decaimiento temporal) ajustado sobre los partidos ya
  jugados; predice los próximos. La distribución de marcadores da 1X2,
  over/under y BTTS de forma coherente.
- **Validación**: backtest walk-forward sin fuga de información sobre una liga
  histórica (Premier 2015/16), evaluado con log loss y Brier. Demuestra que el
  motor está calibrado y bate a un baseline ingenuo.
- **Auto-aprendizaje sin PC encendido**: un robot de GitHub Actions ejecuta la
  tubería cada día, reentrena y guarda las predicciones; la web solo las lee.

## Estructura
```
futbol-predict/
├── app.py                      # interfaz Streamlit (calendario, predicción, validación)
├── run.py                      # tubería: ingesta multi-liga + modelo + validación
├── requirements.txt
├── .env.example                # plantilla de la clave (la real nunca se sube)
├── .gitignore
├── .github/workflows/refresh.yml  # robot de auto-refresco diario
├── .streamlit/config.toml
├── data/tables/                # datos en Parquet (los actualiza el robot)
└── src/
    ├── config.py               # rutas, clave, lista de competiciones
    ├── storage.py              # Parquet + DuckDB (en memoria)
    ├── ingest_footballdata.py  # ingesta multi-liga normalizada
    ├── ingest_statsbomb.py     # datos históricos (validación / xG)
    ├── model_dixon_coles.py    # modelo de goles
    ├── predict_competitions.py # ajuste por competición + predicción
    └── backtest.py             # backtest walk-forward
```

## Poner en marcha el auto-refresco (un único paso)
La clave de football-data.org se guarda **solo** como secreto del robot, nunca
en el repo ni en la app:
1. Saca tu clave gratuita en https://www.football-data.org/client/register
2. En tu repo de GitHub: Settings → Secrets and variables → Actions →
   "New repository secret". Nombre: `FOOTBALL_DATA_TOKEN`. Valor: tu clave.
3. Ve a la pestaña Actions del repo, elige "Refrescar datos y modelo" y pulsa
   "Run workflow" para la primera vez. A partir de ahí corre solo cada día.

Cuando el robot termina, guarda los datos nuevos en el repo y la app desplegada
(Streamlit Community Cloud) se actualiza sola.

## Probar en local (opcional)
```bash
pip install -r requirements.txt
# para datos en vivo, crea .env con FOOTBALL_DATA_TOKEN=tu_clave
streamlit run app.py
```

## Notas
- Tier gratuito de football-data.org: 10 peticiones/minuto, scores con ligero
  retardo (suficiente para un modelo que aprende por jornadas).
- El Mundial tiene poca muestra: hasta que no se juegan suficientes partidos no
  se predice (umbral configurable). El refinamiento bayesiano queda para más
  adelante; las probabilidades irán con su incertidumbre.
- Cita la fuente como StatsBomb si publicas análisis basados en sus datos.
