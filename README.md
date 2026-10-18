# Pronóstico diario de siniestros viales con víctimas en Bogotá D.C.

Proyecto integrador · Técnicas Avanzadas de Modelado en IA · Maestría en IA · Universidad de La Sabana · **Grupo 7**

**Pregunta:** bajo la misma partición temporal y la misma información disponible, ¿una red recurrente (GRU/LSTM) reduce
el error de pronóstico a 1–7 días de los siniestros con heridos o muertos frente a SARIMAX y a una regresión Ridge con rezagos?

## Estructura

```
config/config.yaml        # todos los parámetros fijos: fechas, split, horizonte, rejilla, semillas
data/raw/                 # datos crudos (ver "Datos")
data/processed/           # serie diaria construida por src/data.py
src/data.py               # carga, auditoría, serie diaria + calendario + clima, split temporal
src/baselines.py          # naive estacional, media estacional, Ridge con rezagos, SARIMAX
src/models_dl.py          # GRU / LSTM en PyTorch, entrenamiento con parada temprana
src/metrics.py            # MAE, RMSE, MAPE, RMSLE, prueba de Diebold-Mariano
src/figuras.py            # figuras descriptivas
scripts/run_baselines.py  # E0: líneas base
scripts/run_experiments.py# E1-E5: rejilla, 3 semillas, ablaciones
scripts/run_evaluacion.py # E6-E7: comparación en prueba, Diebold-Mariano, errores, figuras
notebooks/                # notebook de entrega que ejecuta el pipeline paso a paso
results/                  # métricas (json/csv) y predicciones (pkl)
figures/                  # figuras PNG
docs/                     # informes PDF y generadores
tests/                    # pruebas de integridad y de ausencia de fuga
```

## Datos

| Fuente | Archivo | Licencia |
|---|---|---|
| Anuario de Siniestralidad Vial de Bogotá – Secretaría Distrital de Movilidad, Observatorio de Movilidad (https://observatorio.movilidadbogota.gov.co/publicaciones/anuarios): `Base_Anuario_de_Siniestralidad_2024.xlsx` (consolida 2015–2024) y `Base_Anuario_de_Siniestralidad_Vial_2025.xlsx` | `data/raw/siniestros_2015_2024.csv`, `data/raw/siniestros_2025.csv` (extraídos con `scripts/extraer_anuarios.py`; sin dirección ni coordenadas) | Datos abiertos de la SDM; la página no declara licencia explícita: se usan con atribución |
| Open-Meteo Historical Weather API (ERA5), Bogotá 4,60 N −74,05 O, 2015–2025, `precipitation_sum`, `temperature_2m_mean` | `data/raw/openmeteo_bogota_diario.csv` | CC BY 4.0 – *Weather data by Open-Meteo.com* |

Los .xlsx originales (127 MB el de 2024) no se versionan: descárguelos en `data/raw/anuarios_xlsx/` y ejecute
`python -m scripts.extraer_anuarios`. Descarga del clima:
`https://archive-api.open-meteo.com/v1/archive?latitude=4.61&longitude=-74.08&start_date=2015-01-01&end_date=2025-12-31&daily=precipitation_sum,temperature_2m_mean&timezone=America%2FBogota&format=csv`
Fecha de descarga: 18-oct-2026.

**Variable objetivo:** siniestros con heridos o muertos por día (indicador de gravedad a 30 días). Los siniestros
"solo daños" dejan de registrarse de forma comparable desde octubre de 2022, por eso no se usan.

## Reproducir

```bash
pip install -r requirements.txt
python -m scripts.extraer_anuarios # .xlsx del Anuario -> CSV livianos
python -m src.data                 # auditoría + serie diaria  -> results/auditoria.json
python -m scripts.run_baselines    # líneas base               -> results/baselines_metricas.json
python -m scripts.run_baselines indicador  # control: pandemia dentro del entrenamiento
python -m scripts.run_experiments  # GRU/LSTM (~30 min en CPU)  -> results/dl_metricas.json, grid_search.csv
python -m scripts.run_evaluacion   # comparación final         -> results/comparacion_final.csv, figures/
python -m src.figuras              # figuras descriptivas
pytest -q                          # pruebas
```

Semilla global 42; los resultados de redes se reportan como media ± desviación de las semillas {42, 7, 2024}.

## Decisiones clave contra fuga de información

- Partición cronológica: entrenamiento 2015-01-01 → 2022-12-31, validación 2023, prueba 2024-01-01 → 2025-12-31.
- Pandemia (16-mar-2020 → 31-ago-2021) excluida del entrenamiento: se descartan los orígenes cuya ventana la toca
  (en SARIMAX esos días son faltantes). Control "con pandemia + indicador" en `run_experiments` y `run_baselines indicador`.
- Evaluación por origen móvil: en cada día *t* se pronostica *t+1…t+7* solo con datos hasta *t*.
- Escalado con estadísticos de entrenamiento; selección de hiperparámetros y de modelos solo con validación.
- La lluvia futura no se usa (no se conoce al pronosticar); el calendario futuro sí (es conocido de antemano).

## Uso de herramientas de IA

Se usó un asistente de IA (Claude, Anthropic) como apoyo para escribir código y redactar documentos. El equipo revisó,
ejecutó y validó todos los resultados.
