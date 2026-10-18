"""Extrae la hoja de siniestros de las bases del Anuario de Siniestralidad Vial (SDM) a CSV livianos.

Fuente: https://observatorio.movilidadbogota.gov.co/publicaciones/anuarios ("Descargar Excel")
  - Base_Anuario_de_Siniestralidad_2024.xlsx  -> histórico consolidado 2015-2024 (hoja "Siniestros")
  - Base_Anuario_de_Siniestralidad_Vial_2025.xlsx -> año 2025 (hoja "Siniestros")
Los .xlsx (127 MB el de 2024) no se versionan; se guardan en data/raw/anuarios_xlsx/.
Se conservan solo las columnas necesarias (sin dirección ni coordenadas).
Uso: python -m scripts.extraer_anuarios
"""
import csv
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "data" / "raw" / "anuarios_xlsx"
OUT = ROOT / "data" / "raw"

FUENTES = [
    ("Base_Anuario_de_Siniestralidad_2024.xlsx", "siniestros_2015_2024.csv",
     {"Codigo_Accidente": "codigo", "Fecha_Acc": "fecha", "Hora_Acc": "hora", "Localidad": "localidad",
      "Clase_Acc": "clase", "Gravedad_Indicador_Tradicional": "gravedad_tradicional", "Gravedad_indicador_30d": "gravedad_30d"}),
    ("Base_Anuario_de_Siniestralidad_Vial_2025.xlsx", "siniestros_2025.csv",
     {"Codigo_Siniestro": "codigo", "Fecha_Siniestro": "fecha", "Hora_Siniestro": "hora", "Localidad": "localidad",
      "Clase_Siniestro": "clase", "Gravedad_indicador_30d": "gravedad_30d"}),
]

if __name__ == "__main__":
    for xlsx, csv_out, cols in FUENTES:
        ws = openpyxl.load_workbook(XLSX / xlsx, read_only=True)["Siniestros"]
        filas = ws.iter_rows(values_only=True)
        cab = list(next(filas))
        idx = [cab.index(c) for c in cols]
        with open(OUT / csv_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(list(cols.values()))
            n = 0
            for r in filas:
                if r[idx[0]] is None:
                    continue
                w.writerow([r[i].date().isoformat() if hasattr(r[i], "date") else r[i] for i in idx])
                n += 1
        print(csv_out, n)
