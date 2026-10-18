"""Carga, auditoría y construcción de la serie diaria de siniestros + covariables."""
from __future__ import annotations

import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config" / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


VICTIMAS = ("Con Heridos", "Con Muertos")


def load_siniestros(cfg: dict) -> pd.DataFrame:
    """Une las bases del Anuario de Siniestralidad (2015-2024 consolidado + 2025). Un registro por siniestro."""
    partes = [pd.read_csv(ROOT / p, low_memory=False) for p in cfg["data"]["siniestros_csv"]]
    df = pd.concat(partes, ignore_index=True)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.drop_duplicates("codigo", keep="first")  # 3 códigos aparecen en ambas bases
    return df


def auditoria(df: pd.DataFrame) -> dict:
    """Auditoría inicial de calidad sobre los registros crudos."""
    diaria = df.groupby("fecha").size()
    rango = pd.date_range(diaria.index.min(), diaria.index.max())
    anio = df["fecha"].dt.year
    por_anio_grav = df.groupby([anio, "gravedad_30d"]).size().unstack(fill_value=0)
    vict = df[df["gravedad_30d"].isin(VICTIMAS)].groupby("fecha").size().reindex(rango, fill_value=0)
    return {
        "registros": int(len(df)),
        "fecha_min": str(diaria.index.min().date()),
        "fecha_max": str(diaria.index.max().date()),
        "dias_rango": int(len(rango)),
        "dias_sin_registro": int(len(rango.difference(diaria.index))),
        "dias_sin_victimas": int((vict == 0).sum()),
        "nulos_pct": {c: round(float(v) * 100, 2) for c, v in df.isna().mean().items() if v > 0},
        "gravedad": df["gravedad_30d"].value_counts().to_dict(),
        "clase": df["clase"].value_counts().to_dict(),
        "localidades": int(df["localidad"].str.upper().str.strip().nunique()),
        "por_anio_gravedad": {int(k): {str(c): int(v) for c, v in r.items()} for k, r in por_anio_grav.iterrows()},
        "solo_danos_mensual_2022": {str(k.date()): int(v) for k, v in
                                    df[df["gravedad_30d"] == "Solo Daños"].groupby("fecha").size()
                                    .resample("ME").sum().loc["2022-06":"2023-01"].items()},
        "victimas_por_dia_por_anio": {int(k): round(float(v), 1) for k, v in vict.groupby(vict.index.year).mean().items()},
        "gravedad_tradicional_vs_30d_victimas": {
            "tradicional": int(df["gravedad_tradicional"].isin(VICTIMAS).sum()),
            "30d": int(df.loc[df["gravedad_tradicional"].notna(), "gravedad_30d"].isin(VICTIMAS).sum())},
    }


def construir_serie(cfg: dict, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Serie diaria: y (total), por gravedad, clima y calendario."""
    if df is None:
        df = load_siniestros(cfg)
    ini, fin = pd.Timestamp(cfg["data"]["fecha_inicio"]), pd.Timestamp(cfg["data"]["fecha_fin"])
    df = df[(df["fecha"] >= ini) & (df["fecha"] <= fin)]
    idx = pd.date_range(ini, fin, name="fecha")

    s = pd.DataFrame(index=idx)
    s["y"] = df[df["gravedad_30d"].isin(VICTIMAS)].groupby("fecha").size().reindex(idx, fill_value=0)
    s["y_total_registrado"] = df.groupby("fecha").size().reindex(idx, fill_value=0)
    for g, col in [("Solo Daños", "y_danos"), ("Con Heridos", "y_heridos"), ("Con Muertos", "y_muertos")]:
        s[col] = df[df["gravedad_30d"] == g].groupby("fecha").size().reindex(idx, fill_value=0)

    clima = pd.read_csv(ROOT / cfg["data"]["clima_csv"], skiprows=3)
    clima["fecha"] = pd.to_datetime(clima["time"])
    clima = clima.set_index("fecha").reindex(idx)
    s["lluvia_mm"] = clima["precipitation_sum (mm)"].values
    s["temp_c"] = clima["temperature_2m_mean (°C)"].values

    # Calendario (conocido de antemano -> se puede usar para el futuro sin fuga)
    co = holidays.Colombia(years=range(ini.year, fin.year + 1))
    s["dow"] = s.index.dayofweek
    s["festivo"] = s.index.map(lambda d: int(d in co))
    s["puente"] = ((s["festivo"] == 1) & (s["dow"] == 0)).astype(int)
    s["pre_festivo"] = s["festivo"].shift(-1, fill_value=0)
    s["quincena"] = s.index.day.isin([14, 15, 16, 29, 30, 31, 1]).astype(int)
    s["mes"] = s.index.month
    s["diciembre"] = (s["mes"] == 12).astype(int)
    s["confinamiento"] = 0
    for a, b in cfg["confinamiento"]:
        s.loc[a:b, "confinamiento"] = 1
    return s


def mascara_atipica(s: pd.DataFrame, cfg: dict) -> np.ndarray:
    """True en los días del periodo atípico (pandemia)."""
    m = np.zeros(len(s), dtype=bool)
    for a, b in cfg["confinamiento"]:
        m |= (s.index >= a) & (s.index <= b)
    return m


def origenes_entrenables(origs, atip: np.ndarray, W: int, H: int, excluir: bool) -> list[int]:
    """En modo excluir, descarta los orígenes cuya ventana [t-W+1, t+H] toca el periodo atípico."""
    if not excluir:
        return list(origs)
    return [t for t in origs if not atip[t - W + 1 : t + H + 1].any()]


def split(s: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    sp = cfg["split"]
    return {
        "train": s.loc[: sp["train_fin"]],
        "val": s.loc[sp["val_inicio"] : sp["val_fin"]],
        "test": s.loc[sp["test_inicio"] :],
    }


if __name__ == "__main__":
    cfg = load_config()
    raw = load_siniestros(cfg)
    aud = auditoria(raw)
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "auditoria.json", "w", encoding="utf-8") as f:
        json.dump(aud, f, ensure_ascii=False, indent=2)
    serie = construir_serie(cfg, raw)
    out = ROOT / cfg["data"]["processed"]
    out.parent.mkdir(parents=True, exist_ok=True)
    serie.to_csv(out)
    parts = split(serie, cfg)
    print(json.dumps({k: [str(v.index.min().date()), str(v.index.max().date()), len(v)] for k, v in parts.items()}))
    print(serie.describe().T[["mean", "std", "min", "max"]])
