"""Métricas de evaluación y prueba de Diebold-Mariano."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p):
    y = np.asarray(y, float)
    return float(np.mean(np.abs(y - p) / np.maximum(y, 1)) * 100)


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(p, 0)) - np.log1p(y)) ** 2)))


def resumen(y, p) -> dict:
    """y, p: arrays (n_origenes, horizonte)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p), "RMSLE": rmsle(y, p)}
    out["MAE_por_h"] = [mae(y[:, h], p[:, h]) for h in range(y.shape[1])]
    return out


def diebold_mariano(e1, e2, h: int = 1, power: int = 1):
    """DM con varianza HAC (Newey-West) y corrección de Harvey et al. (1997).
    e1, e2: errores de pronóstico (1D) de los modelos 1 y 2. H1: modelo 2 es mejor si estadístico > 0."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p)
