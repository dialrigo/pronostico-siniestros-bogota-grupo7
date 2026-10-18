"""Líneas base: naive estacional, media estacional, regresión Ridge con rezagos y SARIMAX.

Todas se evalúan con el mismo esquema de origen móvil: en cada día t (último dato observado)
se pronostican t+1..t+H usando solo información disponible hasta t (más el calendario futuro,
que se conoce de antemano)."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CAL_FUT = ["festivo", "puente", "pre_festivo", "quincena", "diciembre", "confinamiento"]


def cal_fut(excluir: bool) -> list[str]:
    """Calendario futuro; sin el indicador de confinamiento cuando la pandemia se excluye (sería constante)."""
    return [c for c in CAL_FUT if not (excluir and c == "confinamiento")]


def origenes(s: pd.DataFrame, periodo: pd.DataFrame, H: int) -> list[int]:
    """Índices posicionales t tales que t+1..t+H caen dentro del periodo."""
    ini = s.index.get_loc(periodo.index[0])
    fin = s.index.get_loc(periodo.index[-1])
    return list(range(ini - 1, fin - H + 1))


def objetivos(s: pd.DataFrame, origs: list[int], H: int) -> np.ndarray:
    y = s["y"].values
    return np.stack([y[t + 1 : t + 1 + H] for t in origs])


def naive_estacional(s, origs, H, m=7):
    y = s["y"].values
    return np.stack([[y[t + h - m] for h in range(1, H + 1)] for t in origs]).astype(float)


def media_estacional(s, origs, H, m=7, k=4):
    y = s["y"].values
    return np.stack([[np.mean([y[t + h - m * j] for j in range(1, k + 1)]) for h in range(1, H + 1)] for t in origs])


# ---------------- Ridge con rezagos (ML clásico, pronóstico directo por horizonte) -------------
def _x_ridge(s, t, h, L=28, cal=CAL_FUT):
    y = np.log1p(s["y"].values)
    lags = y[t - L + 1 : t + 1][::-1]
    d = s.iloc[t + h]
    dow = np.eye(7)[int(d["dow"])]
    return np.concatenate([lags, dow, d[cal].values.astype(float)])


class RidgeDirecto:
    def __init__(self, H, L=28, alpha=1.0, cal=CAL_FUT):
        self.H, self.L, self.alpha, self.cal = H, L, alpha, list(cal)

    def fit(self, s, origs_train):
        self.models, self.scalers = [], []
        y = np.log1p(s["y"].values)
        origs = [t for t in origs_train if t - self.L + 1 >= 0]
        self.n_origenes = len(origs)
        for h in range(1, self.H + 1):
            X = np.stack([_x_ridge(s, t, h, self.L, self.cal) for t in origs])
            sc = StandardScaler().fit(X)
            m = Ridge(alpha=self.alpha).fit(sc.transform(X), y[np.array(origs) + h])
            self.models.append(m)
            self.scalers.append(sc)
        return self

    def predict(self, s, origs):
        out = np.zeros((len(origs), self.H))
        for h in range(1, self.H + 1):
            X = np.stack([_x_ridge(s, t, h, self.L, self.cal) for t in origs])
            out[:, h - 1] = self.models[h - 1].predict(self.scalers[h - 1].transform(X))
        return np.expm1(out)


# ---------------- SARIMAX ----------------
class Sarimax:
    def __init__(self, H, order=(2, 0, 1), sorder=(1, 0, 1, 7), exog=("festivo", "puente", "pre_festivo", "diciembre", "confinamiento")):
        self.H, self.order, self.sorder, self.exog = H, order, sorder, list(exog)

    def fit(self, s, fin_train: int, atip=None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = np.log1p(s["y"].values.astype(float))
        if atip is not None:
            y[atip] = np.nan  # el filtro de Kalman trata el periodo atípico como faltante
        self.y_hist = y
        y = y[: fin_train + 1]
        X = s[self.exog].values[: fin_train + 1].astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.res = SARIMAX(y, exog=X, order=self.order, seasonal_order=self.sorder, trend="c").fit(disp=False, maxiter=200)
        return self

    def predict(self, s, origs):
        y = self.y_hist
        X = s[self.exog].values.astype(float)
        out = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for t in origs:
                r = self.res.apply(y[: t + 1], exog=X[: t + 1])  # mismos parámetros, estado filtrado hasta t
                out.append(r.forecast(self.H, exog=X[t + 1 : t + 1 + self.H]))
        return np.expm1(np.array(out))


def evaluar_baselines(s, parts, H, cfg):
    from .data import mascara_atipica, origenes_entrenables
    from .metrics import resumen

    excluir = cfg["pandemia"]["modo"] == "excluir"
    atip = mascara_atipica(s, cfg)
    fin_train = s.index.get_loc(parts["train"].index[-1])
    o_tr = origenes_entrenables(range(28, fin_train - H + 1), atip, 28, H, excluir)
    res, preds = {}, {}
    for nombre in ["val", "test"]:
        o = origenes(s, parts[nombre], H)
        Y = objetivos(s, o, H)
        preds[nombre] = {"y": Y, "origenes": o}
        for mod, fn in [("Naive estacional (t-7)", naive_estacional), ("Media estacional (4 sem.)", media_estacional)]:
            t0 = time.time()
            P = fn(s, o, H)
            preds[nombre][mod] = P
            res.setdefault(mod, {})[nombre] = resumen(Y, P) | {"seg_pred": time.time() - t0}
    t0 = time.time()
    rid = RidgeDirecto(H, cal=cal_fut(excluir)).fit(s, o_tr)
    t_fit = time.time() - t0
    t0 = time.time()
    ex = [c for c in ("festivo", "puente", "pre_festivo", "diciembre", "confinamiento") if not (excluir and c == "confinamiento")]
    sar = Sarimax(H, exog=ex).fit(s, fin_train, atip if excluir else None)
    t_fit_s = time.time() - t0
    for nombre in ["val", "test"]:
        o, Y = preds[nombre]["origenes"], preds[nombre]["y"]
        for mod, m, tf in [("Ridge con rezagos", rid, t_fit), ("SARIMAX", sar, t_fit_s)]:
            t0 = time.time()
            P = m.predict(s, o)
            preds[nombre][mod] = P
            res.setdefault(mod, {})[nombre] = resumen(Y, P) | {"seg_entreno": tf, "seg_pred": time.time() - t0}
    res["SARIMAX"]["params"] = {k: float(v) for k, v in zip(sar.res.model.param_names, sar.res.params)}
    res["_info"] = {"modo_pandemia": cfg["pandemia"]["modo"], "origenes_train_ridge": rid.n_origenes,
                    "dias_atipicos": int(atip.sum())}
    return res, preds
