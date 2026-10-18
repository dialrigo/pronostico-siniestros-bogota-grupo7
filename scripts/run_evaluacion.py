"""Comparación final en prueba: tabla consolidada, Diebold-Mariano, análisis de errores y figuras.
Uso: python -m scripts.run_evaluacion   (requiere run_baselines y run_experiments)"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402

from src.data import construir_serie, load_config  # noqa: E402
from src.figuras import C, _guardar  # noqa: E402
from src.metrics import diebold_mariano, mae  # noqa: E402

cfg = load_config()
s = construir_serie(cfg)
R = ROOT / "results"
B = json.load(open(R / "baselines_metricas.json"))
Bp = pickle.load(open(R / "baselines_preds.pkl", "rb"))
D = json.load(open(R / "dl_metricas.json"))
Dp = pickle.load(open(R / "dl_curvas_preds.pkl", "rb"))
Y = Bp["test"]["y"]
o_te = Bp["test"]["origenes"]
assert o_te == Dp["o_te"], "los orígenes deben coincidir"

# ---------- tabla consolidada ----------
filas = []
for m in ["Naive estacional (t-7)", "Media estacional (4 sem.)", "Ridge con rezagos", "SARIMAX"]:
    r = B[m]  # noqa
    filas.append({"modelo": m, "tipo": "Línea base", "val_MAE": r["val"]["MAE"], "test_MAE": r["test"]["MAE"], "test_MAE_sd": 0.0,
                  "test_RMSE": r["test"]["RMSE"], "test_MAPE": r["test"]["MAPE"], "test_RMSLE": r["test"]["RMSLE"],
                  "seg_entreno": r["test"].get("seg_entreno", 0.0), "seg_pred_test": r["test"]["seg_pred"], "parametros": None})
for m, r in D.items():
    filas.append({"modelo": m, "tipo": "Ablación" if ("sin" in m or "pandemia" in m) else "Avanzado", "val_MAE": r["val_MAE"][0], "test_MAE": r["test_MAE"][0],
                  "test_MAE_sd": r["test_MAE"][1], "test_RMSE": r["test_RMSE"][0], "test_MAPE": r["test_MAPE"][0],
                  "test_RMSLE": r["test_RMSLE"][0], "seg_entreno": r["seg_entreno"], "seg_pred_test": r["seg_pred_test"],
                  "parametros": r["n_parametros"]})
tab = pd.DataFrame(filas)
tab.to_csv(R / "comparacion_final.csv", index=False)

# mejor línea base según VALIDACIÓN (no prueba)
base = tab[tab.tipo == "Línea base"].sort_values("val_MAE").iloc[0]["modelo"]
P = {m: Bp["test"][m] for m in ["Ridge con rezagos", "SARIMAX", "Naive estacional (t-7)"]}
P.update({m: Dp["preds"][m]["test"] for m in D})

# ---------- E7: combinación mejor red + mejor línea base; ambos elegidos y el peso fijado SOLO en validación ----------
DL = min(["GRU", "LSTM"], key=lambda m: D[m]["val_MAE"][0])
COMB = f"Combinación {DL} + {base}"
Yv = Bp["val"]["y"]
pv_gru, pv_base = Dp["preds"][DL]["val"], Bp["val"][base]
pesos = np.round(np.arange(0, 1.01, 0.1), 2)
mae_w = {float(w): mae(Yv, w * pv_gru + (1 - w) * pv_base) for w in pesos}
w_opt = min(mae_w, key=mae_w.get)
P[COMB] = w_opt * P[DL] + (1 - w_opt) * P[base]
from src.metrics import resumen  # noqa: E402

rc = resumen(Y, P[COMB])
if 0 < w_opt < 1:  # si el peso es 0 o 1 la combinación es idéntica a un modelo existente
    tab = pd.concat([tab, pd.DataFrame([{"modelo": COMB, "tipo": "Avanzado", "val_MAE": mae_w[w_opt],
                                     "test_MAE": rc["MAE"], "test_MAE_sd": 0.0, "test_RMSE": rc["RMSE"], "test_MAPE": rc["MAPE"],
                                     "test_RMSLE": rc["RMSLE"], "seg_entreno": None, "seg_pred_test": None, "parametros": None}])],
                ignore_index=True)
tab.to_csv(R / "comparacion_final.csv", index=False)
json.dump({"pesos_val_MAE": mae_w, "w_red": w_opt, "red": DL, "base": base}, open(R / "combinacion.json", "w"), indent=2)

# ---------- Diebold-Mariano (errores absolutos; promedio por origen sobre h=1..7, HAC con h=7) ----------
dm = []
for m in ["GRU", "LSTM"] + ([COMB] if 0 < w_opt < 1 else []):
    for b in ["SARIMAX", "Ridge con rezagos"]:
        e_b = np.abs(Y - P[b]).mean(axis=1)
        e_m = np.abs(Y - P[m]).mean(axis=1)
        st, p = diebold_mariano(e_b, e_m, h=7)
        st1, p1 = diebold_mariano(Y[:, 0] - P[b][:, 0], Y[:, 0] - P[m][:, 0], h=1)
        dm.append({"modelo": m, "vs": b, "MAE_modelo": mae(Y, P[m]), "MAE_base": mae(Y, P[b]),
                   "mejora_pct": (1 - mae(Y, P[m]) / mae(Y, P[b])) * 100, "DM_h1a7": st, "p_h1a7": p, "DM_h1": st1, "p_h1": p1})
dm = pd.DataFrame(dm)
dm.to_csv(R / "diebold_mariano.csv", index=False)

# ---------- análisis de errores ----------
fechas = s.index.values[np.array(o_te)[:, None] + np.arange(1, 8)[None, :]]
err = pd.DataFrame({"fecha": fechas.ravel(), "y": Y.ravel()})
for m in ["GRU", "LSTM", "SARIMAX", "Ridge con rezagos"]:
    err[m] = (P[m] - Y).ravel()
err = err.merge(s[["dow", "festivo", "lluvia_mm"]], left_on="fecha", right_index=True)
dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
por_dow = err.groupby("dow")[["GRU", "LSTM", "SARIMAX", "Ridge con rezagos"]].apply(lambda d: d.abs().mean())
por_dow.index = dias
por_fest = err.groupby("festivo")[["GRU", "LSTM", "SARIMAX", "Ridge con rezagos"]].apply(lambda d: d.abs().mean())
sesgo = err[["GRU", "LSTM", "SARIMAX", "Ridge con rezagos"]].mean()
lluvia_alta = err["lluvia_mm"] >= err["lluvia_mm"].quantile(0.9)
por_lluvia = err.groupby(lluvia_alta)[["GRU", "SARIMAX", "Ridge con rezagos"]].apply(lambda d: d.abs().mean())
peores = err.assign(abs_gru=err[DL].abs()).groupby("fecha").agg(y=("y", "first"), mae_gru=("abs_gru", "mean"), festivo=("festivo", "first")).sort_values("mae_gru", ascending=False).head(6)
por_anio = err.groupby(pd.to_datetime(err.fecha).dt.year)[["GRU", "LSTM", "SARIMAX", "Ridge con rezagos"]].apply(lambda d: d.abs().mean())
sesgo_anio = err.groupby(pd.to_datetime(err.fecha).dt.year)[["LSTM", "SARIMAX", "Ridge con rezagos"]].mean()
analisis = {"mejor_base_por_validacion": base, "mejor_red_por_validacion": DL,
            "mae_por_anio": por_anio.round(2).to_dict(), "sesgo_por_anio": sesgo_anio.round(2).to_dict(), "por_dow": por_dow.round(2).to_dict(), "por_festivo": por_fest.round(2).to_dict(),
            "sesgo_medio": sesgo.round(2).to_dict(), "lluvia_p90": por_lluvia.round(2).to_dict(),
            "peores_dias_mejor_red": {str(pd.Timestamp(k).date()): {"y": int(v.y), "mae": round(v.mae_gru, 1), "festivo": int(v.festivo)} for k, v in peores.iterrows()}}
json.dump(analisis, open(R / "analisis_errores.json", "w"), ensure_ascii=False, indent=2, default=str)

# ---------- figuras ----------
# Curvas de aprendizaje
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
for ax, m in zip(axs, ["GRU", "LSTM"]):
    h = Dp["curvas"][m]
    ep = np.arange(1, len(h["train"]) + 1)
    ax.plot(ep, h["train"], color=C["azul"], lw=2, label="Entrenamiento")
    ax.plot(ep, h["val"], color=C["naranja"], lw=2, label="Validación")
    b = int(np.argmin(h["val"])) + 1
    ax.axvline(b, color=C["gris"], lw=1, ls="--")
    ax.text(b + 1, 0.93, f"mejor época {b}", fontsize=7, color=C["tinta2"], transform=ax.get_xaxis_transform())
    ax.set_title(f"{m} ({D[m]['config']['W']} días, {D[m]['config']['hidden']} unidades)", loc="left")
    ax.set_xlabel("Época")
axs[0].set_ylabel("Pérdida MSE (escala log estandarizada)")
axs[1].legend(fontsize=7.5, loc="upper right")
_guardar(fig, "fig4_curvas_aprendizaje.png")

# Pronóstico h=1 en prueba (ventana sep-dic 2025 para legibilidad)
fig, ax = plt.subplots(figsize=(7.2, 2.6))
f1 = s.index[np.array(o_te) + 1]
w = (f1 >= "2025-09-01")
ax.plot(f1[w], Y[w, 0], color=C["tinta"], lw=1.2, label="Real")
ax.plot(f1[w], P[DL][w, 0], color=C["azul"], lw=2, label=DL)
ax.plot(f1[w], P[base][w, 0], color=C["naranja"], lw=1.5, ls="--", label=base)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax.set_ylabel("Con víctimas por día")
ax.set_title("Prueba (sep–dic 2025): pronóstico a 1 día vs. valor real", loc="left")
ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
_guardar(fig, "fig5_pronostico_test.png")

# MAE por horizonte
fig, ax = plt.subplots(figsize=(4.4, 2.6))
hs = np.arange(1, 8)
for m, col, ls in [("GRU", C["azul"], "-"), ("LSTM", C["violeta"], "-"), ("SARIMAX", C["naranja"], "--"), ("Ridge con rezagos", C["aqua"], "--")]:
    v = D[m]["test_MAE_por_h"] if m in D else [mae(Y[:, h], P[m][:, h]) for h in range(7)]  # redes: media de 3 semillas
    ax.plot(hs, v, color=col, lw=2, ls=ls, marker="o", ms=4, label=m)
ax.set_xlabel("Horizonte (días)")
ax.set_ylabel("MAE prueba")
ax.set_title("Error por horizonte (redes: media de 3 semillas)", loc="left")
ax.legend(fontsize=7, ncol=2)
_guardar(fig, "fig6_mae_horizonte.png")

# Comparación final (barras con desviación)
t2 = tab[tab.tipo != "Ablación"].sort_values("test_MAE", ascending=False)
fig, ax = plt.subplots(figsize=(4.8, 2.5))
cols = [C["azul"] if t == "Avanzado" else C["gris"] for t in t2.tipo]
ax.barh(t2.modelo, t2.test_MAE, xerr=t2.test_MAE_sd, color=cols, height=0.55, error_kw={"lw": 1, "capsize": 2})
for k, (v, sd) in enumerate(zip(t2.test_MAE, t2.test_MAE_sd)):
    ax.text(v + sd + 0.1, k, f"{v:.2f}", va="center", fontsize=7.5)
ax.set_xlabel("MAE en prueba (siniestros con víctimas/día)")
ax.set_xlim(0, t2.test_MAE.max() * 1.18)
ax.grid(axis="y", visible=False)
ax.set_title("Comparación final (redes: media de 3 semillas ± d.e.)", loc="left")
_guardar(fig, "fig7_comparacion_final.png")

# Ablaciones (ambas arquitecturas, validación y prueba)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
var = ["", " sin lluvia", " sin calendario futuro", " con pandemia + indicador"]
etq = ["Completo", "Sin lluvia", "Sin calendario futuro", "Con pandemia + indicador"]
for ax, parte in zip(axs, ["val", "test"]):
    yv = np.arange(len(var))
    for k, (m, col) in enumerate([("GRU", C["azul"]), ("LSTM", C["violeta"])]):
        v = [D[m + x][f"{parte}_MAE"][0] for x in var]
        sd = [D[m + x][f"{parte}_MAE"][1] for x in var]
        ax.barh(yv + (k - 0.5) * 0.36, v, xerr=sd, height=0.32, color=col, label=m, error_kw={"lw": 1, "capsize": 2})
        for y0, vv in zip(yv, v):
            ax.text(vv + 0.1, y0 + (k - 0.5) * 0.36, f"{vv:.2f}", va="center", fontsize=6.5)
    ax.set_yticks(yv, etq)
    ax.set_xlim(0, max(D[m + x][f"{parte}_MAE"][0] for m in ["GRU", "LSTM"] for x in var) * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_title("Validación" if parte == "val" else "Prueba", loc="left")
    ax.set_xlabel("MAE (siniestros/día)")
axs[0].invert_yaxis()
fig.legend(*axs[0].get_legend_handles_labels(), fontsize=7.5, loc="upper center", ncol=2, bbox_to_anchor=(0.62, 1.04))
_guardar(fig, "fig8_ablaciones.png")

print(tab.round(3).to_string())
print(dm.round(4).to_string())
print(json.dumps(analisis, ensure_ascii=False, indent=1, default=str))
