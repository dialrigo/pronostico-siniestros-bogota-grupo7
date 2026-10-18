"""Experimentos controlados con GRU/LSTM.  Uso: python -m scripts.run_experiments

1) Búsqueda en rejilla (semilla fija) -> selección por MAE de VALIDACIÓN (nunca con prueba).
2) Mejor configuración por arquitectura re-entrenada con 3 semillas -> media ± desviación.
3) Ablaciones sobre la mejor GRU/LSTM: sin lluvia, sin calendario futuro (un factor a la vez)."""
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import objetivos, origenes  # noqa: E402
from src.data import construir_serie, load_config, mascara_atipica, origenes_entrenables, split  # noqa: E402
from src.metrics import resumen  # noqa: E402
from src.models_dl import Escalador, construir_tensores, entrenar, predecir  # noqa: E402

cfg = load_config()
dl = cfg["modelos_dl"]
H = cfg["pronostico"]["horizonte"]
s = construir_serie(cfg)
parts = split(s, cfg)
fin_train = s.index.get_loc(parts["train"].index[-1])
EXCLUIR = cfg["pandemia"]["modo"] == "excluir"
atip = mascara_atipica(s, cfg)
Wmax = max(dl["ventana_grid"])
o_todos = list(range(Wmax, fin_train - H + 1))  # mismos orígenes para todas las ventanas
o_tr = origenes_entrenables(o_todos, atip, Wmax, H, EXCLUIR)
esc = Escalador().fit(s, fin_train, atip if EXCLUIR else None)
esc_ind = Escalador().fit(s, fin_train)  # control: pandemia dentro del entrenamiento con indicador
print(f"orígenes de entrenamiento: {len(o_tr)} (de {len(o_todos)}; modo {cfg['pandemia']['modo']})", flush=True)
o_va, o_te = origenes(s, parts["val"], H), origenes(s, parts["test"], H)
Y_va, Y_te = objetivos(s, o_va, H), objetivos(s, o_te, H)
OUT = ROOT / "results"


def corrida(tipo, W, hidden, lr, dropout, seed, lluvia=True, cal=True, test=False, pandemia_indicador=False):
    conf = pandemia_indicador or not EXCLUIR
    origs, e = (o_todos, esc_ind) if pandemia_indicador else (o_tr, esc)
    tr = construir_tensores(s, origs, W, H, e, lluvia, cal, conf)
    va = construir_tensores(s, o_va, W, H, e, lluvia, cal, conf)
    m, hist, info = entrenar(tipo, tr, va, H, hidden, lr, dropout, dl["batch_size"], dl["max_epochs"], dl["paciencia"], seed)
    P_va, _ = predecir(m, va, e)
    r = {"val": resumen(Y_va, P_va), **info}
    preds = {"val": P_va}
    if test:
        te = construir_tensores(s, o_te, W, H, e, lluvia, cal, conf)
        P_te, tp = predecir(m, te, e)
        r["test"] = resumen(Y_te, P_te)
        r["seg_pred_test"] = tp
        preds["test"] = P_te
    return r, hist, preds


# 1) Rejilla
filas = []
for tipo in ["GRU", "LSTM"]:
    for W, hd, lr, dp in itertools.product(dl["ventana_grid"], dl["hidden_grid"], dl["lr_grid"], dl["dropout_grid"]):
        r, _, _ = corrida(tipo, W, hd, lr, dp, cfg["seed"])
        filas.append({"modelo": tipo, "ventana": W, "hidden": hd, "lr": lr, "dropout": dp, "val_MAE": r["val"]["MAE"],
                      "val_RMSE": r["val"]["RMSE"], "epocas": r["epocas"], "seg": r["seg_entreno"], "params": r["n_parametros"]})
        print(filas[-1], flush=True)
grid = pd.DataFrame(filas)
grid.to_csv(OUT / "grid_search.csv", index=False)

# 2) Mejores configuraciones con 3 semillas + 3) ablaciones
finales, curvas, preds_finales = {}, {}, {}
for tipo in ["GRU", "LSTM"]:
    best = grid[grid.modelo == tipo].sort_values("val_MAE").iloc[0]
    conf = dict(W=int(best.ventana), hidden=int(best.hidden), lr=float(best.lr), dropout=float(best.dropout))
    variantes = {f"{tipo}": dict(lluvia=True, cal=True), f"{tipo} sin lluvia": dict(lluvia=False, cal=True),
                 f"{tipo} sin calendario futuro": dict(lluvia=True, cal=False)}
    if EXCLUIR:
        variantes[f"{tipo} con pandemia + indicador"] = dict(lluvia=True, cal=True, pandemia_indicador=True)
    for nombre, v in variantes.items():
        rs = []
        for sd in dl["semillas_finales"]:
            r, hist, pr = corrida(tipo, conf["W"], conf["hidden"], conf["lr"], conf["dropout"], sd, test=True, **v)
            rs.append(r)
            if sd == cfg["seed"]:
                curvas[nombre] = hist
                preds_finales[nombre] = pr
        agg = {"config": conf | v, "semillas": dl["semillas_finales"],
               "n_origenes_train": len(o_todos) if v.get("pandemia_indicador") else len(o_tr)}
        for parte in ["val", "test"]:
            for k in ["MAE", "RMSE", "MAPE", "RMSLE"]:
                vals = [r[parte][k] for r in rs]
                agg[f"{parte}_{k}"] = [float(np.mean(vals)), float(np.std(vals))]
            agg[f"{parte}_MAE_por_h"] = np.mean([r[parte]["MAE_por_h"] for r in rs], axis=0).tolist()
        agg["seg_entreno"] = float(np.mean([r["seg_entreno"] for r in rs]))
        agg["seg_pred_test"] = float(np.mean([r["seg_pred_test"] for r in rs]))
        agg["epocas"] = [r["epocas"] for r in rs]
        agg["n_parametros"] = rs[0]["n_parametros"]
        finales[nombre] = agg
        print(nombre, {k: agg[k] for k in ["val_MAE", "test_MAE", "test_RMSE"]}, flush=True)

with open(OUT / "dl_metricas.json", "w", encoding="utf-8") as f:
    json.dump(finales, f, ensure_ascii=False, indent=2)
with open(OUT / "dl_curvas_preds.pkl", "wb") as f:
    pickle.dump({"curvas": curvas, "preds": preds_finales, "o_va": o_va, "o_te": o_te}, f)
print("OK")
