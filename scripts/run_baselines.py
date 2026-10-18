"""Ejecuta las líneas base y guarda métricas y pronósticos.  Uso: python -m scripts.run_baselines"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import evaluar_baselines  # noqa: E402
from src.data import construir_serie, load_config, split  # noqa: E402

cfg = load_config()
MODO = sys.argv[1] if len(sys.argv) > 1 else cfg["pandemia"]["modo"]
cfg["pandemia"]["modo"] = MODO
SUF = "" if MODO == "excluir" else "_" + MODO
np.random.seed(cfg["seed"])
s = construir_serie(cfg)
parts = split(s, cfg)
H = cfg["pronostico"]["horizonte"]
res, preds = evaluar_baselines(s, parts, H, cfg)
(ROOT / "results").mkdir(exist_ok=True)
with open(ROOT / "results" / f"baselines_metricas{SUF}.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
with open(ROOT / "results" / f"baselines_preds{SUF}.pkl", "wb") as f:
    pickle.dump(preds, f)
for m, r in res.items():
    if m.startswith("_"):
        continue
    print(f"{m:28s} VAL MAE {r['val']['MAE']:6.2f} RMSE {r['val']['RMSE']:6.2f} | TEST MAE {r['test']['MAE']:6.2f} RMSE {r['test']['RMSE']:6.2f} MAPE {r['test']['MAPE']:5.1f}%")
