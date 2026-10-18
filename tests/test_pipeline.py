"""Pruebas mínimas de integridad y de ausencia de fuga de información.  Uso: pytest -q"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baselines import naive_estacional, origenes  # noqa: E402
from src.data import construir_serie, load_config, split  # noqa: E402
from src.metrics import diebold_mariano, mae  # noqa: E402
from src.models_dl import Escalador, construir_tensores  # noqa: E402

cfg = load_config()
s = construir_serie(cfg)
parts = split(s, cfg)


def test_serie_continua_y_sin_nulos():
    assert s.index.is_monotonic_increasing
    assert (s.index.to_series().diff().dropna().dt.days == 1).all()
    assert not s[["y", "lluvia_mm", "festivo"]].isna().any().any()


def test_split_cronologico_sin_solape():
    assert parts["train"].index.max() < parts["val"].index.min() <= parts["val"].index.max() < parts["test"].index.min()


def test_origenes_no_miran_fuera_del_periodo():
    H = cfg["pronostico"]["horizonte"]
    o = origenes(s, parts["test"], H)
    assert s.index[o[0] + 1] == parts["test"].index[0]
    assert s.index[o[-1] + H] == parts["test"].index[-1]


def test_escalador_usa_solo_train():
    fin = s.index.get_loc(parts["train"].index[-1])
    e = Escalador().fit(s, fin)
    assert np.isclose(e.my, np.log1p(parts["train"]["y"]).mean())


def test_entrada_dl_no_contiene_futuro():
    fin = s.index.get_loc(parts["train"].index[-1])
    e = Escalador().fit(s, fin)
    t = 1000
    Xs, Xf, Y = construir_tensores(s, [t], 14, 7, e)
    assert np.isclose(Xs[0, -1, 0].item(), e.y(s["y"].values[t]), atol=1e-5)  # último paso = día t
    assert np.isclose(Y[0, 0].item(), e.y(s["y"].values[t + 1]), atol=1e-5)


def test_metricas_y_dm():
    o = origenes(s, parts["test"], 7)
    p = naive_estacional(s, o, 7)
    assert mae(p, p) == 0
    st, pv = diebold_mariano(np.random.rand(100) + 1, np.random.rand(100))
    assert st > 0 and pv < 0.05


def test_pandemia_excluida_del_entrenamiento():
    from src.data import mascara_atipica, origenes_entrenables

    atip = mascara_atipica(s, cfg)
    fin = s.index.get_loc(parts["train"].index[-1])
    o = origenes_entrenables(range(28, fin - 6), atip, 28, 7, True)
    assert all(not atip[t - 27 : t + 8].any() for t in o)
    assert not atip[s.index.get_loc(parts["val"].index[0]) :].any()  # validación y prueba no tocan la pandemia


def test_objetivo_solo_victimas():
    assert (s["y"] == s["y_heridos"] + s["y_muertos"]).all()
