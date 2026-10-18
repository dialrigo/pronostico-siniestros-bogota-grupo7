"""Figuras del informe (PNG, 200 dpi)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
C = {"azul": "#2a78d6", "naranja": "#eb6834", "aqua": "#1baf7a", "violeta": "#6250d6", "gris": "#9a9893",
     "tinta": "#0b0b0b", "tinta2": "#52514e", "grid": "#e4e3df", "superficie": "#fcfcfb"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": C["grid"], "axes.labelcolor": C["tinta2"],
    "xtick.color": C["tinta2"], "ytick.color": C["tinta2"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": C["tinta"],
    "legend.frameon": False,
})


def _guardar(fig, nombre):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / nombre, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_registro(s):
    """Quiebre de registro: solo daños vs siniestros con víctimas (promedio diario por mes)."""
    m = s[["y_danos", "y"]].resample("ME").mean()
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    md = m.loc[:"2024-12-31", "y_danos"]  # la base 2025 solo publica siniestros con víctimas
    ax.plot(md.index, md, color=C["naranja"], lw=2, label="Solo daños (hasta 2024)")
    ax.plot(m.index, m["y"], color=C["azul"], lw=2, label="Con heridos o muertos (objetivo)")
    ax.axvline(pd.Timestamp("2022-10-01"), color=C["gris"], lw=1, ls="--")
    ax.text(pd.Timestamp("2022-11-01"), 60, "oct-2022: los solo daños\ncasi dejan de registrarse", fontsize=7.5, color=C["tinta2"])
    ax.set_ylabel("Promedio diario (por mes)")
    ax.set_ylim(0, 85)
    ax.legend(loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Siniestros registrados en Bogotá por gravedad, 2015–2025", loc="left")
    _guardar(fig, "fig0_quiebre_registro.png")


def fig_serie(s, cfg):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(s.index, s["y"], color=C["azul"], lw=0.5, alpha=0.35, label="Diario")
    ax.plot(s.index, s["y"].rolling(28, center=True).mean(), color=C["azul"], lw=2, label="Media móvil 28 días")
    sp = cfg["split"]
    a, b = cfg["confinamiento"][0]
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=C["gris"], alpha=0.25, lw=0, label="Pandemia (excluida del entrenamiento)")
    ax.axvspan(pd.Timestamp(sp["val_inicio"]), pd.Timestamp(sp["val_fin"]), color=C["naranja"], alpha=0.18, lw=0,
               label="Validación (2023)")
    ax.axvspan(pd.Timestamp(sp["test_inicio"]), s.index[-1], color=C["aqua"], alpha=0.18, lw=0,
               label="Prueba (2024–2025)")
    ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-01-15"))
    ax.set_ylim(0, 65)
    ax.set_ylabel("Siniestros con víctimas/día")
    ax.legend(loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Siniestros con víctimas por día en Bogotá D.C. (2015–2025) y partición temporal", loc="left")
    _guardar(fig, "fig1_serie_split.png")


def fig_perfil(s):
    t = s.loc[: "2020-03-19"]
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    m = t[t.festivo == 0].groupby("dow")["y"].mean()
    fest = t[t.festivo == 1]["y"].mean()
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    vals = list(m.values) + [fest]
    labs = dias + ["Festivo"]
    cols = [C["azul"]] * 7 + [C["naranja"]]
    ax.bar(labs, vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.0f}", ha="center", fontsize=8, color=C["tinta"])
    ax.set_ylabel("Con víctimas, promedio diario")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Patrón semanal y efecto festivo (2015 – feb. 2020)", loc="left")
    _guardar(fig, "fig2_perfil_semanal.png")


def fig_baselines(res):
    mods = [m for m in res if not m.startswith("_") and "test" in res[m]]
    v = [res[m]["test"]["MAE"] for m in mods]
    orden = np.argsort(v)[::-1]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.barh([mods[i] for i in orden], [v[i] for i in orden], color=C["azul"], height=0.55)
    for k, i in enumerate(orden):
        ax.text(v[i] + 0.1, k, f"{v[i]:.2f}", va="center", fontsize=8)
    ax.set_xlabel("MAE en prueba (siniestros con víctimas/día)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v) * 1.15)
    ax.set_title("Líneas base: error absoluto medio, horizonte 1–7 días", loc="left")
    _guardar(fig, "fig3_baselines.png")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    from src.data import construir_serie, load_config

    cfg = load_config()
    s = construir_serie(cfg)
    fig_registro(s)
    fig_serie(s, cfg)
    fig_perfil(s)
    fig_baselines(json.load(open(ROOT / "results" / "baselines_metricas.json")))
    print("ok")
