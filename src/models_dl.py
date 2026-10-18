"""Pipeline de redes recurrentes (GRU / LSTM) en PyTorch para pronóstico multi-horizonte.

Entrada en el origen t:
  - secuencia pasada (t-W+1..t): log1p(y) estandarizado, [lluvia_mm estandarizada], seno/coseno del día
    de la semana, festivo, confinamiento
  - calendario futuro conocido (t+1..t+H): one-hot día de semana + festivo, puente, pre_festivo,
    quincena, diciembre, confinamiento
Salida: log1p(y) estandarizado para t+1..t+H (vector de H valores).
Los estadísticos de escalado se calculan SOLO con el conjunto de entrenamiento."""
from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import CAL_FUT

torch.set_num_threads(2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Escalador:
    def fit(self, s: pd.DataFrame, fin_train: int, excluir=None):
        """Estadísticos solo con entrenamiento (y sin el periodo atípico si se pasa la máscara `excluir`)."""
        keep = np.ones(fin_train + 1, dtype=bool) if excluir is None else ~excluir[: fin_train + 1]
        ly = np.log1p(s["y"].values[: fin_train + 1])[keep]
        ll = s["lluvia_mm"].values[: fin_train + 1][keep]
        self.my, self.sy = ly.mean(), ly.std()
        self.ml, self.sl = ll.mean(), ll.std()
        return self

    def y(self, v):
        return (np.log1p(v) - self.my) / self.sy

    def inv_y(self, z):
        return np.expm1(z * self.sy + self.my)

    def lluvia(self, v):
        return (v - self.ml) / self.sl


def construir_tensores(s, origs, W, H, esc: Escalador, usar_lluvia=True, usar_cal_fut=True, usar_conf=True):
    """usar_conf=False (pandemia excluida): el indicador de confinamiento sería constante y se omite."""
    y = esc.y(s["y"].values)
    dow = s["dow"].values
    pasado = [y, np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), s["festivo"].values]
    if usar_conf:
        pasado.append(s["confinamiento"].values)
    if usar_lluvia:
        pasado.append(esc.lluvia(s["lluvia_mm"].values))
    P = np.stack(pasado, axis=1).astype(np.float32)
    cal = [c for c in CAL_FUT if usar_conf or c != "confinamiento"]
    F = np.concatenate([np.eye(7)[dow], s[cal].values], axis=1).astype(np.float32)
    Xs = np.stack([P[t - W + 1 : t + 1] for t in origs])
    Xf = np.stack([F[t + 1 : t + 1 + H].reshape(-1) for t in origs])
    if not usar_cal_fut:
        Xf = np.zeros_like(Xf)
    Y = np.stack([y[t + 1 : t + 1 + H] for t in origs]).astype(np.float32)
    return torch.tensor(Xs), torch.tensor(Xf), torch.tensor(Y)


class RedRecurrente(nn.Module):
    def __init__(self, tipo: str, n_in: int, n_fut: int, hidden: int, H: int, dropout: float, capas: int = 1):
        super().__init__()
        rnn = nn.GRU if tipo == "GRU" else nn.LSTM
        self.rnn = rnn(n_in, hidden, num_layers=capas, batch_first=True, dropout=dropout if capas > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_fut, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, H)
        )

    def forward(self, xs, xf):
        _, h = self.rnn(xs)
        h = h[0] if isinstance(h, tuple) else h
        return self.head(torch.cat([h[-1], xf], dim=1))


def entrenar(tipo, datos_tr, datos_va, H, hidden, lr, dropout, batch=64, max_epochs=300, paciencia=25, seed=42, wd=1e-4):
    set_seed(seed)
    Xs, Xf, Y = datos_tr
    modelo = RedRecurrente(tipo, Xs.shape[2], Xf.shape[1], hidden, H, dropout)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()
    hist = {"train": [], "val": []}
    mejor, mejor_estado, sin_mejora = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(max_epochs):
        modelo.train()
        perm = torch.randperm(len(Y), generator=g)
        tot = 0.0
        for i in range(0, len(Y), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(modelo(Xs[idx], Xf[idx]), Y[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        modelo.eval()
        with torch.no_grad():
            lv = loss_fn(modelo(*datos_va[:2]), datos_va[2]).item()
        hist["train"].append(tot / len(Y))
        hist["val"].append(lv)
        sched.step(lv)
        if lv < mejor - 1e-5:
            mejor, sin_mejora = lv, 0
            mejor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    info = {"epocas": ep + 1, "mejor_epoca": int(np.argmin(hist["val"])) + 1, "seg_entreno": time.time() - t0,
            "n_parametros": int(sum(p.numel() for p in modelo.parameters()))}
    return modelo, hist, info


def predecir(modelo, datos, esc: Escalador):
    modelo.eval()
    t0 = time.time()
    with torch.no_grad():
        z = modelo(datos[0], datos[1]).numpy()
    return esc.inv_y(z), time.time() - t0
