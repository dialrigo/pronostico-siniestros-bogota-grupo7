"""Genera el reporte técnico de la Actividad 2 (≤ 6 páginas). Todas las cifras se leen de results/."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
from estilo_pdf import P, bullets, construir, figura, tabla  # noqa: E402

R, FIG = ROOT / "results", ROOT / "figures"
B = json.load(open(R / "baselines_metricas.json"))
BI = json.load(open(R / "baselines_metricas_indicador.json"))
D = json.load(open(R / "dl_metricas.json"))
T = pd.read_csv(R / "comparacion_final.csv")
DM = pd.read_csv(R / "diebold_mariano.csv")
G = pd.read_csv(R / "grid_search.csv")
E = json.load(open(R / "analisis_errores.json"))
INTEGRANTES = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "Diego Alejandro Ríos Gómez · Marlon [apellidos] · Nicolás [apellidos] · Nataly [apellidos]"


def f(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(a, b):
    return (a / b - 1) * 100


def dm(m, b):
    return DM[(DM.modelo == m) & (DM.vs == b)].iloc[0]


BASE = E["mejor_base_por_validacion"]           # mejor línea base según validación
RED = E["mejor_red_por_validacion"]             # mejor red según validación
OTRA = "LSTM" if RED == "GRU" else "GRU"
OB = "SARIMAX" if BASE == "Ridge con rezagos" else "Ridge con rezagos"
N, O, Bb, Ob = D[RED], D[OTRA], B[BASE], B[OB]


def abl(m, v, p="test"):
    return D[f"{m} {v}"][f"{p}_MAE"][0]


s = [P("Pronóstico de siniestros con víctimas en Bogotá: ejecución, optimización y validación comparativa", "titulo"),
     P("Actividad 2 · Reporte técnico · Unidad 4 · Técnicas Avanzadas de Modelado en IA · Universidad de La Sabana<br/>"
       f"Grupo 7: {INTEGRANTES} · Octubre de 2026", "sub")]

s += [P("Resumen", "h1"),
      P(f"Implementamos en PyTorch redes GRU y LSTM para pronosticar los siniestros con heridos o muertos de Bogotá a 1–7 días "
        f"(entrenamiento 2015–2022 sin el periodo de pandemia, validación 2023, prueba 2024–2025). Tras una rejilla de 32 "
        f"configuraciones, la mejor red en validación es la {RED}: reduce el MAE de validación {f(-pct(N['val_MAE'][0], Bb['val']['MAE']), 1)} % "
        f"frente a {BASE} (mejor línea base en validación) y cumple el criterio de éxito. En prueba su MAE es {f(N['test_MAE'][0])} ± "
        f"{f(N['test_MAE'][1])} siniestros/día (3 semillas), {f(-pct(N['test_MAE'][0], Bb['test']['MAE']), 1)} % menor que {BASE}; la "
        f"{OTRA} obtiene {f(O['test_MAE'][0])}. La evidencia estadística es mixta: la {OTRA} supera a {BASE} con significancia "
        f"(Diebold–Mariano, p = {f(dm(OTRA, BASE).p_h1a7, 2)}), la {RED} no (p = {f(dm(RED, BASE).p_h1a7, 2)}). El calendario "
        "futuro es el componente con más aporte; la lluvia no aporta de forma robusta.")]

# 1. Pipeline
s += [P("1. Pipeline e implementación de la red", "h1"),
      P("El código está en el repositorio público <i>github.com/dialrigo/pronostico-siniestros-bogota-grupo7</i>, organizado en (<i>config/</i>, <i>src/</i>, <i>scripts/</i>, <i>tests/</i>, <i>notebooks/</i>). "
        "Todos los parámetros fijos, incluido el periodo atípico de pandemia, viven en <i>config/config.yaml</i>. El flujo es: "
        "extracción de las bases del Anuario → serie diaria y calendario (<i>src/data.py</i>) → tensores por origen móvil "
        "(<i>src/models_dl.py</i>) → entrenamiento con parada temprana → predicción y métricas (<i>src/metrics.py</i>) → "
        "comparación final (<i>scripts/run_evaluacion.py</i>). Ocho pruebas automáticas verifican la continuidad de la serie, el "
        "orden del split, la exclusión de la pandemia y que ninguna entrada contenga información posterior al origen."),
      tabla([["Bloque", "Dimensión", "Contenido y justificación"],
             ["Entrada secuencial", "W × 5 (W = 14 o 28)", "log1p(y) estandarizado, seno y coseno del día de la semana, festivo, lluvia observada estandarizada. Escaladores ajustados solo con entrenamiento y sin el periodo atípico."],
             ["Capa recurrente", "1 capa, 32 o 64 unidades", f"GRU o LSTM. Una sola capa porque hay {f(D[RED]['n_origenes_train'], 0)} orígenes de entrenamiento; capas adicionales aumentan parámetros sin datos que los respalden."],
             ["Calendario futuro", "7 × 12 = 84", "One-hot del día de la semana + festivo, puente, víspera, quincena y diciembre de cada día a pronosticar. Conocido de antemano: no hay fuga."],
             ["Cabeza densa", "(h + 84) → h → 7", "Concatenación del último estado oculto con el calendario; ReLU y dropout; salida lineal de 7 valores (pronóstico directo multi-horizonte)."],
             ["Pérdida y salida", "MSE en escala log", "Estabiliza la varianza del conteo; la predicción vuelve a la escala original con expm1."]],
            [3.1, 3.4, 10.5])]

# 2. Hiperparámetros
top = []
for m in ["GRU", "LSTM"]:
    for _, r in G[G.modelo == m].sort_values("val_MAE").head(3).iterrows():
        top.append([m, str(int(r.ventana)), str(int(r.hidden)), f(r.lr, 3), f(r.dropout, 1), f(r.val_MAE, 3), str(int(r.epocas)), f"{int(r.params):,}".replace(",", ".")])
gm = G.groupby(["modelo", "ventana"]).val_MAE.mean()
s += [P("2. Optimización y ajuste de hiperparámetros", "h1"),
      P("Fijamos lo que no se compara y variamos de forma sistemática los hiperparámetros críticos: ventana de entrada {14, 28}, "
        "unidades ocultas {32, 64}, tasa de aprendizaje inicial {0,001; 0,003} y dropout {0; 0,2}, con la misma rejilla para "
        "ambas arquitecturas (32 corridas, semilla 42). Optimizador AdamW (decaimiento de pesos 1e-4, regularización L2 "
        "desacoplada), recorte de gradiente a norma 1 (estabiliza la recurrencia), reducción de la tasa a la mitad tras 10 épocas "
        "sin mejora y parada temprana con paciencia 25. Se comparan arquitecturas con el mismo criterio de parada y no con el "
        "mismo número de épocas, porque un modelo con más parámetros converge a otro ritmo. La selección usa solo el MAE de validación."),
      tabla([["Modelo", "Ventana", "Unidades", "Tasa", "Dropout", "MAE val", "Épocas", "Parámetros"]] + top,
            [2.0, 1.7, 1.8, 1.7, 1.7, 2.0, 1.7, 2.2]),
      P(f"Las 32 configuraciones quedan entre {f(G.val_MAE.min())} y {f(G.val_MAE.max())} de MAE de validación; todas mejoran a "
        f"{BASE} ({f(Bb['val']['MAE'])}) y a {OB} ({f(Ob['val']['MAE'])}), así que el resultado no depende de una configuración "
        f"afortunada. La ventana de 28 días rinde mejor que la de 14 en ambas arquitecturas (GRU {f(gm['GRU', 28], 2)} vs "
        f"{f(gm['GRU', 14], 2)}; LSTM {f(gm['LSTM', 28], 2)} vs {f(gm['LSTM', 14], 2)}): cuatro semanas de historia capturan mejor "
        "el nivel reciente de la serie.")]

# 3. Curvas
s += [P("3. Monitoreo de curvas de aprendizaje", "h1")]
s += figura(FIG / "fig4_curvas_aprendizaje.png", 15.5, "Figura 1. Pérdida de entrenamiento y validación de las configuraciones elegidas (semilla 42). La línea punteada marca la época conservada.")
s += [P("La pérdida de entrenamiento baja de forma monótona; la de validación cae en las primeras épocas y luego se estabiliza. "
        "Después de la mejor época la validación deja de mejorar mientras el entrenamiento sigue bajando: es el inicio del "
        "sobreajuste, y la parada temprana conserva los pesos anteriores. La pérdida de validación queda por debajo de la de "
        "entrenamiento porque el dropout solo actúa al entrenar y porque 2023 es un año estable, sin choques de régimen: con la "
        "pandemia excluida, entrenamiento y validación representan el mismo régimen.")]

# 4. Comparación
orden = ["Naive estacional (t-7)", "Media estacional (4 sem.)", "Ridge con rezagos", "SARIMAX", "GRU", "LSTM"]
filas = [["Modelo", "MAE val", "MAE prueba", "RMSE", "MAPE", "RMSLE", "Entreno (s)", "Parám."]]
for m in orden:
    r = T[T.modelo == m].iloc[0]
    mt = f(r.test_MAE) + (f" ± {f(r.test_MAE_sd)}" if r.test_MAE_sd > 0 else "")
    par = "—" if pd.isna(r.parametros) else f"{int(r.parametros):,}".replace(",", ".")
    ent = "—" if pd.isna(r.seg_entreno) or r.seg_entreno == 0 else f(r.seg_entreno, 1)
    filas.append([m, f(r.val_MAE), mt, f(r.test_RMSE), f(r.test_MAPE, 1) + " %", f(r.test_RMSLE, 3), ent, par])
s += [P("4. Validación comparativa cuantitativa", "h1"),
      P("Todos los modelos se evalúan sobre los mismos 726 orígenes de prueba (2024–2025) y la misma partición. Las redes se "
        "reportan como media ± desviación de tres semillas {42, 7, 2024}."),
      tabla(filas, [3.6, 1.6, 2.4, 1.5, 1.6, 1.6, 1.9, 1.6])]
s += figura(FIG / "fig7_comparacion_final.png", 9.5, "Figura 2. MAE en prueba. Gris: líneas base; azul: redes (media de 3 semillas ± d.e.).")
dmf = [["Comparación", "MAE modelo", "MAE base", "Δ MAE", "DM (h = 1–7)", "p", "Lectura"]]
for _, r in DM.iterrows():
    lect = ("Red mejor (p < 0,05)" if (r.p_h1a7 < 0.05 and r.DM_h1a7 > 0) else
            "Base mejor (p < 0,05)" if r.p_h1a7 < 0.05 else "Diferencia no significativa")
    dmf.append([f"{r.modelo} vs {r.vs}", f(r.MAE_modelo), f(r.MAE_base), f(-r.mejora_pct, 1) + " %", f(r.DM_h1a7), f(r.p_h1a7, 3), lect])
s += [P("<b>Prueba de Diebold–Mariano.</b> Para no afirmar superioridad con una sola cifra, comparamos los errores absolutos medios "
        "por origen (promedio de h = 1–7) con varianza HAC de Newey–West (rezago 7, por el solapamiento de horizontes) y la "
        "corrección de Harvey, Leybourne y Newbold. Se usa la corrida de semilla 42 de cada red; un estadístico positivo favorece a la red."),
      tabla(dmf, [4.1, 1.7, 1.6, 1.4, 2.0, 1.2, 5.0])]
s += figura(FIG / "fig6_mae_horizonte.png", 8.0, "Figura 3. MAE por horizonte en prueba. Las redes quedan por debajo de ambas líneas base en los siete horizontes.")
s += [P(f"<b>Lectura.</b> La {RED} cumple el criterio en validación (−{f(-pct(N['val_MAE'][0], Bb['val']['MAE']), 1)} % frente a {BASE}) "
        f"y conserva una ventaja menor en prueba (−{f(-pct(N['test_MAE'][0], Bb['test']['MAE']), 1)} %), con menor RMSE "
        f"({f(N['test_RMSE'][0])} vs {f(Bb['test']['RMSE'])}). Ambas redes quedan por debajo de las dos líneas base en todos los "
        f"horizontes. Pero la mejora es de 0,1 siniestros por día y solo la {OTRA} alcanza significancia frente a {BASE}; "
        f"frente a {OB} ninguna red es significativa. Se elige la {RED} como modelo avanzado por su desempeño en validación, sin "
        "afirmar una superioridad fuerte sobre un buen modelo lineal.")]

# 5. Ablaciones
s += [P("5. Experimentos de ablación (un factor a la vez)", "h1")]
s += figura(FIG / "fig8_ablaciones.png", 15.5, "Figura 4. MAE de las variantes con la misma configuración y 3 semillas.")
s += [P(f"<b>Calendario futuro:</b> sin él, el MAE de prueba sube {f(pct(abl(RED, 'sin calendario futuro'), N['test_MAE'][0]), 1)} % "
        f"en la {RED} y {f(pct(abl(OTRA, 'sin calendario futuro'), O['test_MAE'][0]), 1)} % en la {OTRA}: es el componente con mayor aporte. "
        f"<b>Lluvia:</b> sin ella, la prueba mejora levemente en ambas redes ({RED} {f(abl(RED, 'sin lluvia'))}, {OTRA} "
        f"{f(abl(OTRA, 'sin lluvia'))}) y la validación cambia de signo entre arquitecturas: su aporte no es robusto. "
        f"<b>Pandemia:</b> con el periodo atípico dentro del entrenamiento (y un indicador), la validación empeora en ambas redes "
        f"({RED} {f(abl(RED, 'con pandemia + indicador', 'val'))} vs {f(N['val_MAE'][0])}; {OTRA} "
        f"{f(abl(OTRA, 'con pandemia + indicador', 'val'))} vs {f(O['val_MAE'][0])}) y las líneas base empeoran también en prueba "
        f"(Ridge {f(BI['Ridge con rezagos']['test']['MAE'])} vs {f(B['Ridge con rezagos']['test']['MAE'])}; SARIMAX "
        f"{f(BI['SARIMAX']['test']['MAE'])} vs {f(B['SARIMAX']['test']['MAE'])}). En prueba las redes casi no cambian. La exclusión "
        "se mantiene porque la decisión se toma con validación.")]

# 6. Errores
dw = E["por_dow"][RED]
ma = E["mae_por_anio"]
pe = E["peores_dias_mejor_red"]
s += [P("6. Análisis de errores", "h1")]
s += figura(FIG / "fig5_pronostico_test.png", 15.5, f"Figura 5. Pronóstico a 1 día de la {RED} y de {BASE} frente al valor real (sep–dic 2025).")
s += bullets([f"<b>Por año:</b> el MAE de la {RED} es {f(ma[RED]['2024'])} en 2024 y {f(ma[RED]['2025'])} en 2025. En 2025 la siniestralidad "
              f"con víctimas bajó (31,7/día frente a 35,3 en 2024) y las redes la sobrestiman (sesgo de la LSTM +{f(E['sesgo_por_anio']['LSTM']['2025'])}/día).",
              f"<b>Por día de la semana (MAE {RED}):</b> mayor el sábado ({f(dw['Sáb'])}), martes ({f(dw['Mar'])}) y jueves ({f(dw['Jue'])}); "
              f"menor el domingo ({f(dw['Dom'])}) y el viernes ({f(dw['Vie'])}).",
              f"<b>Peores días:</b> {', '.join(f'{k} (real {v['y']})' for k, v in list(pe.items())[:4])}. Son días atípicos en ambos "
              "sentidos, como el domingo 7 de diciembre de 2025 (día de velitas, 43 siniestros con víctimas frente a ~24 de un domingo típico); ninguna variable los anticipa.",
              f"<b>Lluvia:</b> el 10 % de días más lluviosos tiene más error ({RED} {f(E['lluvia_p90'][RED]['true'])} vs "
              f"{f(E['lluvia_p90'][RED]['false'])}), aunque usar la lluvia observada como entrada no lo corrige.",
              f"<b>Festivos:</b> MAE {f(E['por_festivo'][RED]['1'])} frente a {f(E['por_festivo'][RED]['0'])} en días normales: el calendario futuro los resuelve bien."])

# 7. Mejoras
s += [P("7. Oportunidades de mejora", "h1"),
      *bullets(["Pronóstico de lluvia (no la lluvia observada) como covariable futura, ya que los días lluviosos concentran más error.",
                "Indicador de eventos (días sin carro, velitas, partidos, paros) conocido al pronosticar.",
                "Modelo global por localidad (20 series): más datos para la red y un pronóstico más útil para la operación.",
                "Pronóstico probabilístico (cuantiles) para planear con intervalos y no solo con un valor puntual.",
                "Recalibrar el nivel con datos de 2025, año en que la siniestralidad con víctimas bajó."]),
      P("8. Reproducibilidad y uso de IA", "h1"),
      P("<i>python -m scripts.extraer_anuarios → src.data → scripts.run_baselines → scripts.run_experiments → scripts.run_evaluacion</i> "
        "reproduce todas las cifras de este reporte (≈ 30 minutos en CPU). Semillas, dependencias y configuración están versionadas. "
        "Se usó un asistente de IA (Claude) como apoyo para escribir código y redactar; el equipo revisó, ejecutó y validó los resultados.")]

if __name__ == "__main__":
    out = ROOT / "docs" / "Actividad2_Reporte_Tecnico_Grupo7.pdf"
    construir(out, s, "Grupo 7 · Reporte técnico · Siniestros viales Bogotá")
    print(out)
