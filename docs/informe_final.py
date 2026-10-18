"""Informe técnico final consolidado (Actividad 3, 8–12 páginas): reúne el plan (Act. 1) y la ejecución (Act. 2)
y agrega ficha del modelo, ficha de datos, conclusiones, reproducibilidad y contribuciones."""
import json
import re
import sys
from pathlib import Path

from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
import act1_plan as A1  # noqa: E402
import act2_reporte as A2  # noqa: E402
from estilo_pdf import ST, P, bullets, construir, tabla  # noqa: E402

INTEGRANTES = sys.argv[1] if len(sys.argv) > 1 else A1.INTEGRANTES
R = ROOT / "results"
D = json.load(open(R / "dl_metricas.json"))
B = json.load(open(R / "baselines_metricas.json"))
f = A1.f


def secciones(story):
    """Divide una lista de flowables en {numero_o_titulo: [flowables]} usando los encabezados h1."""
    out, actual = {}, None
    for fl in story:
        if isinstance(fl, Paragraph) and fl.style.name == "h1":
            m = re.match(r"^(\d+)\. ", fl.text)
            actual = m.group(1) if m else fl.text
            out[actual] = [fl]
        elif actual is not None:
            out[actual].append(fl)
    return out


S1, S2 = secciones(A1.s), secciones(A2.s)
RED, BASE, OTRA, OB = A2.RED, A2.BASE, A2.OTRA, A2.OB
N, Bb = A2.N, A2.Bb
pct, abl, dm = A2.pct, A2.abl, A2.dm
AU = A1.A
E = A2.E
cfg = N["config"]

s = [P("Pronóstico de siniestros viales con víctimas en Bogotá D.C. con redes recurrentes: informe técnico final", "titulo"),
     P("Actividad 3 · Proyecto integrador · Unidad 4 · Técnicas Avanzadas de Modelado en IA · Maestría en IA · Universidad de La Sabana<br/>"
       f"Grupo 7: {INTEGRANTES} · Octubre de 2026", "sub"),
     P("Resumen ejecutivo", "h1"),
     P("Pronosticamos los siniestros viales con heridos o muertos de Bogotá para cada uno de los próximos 7 días, como apoyo a la "
       f"planeación operativa de movilidad y atención de emergencias. Con las bases del Anuario de Siniestralidad Vial 2015–2025 "
       f"({f(AU['registros'], 0)} siniestros) comparamos cuatro líneas base con redes GRU y LSTM bajo la misma partición temporal, "
       "el mismo esquema de origen móvil y con la pandemia excluida del entrenamiento."),
     *bullets([f"<b>Resultado:</b> la {RED}, mejor red en validación, obtiene un MAE de {f(N['val_MAE'][0])} en validación ({BASE} "
               f"{f(Bb['val']['MAE'])}, −{f(-pct(N['val_MAE'][0], Bb['val']['MAE']), 1)} %) y de {f(N['test_MAE'][0])} ± {f(N['test_MAE'][1])} "
               f"siniestros con víctimas/día en prueba 2024–2025 ({BASE} {f(Bb['test']['MAE'])}, −{f(-pct(N['test_MAE'][0], Bb['test']['MAE']), 1)} %).",
               f"<b>Evidencia:</b> mixta. La {OTRA} supera a {BASE} con significancia (Diebold–Mariano, p = {f(dm(OTRA, BASE).p_h1a7, 2)}); "
               f"la {RED} no (p = {f(dm(RED, BASE).p_h1a7, 2)}); frente a {OB} ninguna red es significativa.",
               f"<b>Aprendizaje:</b> el calendario futuro aporta más que la arquitectura (+{f(pct(abl(RED, 'sin calendario futuro'), N['test_MAE'][0]), 1)} % "
               "de error sin él); la lluvia observada no aporta de forma robusta; excluir la pandemia mejora la validación.",
               "<b>Datos:</b> los siniestros solo con daños dejaron de registrarse de forma comparable en octubre de 2022; por eso el objetivo son los siniestros con víctimas.",
               f"<b>Recomendación de uso:</b> apoyo semanal a la planeación con revisión humana; la {RED} es la opción más precisa y la Ridge una alternativa simple con desempeño cercano."])]

# Formulación y datos (Actividad 1)
for k in ["1", "2", "3", "4", "5", "6", "7", "8"]:
    s += S1[k]
# Ejecución (Actividad 2)
for k in ["1", "2", "3", "4", "5", "6"]:
    s += S2[k]

# Ficha del modelo
s += [P("Ficha del modelo (model card)", "h1"),
      tabla([["Campo", "Contenido"],
             ["Modelo", f"{RED} de 1 capa ({cfg['hidden']} unidades, ventana de {cfg['W']} días, dropout {f(cfg['dropout'], 1)}) + cabeza densa con calendario futuro; "
                        f"{N['n_parametros']:,} parámetros; PyTorch.".replace(",", ".")],
             ["Uso previsto", "Pronóstico de siniestros con heridos o muertos por día en Bogotá a 1–7 días, para dimensionar recursos de tránsito y atención con revisión humana."],
             ["Usos no previstos", "Decisiones automáticas de asignación, evaluación de la seguridad de una zona o persona, pronóstico por localidad u hora, siniestros solo con daños."],
             ["Entradas", f"{cfg['W']} días de historia de conteos, lluvia observada, día de la semana y festivos; calendario de los 7 días futuros."],
             ["Datos de entrenamiento", "Anuario de Siniestralidad Vial (SDM), 2015–2022 sin el periodo 16-mar-2020 a 31-ago-2021; validación 2023."],
             ["Desempeño (prueba)", f"MAE {f(N['test_MAE'][0])} ± {f(N['test_MAE'][1])}; RMSE {f(N['test_RMSE'][0])}; MAPE {f(N['test_MAPE'][0], 1)} % (2024–2025, 3 semillas)."],
             ["Costo", f"Entrenamiento ≈ {f(N['seg_entreno'], 0)} s en CPU; inferencia de 726 orígenes &lt; 1 s."],
             ["Limitaciones", f"No anticipa días atípicos (velitas, eventos); sobrestima en 2025, año con menos siniestralidad; mayor error en días muy lluviosos."],
             ["Mantenimiento", "Monitorear el MAE semanal; reentrenar cada año o ante cambios de política de movilidad o del registro de siniestros."]],
            [3.2, 13.8])]
# Riesgos (Actividad 1, sección 9)
s += S1["9"]

# Conclusiones diferenciando resultados, interpretación y conclusiones
s += [P("Resultados, interpretación y conclusiones", "h1"),
      P("Resultados", "h2"),
      P(f"La {RED} reduce el MAE frente a {BASE} en validación ({f(N['val_MAE'][0])} vs {f(Bb['val']['MAE'])}, −{f(-pct(N['val_MAE'][0], Bb['val']['MAE']), 1)} %) "
        f"y en prueba ({f(N['test_MAE'][0])} vs {f(Bb['test']['MAE'])}, −{f(-pct(N['test_MAE'][0], Bb['test']['MAE']), 1)} %). Ambas redes quedan por "
        f"debajo de las dos líneas base en los siete horizontes. Sin calendario futuro el error sube {f(pct(abl(RED, 'sin calendario futuro'), N['test_MAE'][0]), 1)} %; "
        "sin lluvia el efecto es pequeño y de signo cambiante; con la pandemia dentro del entrenamiento la validación empeora."),
      P("Interpretación", "h2"),
      P("La estructura principal del problema (estacionalidad semanal y festivos) la captura bien un modelo lineal con rezagos. "
        "La red agrega una mejora pequeña pero consistente en todos los horizontes, de unos 0,1 siniestros por día, que solo en un "
        "caso alcanza significancia estadística. Su costo de ajuste (rejilla, semillas, parada temprana) es mayor."),
      P("Conclusiones", "h2"),
      *bullets(["Respondiendo la pregunta experimental: la red recurrente reduce el error frente a SARIMAX y Ridge, pero la mejora es pequeña y la evidencia estadística es mixta.",
                "La mejora más grande proviene de la información correcta (calendario futuro sin fuga) y de datos consistentes (objetivo con víctimas, pandemia excluida), no de la complejidad del modelo.",
                "El siguiente paso con mayor potencial es un modelo global por localidad, pronóstico de lluvia como covariable y un indicador de eventos."])]

# Reproducibilidad
s += [P("Reproducibilidad del repositorio", "h1"),
      tabla([["Elemento", "Detalle"],
             ["Estructura", "config/ · src/ · scripts/ · notebooks/ · results/ · figures/ · docs/ · tests/ · README.md · requirements.txt"],
             ["Ejecución", "pip install -r requirements.txt → python -m scripts.extraer_anuarios → python -m src.data → python -m scripts.run_baselines → python -m scripts.run_experiments → python -m scripts.run_evaluacion → pytest -q"],
             ["Semillas", "Global 42; finales {42, 7, 2024}; permutación de lotes con generador sembrado."],
             ["Dependencias", "Python 3.13; pandas 3.0, numpy 2.5, scikit-learn 1.9, statsmodels 0.15, torch 2.14, holidays 0.106 (versiones fijadas)."],
             ["Datos", "Crudos en data/raw con enlace, licencia y fecha de descarga en el README."],
             ["Repositorio", "https://github.com/dialrigo/pronostico-siniestros-bogota-grupo7"]],
            [3.0, 14.0])]
s += [P("Contribución del equipo", "h1"), S1["11"][-1]]  # tabla de roles
s += [P("Declaración de uso de herramientas de IA", "h1"),
      P("El equipo usó un asistente de IA (Claude, de Anthropic) como apoyo para escribir código, generar figuras y redactar borradores. "
        "Todas las decisiones de diseño experimental fueron revisadas por el equipo, el código se ejecutó y las cifras de este informe "
        "provienen de los archivos de resultados del repositorio.")]
s += S1["Referencias"]

# renumerar secciones y figuras
final, n_sec, n_fig = [], 0, 0
for fl in s:
    if isinstance(fl, Paragraph) and fl.style.name == "h1" and fl.text not in ("Resumen ejecutivo", "Referencias"):
        n_sec += 1
        fl = Paragraph(f"{n_sec}. " + re.sub(r"^\d+\. ", "", fl.text), ST["h1"])
    elif isinstance(fl, Paragraph) and fl.style.name == "cap" and re.match(r"^Figura \d+\.", fl.text):
        n_fig += 1
        fl = Paragraph(re.sub(r"^Figura \d+\.", f"Figura {n_fig}.", fl.text), ST["cap"])
    final.append(fl)

out = ROOT / "docs" / "Actividad3_Informe_Final_Grupo7.pdf"
construir(out, final, "Grupo 7 · Informe final · Siniestros viales Bogotá")
print(out)
