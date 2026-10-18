"""Compone el informe final (Actividad 3) en Word a partir de las versiones revisadas por el equipo de las
Actividades 1 y 2 (docs/Actividad1_*.docx y docs/Actividad2_*.docx) y agrega las secciones propias del informe final.
Salida: docs/Actividad3_Informe_Final_Grupo7.docx (y PDF con LibreOffice si está disponible)."""
import copy
import json
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.shared import Cm, Pt
from docx.table import Table
from docx.text.paragraph import Paragraph
from docxcompose.composer import Composer

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
A1 = DOCS / "Actividad1_Plan_Experimental_Grupo7.docx"
A2 = DOCS / "Actividad2_Reporte_Tecnico_Grupo7.docx"
SALIDA = DOCS / "Actividad3_Informe_Final_Grupo7.docx"

R = ROOT / "results"
N = json.load(open(R / "dl_metricas.json"))
B = json.load(open(R / "baselines_metricas.json"))


def f(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------- 1. unir Act1 + Act2 conservando imágenes y estilos ----------
base = Document(A1)
comp = Composer(base)
comp.append(Document(A2))
doc = comp.doc
body = doc.element.body
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def bloques():
    out = []
    for el in body.iterchildren():
        if el.tag == W + "p":
            out.append(("p", el, Paragraph(el, doc)))
        elif el.tag == W + "tbl":
            out.append(("t", el, Table(el, doc)))
    return out


def es_titulo(p):
    return p.style.name == "Heading 1" and re.match(r"^(\d+\.\s|Resumen|Referencias)", p.text.strip())


# corregir estilos heredados: párrafos de preprocesamiento con estilo de título y "3. Monitoreo" sin estilo
for k, el, o in bloques():
    if k == "p":
        t = o.text.strip()
        if o.style.name == "Heading 1" and not re.match(r"^(\d+\.\s|Resumen|Referencias)", t):
            if t:
                o.style = doc.styles["List Bullet"]
            else:
                body.remove(el)
        elif o.style.name == "Normal" and re.match(r"^\d+\.\s+Monitoreo", t):
            o.style = doc.styles["Heading 1"]

# partir en secciones por título: clave = (origen, número o nombre)
secs, origen, actual, cabecera = {}, 1, None, {1: [], 2: []}
for k, el, o in bloques():
    if k == "p" and o.text.startswith("Pronóstico de siniestros viales") and actual is not None:
        origen, actual = 2, None
    if k == "p" and es_titulo(o):
        m = re.match(r"^(\d+)\.", o.text.strip())
        actual = (origen, m.group(1) if m else o.text.strip())
        secs[actual] = [el]
    elif actual is None:
        cabecera[origen].append(el)
    else:
        secs[actual].append(el)

# ---------- 2. utilidades para crear contenido nuevo con los estilos del documento ----------
TPL_TABLA = secs[(1, "7")][next(i for i, e in enumerate(secs[(1, "7")]) if e.tag == W + "tbl")]


def par(texto, estilo="Normal", negrita_hasta=None):
    p = doc.add_paragraph(style=estilo)
    if negrita_hasta and texto.startswith(negrita_hasta):
        p.add_run(negrita_hasta).bold = True
        p.add_run(texto[len(negrita_hasta):])
    else:
        p.add_run(texto)
    if estilo != "Heading 1":
        for r in p.runs:
            r.font.name = "Helvetica"
    el = p._p
    body.remove(el)
    return el


def titulo(texto):
    return par(texto, "Heading 1")


def bullet(texto, negrita_hasta=None):
    return par(texto, "List Bullet", negrita_hasta)


def tabla(filas):
    """Clona el formato de una tabla de 2 columnas del documento (encabezado + filas)."""
    t = copy.deepcopy(TPL_TABLA)
    tb = Table(t, doc)
    while len(tb.rows) > 2:
        t.remove(tb.rows[-1]._tr)
    fila_tpl = copy.deepcopy(tb.rows[1]._tr)
    t.remove(tb.rows[1]._tr)
    for _ in filas[1:]:
        t.append(copy.deepcopy(fila_tpl))
    tb = Table(t, doc)
    anchos = [Cm(3.4), Cm(14.2)]
    for col, w in zip(t.findall(W + "tblGrid/" + W + "gridCol"), anchos):
        col.set(W + "w", str(int(w.twips)))
    for row in tb.rows:
        for c, w in zip(row.cells, anchos):
            c.width = w
    for i, fila in enumerate(filas):
        for j, txt in enumerate(fila):
            celda = tb.rows[i].cells[j]
            ps = celda.paragraphs
            for extra in ps[1:]:
                extra._p.getparent().remove(extra._p)
            p = ps[0]
            runs = p.runs
            if runs:
                runs[0].text = txt
                for r in runs[1:]:
                    r._r.getparent().remove(r._r)
            else:
                p.add_run(txt)
    return t


g, l = N["GRU"], N["LSTM"]
rb = B["Ridge con rezagos"]
cfg = g.get("config", {})
pct = lambda a, b: (a - b) / b * 100

# ---------- 3. secciones nuevas ----------
resumen = [
    titulo("Resumen ejecutivo"),
    par("Pronosticamos el número diario de siniestros viales con heridos o muertos en Bogotá para cada uno de los próximos 7 días, "
        "como apoyo a la planeación operativa de movilidad y atención de emergencias. Con las bases del Anuario de Siniestralidad "
        "Vial 2015–2025 (290.184 siniestros) comparamos cuatro líneas base con redes GRU y LSTM, con la misma partición temporal, "
        "el mismo esquema de origen móvil y la pandemia excluida del entrenamiento."),
    bullet(f"Resultado: la GRU, mejor red en validación, obtiene un MAE de {f(g['val_MAE'][0])} en validación (Ridge con rezagos "
           f"{f(rb['val']['MAE'])}, −{f(-pct(g['val_MAE'][0], rb['val']['MAE']), 1)} %) y de {f(g['test_MAE'][0])} ± {f(g['test_MAE'][1])} "
           f"siniestros con víctimas por día en prueba 2024–2025 (Ridge {f(rb['test']['MAE'])}, −{f(-pct(g['test_MAE'][0], rb['test']['MAE']), 1)} %).",
           "Resultado:"),
    bullet("Evidencia: mixta. La LSTM supera a Ridge con significancia (Diebold–Mariano, p = 0,04); la GRU no (p = 0,39); "
           "frente a SARIMAX ninguna red es significativa.", "Evidencia:"),
    bullet("Aprendizaje: el calendario futuro aporta más que la arquitectura (+7,8 % de error sin él); la lluvia observada no aporta "
           "de forma robusta; excluir la pandemia mejora la validación.", "Aprendizaje:"),
    bullet("Datos: los siniestros solo con daños dejaron de registrarse de forma comparable en octubre de 2022; por eso el objetivo "
           "son los siniestros con víctimas.", "Datos:"),
    bullet("Recomendación de uso: apoyo semanal a la planeación con revisión humana; la GRU es la opción más precisa y Ridge una "
           "alternativa simple con desempeño cercano.", "Recomendación de uso:"),
]

ficha = [
    titulo("Ficha del modelo (model card)"),
    tabla([["Campo", "Contenido"],
           ["Modelo", f"GRU de 1 capa (32 unidades, ventana de 28 días, dropout 0,2) + cabeza densa con calendario futuro; "
                      f"{g['n_parametros']:,} parámetros; PyTorch.".replace(",", ".")],
           ["Uso previsto", "Pronóstico de siniestros con heridos o muertos por día en Bogotá a 1–7 días, para dimensionar recursos de tránsito y atención, con revisión humana."],
           ["Usos no previstos", "Asignación automática de recursos, evaluación de la seguridad de una zona o persona, pronóstico por localidad u hora, siniestros solo con daños."],
           ["Entradas", "28 días de historia de conteos, lluvia observada, día de la semana y festivos; calendario de los 7 días futuros."],
           ["Datos de entrenamiento", "Anuario de Siniestralidad Vial (SDM), 2015–2022 sin el periodo 16-mar-2020 a 31-ago-2021; validación 2023."],
           ["Desempeño (prueba)", f"MAE {f(g['test_MAE'][0])} ± {f(g['test_MAE'][1])}; RMSE {f(g['test_RMSE'][0])}; MAPE {f(g['test_MAPE'][0], 1)} % (2024–2025, 3 semillas)."],
           ["Costo", f"Entrenamiento ≈ {f(g['seg_entreno'], 0)} s en CPU; inferencia de 726 orígenes en menos de 1 s."],
           ["Limitaciones", "No anticipa días atípicos (velitas, eventos); sobrestima en 2025, año con menos siniestralidad; más error en días muy lluviosos."],
           ["Mantenimiento", "Monitorear el MAE semanal; reentrenar cada año o ante cambios de política de movilidad o del registro de siniestros."]]),
]

conclusiones = [
    titulo("Resultados, interpretación y conclusiones"),
    par("Resultados. La GRU reduce el MAE frente a Ridge con rezagos en validación (4,96 vs 5,23, −5,1 %) y en prueba "
        "(5,35 vs 5,47, −2,3 %). Ambas redes quedan por debajo de las dos líneas base en los siete horizontes. Sin calendario futuro "
        "el error sube 7,8 %; sin lluvia el efecto es pequeño y de signo cambiante; con la pandemia dentro del entrenamiento la "
        "validación empeora.", negrita_hasta="Resultados."),
    par("Interpretación. La estructura principal del problema (estacionalidad semanal y festivos) la captura bien un modelo lineal "
        "con rezagos. La red agrega una mejora pequeña pero consistente en todos los horizontes, de unos 0,1 siniestros por día, que "
        "solo en un caso alcanza significancia estadística, con un costo de ajuste mayor (rejilla, semillas, parada temprana).",
        negrita_hasta="Interpretación."),
    par("Conclusiones.", negrita_hasta="Conclusiones."),
    bullet("Respuesta a la pregunta experimental: las redes GRU y LSTM predicen con algo más de precisión que SARIMAX y Ridge, "
           "pero la mejora es pequeña y la evidencia estadística es mixta."),
    bullet("La mejora más grande proviene de la información correcta (calendario futuro sin fuga) y de datos consistentes "
           "(objetivo con víctimas, pandemia excluida), no de la complejidad del modelo."),
    bullet("Siguiente paso con más potencial: modelo global por localidad, pronóstico de lluvia como covariable e indicador de eventos."),
]

reproducibilidad = [
    titulo("Reproducibilidad del repositorio"),
    tabla([["Elemento", "Detalle"],
           ["Repositorio", "https://github.com/dialrigo/pronostico-siniestros-bogota-grupo7"],
           ["Estructura", "config/ · src/ · scripts/ · notebooks/ · results/ · figures/ · docs/ · tests/ · README.md · requirements.txt"],
           ["Ejecución", "pip install -r requirements.txt → python -m scripts.extraer_anuarios → python -m src.data → python -m scripts.run_baselines → "
                         "python -m scripts.run_experiments → python -m scripts.run_evaluacion → pytest -q (≈ 30 min en CPU). "
                         "El notebook notebooks/proyecto_siniestros_grupo7.ipynb muestra el flujo completo con resultados."],
           ["Semillas", "Global 42; resultados finales con {42, 7, 2024}, reportados como media ± desviación."],
           ["Datos", "Crudos en data/raw con enlace, licencia y fecha de descarga en el README; parámetros fijos en config/config.yaml."]]),
]

ia = [
    titulo("Declaración de uso de herramientas de IA"),
    par("El equipo usó un asistente de IA (Claude, de Anthropic) como apoyo para escribir código, generar figuras y redactar borradores. "
        "Todas las decisiones de diseño experimental fueron revisadas por el equipo, el código se ejecutó y las cifras de este informe "
        "provienen de los archivos de resultados del repositorio."),
]

# contribución: tabla de roles de la sección 11 de la Actividad 1
roles = [e for e in secs[(1, "11")] if e.tag == W + "tbl"][-1]
contribucion = [titulo("Contribución del equipo"), roles]

# línea base (Act1 §8) sin la figura repetida
s8 = [e for e in secs[(1, "8")] if not (e.tag == W + "p" and (e.xpath(".//w:drawing") or Paragraph(e, doc).text.startswith("Figura")))]

# ---------- 4. orden final ----------
cab = cabecera[1]
for el in cab:
    p = Paragraph(el, doc)
    if p.text.startswith("Actividad 1"):
        for r in p.runs:
            r.text = r.text.replace("Actividad 1 · Diseño experimental y formulación del proyecto integrador",
                                    "Actividad 3 · Informe técnico final del proyecto integrador")
        if "Actividad 1" in p.text:  # el texto podía venir partido en varios runs
            p.runs[0].text = p.text.replace("Actividad 1 · Diseño experimental y formulación del proyecto integrador",
                                            "Actividad 3 · Informe técnico final del proyecto integrador")
            for r in p.runs[1:]:
                r.text = ""

orden = (cab + resumen
         + sum((secs[(1, k)] for k in ["1", "2", "3", "4", "5", "6", "7"]), []) + s8
         + sum((secs[(2, k)] for k in ["1", "2", "3", "4", "5", "6", "7"]), [])
         + ficha + secs[(1, "9")] + conclusiones + reproducibilidad + contribucion + ia
         + secs[(1, "Referencias")])

sect = body.find(W + "sectPr")
for el in list(body):
    if el is not sect:
        body.remove(el)
for el in orden:
    body.insert(len(body) - 1, el)

# renumerar títulos y figuras; quitar párrafos vacíos que sobran después de tablas
n_sec = n_fig = 0
for k, el, o in bloques():
    if k != "p":
        continue
    t = o.text.strip()
    if o.style.name == "Heading 1" and t not in ("Resumen ejecutivo", "Referencias"):
        n_sec += 1
        nuevo = f"{n_sec}. " + re.sub(r"^\d+\.\s*", "", t)
        o.runs[0].text = nuevo
        for r in o.runs[1:]:
            r.text = ""
    elif re.match(r"^Figura \d+\.", t):
        n_fig += 1
        for r in o.runs:
            if re.search(r"Figura\s*\d+", r.text):
                r.text = re.sub(r"Figura\s*\d+", f"Figura {n_fig}", r.text, count=1)
                break
            m = re.match(r"^\s*\d+\.?", r.text)
            if r.text.strip().startswith("Figura") is False and m and n_fig:
                pass

# el texto "Figura 4." de la Act. 1 §8 se eliminó; las referencias cruzadas en el texto no se tocan
doc.core_properties.title = "Informe técnico final – Grupo 7"
doc.save(SALIDA)
print(SALIDA)

try:
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(DOCS), str(SALIDA)],
                   check=True, capture_output=True, timeout=180)
    print(SALIDA.with_suffix(".pdf"))
except Exception as e:  # noqa: BLE001
    print("PDF no generado:", e)
