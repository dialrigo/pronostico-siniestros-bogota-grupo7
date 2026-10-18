"""Genera el PDF de la Actividad 1: plan experimental del proyecto integrador."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))
from estilo_pdf import P, Spacer, bullets, construir, figura, tabla  # noqa: E402

A = json.load(open(ROOT / "results" / "auditoria.json"))
B = json.load(open(ROOT / "results" / "baselines_metricas.json"))
BI = json.load(open(ROOT / "results" / "baselines_metricas_indicador.json"))
FIG = ROOT / "figures"
INTEGRANTES = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "Diego Alejandro Ríos Gómez · Marlon [apellidos] · Nicolás [apellidos] · Nataly [apellidos]"


def f(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


g = A["gravedad"]
vic = g["Con Heridos"] + g["Con Muertos"]
pa = A["por_anio_gravedad"]
vpd = A["victimas_por_dia_por_anio"]
n_tr_orig = B["_info"]["origenes_train_ridge"]

s = []
s += [P("Pronóstico de siniestros viales con víctimas en Bogotá D.C. con redes recurrentes", "titulo"),
      P("Actividad 1 · Diseño experimental y formulación del proyecto integrador · Unidad 4<br/>"
        "Técnicas Avanzadas de Modelado en IA · Maestría en Inteligencia Artificial · Universidad de La Sabana<br/>"
        f"Grupo 7: {INTEGRANTES} · Octubre de 2026", "sub")]

# 1. Problema
s += [P("1. Contexto, problema y alcance", "h1"),
      P(f"Bogotá registra entre 30 y 36 siniestros viales <b>con heridos o muertos</b> por día (2024: {f(vpd['2024'], 1)}; "
        f"2025: {f(vpd['2025'], 1)}). Son los eventos que movilizan ambulancias, agentes y equipos de atención, y los que "
        "persigue la política distrital de Visión Cero. La Secretaría Distrital de Movilidad y la Policía de Tránsito "
        "programan esos recursos con días de anticipación, hoy con promedios históricos. <b>Problema:</b> pronosticar el número "
        "de siniestros con víctimas en Bogotá para cada uno de los próximos 7 días."),
      P("<b>Usuario y contexto de uso:</b> analistas de planeación operativa de movilidad y atención de emergencias (uso "
        "agregado de ciudad). <b>Alcance:</b> serie diaria de la ciudad, horizonte 1–7 días, pronóstico puntual. "
        "<b>Fuera de alcance:</b> siniestros solo con daños (su registro cambió en 2022, sección 4), pronóstico por localidad u "
        "hora, causalidad y asignación automática de recursos."),
      P("<b>Pregunta experimental.</b> Bajo la misma partición temporal y la misma información disponible en cada origen de "
        "pronóstico, ¿una red recurrente (GRU o LSTM) que combina la historia reciente con el calendario futuro reduce el MAE "
        "a 1–7 días frente a SARIMAX y a una regresión Ridge con rezagos? Preguntas secundarias, un factor a la vez: ¿aporta "
        "la lluvia observada?, ¿aporta el calendario futuro?, ¿conviene excluir la pandemia del entrenamiento?")]

# 2. Datos
s += [P("2. Procedencia, licencia y composición de los datos (datasheet)", "h1"),
      tabla([["Atributo", "Siniestros (variable objetivo)", "Clima diario"],
             ["Fuente", "Anuario de Siniestralidad Vial de Bogotá, Secretaría Distrital de Movilidad (SDM), Observatorio de "
              "Movilidad. Bases en Excel construidas desde el SIGAT con los informes policiales (IPAT): la base 2024 consolida "
              "2015–2024 y la base 2025 el año 2025.", "Open-Meteo Historical Weather API (reanálisis ERA5), punto 4,60 N; −74,05 O; 2.586 m."],
             ["Licencia", "Datos abiertos publicados por la SDM; la página de descarga no declara una licencia explícita. Se usan con "
              "atribución a la SDM y no se redistribuyen direcciones ni coordenadas.", "CC BY 4.0 (atribución: “Weather data by Open-Meteo.com”)."],
             ["Composición", f"{f(A['registros'], 0)} siniestros únicos (hoja “Siniestros”): fecha, hora, localidad ({A['localidades']}), "
              f"clase y gravedad a 30 días: solo daños {f(g['Solo Daños'], 0)}, con heridos {f(g['Con Heridos'], 0)}, con muertos "
              f"{f(g['Con Muertos'], 0)}. Se usan {f(vic, 0)} siniestros con víctimas.",
              "4.018 días (2015–2025): precipitación diaria (mm) y temperatura media (°C). Sin nulos."],
             ["Cobertura", f"{A['fecha_min']} a {A['fecha_max']}; {f(A['dias_rango'], 0)} días, ninguno sin registro.", "Misma cobertura."],
             ["Limitaciones", "Solo siniestros con informe policial (subregistro de eventos leves); los solo daños dejan de "
              "registrarse en forma comparable desde oct-2022; la base 2025 solo incluye siniestros con víctimas; 0,05 % sin gravedad.",
              "Es un punto de reanálisis, no una estación: suaviza la lluvia localizada."]],
            [2.4, 8.6, 6.0]),
      P("La primera versión del proyecto usó el “Histórico Siniestros Bogotá D.C.” de Datos Abiertos Bogotá, que no se actualiza "
        "desde octubre de 2021. Se reemplazó por las bases del Anuario, que llegan a diciembre de 2025.", "p")]

# 3. Objetivo y predictores
s += [P("3. Variable objetivo y predictores", "h1"),
      *bullets(["<b>Objetivo:</b> y<sub>t+h</sub> = siniestros con heridos o muertos (indicador de gravedad a 30 días) del día t+h, "
                "h = 1…7 (pronóstico multi-horizonte directo).",
                "<b>Historia (hasta el día t):</b> conteos rezagados (ventana de 14 o 28 días), lluvia observada, festivos.",
                "<b>Calendario futuro (conocido de antemano, sin fuga):</b> día de la semana, festivo (calendario oficial "
                "colombiano), lunes festivo (puente), víspera de festivo, días de quincena y diciembre.",
                "<b>Lluvia futura:</b> no se usa, porque no se conoce al pronosticar; solo la lluvia ya observada."])]
s += figura(FIG / "fig2_perfil_semanal.png", 8.2,
            "Figura 1. El calendario domina la señal: ~34 siniestros con víctimas entre semana, 24 el domingo y 20 en festivo. "
            "La correlación diaria con la lluvia es casi nula (r ≈ −0,04), por lo que su aporte se probará en vez de suponerse.")

# 4. Auditoría
s += [P("4. Auditoría inicial de calidad y decisiones", "h1")]
s += figura(FIG / "fig0_quiebre_registro.png", 15.0,
            "Figura 2. Desde octubre de 2022 los siniestros solo con daños caen a ~3 por día, mientras los siniestros con víctimas "
            "se mantienen estables: es un cambio en el registro, no en la siniestralidad.")
s += [tabla([["Verificación", "Hallazgo", "Decisión"],
             ["Cambio de registro", f"Solo daños: {f(pa['2021']['Solo Daños'], 0)} en 2021, {f(pa['2023']['Solo Daños'], 0)} en 2023; "
              f"con víctimas: {f(vpd['2021'], 1)} → {f(vpd['2023'], 1)} por día.", "Objetivo = siniestros con víctimas; el total no es comparable en el tiempo."],
             ["Duplicados", "Códigos únicos en cada base; 3 códigos aparecen en las bases 2024 y 2025.", "Se conserva la primera aparición."],
             ["Continuidad", f"{A['dias_sin_registro']} días sin registro y {A['dias_sin_victimas']} días sin víctimas en 4.018.", "No requiere imputación."],
             ["Gravedad", "Dos indicadores (tradicional y a 30 días) difieren en 88 de 118.536 siniestros con víctimas (0,07 %); 2025 solo trae el de 30 días.",
              "Se usa el indicador a 30 días en todos los años."],
             ["Pandemia", f"Mar-2020 a ago-2021: promedio de 11 víctimas/día en abr-2020, cuarentenas y paro; el nivel normal (~36) vuelve en sep-2021.",
              f"Periodo atípico ({B['_info']['dias_atipicos']} días) excluido del entrenamiento; control con indicador."],
             ["Nivel por año", "2015–2017 ≈ 30/día; 2018–2019 y 2022–2024 ≈ 35/día; 2025 = 31,7/día.", "Se reporta; la prueba (2024–2025) incluye esa baja."]],
            [2.8, 8.0, 6.2])]

# 5. Partición
s += [P("5. Esquema de partición y prevención de fuga de información", "h1"),
      P("La unidad sobre la que se quiere generalizar es el <b>intervalo temporal futuro</b>, así que la partición es cronológica "
        "y sin mezcla: el pasado entrena y el futuro evalúa. Validación y prueba son años completos, por lo que incluyen todas "
        "las estaciones, festivos y diciembre."),
      tabla([["Conjunto", "Periodo", "Días", "Uso"],
             ["Entrenamiento", "2015-01-01 a 2022-12-31", "2.922", f"Ajuste de parámetros y escaladores. Sin el periodo atípico: {f(n_tr_orig, 0)} orígenes útiles."],
             ["Validación", "2023-01-01 a 2023-12-31", "365", "Parada temprana, selección de hiperparámetros y de modelo."],
             ["Prueba", "2024-01-01 a 2025-12-31", "731", "Evaluación final; se usa una sola vez, al terminar."]],
            [2.8, 4.2, 1.3, 8.7])]
s += figura(FIG / "fig1_serie_split.png", 16.5, "Figura 3. Serie diaria con víctimas, media móvil de 28 días, periodo atípico excluido y partición temporal.")
s += bullets(["<b>Origen móvil:</b> en cada día t de validación o prueba se pronostican t+1…t+7 solo con datos hasta t "
              "(360 orígenes en validación y 726 en prueba). Todos los modelos usan exactamente los mismos orígenes.",
              "<b>Pandemia:</b> se descartan los orígenes de entrenamiento cuya ventana de entrada u objetivos tocan el 16-mar-2020 "
              "a 31-ago-2021; en SARIMAX esos días se tratan como faltantes en el filtro de Kalman.",
              "Normalización calculada <b>solo con entrenamiento</b> (sin el periodo atípico).",
              "Ningún rasgo usa información posterior al origen, excepto el calendario, que es conocido de antemano.",
              "Selección de hiperparámetros y de modelo <b>solo con validación</b>; la prueba no se mira hasta el final."])

# 6. Preprocesamiento
s += [P("6. Preprocesamiento", "h1"),
      P("Extracción de la hoja de siniestros de cada base (<i>scripts/extraer_anuarios.py</i>, sin dirección ni coordenadas); "
        "unión 2015–2024 + 2025 y deduplicación por código; conteo diario de siniestros con víctimas; transformación log1p "
        "(estabiliza la varianza); estandarización con estadísticos de entrenamiento; seno/coseno del día de la semana en la "
        "secuencia y one-hot en el calendario futuro; ventanas deslizantes (W = 14 o 28 días). Festivos con la librería <i>holidays</i>.")]

# 7. Métricas
s += [P("7. Métricas de éxito", "h1"),
      tabla([["Métrica", "Por qué"],
             ["MAE (principal)", "Se interpreta directamente en siniestros con víctimas por día, la unidad con la que se dimensionan recursos."],
             ["RMSE", "Penaliza más los días con error grande, los operativamente costosos (quedarse corto un día pico)."],
             ["MAPE", "Error relativo para usuarios no técnicos; estable porque el conteo diario casi nunca baja de 10."],
             ["RMSLE", "Error en escala logarítmica, coherente con la pérdida de entrenamiento (MSE sobre log1p)."]],
            [3.2, 13.8]),
      P("<b>Criterio de éxito:</b> el modelo avanzado es mejor si reduce el MAE de validación en al menos 5 % frente a la mejor "
        "línea base, la mejora se mantiene en prueba (promedio de 3 semillas) y la prueba de Diebold–Mariano sobre los errores "
        "absolutos es significativa (p &lt; 0,05). Se reporta también el costo: tiempo de entrenamiento e inferencia y parámetros.")]

# 8. Línea base
orden = ["Naive estacional (t-7)", "Media estacional (4 sem.)", "Ridge con rezagos", "SARIMAX"]
desc = {"Naive estacional (t-7)": "Repite el valor del mismo día de la semana anterior.",
        "Media estacional (4 sem.)": "Promedio de las 4 semanas previas para el mismo día de la semana.",
        "Ridge con rezagos": "Un modelo lineal por horizonte: 28 rezagos de log1p(y) + calendario del día objetivo.",
        "SARIMAX": "SARIMAX(2,0,1)(1,0,1)₇ en log1p(y) con festivo, puente, víspera y diciembre como exógenas."}
filas = [["Línea base", "Descripción", "MAE val", "MAE prueba", "RMSE prueba", "MAPE prueba"]]
for m in orden:
    filas.append([m, desc[m], f(B[m]["val"]["MAE"]), f(B[m]["test"]["MAE"]), f(B[m]["test"]["RMSE"]), f(B[m]["test"]["MAPE"], 1) + " %"])
mejor = min(["Ridge con rezagos", "SARIMAX"], key=lambda m: B[m]["val"]["MAE"])
s += [P("8. Línea base implementada y primeras métricas", "h1"),
      P("Las cuatro líneas base están implementadas y son reproducibles (<i>python -m scripts.run_baselines</i>), con el mismo "
        "esquema de origen móvil a 1–7 días."),
      tabla(filas, [3.3, 7.0, 1.6, 1.7, 1.7, 1.7])]
s += figura(FIG / "fig3_baselines.png", 9.0, "Figura 4. MAE en prueba de las líneas base. Los modelos con calendario superan a las heurísticas.")
s += [P(f"<b>Lectura.</b> Ridge ({f(B['Ridge con rezagos']['val']['MAE'])}) y SARIMAX ({f(B['SARIMAX']['val']['MAE'])}) están "
        f"prácticamente empatados en validación y mejoran a la heurística naive en ~30 %. La mejor línea base en validación es "
        f"{mejor}; el umbral de éxito del modelo avanzado es un MAE de validación ≤ {f(B[mejor]['val']['MAE'] * 0.95)}. "
        f"<b>Control de pandemia:</b> con el periodo atípico dentro del entrenamiento (y un indicador), el MAE de prueba empeora en "
        f"ambas líneas base (Ridge {f(BI['Ridge con rezagos']['test']['MAE'])} vs {f(B['Ridge con rezagos']['test']['MAE'])}; SARIMAX "
        f"{f(BI['SARIMAX']['test']['MAE'])} vs {f(B['SARIMAX']['test']['MAE'])}), lo que respalda excluirlo.")]

# 9. Riesgos
s += [P("9. Riesgos de sesgo, privacidad, seguridad y uso indebido", "h1"),
      tabla([["Riesgo", "Descripción", "Mitigación"],
             ["Sesgo de medición", "Solo hay eventos con informe policial; el registro cambió en 2022 y puede volver a cambiar.",
              "Objetivo restringido a siniestros con víctimas, el más estable; monitorear el registro y declararlo."],
             ["Cambio de régimen", "Pandemia, paros y medidas de movilidad cambian la relación histórica.",
              "Periodo atípico excluido, validación en 2023, monitoreo del error y reentrenamiento."],
             ["Privacidad", "Las bases incluyen direcciones, coordenadas y, en otras hojas, edad y sexo de víctimas.",
              "Solo se extraen fecha, hora, localidad, clase y gravedad; se publican conteos diarios agregados."],
             ["Uso indebido", "Asignar recursos solo con el pronóstico podría desatender días atípicos (eventos masivos, paros).",
              "Uso como apoyo a la decisión, con error reportado y revisión humana."],
             ["Seguridad", "Repositorio público.", "Sin credenciales ni rutas locales; dependencias fijadas en <i>requirements.txt</i>."]],
            [2.6, 7.2, 7.2]),
      P("La clasificación sigue las funciones del NIST AI RMF 1.0 (mapear, medir, gestionar).", "cap")]

# 10. Arquitectura
s += [P("10. Arquitectura candidata", "h1"),
      P("<b>Red recurrente con cabeza de calendario futuro</b> (PyTorch): un codificador GRU o LSTM de una capa lee la "
        "ventana pasada (log1p(y) estandarizado, seno/coseno del día de la semana, festivo y lluvia observada). Su último estado "
        "oculto se concatena con el calendario de los 7 días a pronosticar y pasa por una capa densa (ReLU + dropout) que "
        "entrega los 7 valores a la vez. Pérdida MSE en escala log, AdamW (weight decay 1e-4), recorte de gradiente, "
        "reducción de la tasa en meseta y parada temprana por pérdida de validación."),
      P("<b>Experimentos controlados (un factor a la vez):</b> E1 GRU vs LSTM con la misma rejilla; E2 rejilla: ventana {14, 28}, "
        "unidades {32, 64}, tasa {0,001; 0,003}, dropout {0; 0,2} (16 configuraciones por arquitectura); E3 sin lluvia; E4 sin "
        "calendario futuro; E5 con la pandemia dentro del entrenamiento (indicador); E6 robustez con 3 semillas; E7 "
        "Diebold–Mariano frente a las líneas base. Las arquitecturas se comparan con el mismo criterio de parada, no con el mismo número de épocas.")]

# 11. Cronograma y roles
s += [P("11. Cronograma y roles", "h1"),
      tabla([["Fecha", "Hito", "Entregable"],
             ["18-oct", "Formulación, auditoría, cambio de fuente y líneas base", "Actividad 1 (este documento)"],
             ["19 a 21-oct", "Rejilla GRU/LSTM, ablaciones y curvas de aprendizaje", "Notebook / repositorio"],
             ["22 a 23-oct", "Comparación en prueba, Diebold–Mariano, análisis de errores", "Reporte técnico (Actividad 2)"],
             ["24-oct", "Informe consolidado, ficha del modelo, presentación y ensayo", "Informe y presentación"],
             ["25-oct", "Entrega final y sustentación", "Actividades 2 y 3"]],
            [2.4, 8.6, 6.0]),
      Spacer(1, 4),
      tabla([["Integrante", "Rol principal", "Responsabilidades"],
             ["Diego Ríos", "Líder técnico y datos", "Formulación, auditoría, pipeline de datos, integración del informe."],
             ["Marlon", "Modelos clásicos", "Líneas base, SARIMAX, Ridge, prueba de Diebold–Mariano."],
             ["Nicolás", "Modelos profundos", "GRU/LSTM, rejilla de hiperparámetros, curvas de aprendizaje."],
             ["Nataly", "Evaluación y comunicación", "Análisis de errores, riesgos y ética, ficha del modelo, presentación."]],
            [2.6, 3.6, 10.8])]

# 12. Reproducibilidad
s += [P("12. Criterios de reproducibilidad", "h1"),
      *bullets(["Repositorio con estructura separada: <i>config/</i> (parámetros fijos en <i>config.yaml</i>, incluido el periodo "
                "atípico), <i>src/</i>, <i>scripts/</i>, <i>notebooks/</i>, <i>results/</i>, <i>figures/</i>, <i>tests/</i>, "
                "<i>README.md</i> y <i>requirements.txt</i>.",
                "Semilla global 42; resultados finales con semillas {42, 7, 2024}; se reportan media y desviación.",
                "Mismos orígenes de pronóstico y misma partición para todos los modelos; métricas y predicciones en <i>results/</i>.",
                "Ejecución: <i>scripts.extraer_anuarios</i> → <i>src.data</i> → <i>scripts.run_baselines</i> → "
                "<i>scripts.run_experiments</i> → <i>scripts.run_evaluacion</i>.",
                "Datos crudos descargables de sus fuentes oficiales; el README indica enlaces, licencia y fecha de descarga.",
                "Uso de herramientas de IA: se usó un asistente de IA (Claude) como apoyo en la escritura de código y la redacción; "
                "el equipo revisó y validó los resultados."])]

s += [P("Referencias", "h1"),
      *[P(r, "ref") for r in [
          "Secretaría Distrital de Movilidad. Anuario de Siniestralidad Vial de Bogotá 2024 y 2025 (bases en Excel). Observatorio de Movilidad de Bogotá. https://observatorio.movilidadbogota.gov.co/publicaciones/anuarios",
          "Zippenfenig, P. (2023). Open-Meteo.com Weather API. https://open-meteo.com",
          "Gebru, T. et al. (2021). Datasheets for Datasets. Communications of the ACM, 64(12), 86–92.",
          "Zinkevich, M. Rules of Machine Learning: Best Practices for ML Engineering. Google.",
          "Scikit-learn Developers. Common pitfalls and recommended practices.",
          "NIST (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0). NIST AI 100-1.",
          "Diebold, F. X. y Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics, 13(3), 253–263."]]]

if __name__ == "__main__":
    out = ROOT / "docs" / "Actividad1_Plan_Experimental_Grupo7.pdf"
    construir(out, s, "Grupo 7 · Plan experimental · Siniestros viales Bogotá")
    print(out)
