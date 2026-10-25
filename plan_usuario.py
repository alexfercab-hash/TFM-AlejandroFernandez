"""
Traduce el resultado de la optimización en un plan que el usuario puede seguir sin
domótica ni automatización: qué electrodoméstico usar, en qué franja horaria y cómo
manejar la climatización, día a día, para conseguir el ahorro calculado.

Entrada : resultados del MILP y del baseline (CSV que genera main.py).
Salida  : resultados/plan_usuario.json  (lo usa el dashboard)
          resultados/plan_usuario.md    (versión legible/imprimible)

Uso:
    python plan_usuario.py
"""

import json
import os

import numpy as np
import pandas as pd

DT_HORAS = 0.25
UMBRAL_FUNCIONANDO_KW = 0.15   # por debajo de esta potencia se considera que la climatización "no trabaja" (solo consumo residual)


def potencias_max_clima(parametros: dict) -> float:
    return parametros["potencia_max_climatizacion_kW"]
NOMBRE_PERIODO = {"P1": "punta", "P2": "llano", "P3": "valle"}
HORAS_PERIODO_LABORABLE = {
    "P3": "de 00:00 a 08:00",
    "P2": "de 08:00 a 10:00, de 14:00 a 18:00 y de 22:00 a 24:00",
    "P1": "de 10:00 a 14:00 y de 18:00 a 22:00",
}
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

CARGAS = {
    "Lavadora": "estado_lavadora",
    "Lavavajillas": "estado_lavavajillas",
    "Calentador de agua": "estado_calentador_agua",
}

REGLAS = [
    {"titulo": "Electrodomésticos en horas valle",
     "texto": "Usa la lavadora, el lavavajillas y el calentador de agua entre las 00:00 y las 08:00 "
              "(días laborables). Sábados, domingos y festivos nacionales todo el día es valle: da igual la hora."},
    {"titulo": "Mismos usos, distinta hora",
     "texto": "No tienes que usar menos los aparatos: haz los mismos usos de siempre, pero programados "
              "para la franja valle del mismo día."},
    {"titulo": "Climatización fuera de la punta",
     "texto": "Evita la climatización en las horas punta (10-14 h y 18-22 h laborables) y concéntrala en las "
              "horas llano (8-10, 14-18 y 22-24 h). Esta parte supone que tu vivienda mantiene la temperatura "
              "unas horas; ajústalo a tu confort."},
]

COMO_APLICARLO = [
    {"aparato": "Lavadora y lavavajillas",
     "como": "Usa el INICIO DIFERIDO del propio electrodoméstico: lo cargas por la tarde y programas que "
             "arranque de madrugada. Si no lo tiene, un enchufe con temporizador programable."},
    {"aparato": "Calentador de agua (termo)",
     "como": "Un programador horario o un enchufe temporizador que lo active solo en la franja valle."},
    {"aparato": "Climatización",
     "como": "La programación horaria de tu termostato o aire acondicionado: que trabaje en las horas llano "
             "y la deje de forzar durante la punta."},
]


def _rangos(df: pd.DataFrame) -> list:
    """Agrupa intervalos consecutivos de 15 min en rangos 'HH:MM-HH:MM (franja)'."""
    if df.empty:
        return []
    filas = list(df.sort_values("timestamp")[["timestamp", "periodo_tarifario"]].itertuples(index=False))
    rangos, (ini, per), ant = [], filas[0], filas[0][0]
    for t, p in filas[1:]:
        if (t - ant) == pd.Timedelta(minutes=15) and p == per:
            ant = t
            continue
        rangos.append((ini, ant, per))
        ini, per, ant = t, p, t
    rangos.append((ini, ant, per))
    return [f"{a.strftime('%H:%M')}-{(b + pd.Timedelta(minutes=15)).strftime('%H:%M')} ({NOMBRE_PERIODO[p]})"
            for a, b, p in rangos]


def _bloques(df: pd.DataFrame) -> list:
    """Bloques consecutivos de intervalos de 15 min como [{'inicio': 'HH:MM', 'fin': 'HH:MM'}]."""
    if df.empty:
        return []
    ts = list(df.sort_values("timestamp")["timestamp"])
    bloques, ini, ant = [], ts[0], ts[0]
    for t in ts[1:]:
        if (t - ant) == pd.Timedelta(minutes=15):
            ant = t
            continue
        bloques.append((ini, ant)); ini = ant = t
    bloques.append((ini, ant))
    return [{"inicio": a.strftime("%H:%M"), "fin": (b + pd.Timedelta(minutes=15)).strftime("%H:%M")} for a, b in bloques]


def _tipo_dia(g: pd.DataFrame) -> str:
    if g["timestamp"].iloc[0].dayofweek >= 5:
        return "fin de semana"
    if (g["periodo_tarifario"] == "P3").all():
        return "festivo nacional"
    return "laborable"


CLAVE_PARAMETRO = {"Lavadora": "lavadora", "Lavavajillas": "lavavajillas", "Calentador de agua": "calentador_agua"}
ID_CARGA = {"Lavadora": "lavadora", "Lavavajillas": "lavavajillas", "Calentador de agua": "calentador_agua"}


def generar_plan(ruta_milp: str, ruta_baseline: str, ruta_parametros: str = None) -> dict:
    parametros = None
    if ruta_parametros and os.path.exists(ruta_parametros):
        with open(ruta_parametros, encoding="utf-8") as f:
            parametros = json.load(f)
    potencias = parametros["potencias_nominales_kW"] if parametros else None
    m = pd.read_csv(ruta_milp, parse_dates=["timestamp"])
    b = pd.read_csv(ruta_baseline, parse_dates=["timestamp"])
    b["periodo_tarifario"] = m["periodo_tarifario"].values
    b["precio"] = m["precio_energia_eur_kWh"].values
    m["precio"] = m["precio_energia_eur_kWh"]
    m["dia"] = m["timestamp"].dt.date
    b["dia"] = b["timestamp"].dt.date

    coste_b = (b["consumo_total_kW"] * DT_HORAS * b["precio"]).sum()
    coste_m = (m["consumo_total_kW"] * DT_HORAS * m["precio"]).sum()
    ahorro_hvac = ((b["P_climatizacion_kW"] - m["P_climatizacion_kW"]) * DT_HORAS * m["precio"]).sum()
    ahorro_total = coste_b - coste_m

    dias = []
    for dia, gm in m.groupby("dia"):
        gb = b[b["dia"] == dia]
        tipo = _tipo_dia(gm)
        fecha = pd.Timestamp(dia)
        cargas = []
        for nombre, col in CARGAS.items():
            usos = int((gm[col] > 0.5).sum())
            antes = gb[gb[col] > 0.5]
            despues = gm[gm[col] > 0.5]
            if usos == 0:
                recomendacion = "No se usa este día."
            elif tipo != "laborable":
                recomendacion = "Sin cambios: todo el día es valle, a cualquier hora."
            else:
                periodos = [p for p in ["P3", "P2", "P1"] if (despues["periodo_tarifario"] == p).any()]
                recomendacion = " y ".join(
                    f"{HORAS_PERIODO_LABORABLE[p]} ({NOMBRE_PERIODO[p]})" for p in periodos)
                recomendacion = ("Prográmala " if nombre == "Lavadora" else "Prográmalo ") + recomendacion + "."
            ahorro_dia = None
            if potencias:
                p = potencias[CLAVE_PARAMETRO[nombre]]
                ahorro_dia = round(float(((gb[col].to_numpy() - gm[col].to_numpy()) * p * DT_HORAS * gm["precio"].to_numpy()).sum()), 4)
            cargas.append({
                "id": ID_CARGA[nombre], "ahorro_eur": ahorro_dia,
                "nombre": nombre, "usos_de_15_min": usos,
                "antes": _rangos(antes),
                "bloques_recomendados": _bloques(despues),
                "periodos_antes": sorted({NOMBRE_PERIODO[p] for p in antes["periodo_tarifario"]}),
                "recomendacion": recomendacion,
            })

        kwh_b = (gb.groupby("periodo_tarifario")["P_climatizacion_kW"].sum() * DT_HORAS).round(2).to_dict()
        kwh_m = (gm.groupby("periodo_tarifario")["P_climatizacion_kW"].sum() * DT_HORAS).round(2).to_dict()
        punta_antes, punta_despues = kwh_b.get("P1", 0.0), kwh_m.get("P1", 0.0)
        if tipo == "laborable" and punta_antes - punta_despues > 0.05:
            texto_clima = (f"Evita la climatización en punta: de {punta_antes:.1f} kWh a {punta_despues:.1f} kWh. "
                           f"Concéntrala en las horas llano (8-10, 14-18 y 22-24 h).")
        else:
            texto_clima = "Sin cambios: no hay horas punta que evitar este día."

        c_b = float((gb["consumo_total_kW"] * DT_HORAS * gb["precio"]).sum())
        c_m = float((gm["consumo_total_kW"] * DT_HORAS * gm["precio"]).sum())
        dias.append({
            "fecha": str(dia),
            "nombre": f"{DIAS[fecha.dayofweek]} {fecha.day}",
            "tipo": tipo,
            "cargas": cargas,
            "climatizacion": {"kwh_antes": kwh_b, "kwh_despues": kwh_m, "texto": texto_clima,
                              "ahorro_eur": round(float(((gb["P_climatizacion_kW"].to_numpy() - gm["P_climatizacion_kW"].to_numpy())
                                                         * DT_HORAS * gm["precio"].to_numpy()).sum()), 4)},
            "coste_historico": round(c_b, 4), "coste_recomendado": round(c_m, 4),
            "ahorro_eur": round(c_b - c_m, 4),
        })

    por_carga, base = None, None
    if potencias:
        por_carga = {}
        suma_ahorros = 0.0
        for nombre, col in CARGAS.items():
            p = potencias[CLAVE_PARAMETRO[nombre]]
            kwh = float(m[col].sum() * p * DT_HORAS)
            c_b = float((b[col] * p * DT_HORAS * b["precio"]).sum())
            c_m = float((m[col] * p * DT_HORAS * m["precio"]).sum())
            por_carga[ID_CARGA[nombre]] = {"nombre": nombre, "kwh_semana": round(kwh, 3),
                "coste_historico": round(c_b, 4), "coste_recomendado": round(c_m, 4),
                "ahorro_eur": round(c_b - c_m, 4),
                "ahorro_pct": round(100 * (c_b - c_m) / c_b, 2) if c_b > 0 else 0.0}
            suma_ahorros += c_b - c_m
        kwh_h = float(m["P_climatizacion_kW"].sum() * DT_HORAS)
        funcionando = b["P_climatizacion_kW"][b["P_climatizacion_kW"] > UMBRAL_FUNCIONANDO_KW]
        potencia_media_clima = float(funcionando.mean()) if len(funcionando) else float(potencias_max_clima(parametros))
        ch_b = float((b["P_climatizacion_kW"] * DT_HORAS * b["precio"]).sum())
        ch_m = float((m["P_climatizacion_kW"] * DT_HORAS * m["precio"]).sum())
        por_carga["climatizacion"] = {"nombre": "Climatización", "kwh_semana": round(kwh_h, 3),
            "potencia_media_funcionando_kW": round(potencia_media_clima, 3),
            "coste_historico": round(ch_b, 4), "coste_recomendado": round(ch_m, 4),
            "ahorro_eur": round(ch_b - ch_m, 4), "ahorro_pct": round(100 * (ch_b - ch_m) / ch_b, 2) if ch_b > 0 else 0.0}
        suma_ahorros += ch_b - ch_m
        assert abs(suma_ahorros - ahorro_total) < 1e-6, "los ahorros por carga no suman el ahorro total"

        potencia_apar = sum(potencias[CLAVE_PARAMETRO[n]] * m[col] for n, col in CARGAS.items())
        carga_base = (m["consumo_total_kW"] - m["P_climatizacion_kW"] - potencia_apar).clip(lower=0)
        kwh_base = float(carga_base.sum() * DT_HORAS)
        base = {"kwh_semana": round(kwh_base, 2),
                "coste": round(float((carga_base * DT_HORAS * m["precio"]).sum()), 3),
                "porcentaje_consumo": round(100 * kwh_base / float(m["consumo_total_kW"].sum() * DT_HORAS), 1)}

    primero, ultimo = m["timestamp"].iloc[0], m["timestamp"].iloc[-1]
    return {
        "parametros": parametros,
        "precios_eur_kWh": parametros["precios_eur_kWh"] if parametros else None,
        "por_carga": por_carga,
        "carga_base": base,
        "semana": f"{primero.day}-{ultimo.day} {MESES[ultimo.month - 1]} {ultimo.year}",
        "resumen": {
            "coste_historico": round(float(coste_b), 4), "coste_recomendado": round(float(coste_m), 4),
            "ahorro_eur": round(float(ahorro_total), 4),
            "ahorro_pct": round(100 * float(ahorro_total) / float(coste_b), 2),
            "ahorro_climatizacion_eur": round(float(ahorro_hvac), 4),
            "ahorro_electrodomesticos_eur": round(float(ahorro_total - ahorro_hvac), 4),
        },
        "reglas": REGLAS,
        "como_aplicarlo": COMO_APLICARLO,
        "dias": dias,
    }


def a_markdown(plan: dict) -> str:
    r = plan["resumen"]
    lineas = [
        f"# Plan semanal para el usuario ({plan['semana']})", "",
        f"Si sigues este plan pagarías **{r['coste_recomendado']:.2f} €** en vez de **{r['coste_historico']:.2f} €** "
        f"(ahorro de **{r['ahorro_eur']:.2f} €**, {r['ahorro_pct']:.1f} %), con el mismo consumo y los mismos usos.", "",
        f"- Ahorro por mover los electrodomésticos: **{r['ahorro_electrodomesticos_eur']:.2f} €** "
        f"(no depende de ninguna hipótesis).",
        f"- Ahorro por mover la climatización: **{r['ahorro_climatizacion_eur']:.2f} €** "
        f"(depende de que la vivienda mantenga la temperatura unas horas).", "",
        "## Reglas", ""]
    lineas += [f"{i}. **{x['titulo']}.** {x['texto']}" for i, x in enumerate(plan["reglas"], 1)]
    lineas += ["", "## Cómo hacerlo sin domótica", ""]
    lineas += [f"- **{x['aparato']}:** {x['como']}" for x in plan["como_aplicarlo"]]
    lineas += ["", "## Agenda de la semana", ""]
    for d in plan["dias"]:
        lineas += [f"### {d['nombre'].capitalize()} ({d['tipo']}) — ahorro {d['ahorro_eur']:.2f} €", ""]
        for c in d["cargas"]:
            antes = ", ".join(c["antes"]) if c["antes"] else "—"
            n = c["usos_de_15_min"]
            usos = "1 uso" if n == 1 else f"{n} usos"
            lineas.append(f"- **{c['nombre']}** ({usos}). En tu histórico: {antes}. {c['recomendacion']}")
        lineas += [f"- **Climatización.** {d['climatizacion']['texto']}", ""]
    lineas += ["---", "Nota: cada \"uso\" es un intervalo de 15 min, la resolución de los datos de partida.",
               "El plan se genera con la solución del MILP (óptimo garantizado) y con los mismos usos por día que tu histórico."]
    return "\n".join(lineas)


def main(ruta_milp="resultados/resultados_MILP.csv", ruta_baseline="resultados/resultados_Baseline.csv",
         carpeta_salida="resultados"):
    plan = generar_plan(ruta_milp, ruta_baseline)
    os.makedirs(carpeta_salida, exist_ok=True)
    with open(os.path.join(carpeta_salida, "plan_usuario.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    md = a_markdown(plan)
    with open(os.path.join(carpeta_salida, "plan_usuario.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return plan


if __name__ == "__main__":
    main()
