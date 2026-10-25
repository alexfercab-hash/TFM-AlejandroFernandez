"""
Igual que main.py pero con alto presupuesto.
"""

import os
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from procesado_datos import preparar_datos
from Modelo_energetico import crear_modelo_energetico, periodo_inicio_desde_fecha, FECHA_SEMANA_REFERENCIA, PERIODOS_POR_DIA
from Restricciones import definir_restricciones
from Funcion_objetivo import definir_funcion_objetivo

from MILP import resolver_milp
from GA_alto_presupuesto import ejecutar_ga
from PSO_alto_presupuesto import ejecutar_pso
from SA_alto_presupuesto import ejecutar_sa


CARPETA_RESULTADOS = "resultados_alto_presupuesto"
TOLERANCIA_NUMERICA = 1e-3   # margen para comparaciones con "<=" / "==" en variables continuas



# 1-4. CONSTRUCCIÓN DEL MODELO ENERGÉTICO COMPLETO


def construir_modelo_completo(house_id: int = None, fecha_inicio_semana: str = FECHA_SEMANA_REFERENCIA):

    print("=" * 70)
    print(f"PASO 1-4: Construcción del modelo energético "
          f"(vivienda={house_id if house_id is not None else 'por defecto'}, semana desde {fecha_inicio_semana})")
    print("=" * 70)

    if house_id is not None:
        df_preparado = preparar_datos(house_id=house_id)
    else:
        df_preparado = preparar_datos()

    periodo_inicio = periodo_inicio_desde_fecha(df_preparado, fecha_inicio_semana)
    modelo = crear_modelo_energetico(df_preparado, periodo_inicio=periodo_inicio)
    modelo.historico = df_preparado     # histórico completo de la vivienda (lo usa consigna_climatizacion.py)
    modelo = definir_restricciones(modelo, datos_historicos=df_preparado)
    modelo = definir_funcion_objetivo(modelo)

    return modelo



# BASELINE: CONSUMO REAL SIN OPTIMIZAR (tal cual ocurrió en el dataset)

def construir_resultado_baseline(modelo) -> dict:
   
    datos = modelo.datos.reset_index(drop=True)
    dt = modelo.dt_horas
    otros = modelo.cargas_controlables["otros_dispositivos"]["dispositivos"]

    estado_lavadora = datos["washing_machine_on"].to_numpy()
    estado_lavavajillas = datos["dishwasher_on"].to_numpy()
    estado_calentador = datos["water_heater_on"].to_numpy()

    consumo_total = (
        datos["base_load_kW"].to_numpy()
        + datos["hvac_power_kW"].to_numpy()
        + otros["lavadora"]["potencia_nominal_kW"] * estado_lavadora
        + otros["lavavajillas"]["potencia_nominal_kW"] * estado_lavavajillas
        + otros["calentador_agua"]["potencia_nominal_kW"] * estado_calentador
    )

    # Misma fórmula que la función objetivo: potencia (kW) x 0,25 h x precio (EUR/kWh)
    precio = datos["precio_energia_eur_kWh"].to_numpy()
    coste = float(np.sum(consumo_total * dt * precio))
    consumo_kWh = float(np.sum(consumo_total) * dt)

    # Comprobación de la reconstrucción frente a la demanda real del dataset
    demanda_real = datos["demanda_sin_ev_kW"].to_numpy()
    consumo_real_kWh = float(np.sum(demanda_real) * dt)
    coste_real = float(np.sum(demanda_real * dt * precio))

    df_baseline = pd.DataFrame({
        "periodo": np.arange(len(datos)),
        "timestamp": datos["timestamp"],
        "P_climatizacion_kW": datos["hvac_power_kW"],
        "estado_lavadora": estado_lavadora,
        "estado_lavavajillas": estado_lavavajillas,
        "estado_calentador_agua": estado_calentador,
        "consumo_total_kW": consumo_total,
    })

    resumen_baseline = {
        "algoritmo": "Baseline",
        "coste_economico_total": coste,
        "consumo_energetico_total_kWh": consumo_kWh,
        "potencia_contratada_kW": modelo.potencia_contratada_kW,
        "n_periodos": len(datos),
        "validacion_vs_demanda_real": {
            "consumo_real_kWh": consumo_real_kWh,
            "coste_real": coste_real,
            "dif_consumo_pct": 100 * (consumo_kWh - consumo_real_kWh) / consumo_real_kWh,
            "dif_coste_pct": 100 * (coste - coste_real) / coste_real,
        },
    }

    print(f"[Baseline] Coste histórico: {coste:.2f} € | Consumo: {consumo_kWh:.2f} kWh "
          f"(frente a la demanda real del dataset: {coste_real:.2f} € / {consumo_real_kWh:.2f} kWh; "
          f"diferencia {resumen_baseline['validacion_vs_demanda_real']['dif_coste_pct']:+.2f} % en coste)")

    return {"df": df_baseline, "resumen": resumen_baseline, "tiempo_s": 0.0}



# 5-6. EJECUCIÓN DE LOS ALGORITMOS Y RECOGIDA DE SOLUCIONES

def ejecutar_algoritmos(modelo) -> dict:
   
    resultados = {}

    print("\n" + "=" * 70)
    print("PASO 5: Ejecución de los algoritmos de optimización")
    print("=" * 70)

    # --- Baseline (referencia, no es un algoritmo de optimización) ---
    print("\n--- Baseline (consumo real, sin optimizar) ---")
    resultados["Baseline"] = construir_resultado_baseline(modelo)

    # --- MILP ---
    print("\n--- MILP ---")
    t0 = time.perf_counter()
    _, df_milp, resumen_milp = resolver_milp(modelo, carpeta_salida=CARPETA_RESULTADOS)
    tiempo_milp = time.perf_counter() - t0
    resultados["MILP"] = {"df": df_milp, "resumen": resumen_milp, "tiempo_s": tiempo_milp}

    # --- GA ---
    print("\n--- GA ---")
    t0 = time.perf_counter()
    df_ga, resumen_ga, historial_ga = ejecutar_ga(modelo)
    tiempo_ga = time.perf_counter() - t0
    resultados["GA"] = {"df": df_ga, "resumen": resumen_ga, "tiempo_s": tiempo_ga, "historial": historial_ga}

    # --- PSO ---
    print("\n--- PSO ---")
    t0 = time.perf_counter()
    df_pso, resumen_pso, historial_pso = ejecutar_pso(modelo)
    tiempo_pso = time.perf_counter() - t0
    resultados["PSO"] = {"df": df_pso, "resumen": resumen_pso, "tiempo_s": tiempo_pso, "historial": historial_pso}

    # --- SA ---
    print("\n--- SA ---")
    t0 = time.perf_counter()
    df_sa, resumen_sa, historial_sa = ejecutar_sa(modelo)
    tiempo_sa = time.perf_counter() - t0
    resultados["SA"] = {"df": df_sa, "resumen": resumen_sa, "tiempo_s": tiempo_sa, "historial": historial_sa}

    for nombre, r in resultados.items():
        print(f"[Main] {nombre} finalizado en {r['tiempo_s']:.2f} s.")

    return resultados



# 7. EVALUACIÓN DE LAS SOLUCIONES

def evaluar_cumplimiento_restricciones(df_resultados: pd.DataFrame, modelo) -> dict:
    
    n_periodos = len(modelo.T)
    n_dias = n_periodos // PERIODOS_POR_DIA
    servicio = modelo.servicio
    ventana = servicio["ventana_periodos"]
    dt = modelo.dt_horas

    # --- Usos diarios de los electrodomésticos ---
    columnas = {"lavadora": "estado_lavadora", "lavavajillas": "estado_lavavajillas",
                "calentador_agua": "estado_calentador_agua"}
    usos_correctos = {}
    for nombre, col in columnas.items():
        usos_resultado = df_resultados[col].to_numpy().reshape(n_dias, PERIODOS_POR_DIA).sum(axis=1)
        usos_correctos[nombre] = bool(np.all(np.round(usos_resultado) == servicio["usos_por_dia"][nombre]))

    # --- Confort de la climatización (energía mínima por ventana) ---
    p_clim = df_resultados["P_climatizacion_kW"].to_numpy()
    energia_ventana = p_clim.reshape(-1, ventana).sum(axis=1) * dt
    deficit = np.maximum(servicio["energia_hvac_ventana_kWh"] - energia_ventana, 0.0)
    incumplimientos_confort = int(np.sum(deficit > TOLERANCIA_NUMERICA))
    p_max_hvac = modelo.cargas_controlables["climatizacion"]["potencia_max_kW"]
    supera_potencia_hvac = bool(np.any(p_clim > p_max_hvac + TOLERANCIA_NUMERICA))

    # --- Potencia contratada ---
    exceso_potencia = np.maximum(df_resultados["consumo_total_kW"].to_numpy() - modelo.potencia_contratada_kW, 0.0)
    max_exceso_potencia = float(exceso_potencia.max())
    incumple_potencia_contratada = max_exceso_potencia > TOLERANCIA_NUMERICA

    cumple_todo = (
        all(usos_correctos.values())
        and incumplimientos_confort == 0
        and not supera_potencia_hvac
        and not incumple_potencia_contratada
    )

    return {
        "cumple_todas_las_restricciones": bool(cumple_todo),
        "usos_diarios_correctos": usos_correctos,
        "ventanas_confort_incumplidas": incumplimientos_confort,
        "climatizacion_supera_potencia_max": supera_potencia_hvac,
        "potencia_contratada_excedida": bool(incumple_potencia_contratada),
        "potencia_contratada_max_exceso_kW": round(max_exceso_potencia, 4),
    }


def evaluar_soluciones_obtenidas(resultados: dict, modelo) -> pd.DataFrame:
    
    print("\n" + "=" * 70)
    print("PASO 7: Evaluación de las soluciones")
    print("=" * 70)

    filas = []
    coste_baseline = resultados.get("Baseline", {}).get("resumen", {}).get("coste_economico_total")
    consumo_baseline = resultados.get("Baseline", {}).get("resumen", {}).get("consumo_energetico_total_kWh")

    for nombre, r in resultados.items():
        df = r["df"]
        resumen = r["resumen"]
        cumplimiento = evaluar_cumplimiento_restricciones(df, modelo)
        resultados[nombre]["cumplimiento"] = cumplimiento  # se guarda también para el JSON detallado

        coste = resumen["coste_economico_total"]
        consumo = resumen["consumo_energetico_total_kWh"]

        ahorro_coste_pct = None
        ahorro_consumo_pct = None
        if nombre != "Baseline":
            if coste_baseline:
                ahorro_coste_pct = round(100 * (coste_baseline - coste) / coste_baseline, 2)
            if consumo_baseline:
                ahorro_consumo_pct = round(100 * (consumo_baseline - consumo) / consumo_baseline, 2)

        filas.append({
            "algoritmo": nombre,
            "coste_economico": round(coste, 2),
            "ahorro_coste_vs_baseline_%": ahorro_coste_pct,
            "consumo_energetico_kWh": round(consumo, 2),
            "ahorro_consumo_vs_baseline_%": ahorro_consumo_pct,
            "pico_demanda_kW": round(float(df["consumo_total_kW"].max()), 3),
            "tiempo_ejecucion_s": round(r["tiempo_s"], 3),
            "cumple_todas_las_restricciones": cumplimiento["cumple_todas_las_restricciones"],
            "potencia_contratada_max_exceso_kW": cumplimiento["potencia_contratada_max_exceso_kW"],
        })

        print(f"[Evaluación] {nombre}: coste={filas[-1]['coste_economico']}"
              f"{f' (ahorro coste {ahorro_coste_pct}%)' if ahorro_coste_pct is not None else ''}, "
              f"consumo={filas[-1]['consumo_energetico_kWh']} kWh"
              f"{f' (ahorro consumo {ahorro_consumo_pct}%)' if ahorro_consumo_pct is not None else ''}, "
              f"pico={filas[-1]['pico_demanda_kW']} kW, "
              f"cumple restricciones={cumplimiento['cumple_todas_las_restricciones']}, "
              f"tiempo={filas[-1]['tiempo_ejecucion_s']} s")

    return pd.DataFrame(filas)



# 8. COMPARACIÓN DE LOS ALGORITMOS

def comparar_algoritmos(tabla_comparativa: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("PASO 8: Comparación de los algoritmos")
    print("=" * 70)
    print(tabla_comparativa.to_string(index=False))

    factibles = tabla_comparativa[
        (tabla_comparativa["cumple_todas_las_restricciones"]) & (tabla_comparativa["algoritmo"] != "Baseline")
    ]
    algoritmos_optimizacion = tabla_comparativa[tabla_comparativa["algoritmo"] != "Baseline"]

    if factibles.empty:
        print("\n[Aviso] Ningún algoritmo de optimización cumplió todas las restricciones; revisa el modelo o las tolerancias.")
        return

    mejor_coste = factibles.loc[factibles["coste_economico"].idxmin()]
    mejor_tiempo = algoritmos_optimizacion.loc[algoritmos_optimizacion["tiempo_ejecucion_s"].idxmin()]
    menor_pico = factibles.loc[factibles["pico_demanda_kW"].idxmin()]

    print(f"\nMejor coste (entre soluciones factibles): {mejor_coste['algoritmo']} ({mejor_coste['coste_economico']})")
    print(f"Menor pico de demanda (entre factibles): {menor_pico['algoritmo']} ({menor_pico['pico_demanda_kW']} kW)")
    print(f"Más rápido: {mejor_tiempo['algoritmo']} ({mejor_tiempo['tiempo_ejecucion_s']} s)")

    if "Baseline" in tabla_comparativa["algoritmo"].values:
        ahorro_coste_mejor = tabla_comparativa.set_index("algoritmo").loc[mejor_coste["algoritmo"], "ahorro_coste_vs_baseline_%"]
        ahorro_consumo_mejor = tabla_comparativa.set_index("algoritmo").loc[mejor_coste["algoritmo"], "ahorro_consumo_vs_baseline_%"]
        print(f"Ahorro de coste del mejor método frente al consumo real sin optimizar: {ahorro_coste_mejor}%")
        print(f"Ahorro de consumo energético (kWh) del mejor método frente al baseline: {ahorro_consumo_mejor}%")



# 9. GENERACIÓN Y ALMACENAMIENTO DE RESULTADOS

def generar_graficas(resultados: dict, tabla_comparativa: pd.DataFrame, carpeta_salida: str) -> None:
   
    algoritmos = tabla_comparativa["algoritmo"].tolist()
    algoritmos_optimizacion = tabla_comparativa[tabla_comparativa["algoritmo"] != "Baseline"]

    # --- Coste, pico y tiempo (barras) ---
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4))

    ejes[0].bar(algoritmos, tabla_comparativa["coste_economico"], color="#4C72B0")
    ejes[0].set_title("Coste económico total (semanal)")
    ejes[0].set_ylabel("Coste semanal (€)")

    ejes[1].bar(algoritmos, tabla_comparativa["pico_demanda_kW"], color="#DD8452")
    ejes[1].set_title("Pico de demanda")
    ejes[1].set_ylabel("kW")

    ejes[2].bar(algoritmos, tabla_comparativa["tiempo_ejecucion_s"], color="#55A868")
    ejes[2].set_title("Tiempo de ejecución")
    ejes[2].set_ylabel("segundos")
    ejes[2].set_yscale("log")

    fig.tight_layout()
    ruta_barras = os.path.join(carpeta_salida, "comparacion_metodos.png")
    fig.savefig(ruta_barras, dpi=150)
    plt.close(fig)

    # --- Consumo energético (kWh) y % de ahorro de consumo frente al baseline ---
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4))

    ejes[0].bar(algoritmos, tabla_comparativa["consumo_energetico_kWh"], color="#8172B2")
    ejes[0].set_title("Consumo energético total")
    ejes[0].set_ylabel("kWh")

    colores = ["#55A868" if v is not None and v >= 0 else "#C44E52"
               for v in algoritmos_optimizacion["ahorro_consumo_vs_baseline_%"]]
    ejes[1].bar(algoritmos_optimizacion["algoritmo"], algoritmos_optimizacion["ahorro_consumo_vs_baseline_%"], color=colores)
    ejes[1].axhline(0, color="black", linewidth=0.8)
    ejes[1].set_title("Ahorro de consumo (%) vs. Baseline\n(sin optimizar)")
    ejes[1].set_ylabel("Ahorro de consumo (%)")

    fig.tight_layout()
    ruta_consumo = os.path.join(carpeta_salida, "comparacion_consumo_ahorro.png")
    fig.savefig(ruta_consumo, dpi=150)
    plt.close(fig)

    # --- Convergencia de los métodos metaheurísticos (GA, PSO, SA) ---
    fig, eje = plt.subplots(figsize=(8, 5))
    for nombre in ["GA", "PSO", "SA"]:
        if "historial" in resultados[nombre]:
            eje.plot(resultados[nombre]["historial"], label=nombre)
    eje.axhline(tabla_comparativa.set_index("algoritmo").loc["MILP", "coste_economico"],
                color="black", linestyle="--", label="Óptimo MILP")
    eje.set_xlabel("Iteración / generación")
    eje.set_ylabel("Mejor fitness (coste)")
    eje.set_title("Convergencia de los métodos metaheurísticos")
    eje.legend()
    fig.tight_layout()
    ruta_convergencia = os.path.join(carpeta_salida, "convergencia_metaheuristicas.png")
    fig.savefig(ruta_convergencia, dpi=150)
    plt.close(fig)

    print(f"[Main] Gráficas guardadas en '{ruta_barras}' y '{ruta_convergencia}'.")


def guardar_resultados_finales(resultados: dict, tabla_comparativa: pd.DataFrame, carpeta_salida: str) -> None:
    """Guarda en disco: la tabla comparativa (CSV), el detalle completo (JSON) y los resultados de cada algoritmo (CSV)."""
    os.makedirs(carpeta_salida, exist_ok=True)

    tabla_comparativa.to_csv(os.path.join(carpeta_salida, "comparacion_metodos.csv"), index=False)

    detalle = {}
    for nombre, r in resultados.items():
        r["df"].to_csv(os.path.join(carpeta_salida, f"resultados_{nombre}.csv"), index=False)
        detalle[nombre] = {"resumen": r["resumen"], "cumplimiento": r["cumplimiento"], "tiempo_s": r["tiempo_s"]}

    with open(os.path.join(carpeta_salida, "detalle_resultados.json"), "w", encoding="utf-8") as f:
        json.dump(detalle, f, indent=2, ensure_ascii=False, default=str)

    print(f"[Main] Resultados detallados guardados en '{carpeta_salida}/'.")


def guardar_parametros_cargas(modelo, carpeta_salida: str) -> None:
    
    from tarifa_2_0TD import PRECIOS_EUR_KWH
    cargas = modelo.cargas_controlables
    otros = cargas["otros_dispositivos"]["dispositivos"]
    parametros = {
        "potencias_nominales_kW": {n: d["potencia_nominal_kW"] for n, d in otros.items()},
        "potencia_max_climatizacion_kW": cargas["climatizacion"]["potencia_max_kW"],
        "ventana_confort_horas": modelo.servicio["ventana_periodos"] * 15 / 60,
        "potencia_contratada_kW": modelo.potencia_contratada_kW,
        "precios_eur_kWh": PRECIOS_EUR_KWH,
        "semana_inicio": str(modelo.datos["timestamp"].iloc[0]),
        "semana_fin": str(modelo.datos["timestamp"].iloc[-1]),
    }
    from consigna_climatizacion import analizar
    parametros["consigna_climatizacion"] = analizar(modelo.historico, modelo.datos)
    with open(os.path.join(carpeta_salida, "parametros_cargas.json"), "w", encoding="utf-8") as f:
        json.dump(parametros, f, indent=2, ensure_ascii=False)



# ORQUESTACIÓN COMPLETA (PSEUDOCÓDIGO 7)

def main():
    tiempo_inicio_total = time.perf_counter()

    modelo = construir_modelo_completo()
    resultados = ejecutar_algoritmos(modelo)
    tabla_comparativa = evaluar_soluciones_obtenidas(resultados, modelo)
    comparar_algoritmos(tabla_comparativa)

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    generar_graficas(resultados, tabla_comparativa, CARPETA_RESULTADOS)
    guardar_resultados_finales(resultados, tabla_comparativa, CARPETA_RESULTADOS)
    guardar_parametros_cargas(modelo, CARPETA_RESULTADOS)

    print("\n" + "=" * 70)
    print(f"Pipeline completo ejecutado en {time.perf_counter() - tiempo_inicio_total:.2f} s.")
    print("=" * 70)

    return resultados, tabla_comparativa


if __name__ == "__main__":
    resultados, tabla_comparativa = main()
