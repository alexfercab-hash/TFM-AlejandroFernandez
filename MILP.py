"""
Construcción y resolución del modelo de optimización mediante Programación
Lineal Entera-Mixta (MILP).

Sigue el Pseudocódigo 3:
    1. Recibir modelo, función objetivo y restricciones.
    2. Construir modelo MILP.
    3. Ejecutar solver.
    4. Obtener solución óptima.
    5. Evaluar y guardar resultados.

"""

import os
import json
import pandas as pd
import pyomo.environ as pyo


# CONFIGURACIÓN DEL SOLVER

SOLVER_NOMBRE = "cbc"
CARPETA_RESULTADOS = "resultados"
TIEMPO_LIMITE_SEG = None     # None = sin límite; ej. 120 para limitar a 2 min
GAP_RELATIVO = None          # None = por defecto del solver; ej. 0.01 para 1% de holgura



# 1. CONSTRUCCIÓN DEL MODELO MILP

def construir_modelo_milp(modelo: pyo.ConcreteModel) -> pyo.ConcreteModel:
    """
    Recibe el modelo energético con restricciones y función objetivo ya
    definidas, y comprueba que está correctamente formado como problema MILP
    (mezcla de variables continuas y binarias, función objetivo y
    restricciones presentes) antes de pasarlo al solver.
    """
    if not hasattr(modelo, "objetivo"):
        raise ValueError("El modelo no tiene función objetivo definida. "
                          "Ejecuta antes Funcion_objetivo.definir_funcion_objetivo().")

    n_continuas = 0
    n_binarias = 0
    for var in modelo.component_objects(pyo.Var, active=True):
        for indice in var:
            if var[indice].domain is pyo.Binary:
                n_binarias += 1
            else:
                n_continuas += 1

    n_restricciones = sum(len(c) for c in modelo.component_objects(pyo.Constraint, active=True))

    print("[MILP] Modelo MILP construido:")
    print(f"  - Variables continuas: {n_continuas}")
    print(f"  - Variables binarias: {n_binarias}")
    print(f"  - Restricciones totales: {n_restricciones}")
    print(f"  - Función objetivo: {modelo.objetivo.expr.__class__.__name__} "
          f"({'minimizar' if modelo.objetivo.sense == pyo.minimize else 'maximizar'})")

    if n_binarias == 0:
        print("  [Aviso] No hay variables binarias: el problema es en realidad un LP puro, no un MILP.")

    return modelo



# 2. EJECUCIÓN DEL SOLVER


def ejecutar_solver(modelo: pyo.ConcreteModel, solver_nombre: str = SOLVER_NOMBRE,
                     tiempo_limite_seg: float = TIEMPO_LIMITE_SEG,
                     gap_relativo: float = GAP_RELATIVO):
    """Ejecuta el solver MILP sobre el modelo y devuelve el objeto de resultados de Pyomo."""
    solver = pyo.SolverFactory(solver_nombre)
    if not solver.available():
        raise RuntimeError(f"El solver '{solver_nombre}' no está disponible en este entorno.")

    opciones = {}
    if tiempo_limite_seg is not None:
        opciones["sec"] = tiempo_limite_seg
    if gap_relativo is not None:
        opciones["ratioGap"] = gap_relativo

    print(f"[MILP] Ejecutando solver '{solver_nombre}'"
          f"{f' (límite {tiempo_limite_seg}s)' if tiempo_limite_seg else ''}...")
    resultado = solver.solve(modelo, tee=False, options=opciones)
    return resultado



# 3. OBTENCIÓN DE LA SOLUCIÓN ÓPTIMA


def obtener_solucion_optima(modelo: pyo.ConcreteModel, resultado) -> dict:
    """
    Comprueba el estado de la resolución y extrae un resumen de la solución óptima.
    Lanza un aviso (sin detener la ejecución) si el solver no alcanzó el óptimo.
    """
    estado = str(resultado.solver.status)
    condicion = str(resultado.solver.termination_condition)

    if resultado.solver.termination_condition != pyo.TerminationCondition.optimal:
        print(f"[MILP] AVISO: el solver terminó con condición '{condicion}' (no óptima). "
              f"Revisa la factibilidad del modelo o los límites de tiempo/gap.")
    else:
        print(f"[MILP] Solución óptima encontrada (estado: {estado}).")

    resumen = {
        "estado_solver": estado,
        "condicion_terminacion": condicion,
        "coste_economico_total": pyo.value(modelo.coste_energetico_total),
        "consumo_energetico_total_kWh": pyo.value(modelo.consumo_energetico_total_kWh),
        "potencia_contratada_kW": modelo.potencia_contratada_kW,
        "n_periodos": len(modelo.T),
    }

    print(f"[MILP] Coste económico total: {resumen['coste_economico_total']:.2f} €")
    print(f"[MILP] Consumo energético total: {resumen['consumo_energetico_total_kWh']:.2f} kWh")

    return resumen



# 4. EVALUACIÓN Y GUARDADO DE RESULTADOS

def extraer_resultados_temporales(modelo: pyo.ConcreteModel) -> pd.DataFrame:
  
    filas = []
    for t in modelo.T:
        filas.append({
            "periodo": t,
            "timestamp": modelo.datos["timestamp"].iloc[t],
            "P_climatizacion_kW": pyo.value(modelo.P_climatizacion[t]),
            "estado_lavadora": round(pyo.value(modelo.estado_lavadora[t])),
            "estado_lavavajillas": round(pyo.value(modelo.estado_lavavajillas[t])),
            "estado_calentador_agua": round(pyo.value(modelo.estado_calentador_agua[t])),
            "consumo_total_kW": pyo.value(modelo.consumo_total_kW[t]),
            "periodo_tarifario": modelo.datos["periodo_tarifario"].iloc[t],
            "precio_energia_eur_kWh": pyo.value(modelo.precio_energia_eur_kWh[t]),
        })
    return pd.DataFrame(filas)


def guardar_resultados(df_resultados: pd.DataFrame, resumen: dict,
                        carpeta_salida: str = CARPETA_RESULTADOS) -> None:
    """Guarda en disco la serie temporal de resultados (CSV) y el resumen (JSON)."""
    os.makedirs(carpeta_salida, exist_ok=True)

    ruta_csv = os.path.join(carpeta_salida, "resultados_optimizacion.csv")
    ruta_json = os.path.join(carpeta_salida, "resumen_resultados.json")

    df_resultados.to_csv(ruta_csv, index=False)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)

    print(f"[MILP] Resultados guardados en '{ruta_csv}' y '{ruta_json}'.")



# COMPILACIÓN COMPLETA DEL PSEUDOCÓDIGO 3


def resolver_milp(modelo: pyo.ConcreteModel,
                   solver_nombre: str = SOLVER_NOMBRE,
                   carpeta_salida: str = CARPETA_RESULTADOS,
                   tiempo_limite_seg: float = TIEMPO_LIMITE_SEG,
                   gap_relativo: float = GAP_RELATIVO):
   
    modelo = construir_modelo_milp(modelo)
    resultado = ejecutar_solver(modelo, solver_nombre, tiempo_limite_seg, gap_relativo)
    resumen = obtener_solucion_optima(modelo, resultado)
    df_resultados = extraer_resultados_temporales(modelo)
    guardar_resultados(df_resultados, resumen, carpeta_salida)

    return modelo, df_resultados, resumen


if __name__ == "__main__":
    from procesado_datos import preparar_datos
    from Modelo_energetico import crear_modelo_energetico
    from Restricciones import definir_restricciones
    from Funcion_objetivo import definir_funcion_objetivo

    df_preparado = preparar_datos()
    modelo = crear_modelo_energetico(df_preparado)
    modelo = definir_restricciones(modelo, datos_historicos=df_preparado)
    modelo = definir_funcion_objetivo(modelo)

    modelo, df_resultados, resumen = resolver_milp(modelo)

    print("\nResumen final:")
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
    print("\nPrimeras filas de la solución:")
    print(df_resultados.head())
