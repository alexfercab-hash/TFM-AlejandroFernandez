"""
Extiende main.py (Pseudocódigo 7) para comprobar si las conclusiones de la
comparación de algoritmos (MILP vs GA vs PSO vs SA vs Baseline) se sostienen
en más de una vivienda y más de una semana, no solo en el caso concreto
usado hasta ahora (vivienda 26, primera semana).

"""

import os
import io
import json
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import construir_modelo_completo, ejecutar_algoritmos, evaluar_soluciones_obtenidas



# CONFIGURACIÓN DEL ANÁLISIS DE GENERALIZACIÓN

VIVIENDAS = [1, 2, 3, 4, 5]
# Semanas completas de lunes a domingo, sin festivos nacionales (la serie empieza en miércoles
# 1 de enero, y la semana del 1 al 7 contiene dos festivos: se evita).
SEMANAS_INICIO = ["2025-01-13", "2025-01-20", "2025-01-27", "2025-02-03"]
N_SEMANAS = len(SEMANAS_INICIO)
CARPETA_RESULTADOS = "resultados_generalizacion"



# EJECUCIÓN DE UNA COMBINACIÓN (VIVIENDA, SEMANA)

def ejecutar_una_combinacion(house_id: int, semana_idx: int) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo (pasos 1-7) para una vivienda y una semana
    concretas, y devuelve la tabla comparativa de esa combinación (con
    columnas adicionales 'vivienda' y 'semana').
    """
    fecha_inicio_semana = SEMANAS_INICIO[semana_idx]

    buffer_silenciado = io.StringIO()
    with contextlib.redirect_stdout(buffer_silenciado):
        modelo = construir_modelo_completo(house_id=house_id, fecha_inicio_semana=fecha_inicio_semana)
        resultados = ejecutar_algoritmos(modelo)
        tabla = evaluar_soluciones_obtenidas(resultados, modelo)

    tabla.insert(0, "semana", semana_idx + 1)
    tabla.insert(0, "vivienda", house_id)

    mejor = tabla[tabla["algoritmo"] != "Baseline"].sort_values("coste_economico").iloc[0]
    print(f"[Generalización] Vivienda {house_id}, semana {semana_idx + 1}: "
          f"mejor = {mejor['algoritmo']} (coste {mejor['coste_economico']}, "
          f"ahorro coste {mejor['ahorro_coste_vs_baseline_%']}%)")

    return tabla


def ejecutar_analisis_completo(viviendas: list = VIVIENDAS, n_semanas: int = N_SEMANAS) -> pd.DataFrame:
    """Ejecuta todas las combinaciones (vivienda, semana) y concatena los resultados en una única tabla."""
    print("=" * 70)
    print(f"ANÁLISIS DE GENERALIZACIÓN: {len(viviendas)} viviendas x {n_semanas} semanas "
          f"= {len(viviendas) * n_semanas} combinaciones")
    print("=" * 70)

    tablas = []
    for house_id in viviendas:
        for semana_idx in range(n_semanas):
            tabla = ejecutar_una_combinacion(house_id, semana_idx)
            tablas.append(tabla)

    return pd.concat(tablas, ignore_index=True)



# AGREGACIÓN DE RESULTADOS (MEDIA ± DESVIACIÓN POR ALGORITMO)

def agregar_resultados(df_completo: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los resultados de todas las combinaciones (vivienda, semana) por
    algoritmo: media y desviación típica del coste, el ahorro de coste y de
    consumo frente al baseline, el pico de demanda y el tiempo de ejecución,
    además de la proporción de combinaciones en las que se cumplieron todas
    las restricciones.
    """
    resumen = df_completo.groupby("algoritmo").agg(
        coste_medio=("coste_economico", "mean"),
        coste_std=("coste_economico", "std"),
        ahorro_coste_medio_pct=("ahorro_coste_vs_baseline_%", "mean"),
        ahorro_coste_std_pct=("ahorro_coste_vs_baseline_%", "std"),
        pico_medio_kW=("pico_demanda_kW", "mean"),
        pico_std_kW=("pico_demanda_kW", "std"),
        tiempo_medio_s=("tiempo_ejecucion_s", "mean"),
        tasa_cumplimiento=("cumple_todas_las_restricciones", "mean"),
        n_combinaciones=("coste_economico", "count"),
    ).round(3).reset_index()

    resumen["tasa_cumplimiento"] = (resumen["tasa_cumplimiento"] * 100).round(1)

    return resumen



# GRÁFICAS DE GENERALIZACIÓN

def generar_graficas_generalizacion(df_completo: pd.DataFrame, carpeta_salida: str) -> None:
    """Boxplots del coste y del ahorro frente al baseline, por algoritmo, a lo largo de todas las combinaciones."""
    algoritmos_optimizacion = [a for a in df_completo["algoritmo"].unique() if a != "Baseline"]

    fig, ejes = plt.subplots(1, 2, figsize=(12, 5))

    datos_coste = [df_completo[df_completo["algoritmo"] == a]["coste_economico"] for a in algoritmos_optimizacion]
    ejes[0].boxplot(datos_coste, tick_labels=algoritmos_optimizacion)
    ejes[0].set_title("Distribución del coste semanal\n(todas las viviendas/semanas)")
    ejes[0].set_ylabel("Coste semanal (€)")

    datos_ahorro = [df_completo[df_completo["algoritmo"] == a]["ahorro_coste_vs_baseline_%"] for a in algoritmos_optimizacion]
    ejes[1].boxplot(datos_ahorro, tick_labels=algoritmos_optimizacion)
    ejes[1].axhline(0, color="red", linestyle="--", linewidth=1, label="Sin ahorro (=baseline)")
    ejes[1].set_title("Ahorro de COSTE (%) frente al baseline\n(todas las viviendas/semanas)")
    ejes[1].set_ylabel("Ahorro de coste (%)")
    ejes[1].legend()

    fig.tight_layout()
    ruta = os.path.join(carpeta_salida, "generalizacion_boxplots.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"[Generalización] Gráfica guardada en '{ruta}'.")


# GUARDADO DE RESULTADOS

def guardar_resultados_generalizacion(df_completo: pd.DataFrame, resumen: pd.DataFrame,
                                       carpeta_salida: str = CARPETA_RESULTADOS) -> None:
    os.makedirs(carpeta_salida, exist_ok=True)

    df_completo.to_csv(os.path.join(carpeta_salida, "resultados_todas_las_combinaciones.csv"), index=False)
    resumen.to_csv(os.path.join(carpeta_salida, "resumen_agregado_por_algoritmo.csv"), index=False)

    with open(os.path.join(carpeta_salida, "resumen_agregado_por_algoritmo.json"), "w", encoding="utf-8") as f:
        json.dump(resumen.to_dict(orient="records"), f, indent=2, ensure_ascii=False, default=str)

    print(f"[Generalización] Resultados guardados en '{carpeta_salida}/'.")



# ORQUESTACIÓN COMPLETA

def main():
    df_completo = ejecutar_analisis_completo()
    resumen = agregar_resultados(df_completo)

    print("\n" + "=" * 70)
    print("RESUMEN AGREGADO (media ± desviación típica sobre todas las combinaciones)")
    print("=" * 70)
    print(resumen.to_string(index=False))

    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    generar_graficas_generalizacion(df_completo, CARPETA_RESULTADOS)
    guardar_resultados_generalizacion(df_completo, resumen, CARPETA_RESULTADOS)

    return df_completo, resumen


if __name__ == "__main__":
    df_completo, resumen = main()
