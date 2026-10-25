"""
En cada ventana de confort debe entregarse al menos la energía del
histórico, pero puede repartirse libremente dentro de la ventana (inercia térmica).
Este script mide cuánto cambia el ahorro del MILP según el tamaño de esa ventana.

Genera: resultados_sensibilidad/sensibilidad_ventana_confort.csv

Uso:
    python sensibilidad_ventana_confort.py
"""

import copy
import io
import os
import contextlib

import pandas as pd

from procesado_datos import preparar_datos
from Modelo_energetico import crear_modelo_energetico, CARGAS_CONTROLABLES
from Restricciones import definir_restricciones
from Funcion_objetivo import definir_funcion_objetivo
from MILP import resolver_milp
from main import construir_resultado_baseline

VENTANAS_PERIODOS = [1, 8, 16, 32, 96]      # 15 min, 2 h, 4 h, 8 h, 24 h
CARPETA = "resultados_sensibilidad"


def main():
    os.makedirs(CARPETA, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        df = preparar_datos()

    filas = []
    for ventana in VENTANAS_PERIODOS:
        cargas = copy.deepcopy(CARGAS_CONTROLABLES)
        cargas["climatizacion"]["parametros_especificos"]["ventana_confort_periodos"] = ventana
        with contextlib.redirect_stdout(io.StringIO()):
            modelo = crear_modelo_energetico(df, cargas_controlables=cargas)
            modelo = definir_restricciones(modelo, datos_historicos=df)
            modelo = definir_funcion_objetivo(modelo)
            coste_baseline = construir_resultado_baseline(modelo)["resumen"]["coste_economico_total"]
            _, _, resumen = resolver_milp(modelo, carpeta_salida=os.path.join(CARPETA, "_tmp"))
        coste = resumen["coste_economico_total"]
        filas.append({
            "ventana_horas": ventana * 15 / 60,
            "coste_baseline": round(coste_baseline, 3),
            "coste_milp": round(coste, 3),
            "ahorro_pct": round(100 * (coste_baseline - coste) / coste_baseline, 2),
        })

    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(CARPETA, "sensibilidad_ventana_confort.csv"), index=False)
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
