"""

Robustez estadística de las metaheurísticas: GA, PSO y SA son estocásticos, así que
una sola ejecución (semilla 42) no basta. Este script ejecuta cada uno con N_SEMILLAS
semillas distintas, con presupuesto bajo y alto, sobre la semana de referencia, y
resume media, desviación típica, mínimo y máximo del coste y del ahorro frente al
baseline (el MILP es determinista y se ejecuta una vez).

Genera, en 'resultados_semillas/':
    - resultados_por_semilla.csv   (una fila por algoritmo, presupuesto y semilla)
    - resumen_semillas.csv         (media, std, min, max por algoritmo y presupuesto)
    - comparacion_semillas.png     (diagrama de cajas, con MILP y baseline como referencia)

Uso:
    python analisis_semillas.py
"""

import io
import os
import time
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import construir_modelo_completo, construir_resultado_baseline
from MILP import resolver_milp

import GA, PSO, SA
import GA_alto_presupuesto, PSO_alto_presupuesto, SA_alto_presupuesto

N_SEMILLAS = 10
SEMILLAS = list(range(N_SEMILLAS))
CARPETA = "resultados_semillas"


def _silencio():
    return contextlib.redirect_stdout(io.StringIO())


def main():
    os.makedirs(CARPETA, exist_ok=True)

    with _silencio():
        modelo = construir_modelo_completo()
        coste_baseline = construir_resultado_baseline(modelo)["resumen"]["coste_economico_total"]
        _, _, r_milp = resolver_milp(modelo, carpeta_salida=os.path.join(CARPETA, "_tmp_milp"))
    coste_milp = r_milp["coste_economico_total"]
    print(f"Baseline: {coste_baseline:.3f} € | MILP: {coste_milp:.3f} € "
          f"(ahorro {100 * (coste_baseline - coste_milp) / coste_baseline:.2f} %)")

    configuraciones = {
        "bajo": {"GA": GA.ejecutar_ga, "PSO": PSO.ejecutar_pso, "SA": SA.ejecutar_sa},
        "alto": {"GA": GA_alto_presupuesto.ejecutar_ga, "PSO": PSO_alto_presupuesto.ejecutar_pso,
                 "SA": SA_alto_presupuesto.ejecutar_sa},
    }

    filas = []
    for presupuesto, algoritmos in configuraciones.items():
        for nombre, ejecutar in algoritmos.items():
            for semilla in SEMILLAS:
                t0 = time.perf_counter()
                with _silencio():
                    _, resumen, _ = ejecutar(modelo, semilla=semilla)
                tiempo = time.perf_counter() - t0
                coste = resumen["coste_economico_total"]
                filas.append({
                    "presupuesto": presupuesto, "algoritmo": nombre, "semilla": semilla,
                    "coste": coste,
                    "ahorro_vs_baseline_pct": 100 * (coste_baseline - coste) / coste_baseline,
                    "brecha_vs_milp_pct": 100 * (coste - coste_milp) / coste_milp,
                    "penalizacion_potencia": resumen["penalizacion_potencia"],
                    "tiempo_s": tiempo,
                })
            print(f"  {presupuesto:>4} | {nombre}: {N_SEMILLAS} semillas completadas")

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(CARPETA, "resultados_por_semilla.csv"), index=False)

    resumen = df.groupby(["presupuesto", "algoritmo"]).agg(
        coste_medio=("coste", "mean"), coste_std=("coste", "std"),
        coste_min=("coste", "min"), coste_max=("coste", "max"),
        ahorro_medio_pct=("ahorro_vs_baseline_pct", "mean"), ahorro_std_pct=("ahorro_vs_baseline_pct", "std"),
        brecha_media_vs_milp_pct=("brecha_vs_milp_pct", "mean"),
        tiempo_medio_s=("tiempo_s", "mean"),
        semillas_con_penalizacion=("penalizacion_potencia", lambda x: int((x > 1e-9).sum())),
    ).round(3).reset_index()
    resumen.to_csv(os.path.join(CARPETA, "resumen_semillas.csv"), index=False)

    # --- Gráfica: diagramas de caja por algoritmo y presupuesto ---
    fig, ejes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for eje, presupuesto in zip(ejes, ["bajo", "alto"]):
        datos = [df[(df.presupuesto == presupuesto) & (df.algoritmo == a)]["coste"] for a in ["GA", "PSO", "SA"]]
        eje.boxplot(datos, tick_labels=["GA", "PSO", "SA"])
        eje.axhline(coste_milp, color="black", linestyle="--", label=f"Óptimo MILP ({coste_milp:.2f} €)")
        eje.axhline(coste_baseline, color="red", linestyle=":", label=f"Baseline ({coste_baseline:.2f} €)")
        eje.set_title(f"Presupuesto {presupuesto} ({N_SEMILLAS} semillas)")
        eje.set_ylabel("Coste semanal (€)")
        eje.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA, "comparacion_semillas.png"), dpi=150)
    plt.close(fig)

    print("\n" + resumen.to_string(index=False))
    print(f"\nResultados guardados en '{CARPETA}/'.")


if __name__ == "__main__":
    main()
