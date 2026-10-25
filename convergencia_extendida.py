"""
Análisis complementario de convergencia: ejecuta GA y PSO con el mismo número de
vueltas que el SA de bajo presupuesto (5.000) y SIN parada anticipada por
convergencia, para ver si se estancan de verdad o simplemente se les cortó antes.

Genera: resultados/convergencia_GA_PSO_extendida.png

Uso:
    python convergencia_extendida.py
"""

import io
import os
import contextlib

import matplotlib.pyplot as plt

from procesado_datos import preparar_datos
from Modelo_energetico import crear_modelo_energetico
from Restricciones import definir_restricciones
from Funcion_objetivo import definir_funcion_objetivo
from MILP import resolver_milp
from GA import ejecutar_ga
from PSO import ejecutar_pso

N_VUELTAS = 5000
CARPETA = "resultados"


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        df = preparar_datos()
        modelo = crear_modelo_energetico(df)
        modelo = definir_restricciones(modelo, datos_historicos=df)
        modelo = definir_funcion_objetivo(modelo)
        _, _, resumen_milp = resolver_milp(modelo, carpeta_salida=os.path.join(CARPETA, "_tmp_milp"))
        _, resumen_ga, hist_ga = ejecutar_ga(modelo, n_generaciones_max=N_VUELTAS,
                                             generaciones_sin_mejora_max=N_VUELTAS)
        _, resumen_pso, hist_pso = ejecutar_pso(modelo, n_iteraciones_max=N_VUELTAS,
                                                iteraciones_sin_mejora_max=N_VUELTAS)

    print(f"MILP (óptimo): {resumen_milp['coste_economico_total']:.2f} €")
    print(f"GA  {N_VUELTAS} generaciones sin parada anticipada: {resumen_ga['coste_economico_total']:.2f} €")
    print(f"PSO {N_VUELTAS} iteraciones  sin parada anticipada: {resumen_pso['coste_economico_total']:.2f} €")

    os.makedirs(CARPETA, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(hist_ga, label=f"GA ({N_VUELTAS} gen, sin parada anticipada)", color="#4C72B0")
    ax.plot(hist_pso, label=f"PSO ({N_VUELTAS} iter, sin parada anticipada)", color="#DD8452")
    ax.axhline(resumen_milp["coste_economico_total"], color="black", linestyle="--", label="Óptimo MILP")
    ax.set_xlabel("Iteración / generación")
    ax.set_ylabel("Mejor coste semanal (€)")
    ax.set_title(f"GA y PSO con {N_VUELTAS} vueltas, sin parada anticipada por convergencia")
    ax.legend()
    fig.tight_layout()
    ruta = os.path.join(CARPETA, "convergencia_GA_PSO_extendida.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"Gráfica guardada en {ruta}")


if __name__ == "__main__":
    main()
