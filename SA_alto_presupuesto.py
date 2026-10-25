"""
Misma implementación que SA.py con un presupuesto de búsqueda mayor:
hasta 60.000 iteraciones (parada tras 10.000 sin mejora, solo en fase fría).
"""

import numpy as np
import pyomo.environ as pyo

from GA import extraer_parametros, evaluar_soluciones, _construir_df_resultados



# CONFIGURACIÓN DEL SA

N_ITERACIONES_MAX = 60000
ITERACIONES_SIN_MEJORA_MAX = 10000   # criterio de convergencia
TEMPERATURA_MINIMA = 1e-3           # criterio de parada adicional: enfriamiento agotado
FRACCION_T_CONGELACION = 0.01       # el criterio de 'sin mejora' solo se evalúa cuando T <= T0 * esta fracción
TASA_ACEPTACION_INICIAL_DESEADA = 0.8   # para calibrar la temperatura inicial
N_MUESTRAS_CALIBRACION = 40

PROB_PERTURBAR_POTENCIA = 0.03      # prob. de perturbar cada clave de prioridad al generar un vecino
SIGMA_PERTURBACION_POTENCIA = 0.1   # magnitud de esa perturbación

SEMILLA_ALEATORIA = 42



# LÍMITES DE CADA GRUPO DE VARIABLES

def _obtener_limites(parametros: dict) -> dict:
    return {nombre: (0.0, 1.0) for nombre in parametros["grupos"]}


def _a_lote(solucion: dict) -> dict:
    return {clave: valor[None, :] for clave, valor in solucion.items()}


def _evaluar_una_solucion(solucion: dict, parametros: dict) -> dict:
    resultado = evaluar_soluciones(_a_lote(solucion), parametros)
    return {
        "coste": float(resultado["coste"][0]),
        "penalizacion": float(resultado["penalizacion"][0]),
        "fitness": float(resultado["fitness"][0]),
        "decodificado": {clave: valor[0] for clave, valor in resultado["decodificado"].items()},
    }



# 1. GENERACIÓN DE LA SOLUCIÓN INICIAL

def generar_solucion_inicial(parametros: dict, rng: np.random.Generator) -> dict:

    solucion = {nombre: rng.random(parametros["n_periodos"]) for nombre in parametros["grupos"]}
    print("[SA] Solución inicial generada.")
    return solucion


# 2. GENERACIÓN DE UNA SOLUCIÓN VECINA

def generar_solucion_vecina(solucion_actual: dict, parametros: dict, rng: np.random.Generator) -> dict:

    vecina = {clave: valor.copy() for clave, valor in solucion_actual.items()}
    for clave in vecina:
        mascara = rng.random(vecina[clave].shape) < PROB_PERTURBAR_POTENCIA
        ruido = rng.normal(loc=0.0, scale=SIGMA_PERTURBACION_POTENCIA, size=vecina[clave].shape)
        nuevos_valores = np.clip(vecina[clave] + ruido, 0.0, 1.0)
        vecina[clave][mascara] = nuevos_valores[mascara]
    return vecina



# 3. CRITERIO DE ACEPTACIÓN (METROPOLIS)

def aceptar_solucion(fitness_actual: float, fitness_vecina: float, temperatura: float,
                      rng: np.random.Generator) -> bool:

    delta = fitness_vecina - fitness_actual
    if delta <= 0:
        return True
    if temperatura <= 0:
        return False
    probabilidad_aceptacion = np.exp(-delta / temperatura)
    return rng.random() < probabilidad_aceptacion



# CALIBRACIÓN DE LA TEMPERATURA INICIAL

def _calibrar_temperatura_inicial(solucion_inicial: dict, fitness_inicial: float, parametros: dict,
                                   rng: np.random.Generator, n_muestras: int = N_MUESTRAS_CALIBRACION,
                                   tasa_aceptacion_deseada: float = TASA_ACEPTACION_INICIAL_DESEADA) -> float:

    deltas_positivos = []
    for _ in range(n_muestras):
        vecina = generar_solucion_vecina(solucion_inicial, parametros, rng)
        fitness_vecina = _evaluar_una_solucion(vecina, parametros)["fitness"]
        delta = fitness_vecina - fitness_inicial
        if delta > 0:
            deltas_positivos.append(delta)

    if not deltas_positivos:
        return 1.0  # valor de respaldo si no se observó ningún empeoramiento

    delta_medio = float(np.mean(deltas_positivos))
    temperatura_inicial = -delta_medio / np.log(tasa_aceptacion_deseada)
    print(f"[SA] Temperatura inicial calibrada automáticamente: {temperatura_inicial:.2f} "
          f"(delta medio de empeoramiento: {delta_medio:.2f}).")
    return temperatura_inicial



# BUCLE PRINCIPAL DEL SA

def ejecutar_sa(modelo: pyo.ConcreteModel,
                 n_iteraciones_max: int = N_ITERACIONES_MAX,
                 iteraciones_sin_mejora_max: int = ITERACIONES_SIN_MEJORA_MAX,
                 temperatura_minima: float = TEMPERATURA_MINIMA,
                 semilla: int = SEMILLA_ALEATORIA):

    rng = np.random.default_rng(semilla)
    parametros = extraer_parametros(modelo)

    solucion_actual = generar_solucion_inicial(parametros, rng)
    evaluacion_actual = _evaluar_una_solucion(solucion_actual, parametros)
    fitness_actual = evaluacion_actual["fitness"]

    mejor_solucion = {clave: valor.copy() for clave, valor in solucion_actual.items()}
    mejor_fitness = fitness_actual

    temperatura = _calibrar_temperatura_inicial(solucion_actual, fitness_actual, parametros, rng)
    # Enfriamiento geométrico: la temperatura llega a 'temperatura_minima' justo en la última iteración
    alpha = (temperatura_minima / temperatura) ** (1 / n_iteraciones_max)
    # Mientras T > temperatura_congelacion el sistema está "caliente": acepta casi cualquier
    # movimiento y es normal no mejorar el mejor valor. El criterio de convergencia
    # (iteraciones sin mejora) solo debe aplicarse una vez que el sistema se ha enfriado.
    temperatura_congelacion = temperatura * FRACCION_T_CONGELACION

    iteraciones_sin_mejora = 0
    historial_fitness = [mejor_fitness]

    iteracion = 0
    while (iteracion < n_iteraciones_max
           and iteraciones_sin_mejora < iteraciones_sin_mejora_max
           and temperatura > temperatura_minima):

        # a) Generar solución vecina
        solucion_vecina = generar_solucion_vecina(solucion_actual, parametros, rng)

        # b) Evaluar solución
        evaluacion_vecina = _evaluar_una_solucion(solucion_vecina, parametros)

        # c) Aceptar solución según criterio de aceptación
        if aceptar_solucion(fitness_actual, evaluacion_vecina["fitness"], temperatura, rng):
            solucion_actual = solucion_vecina
            fitness_actual = evaluacion_vecina["fitness"]

            if fitness_actual < mejor_fitness - 1e-6:
                mejor_fitness = fitness_actual
                mejor_solucion = {clave: valor.copy() for clave, valor in solucion_actual.items()}
                iteraciones_sin_mejora = 0
            else:
                iteraciones_sin_mejora += 1
        else:
            iteraciones_sin_mejora += 1

        # En fase caliente no se acumula 'sin mejora' (ver temperatura_congelacion)
        if temperatura > temperatura_congelacion:
            iteraciones_sin_mejora = 0

        # d) Reducir temperatura
        temperatura *= alpha

        historial_fitness.append(mejor_fitness)
        iteracion += 1

    if iteraciones_sin_mejora >= iteraciones_sin_mejora_max:
        criterio_parada = "convergencia"
    elif temperatura <= temperatura_minima:
        criterio_parada = "temperatura mínima alcanzada"
    else:
        criterio_parada = "máx. iteraciones"

    print(f"[SA] Parada en la iteración {iteracion} (criterio: {criterio_parada}). "
          f"Temperatura final: {temperatura:.5f}. Mejor fitness: {mejor_fitness:.2f}")

    # Obtener y evaluar la mejor solución encontrada
    evaluacion_final = _evaluar_una_solucion(mejor_solucion, parametros)

    resumen = {
        "algoritmo": "SA",
        "iteraciones_ejecutadas": iteracion,
        "criterio_parada": criterio_parada,
        "temperatura_final": float(temperatura),
        "coste_economico_total": evaluacion_final["coste"],
        "penalizacion_potencia": evaluacion_final["penalizacion"],
        "fitness_final": evaluacion_final["fitness"],
        "consumo_energetico_total_kWh": float(
            np.sum(evaluacion_final["decodificado"]["consumo_total"]) * parametros["dt_horas"]
        ),
        "potencia_contratada_kW": parametros["potencia_contratada"],
        "n_periodos": parametros["n_periodos"],
    }

    print(f"[SA] Coste económico total: {resumen['coste_economico_total']:.2f} €")
    print(f"[SA] Penalización por exceso de potencia: {resumen['penalizacion_potencia']:.2f}")
    print(f"[SA] Consumo energético total: {resumen['consumo_energetico_total_kWh']:.2f} kWh")

    df_resultados = _construir_df_resultados({k: v[None, :] for k, v in evaluacion_final["decodificado"].items()}, parametros)

    return df_resultados, resumen, historial_fitness


if __name__ == "__main__":
    import json
    from procesado_datos import preparar_datos
    from Modelo_energetico import crear_modelo_energetico
    from Restricciones import definir_restricciones
    from Funcion_objetivo import definir_funcion_objetivo

    df_preparado = preparar_datos()
    modelo = crear_modelo_energetico(df_preparado)
    modelo = definir_restricciones(modelo, datos_historicos=df_preparado)
    modelo = definir_funcion_objetivo(modelo)

    df_resultados, resumen, historial_fitness = ejecutar_sa(modelo)

    print("\nResumen final del SA:")
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
    print("\nPrimeras filas de la solución:")
    print(df_resultados.head())
