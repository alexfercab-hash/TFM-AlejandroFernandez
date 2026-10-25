"""
Misma implementación que PSO.py con un presupuesto de búsqueda mayor:
150 partículas, hasta 500 iteraciones (parada tras 100 sin mejora).
"""

import numpy as np
import pyomo.environ as pyo

from GA import (
    extraer_parametros,
    evaluar_soluciones,
    _construir_df_resultados,
    PESO_PENALIZACION_POTENCIA,
)


# CONFIGURACIÓN DEL PSO


N_PARTICULAS = 150
N_ITERACIONES_MAX = 500
ITERACIONES_SIN_MEJORA_MAX = 100   # criterio de convergencia
W_INERCIA_INICIAL = 0.9           # inercia decreciente: exploración al inicio...
W_INERCIA_FINAL = 0.4             # ...explotación al final
C1_COGNITIVO = 1.5                # peso de la atracción hacia el mejor histórico propio (pbest)
C2_SOCIAL = 1.5                   # peso de la atracción hacia el mejor del enjambre (gbest)
FRACCION_VELOCIDAD_MAX = 0.2      # velocidad máxima, como fracción del rango de cada variable
SEMILLA_ALEATORIA = 42



# LÍMITES DE CADA GRUPO DE VARIABLES (necesarios para inicializar y acotar)

def _obtener_limites(parametros: dict) -> dict:
    return {nombre: (0.0, 1.0) for nombre in parametros["grupos"]}



# 1. INICIALIZACIÓN DE PARTÍCULAS

def inicializar_particulas(n_particulas: int, parametros: dict, rng: np.random.Generator) -> dict:
    n_periodos = parametros["n_periodos"]
    limites = _obtener_limites(parametros)

    posiciones = {}
    velocidades = {}
    for clave, (lim_inf, lim_sup) in limites.items():
        forma = (n_particulas, n_periodos)
        posiciones[clave] = rng.uniform(lim_inf, lim_sup, size=forma)
        v_max = FRACCION_VELOCIDAD_MAX * (lim_sup - lim_inf)
        velocidades[clave] = rng.uniform(-v_max, v_max, size=forma)

    print(f"[PSO] Enjambre inicializado: {n_particulas} partículas.")
    return {"posiciones": posiciones, "velocidades": velocidades}



# 2. PREPARAR POSICIONES PARA EVALUACIÓN

def _preparar_para_evaluacion(posiciones: dict, parametros: dict) -> dict:
    limites = _obtener_limites(parametros)
    return {clave: np.clip(valor, *limites[clave]) for clave, valor in posiciones.items()}


def evaluar_particulas(posiciones: dict, parametros: dict) -> dict:
    posiciones_evaluables = _preparar_para_evaluacion(posiciones, parametros)
    return evaluar_soluciones(posiciones_evaluables, parametros)



# 3. ACTUALIZACIÓN DE VELOCIDADES Y POSICIONES

def actualizar_velocidad_y_posicion(posiciones: dict, velocidades: dict, pbest_posiciones: dict,
                                     gbest_posicion: dict, w: float, parametros: dict,
                                     rng: np.random.Generator, c1: float = C1_COGNITIVO,
                                     c2: float = C2_SOCIAL) -> None:
   
    limites = _obtener_limites(parametros)

    for clave in posiciones:
        lim_inf, lim_sup = limites[clave]
        v_max = FRACCION_VELOCIDAD_MAX * (lim_sup - lim_inf)

        r1 = rng.random(posiciones[clave].shape)
        r2 = rng.random(posiciones[clave].shape)

        velocidades[clave][:] = (
            w * velocidades[clave]
            + c1 * r1 * (pbest_posiciones[clave] - posiciones[clave])
            + c2 * r2 * (gbest_posicion[clave][None, :] - posiciones[clave])
        )
        velocidades[clave][:] = np.clip(velocidades[clave], -v_max, v_max)

        posiciones[clave][:] = np.clip(posiciones[clave] + velocidades[clave], lim_inf, lim_sup)



# 4. ACTUALIZACIÓN DE MEJORES SOLUCIONES (pbest / gbest)

def actualizar_mejores_soluciones(posiciones: dict, fitness: np.ndarray,
                                   pbest_posiciones: dict, pbest_fitness: np.ndarray,
                                   gbest_posicion: dict, gbest_fitness: float):
    mejora = fitness < pbest_fitness
    for clave in posiciones:
        pbest_posiciones[clave][mejora] = posiciones[clave][mejora]
    pbest_fitness[mejora] = fitness[mejora]

    idx_mejor_particula = np.argmin(pbest_fitness)
    if pbest_fitness[idx_mejor_particula] < gbest_fitness:
        gbest_fitness = pbest_fitness[idx_mejor_particula]
        gbest_posicion = {clave: valor[idx_mejor_particula].copy() for clave, valor in pbest_posiciones.items()}
        return gbest_posicion, gbest_fitness, True

    return gbest_posicion, gbest_fitness, False


# BUCLE PRINCIPAL DEL PSO

def ejecutar_pso(modelo: pyo.ConcreteModel,
                  n_particulas: int = N_PARTICULAS,
                  n_iteraciones_max: int = N_ITERACIONES_MAX,
                  iteraciones_sin_mejora_max: int = ITERACIONES_SIN_MEJORA_MAX,
                  semilla: int = SEMILLA_ALEATORIA):
   
    rng = np.random.default_rng(semilla)
    parametros = extraer_parametros(modelo)

    enjambre = inicializar_particulas(n_particulas, parametros, rng)
    posiciones, velocidades = enjambre["posiciones"], enjambre["velocidades"]

    # Evaluación inicial: pbest de cada partícula = su posición inicial
    resultado_inicial = evaluar_particulas(posiciones, parametros)
    pbest_posiciones = {clave: valor.copy() for clave, valor in posiciones.items()}
    pbest_fitness = resultado_inicial["fitness"].copy()

    idx_gbest = np.argmin(pbest_fitness)
    gbest_posicion = {clave: valor[idx_gbest].copy() for clave, valor in pbest_posiciones.items()}
    gbest_fitness = pbest_fitness[idx_gbest]

    iteraciones_sin_mejora = 0
    historial_fitness = [gbest_fitness]

    iteracion = 0
    while iteracion < n_iteraciones_max and iteraciones_sin_mejora < iteraciones_sin_mejora_max:
        # Inercia decreciente: más exploración al principio, más explotación al final
        w = W_INERCIA_INICIAL - (W_INERCIA_INICIAL - W_INERCIA_FINAL) * (iteracion / max(n_iteraciones_max - 1, 1))

        actualizar_velocidad_y_posicion(posiciones, velocidades, pbest_posiciones, gbest_posicion, w, parametros, rng)

        resultado = evaluar_particulas(posiciones, parametros)

        gbest_posicion, gbest_fitness_nuevo, hubo_mejora = actualizar_mejores_soluciones(
            posiciones, resultado["fitness"], pbest_posiciones, pbest_fitness, gbest_posicion, gbest_fitness
        )

        if hubo_mejora:
            gbest_fitness = gbest_fitness_nuevo
            iteraciones_sin_mejora = 0
        else:
            iteraciones_sin_mejora += 1

        historial_fitness.append(gbest_fitness)
        iteracion += 1

    criterio_parada = "convergencia" if iteraciones_sin_mejora >= iteraciones_sin_mejora_max else "máx. iteraciones"
    print(f"[PSO] Parada en la iteración {iteracion} (criterio: {criterio_parada}). "
          f"Mejor fitness: {gbest_fitness:.2f}")

    # Obtener y evaluar la mejor solución encontrada (gbest)
    gbest_individual = {clave: valor[None, :] for clave, valor in gbest_posicion.items()}
    evaluacion_final = evaluar_particulas(gbest_individual, parametros)

    resumen = {
        "algoritmo": "PSO",
        "iteraciones_ejecutadas": iteracion,
        "criterio_parada": criterio_parada,
        "coste_economico_total": float(evaluacion_final["coste"][0]),
        "penalizacion_potencia": float(evaluacion_final["penalizacion"][0]),
        "fitness_final": float(evaluacion_final["fitness"][0]),
        "consumo_energetico_total_kWh": float(
            np.sum(evaluacion_final["decodificado"]["consumo_total"][0]) * parametros["dt_horas"]
        ),
        "potencia_contratada_kW": parametros["potencia_contratada"],
        "n_periodos": parametros["n_periodos"],
    }

    print(f"[PSO] Coste económico total: {resumen['coste_economico_total']:.2f} €")
    print(f"[PSO] Penalización por exceso de potencia: {resumen['penalizacion_potencia']:.2f}")
    print(f"[PSO] Consumo energético total: {resumen['consumo_energetico_total_kWh']:.2f} kWh")

    df_resultados = _construir_df_resultados(evaluacion_final["decodificado"], parametros)

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

    df_resultados, resumen, historial_fitness = ejecutar_pso(modelo)

    print("\nResumen final del PSO:")
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
    print("\nPrimeras filas de la solución:")
    print(df_resultados.head())
