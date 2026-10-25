"""
Resolución del problema energético mediante un Algoritmo Genético (GA).

Sigue el Pseudocódigo 4:
    1. Recibir modelo, función objetivo y restricciones.
    2. Generar población inicial.
    3. MIENTRAS no se cumpla el criterio de parada:
        a. Evaluar soluciones.
        b. Seleccionar, cruzar y mutar individuos.
        c. Generar nueva población.
    4. Obtener y evaluar mejor solución.

"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo



# CONFIGURACIÓN DEL GA

TAM_POBLACION = 60
N_GENERACIONES_MAX = 150
GENERACIONES_SIN_MEJORA_MAX = 30     # criterio de convergencia
PROB_CRUCE = 0.8
PROB_MUTACION = 0.1
SIGMA_MUTACION = 0.15                 # magnitud de la mutación "creep" gaussiana
TAM_TORNEO = 3
ELITISMO = 2                          # nº de mejores individuos que pasan intactos
PESO_PENALIZACION_POTENCIA = 100.0    # penalización por kWh que exceda la potencia contratada
SEMILLA_ALEATORIA = 42



# 1. EXTRACCIÓN DE PARÁMETROS DEL MODELO

def extraer_parametros(modelo: pyo.ConcreteModel) -> dict:
   
    df = modelo.datos
    n_periodos = len(modelo.T)
    periodos_por_dia = 96
    dt = modelo.dt_horas
    cargas = modelo.cargas_controlables
    servicio = modelo.servicio

    p_max_hvac = cargas["climatizacion"]["potencia_max_kW"]
    ventana_hvac = servicio["ventana_periodos"]
    k_hvac = servicio["energia_hvac_ventana_kWh"] / (p_max_hvac * dt)

    grupos = {
        "climatizacion": {
            "ventana": ventana_hvac,
            "k": np.minimum(k_hvac, ventana_hvac).astype(float),
            "potencia": p_max_hvac,
        }
    }
    for nombre, disp in cargas["otros_dispositivos"]["dispositivos"].items():
        grupos[nombre] = {
            "ventana": periodos_por_dia,
            "k": servicio["usos_por_dia"][nombre].astype(float),
            "potencia": disp["potencia_nominal_kW"],
        }

    return {
        "n_periodos": n_periodos,
        "periodos_por_dia": periodos_por_dia,
        "n_dias": n_periodos // periodos_por_dia,
        "dt_horas": dt,
        "base_load": df["base_load_kW"].to_numpy(),
        "precio_energia": df["precio_energia_eur_kWh"].to_numpy(),
        "potencia_contratada": modelo.potencia_contratada_kW,
        "grupos": grupos,
        "timestamps": df["timestamp"].to_numpy(),
    }



# 2. GENERACIÓN DE LA POBLACIÓN INICIAL

def generar_poblacion_inicial(tam_poblacion: int, parametros: dict, rng: np.random.Generator) -> dict:
    """Población inicial: para cada carga controlable, una matriz (N, n_periodos) de prioridades aleatorias en [0, 1]."""
    poblacion = {
        nombre: rng.random((tam_poblacion, parametros["n_periodos"]))
        for nombre in parametros["grupos"]
    }
    print(f"[GA] Población inicial generada: {tam_poblacion} individuos.")
    return poblacion


# 3. DECODIFICACIÓN DE INDIVIDUOS

def activar_por_prioridad(claves: np.ndarray, ventana: int, k: np.ndarray) -> np.ndarray:
    
    n_ind, n_periodos = claves.shape
    n_ventanas = n_periodos // ventana
    claves_v = claves.reshape(n_ind, n_ventanas, ventana)
    rango = np.argsort(np.argsort(-claves_v, axis=2), axis=2)      # 0 = máxima prioridad
    h = np.clip(k[None, :, None] - rango, 0.0, 1.0)
    return h.reshape(n_ind, n_periodos)


def decodificar_poblacion(poblacion: dict, parametros: dict) -> dict:
    """Traduce los genes de toda la población a potencias reales (kW) por intervalo y carga."""
    decodificado = {}
    consumo_total = np.broadcast_to(parametros["base_load"], (next(iter(poblacion.values())).shape[0],
                                                                 parametros["n_periodos"])).copy()
    for nombre, g in parametros["grupos"].items():
        h = activar_por_prioridad(poblacion[nombre], g["ventana"], g["k"])
        potencia = h * g["potencia"]
        decodificado[f"act_{nombre}"] = h
        decodificado[f"P_{nombre}"] = potencia
        consumo_total += potencia

    decodificado["consumo_total"] = consumo_total
    return decodificado



# 4. EVALUACIÓN DE SOLUCIONES (FUNCIÓN DE APTITUD)

def evaluar_soluciones(poblacion: dict, parametros: dict) -> dict:
    """
    Evalúa toda la población: coste de energía (igual que Funcion_objetivo.py:
    potencia x 0,25 h x precio) y penalización por exceder la potencia contratada.
    Devuelve coste, penalización y fitness (coste + penalización) por individuo.
    """
    decodificado = decodificar_poblacion(poblacion, parametros)
    dt = parametros["dt_horas"]

    coste = np.sum(decodificado["consumo_total"] * dt * parametros["precio_energia"][None, :], axis=1)

    exceso_potencia = np.maximum(decodificado["consumo_total"] - parametros["potencia_contratada"], 0.0)
    penalizacion = PESO_PENALIZACION_POTENCIA * np.sum(exceso_potencia * dt, axis=1)

    fitness = coste + penalizacion
    return {"decodificado": decodificado, "coste": coste, "penalizacion": penalizacion, "fitness": fitness}


# 5. SELECCIÓN, CRUCE Y MUTACIÓN

def seleccionar(poblacion: dict, fitness: np.ndarray, rng: np.random.Generator, tam_torneo: int = TAM_TORNEO) -> dict:
    """Selección por torneo: elige, para cada hueco de la nueva población, al mejor de 'tam_torneo' candidatos aleatorios."""
    n = fitness.shape[0]
    indices_ganadores = np.empty(n, dtype=int)
    for i in range(n):
        candidatos = rng.integers(0, n, size=tam_torneo)
        indices_ganadores[i] = candidatos[np.argmin(fitness[candidatos])]  # menor fitness = mejor (minimizamos coste)

    return {clave: valor[indices_ganadores] for clave, valor in poblacion.items()}


def cruzar(poblacion: dict, parametros: dict, rng: np.random.Generator, prob_cruce: float = PROB_CRUCE) -> dict:
    """Cruce uniforme: para cada pareja de padres consecutivos, cada gen se hereda de uno u otro al azar."""
    n = next(iter(poblacion.values())).shape[0]
    hijos = {clave: valor.copy() for clave, valor in poblacion.items()}

    for i in range(0, n - 1, 2):
        if rng.random() < prob_cruce:
            for clave in poblacion:
                mascara = rng.random(poblacion[clave].shape[1]) < 0.5
                hijos[clave][i][mascara], hijos[clave][i + 1][mascara] = (
                    poblacion[clave][i + 1][mascara].copy(),
                    poblacion[clave][i][mascara].copy(),
                )
    return hijos


def mutar(poblacion: dict, parametros: dict, rng: np.random.Generator, prob_mutacion: float = PROB_MUTACION) -> dict:
    """
    Con probabilidad prob_mutacion, cada gen recibe una perturbación
    gaussiana pequeña (acotada a [0, 1]), lo que permite ajustar poco a poco el orden de
    prioridades además de explorar.
    """
    poblacion = {clave: valor.copy() for clave, valor in poblacion.items()}
    for clave in poblacion:
        mascara = rng.random(poblacion[clave].shape) < prob_mutacion
        ruido = rng.normal(loc=0.0, scale=SIGMA_MUTACION, size=poblacion[clave].shape)
        nuevos_valores = np.clip(poblacion[clave] + ruido, 0.0, 1.0)
        poblacion[clave][mascara] = nuevos_valores[mascara]
    return poblacion


def generar_nueva_poblacion(poblacion: dict, resultado_evaluacion: dict, parametros: dict,
                             rng: np.random.Generator, elitismo: int = ELITISMO) -> dict:
    """Selecciona, cruza y muta para producir la siguiente generación, conservando los 'elitismo' mejores individuos."""
    fitness = resultado_evaluacion["fitness"]
    n = fitness.shape[0]

    indices_elite = np.argsort(fitness)[:elitismo]

    seleccionados = seleccionar(poblacion, fitness, rng)
    cruzados = cruzar(seleccionados, parametros, rng)
    nueva_poblacion = mutar(cruzados, parametros, rng)

    # Elitismo: los mejores individuos de la generación anterior sustituyen a los primeros de la nueva
    for clave in nueva_poblacion:
        for pos, idx_elite in enumerate(indices_elite):
            nueva_poblacion[clave][pos] = poblacion[clave][idx_elite]

    return nueva_poblacion


# BUCLE PRINCIPAL DEL GA

def ejecutar_ga(modelo: pyo.ConcreteModel,
                 tam_poblacion: int = TAM_POBLACION,
                 n_generaciones_max: int = N_GENERACIONES_MAX,
                 generaciones_sin_mejora_max: int = GENERACIONES_SIN_MEJORA_MAX,
                 semilla: int = SEMILLA_ALEATORIA):
    """
    Ejecuta el flujo completo del Pseudocódigo 4 y devuelve
    (mejor_solucion_decodificada, resumen, historial_fitness).
    """
    rng = np.random.default_rng(semilla)
    parametros = extraer_parametros(modelo)

    poblacion = generar_poblacion_inicial(tam_poblacion, parametros, rng)

    mejor_fitness_global = np.inf
    mejor_individuo_global = None
    generaciones_sin_mejora = 0
    historial_fitness = []

    generacion = 0
    while generacion < n_generaciones_max and generaciones_sin_mejora < generaciones_sin_mejora_max:
        resultado_evaluacion = evaluar_soluciones(poblacion, parametros)

        idx_mejor = np.argmin(resultado_evaluacion["fitness"])
        mejor_fitness_generacion = resultado_evaluacion["fitness"][idx_mejor]
        historial_fitness.append(mejor_fitness_generacion)

        if mejor_fitness_generacion < mejor_fitness_global - 1e-6:
            mejor_fitness_global = mejor_fitness_generacion
            mejor_individuo_global = {clave: valor[idx_mejor].copy() for clave, valor in poblacion.items()}
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        poblacion = generar_nueva_poblacion(poblacion, resultado_evaluacion, parametros, rng)
        generacion += 1

    criterio_parada = "convergencia" if generaciones_sin_mejora >= generaciones_sin_mejora_max else "máx. generaciones"
    print(f"[GA] Parada en la generación {generacion} (criterio: {criterio_parada}). "
          f"Mejor fitness: {mejor_fitness_global:.2f}")

    # Obtener y evaluar la mejor solución encontrada
    mejor_poblacion_individual = {clave: valor[None, :] for clave, valor in mejor_individuo_global.items()}
    evaluacion_final = evaluar_soluciones(mejor_poblacion_individual, parametros)

    resumen = {
        "algoritmo": "GA",
        "generaciones_ejecutadas": generacion,
        "criterio_parada": criterio_parada,
        "coste_economico_total": float(evaluacion_final["coste"][0]),
        "penalizacion_potencia": float(evaluacion_final["penalizacion"][0]),
        "fitness_final": float(evaluacion_final["fitness"][0]),
        "consumo_energetico_total_kWh": float(np.sum(evaluacion_final["decodificado"]["consumo_total"][0]) * parametros["dt_horas"]),
        "potencia_contratada_kW": parametros["potencia_contratada"],
        "n_periodos": parametros["n_periodos"],
    }

    print(f"[GA] Coste económico total: {resumen['coste_economico_total']:.2f} €")
    print(f"[GA] Penalización por exceso de potencia: {resumen['penalizacion_potencia']:.2f}")
    print(f"[GA] Consumo energético total: {resumen['consumo_energetico_total_kWh']:.2f} kWh")

    df_resultados = _construir_df_resultados(evaluacion_final["decodificado"], parametros)

    return df_resultados, resumen, historial_fitness


def _construir_df_resultados(decodificado: dict, parametros: dict) -> pd.DataFrame:
    """Construye un DataFrame con la mejor solución, en el mismo formato que MILP.py, para poder comparar."""
    return pd.DataFrame({
        "periodo": np.arange(parametros["n_periodos"]),
        "timestamp": parametros["timestamps"],
        "P_climatizacion_kW": decodificado["P_climatizacion"][0],
        "estado_lavadora": decodificado["act_lavadora"][0].round().astype(int),
        "estado_lavavajillas": decodificado["act_lavavajillas"][0].round().astype(int),
        "estado_calentador_agua": decodificado["act_calentador_agua"][0].round().astype(int),
        "consumo_total_kW": decodificado["consumo_total"][0],
    })


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

    df_resultados, resumen, historial_fitness = ejecutar_ga(modelo)

    print("\nResumen final del GA:")
    print(json.dumps(resumen, indent=2, ensure_ascii=False, default=str))
    print("\nPrimeras filas de la solución:")
    print(df_resultados.head())
