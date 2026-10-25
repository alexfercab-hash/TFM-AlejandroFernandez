"""
Definición de la función objetivo del problema energético.

El objetivo es MINIMIZAR ÚNICAMENTE EL COSTE
ECONÓMICO DE LA ENERGÍA, con la tarifa española 2.0TD (3 periodos).

En este proyecto todas las restricciones definidas en
Restricciones.py son estrictas (obligatorias): el confort es estricto y los ciclos
de los electrodomésticos son obligatorios. Actualmente no existen
restricciones "blandas" que puedan incumplirse, y el término de penalización es 0.
Se deja preparada la función calcular_penalizaciones() como mecanismo extensible,
por si en el futuro se decide relajar alguna restricción con una variable de holgura.

"""

import pyomo.environ as pyo

from procesado_datos import preparar_datos
from Modelo_energetico import crear_modelo_energetico
from Restricciones import definir_restricciones



# 1. CÁLCULO DEL CONSUMO ENERGÉTICO

def calcular_consumo_energetico(modelo: pyo.ConcreteModel) -> pyo.ConcreteModel:
    
    otros = modelo.cargas_controlables["otros_dispositivos"]["dispositivos"]
    p_lavadora = otros["lavadora"]["potencia_nominal_kW"]
    p_lavavajillas = otros["lavavajillas"]["potencia_nominal_kW"]
    p_calentador = otros["calentador_agua"]["potencia_nominal_kW"]

    # --- Potencia total demandada por la vivienda en cada periodo (kW) ---
    def regla_consumo_total(m, t):
        return (
            m.base_load_kW[t]
            + m.P_climatizacion[t]
            + p_lavadora * m.estado_lavadora[t]
            + p_lavavajillas * m.estado_lavavajillas[t]
            + p_calentador * m.estado_calentador_agua[t]
        )

    modelo.consumo_total_kW = pyo.Expression(modelo.T, rule=regla_consumo_total)

    # --- Consumo energético total del horizonte (kWh), informativo ---
    modelo.consumo_energetico_total_kWh = pyo.Expression(
        expr=sum(modelo.consumo_total_kW[t] for t in modelo.T) * modelo.dt_horas
    )

    print("[Función objetivo] Consumo energético total definido (piso sin FV, sin batería, sin EV).")
    return modelo



# 2. CÁLCULO DEL COSTE ENERGÉTICO

def calcular_coste_energetico(modelo: pyo.ConcreteModel) -> pyo.ConcreteModel:
    """
    Coste de energía del horizonte, en euros con el precio de cada intervalo según la tarifa 2.0TD (tarifa_2_0TD.py).

    Solo se incluye el TÉRMINO DE ENERGÍA. El término de potencia (coste fijo por
    la potencia contratada) no se incluye: no depende de cuándo se consume, así
    que no influye en ninguna decisión del modelo. La potencia contratada actúa
    solo como restricción de potencia máxima por intervalo (Restricciones.py).
    """
    def coste_periodo(t):
        return modelo.consumo_total_kW[t] * modelo.dt_horas * modelo.precio_energia_eur_kWh[t]

    modelo.coste_energetico_total = pyo.Expression(
        expr=sum(coste_periodo(t) for t in modelo.T)
    )

    print("[Función objetivo] Coste de energía definido: potencia x 0,25 h x precio 2.0TD por intervalo.")
    return modelo


# 3. PENALIZACIONES POR INCUMPLIMIENTO DE RESTRICCIONES

def calcular_penalizaciones(modelo: pyo.ConcreteModel) -> pyo.Expression:
  
    penalizacion_total = 0
    print("[Función objetivo] Penalizaciones por incumplimiento de restricciones: 0 "
          "(todas las restricciones del modelo son estrictas).")
    return penalizacion_total



# 4. DEFINICIÓN DE LA FUNCIÓN OBJETIVO

def definir_funcion_objetivo(modelo: pyo.ConcreteModel) -> pyo.ConcreteModel:
  
    modelo = calcular_consumo_energetico(modelo)
    modelo = calcular_coste_energetico(modelo)
    penalizaciones = calcular_penalizaciones(modelo)

    modelo.objetivo = pyo.Objective(
        expr=modelo.coste_energetico_total + penalizaciones,
        sense=pyo.minimize,
    )

    print("[Función objetivo] Función objetivo definida: minimizar coste económico total.")
    return modelo


if __name__ == "__main__":
    df_preparado = preparar_datos()
    modelo = crear_modelo_energetico(df_preparado)
    modelo = definir_restricciones(modelo, datos_historicos=df_preparado)
    modelo = definir_funcion_objetivo(modelo)

    solver = pyo.SolverFactory("cbc")
    resultado = solver.solve(modelo, tee=False)

    print("\nEstado del solver:", resultado.solver.status)
    print("Condición de terminación:", resultado.solver.termination_condition)
    print(f"\nCoste económico total del horizonte (1 semana): {pyo.value(modelo.coste_energetico_total):.2f} €")
    print(f"Consumo energético total del horizonte: {pyo.value(modelo.consumo_energetico_total_kWh):.2f} kWh")
