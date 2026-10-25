"""
Definición de las restricciones del problema energético.

Sigue el Pseudocódigo 2:
    1. Recibir modelo energético y datos.
    2. Definir restricciones de funcionamiento.
    3. Definir restricciones de potencia.
    4. Definir restricciones de confort.
    5. Añadir las restricciones al modelo.
    6. Devolver modelo con restricciones.

Las restricciones obligan al modelo a prestar el mismo servicio que el histórico
real, de modo que el histórico sea una solución factible y la comparación con el
baseline sea justa (el optimizador solo decide cuando consumir):

    FUNCIONAMIENTO  Cada electrodoméstico (lavadora, lavavajillas, calentador) debe
                    usarse, cada día, el mismo número de intervalos que en el
                    histórico; puede recolocarlos en cualquier momento del mismo día.
    CONFORT         La climatización debe entregar, en cada ventana de confort
                    (por defecto 4 h), al menos la energía que entregó el histórico;
                    puede repartirla dentro de la ventana (inercia térmica).
    POTENCIA        La demanda total del piso no puede superar la potencia contratada,
                    calculada a partir del consumo máximo histórico (+ margen).

"""

import math
import pandas as pd
import pyomo.environ as pyo

from procesado_datos import preparar_datos
from Modelo_energetico import crear_modelo_energetico, PERIODOS_POR_DIA



# CONFIGURACIÓN DE RESTRICCIONES

# Margen de seguridad sobre el consumo máximo histórico para fijar la potencia contratada
MARGEN_POTENCIA_CONTRATADA = 1.15



# UTILIDADES

def _calcular_potencia_contratada(datos_historicos: pd.DataFrame, margen: float) -> float:
    """
    Potencia contratada a partir del consumo máximo histórico del piso (sin vehículo
    eléctrico), con un margen de seguridad, redondeada al 0.5 kW superior.
    """
    consumo_max_historico = float(datos_historicos["demanda_sin_ev_kW"].max())
    return math.ceil(consumo_max_historico * margen * 2) / 2


def _nombres_electrodomesticos(modelo: pyo.ConcreteModel) -> dict:
    """Devuelve {nombre_carga: variable_de_estado} de los electrodomésticos del modelo."""
    return {
        "lavadora": modelo.estado_lavadora,
        "lavavajillas": modelo.estado_lavavajillas,
        "calentador_agua": modelo.estado_calentador_agua,
    }



# 1. RESTRICCIONES DE FUNCIONAMIENTO

def definir_restricciones_funcionamiento(modelo: pyo.ConcreteModel):
   
    n_dias = len(modelo.T) // PERIODOS_POR_DIA
    usos_por_dia = modelo.servicio["usos_por_dia"]

    for nombre, estado in _nombres_electrodomesticos(modelo).items():
        usos = usos_por_dia[nombre]

        def regla_usos_diarios(m, dia, estado=estado, usos=usos):
            periodos_del_dia = range(dia * PERIODOS_POR_DIA, (dia + 1) * PERIODOS_POR_DIA)
            return sum(estado[t] for t in periodos_del_dia) == int(usos[dia])

        setattr(modelo, f"restr_usos_diarios_{nombre}",
                pyo.Constraint(pyo.RangeSet(0, n_dias - 1), rule=regla_usos_diarios))
        print(f"[Restricciones] Funcionamiento '{nombre}': usos por día fijados al histórico {usos.tolist()}.")



# 2. RESTRICCIONES DE POTENCIA

def definir_restricciones_potencia(modelo: pyo.ConcreteModel, datos_historicos: pd.DataFrame,
                                    margen: float = MARGEN_POTENCIA_CONTRATADA):
    """Limita la demanda instantánea del piso a la potencia contratada (calculada del histórico)."""
    potencia_contratada = _calcular_potencia_contratada(datos_historicos, margen)
    modelo.potencia_contratada_kW = potencia_contratada

    otros = modelo.cargas_controlables["otros_dispositivos"]["dispositivos"]
    p_lavadora = otros["lavadora"]["potencia_nominal_kW"]
    p_lavavajillas = otros["lavavajillas"]["potencia_nominal_kW"]
    p_calentador = otros["calentador_agua"]["potencia_nominal_kW"]

    def regla_potencia_contratada(m, t):
        demanda_total = (
            m.base_load_kW[t]
            + m.P_climatizacion[t]
            + p_lavadora * m.estado_lavadora[t]
            + p_lavavajillas * m.estado_lavavajillas[t]
            + p_calentador * m.estado_calentador_agua[t]
        )
        return demanda_total <= m.potencia_contratada_kW

    modelo.restr_potencia_contratada = pyo.Constraint(modelo.T, rule=regla_potencia_contratada)
    print(f"[Restricciones] Potencia contratada calculada: {potencia_contratada} kW "
          f"(máximo histórico x{margen}).")



# 3. RESTRICCIONES DE CONFORT

def definir_restricciones_confort(modelo: pyo.ConcreteModel):
    """
    Confort de la climatización: en cada ventana de confort debe entregarse, como
    mínimo, la misma energía que entregó el histórico. El usuario nunca
    recibe menos climatización que la que realmente tuvo; el optimizador solo puede
    reubicarla dentro de la ventana.
    """
    ventana = modelo.servicio["ventana_periodos"]
    energia_requerida = modelo.servicio["energia_hvac_ventana_kWh"]
    n_ventanas = len(energia_requerida)

    def regla_confort_ventana(m, w):
        periodos = range(w * ventana, (w + 1) * ventana)
        return sum(m.P_climatizacion[t] for t in periodos) * m.dt_horas >= float(energia_requerida[w])

    modelo.restr_confort_climatizacion = pyo.Constraint(pyo.RangeSet(0, n_ventanas - 1), rule=regla_confort_ventana)
    print(f"[Restricciones] Confort climatización: energía mínima por ventana de "
          f"{ventana * 15 / 60:g} h igual a la del histórico ({n_ventanas} ventanas).")



# COMPILACIÓN: AÑADIR TODAS LAS RESTRICCIONES AL MODELO

def definir_restricciones(modelo: pyo.ConcreteModel, datos_historicos: pd.DataFrame = None) -> pyo.ConcreteModel:
    if datos_historicos is None:
        datos_historicos = modelo.datos

    definir_restricciones_funcionamiento(modelo)
    definir_restricciones_potencia(modelo, datos_historicos)
    definir_restricciones_confort(modelo)

    n_bloques = sum(1 for _ in modelo.component_objects(pyo.Constraint, active=True))
    print(f"[Restricciones] Modelo con restricciones listo: {n_bloques} bloques de restricciones añadidos.")
    return modelo


if __name__ == "__main__":
    df_preparado = preparar_datos()
    modelo = crear_modelo_energetico(df_preparado)
    modelo = definir_restricciones(modelo, datos_historicos=df_preparado)

    print("\nRestricciones añadidas al modelo:")
    for c in modelo.component_objects(pyo.Constraint, active=True):
        print(f"  - {c.name}: {len(c)} restricciones")
