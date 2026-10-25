"""

Definición del problema energético: qué se está optimizando.

Sigue el Pseudocódigo 2:
    1. Recibir datos preparados.
    2. Definir las cargas controlables.
    3. Definir parámetros de las cargas (potencia, estados de funcionamiento, parámetros específicos).
    4. Definir horizonte temporal.
    5. Definir variables de decisión.
    6. Crear estructura del modelo energético.
    7. Devolver modelo energético.

ESCENARIO: piso en España sin fotovoltaica, sin batería y sin cargador de coche.

Este script solo define la estructura del modelo (conjuntos, parámetros, variables).
Las restricciones se añaden en Restricciones.py y la función objetivo en Funcion_objetivo.py.

"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from procesado_datos import preparar_datos



# CONFIGURACIÓN DEL HORIZONTE TEMPORAL

HORIZON_DIAS = 7
STEP_MINUTOS = 15
PERIODOS_POR_DIA = (24 * 60) // STEP_MINUTOS          # 96
N_PERIODOS = HORIZON_DIAS * PERIODOS_POR_DIA          # 672
DT_HORAS = STEP_MINUTOS / 60                          # duración de cada periodo, en horas

# Semana de referencia: lunes 13 al domingo 19 de enero de 2025 (semana completa, sin
# festivos nacionales). Se evita la semana del 1 al 7 de enero (2 festivos + fin de semana).
FECHA_SEMANA_REFERENCIA = "2025-01-13"

# Ventana de confort de la climatización (nº de intervalos de 15 min; 16 = 4 h).
# En cada ventana debe entregarse al menos la energía de climatización del histórico.
VENTANA_CONFORT_HVAC_PERIODOS = 16



# DEFINICIÓN DE CARGAS CONTROLABLES Y SUS PARÁMETROS

# Cada carga indica:
#   - tipo: "continua" (potencia variable, kW) o "binaria" (encendido/apagado)
#   - columna_datos: columna del dataset asociada
#   - potencia_max_kW / potencia_nominal_kW: se estiman del histórico de la vivienda


CARGAS_CONTROLABLES = {
    "climatizacion": {
        "descripcion": "Climatización (HVAC)",
        "tipo": "continua",
        "columna_datos": "hvac_power_kW",
        "potencia_max_kW": None,                    # se estima del histórico
        "parametros_especificos": {
            "ventana_confort_periodos": VENTANA_CONFORT_HVAC_PERIODOS,
        },
    },
    "otros_dispositivos": {
        "descripcion": "Electrodomésticos desplazables",
        "dispositivos": {
            "lavadora": {
                "tipo": "binaria",
                "columna_datos": "washing_machine_on",
                "potencia_nominal_kW": None,        # se estima del histórico
            },
            "lavavajillas": {
                "tipo": "binaria",
                "columna_datos": "dishwasher_on",
                "potencia_nominal_kW": None,
            },
            "calentador_agua": {
                "tipo": "binaria",
                "columna_datos": "water_heater_on",
                "potencia_nominal_kW": None,
            },
        },
    },
}



# ESTIMACIÓN DE PARÁMETROS DE LAS CARGAS A PARTIR DEL HISTÓRICO

def _estimar_potencia_nominal_binaria(df: pd.DataFrame, columna_flag: str,
                                       otras_columnas_flag: list) -> float:
    """
    Estima la potencia nominal de una carga on/off como la potencia media del
    residuo (demanda del piso - carga base - climatización) en los intervalos donde
    SOLO esa carga está encendida.
    """
    residuo = df["demanda_sin_ev_kW"] - df["base_load_kW"] - df["hvac_power_kW"]

    mascara_solo_esta = (df[columna_flag] == 1)
    for otra in otras_columnas_flag:
        mascara_solo_esta &= (df[otra] == 0)

    valores = residuo[mascara_solo_esta]
    if len(valores) == 0:
        raise ValueError(f"No hay intervalos con solo '{columna_flag}' encendido para estimar su potencia.")
    return float(valores.mean())


def estimar_parametros_desde_datos(df_historico: pd.DataFrame, cargas: dict) -> dict:
  
    cargas = {k: (v.copy() if isinstance(v, dict) else v) for k, v in cargas.items()}
    cargas["climatizacion"] = dict(cargas["climatizacion"])
    cargas["otros_dispositivos"] = dict(cargas["otros_dispositivos"])
    cargas["otros_dispositivos"]["dispositivos"] = {
        k: dict(v) for k, v in cargas["otros_dispositivos"]["dispositivos"].items()
    }

    clim = cargas["climatizacion"]
    if clim["potencia_max_kW"] is None:
        clim["potencia_max_kW"] = round(float(df_historico[clim["columna_datos"]].max()), 3)

    otros = cargas["otros_dispositivos"]["dispositivos"]
    flags = [d["columna_datos"] for d in otros.values()]
    for nombre, disp in otros.items():
        if disp["potencia_nominal_kW"] is None:
            otras = [f for f in flags if f != disp["columna_datos"]]
            disp["potencia_nominal_kW"] = round(
                _estimar_potencia_nominal_binaria(df_historico, disp["columna_datos"], otras), 3
            )
    return cargas



# EXTRACCIÓN DEL SERVICIO HISTÓRICO (lo que el modelo debe seguir prestando)

def extraer_servicio_historico(df_horizonte: pd.DataFrame, cargas: dict) -> dict:
    """
    A partir de la semana de referencia, calcula el servicio que el histórico prestó:
      - usos_por_dia[carga]: nº de intervalos de uso de cada electrodoméstico en cada día.
      - energia_hvac_ventana_kWh: energía de climatización entregada en cada ventana de confort.
    """
    n = len(df_horizonte)
    n_dias = n // PERIODOS_POR_DIA
    otros = cargas["otros_dispositivos"]["dispositivos"]

    usos = {
        nombre: df_horizonte[disp["columna_datos"]].to_numpy().reshape(n_dias, PERIODOS_POR_DIA).sum(axis=1).astype(int)
        for nombre, disp in otros.items()
    }

    ventana = cargas["climatizacion"]["parametros_especificos"]["ventana_confort_periodos"]
    if PERIODOS_POR_DIA % ventana != 0:
        raise ValueError("La ventana de confort debe dividir exactamente un día (96 intervalos).")
    energia_ventana = df_horizonte["hvac_power_kW"].to_numpy().reshape(-1, ventana).sum(axis=1) * DT_HORAS

    return {
        "usos_por_dia": usos,
        "energia_hvac_ventana_kWh": energia_ventana,
        "ventana_periodos": ventana,
    }



# SELECCIÓN DEL HORIZONTE TEMPORAL (1 SEMANA) SOBRE LOS DATOS PREPARADOS

def periodo_inicio_desde_fecha(df: pd.DataFrame, fecha_inicio: str) -> int:
    """Índice del primer intervalo de 'fecha_inicio' (00:00) en la serie preparada."""
    inicio = pd.Timestamp(fecha_inicio)
    posicion = int((df["timestamp"] < inicio).sum())
    if posicion >= len(df) or df["timestamp"].iloc[posicion] != inicio:
        raise ValueError(f"La fecha {fecha_inicio} no está en la serie de datos de esta vivienda.")
    return posicion


def seleccionar_horizonte(df: pd.DataFrame, n_periodos: int = N_PERIODOS, periodo_inicio: int = 0) -> pd.DataFrame:
    """Recorta el dataset preparado a 'n_periodos' a partir de 'periodo_inicio' (horizonte del modelo)."""
    periodo_fin = periodo_inicio + n_periodos
    if len(df) < periodo_fin:
        raise ValueError(
            f"El dataset preparado tiene {len(df)} registros, "
            f"insuficientes para un horizonte de {n_periodos} periodos empezando en el periodo {periodo_inicio}."
        )
    df_horizonte = df.iloc[periodo_inicio:periodo_fin].reset_index(drop=True)
    primero = df_horizonte["timestamp"].iloc[0]
    if primero.hour != 0 or primero.minute != 0:
        raise ValueError("El horizonte debe empezar a las 00:00 para que los días y las ventanas queden alineados.")
    print(f"[Horizonte] Horizonte temporal: {HORIZON_DIAS} días "
          f"({n_periodos} periodos de {STEP_MINUTOS} min), "
          f"de {primero} a {df_horizonte['timestamp'].iloc[-1]} "
          f"(empieza en {primero.day_name()}).")
    return df_horizonte



# CREACIÓN DE LA ESTRUCTURA DEL MODELO ENERGÉTICO (PYOMO)

def crear_modelo_energetico(df_preparado: pd.DataFrame,
                             cargas_controlables: dict = CARGAS_CONTROLABLES,
                             n_periodos: int = N_PERIODOS,
                             periodo_inicio: int = None) -> pyo.ConcreteModel:
    """
    Construye la estructura del modelo energético en Pyomo: conjuntos (horizonte
    temporal), parámetros (carga base y precio) y variables de decisión (potencia de
    la climatización y estado de cada electrodoméstico en cada periodo).

    'periodo_inicio' elige el tramo del histórico que se usa como horizonte.

    """
    # 1) Recibir datos preparados y recortar al horizonte temporal
    if periodo_inicio is None:
        periodo_inicio = periodo_inicio_desde_fecha(df_preparado, FECHA_SEMANA_REFERENCIA)
    df = seleccionar_horizonte(df_preparado, n_periodos, periodo_inicio)

    # 2-3) Parámetros de las cargas (sobre el histórico completo) y servicio de la semana
    cargas = estimar_parametros_desde_datos(df_preparado, cargas_controlables)
    servicio = extraer_servicio_historico(df, cargas)

    print("[Cargas] Cargas controlables definidas:")
    print(f"  - Climatización: potencia máx = {cargas['climatizacion']['potencia_max_kW']} kW, "
          f"ventana de confort = {servicio['ventana_periodos'] * STEP_MINUTOS / 60:g} h")
    for nombre, disp in cargas["otros_dispositivos"]["dispositivos"].items():
        print(f"  - {nombre}: potencia = {disp['potencia_nominal_kW']} kW | "
              f"usos por día en la semana = {servicio['usos_por_dia'][nombre].tolist()}")

    # 6) Crear estructura del modelo energético
    modelo = pyo.ConcreteModel(name="Modelo_Energetico_Vivienda")
    modelo.cargas_controlables = cargas
    modelo.servicio = servicio
    modelo.datos = df
    modelo.dt_horas = DT_HORAS

    # 4) Horizonte temporal: conjunto de periodos T = {0, 1, ..., n_periodos-1}
    modelo.T = pyo.RangeSet(0, n_periodos - 1)

    # --- Parámetros de entrada, indexados por periodo ---
    modelo.base_load_kW = pyo.Param(modelo.T, initialize=dict(enumerate(df["base_load_kW"])))
    modelo.precio_energia_eur_kWh = pyo.Param(modelo.T, initialize=dict(enumerate(df["precio_energia_eur_kWh"])))

    # 5) Variables de decisión
    p_max_hvac = cargas["climatizacion"]["potencia_max_kW"]
    modelo.P_climatizacion = pyo.Var(modelo.T, domain=pyo.NonNegativeReals, bounds=(0, p_max_hvac))

    modelo.estado_lavadora = pyo.Var(modelo.T, domain=pyo.Binary)
    modelo.estado_lavavajillas = pyo.Var(modelo.T, domain=pyo.Binary)
    modelo.estado_calentador_agua = pyo.Var(modelo.T, domain=pyo.Binary)

    print(f"[Modelo] Estructura del modelo energético creada: "
          f"{len(modelo.T)} periodos, 4 variables de decisión por periodo.")

    # 7) Devolver modelo energético
    return modelo


if __name__ == "__main__":
    df_preparado = preparar_datos()
    modelo = crear_modelo_energetico(df_preparado)
    print("\nComponentes del modelo:")
    print([c.name for c in modelo.component_objects()])
