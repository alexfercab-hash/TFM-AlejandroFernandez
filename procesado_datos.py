"""
Preparación de datos energéticos de una vivienda a partir de HEMS_dataset.csv.

Sigue el Pseudocódigo 1:
    1. Cargar los datos energéticos de la vivienda.
    2. Ordenar los registros cronológicamente.
    3. Comprobar la calidad de los datos (ausentes, anómalos, duplicados).
    4. Tratar los datos identificados (corregir/eliminar/imputar).
    5. Seleccionar el periodo de estudio.
    6. Establecer la resolución temporal de los datos.
    7. Caracterizar el consumo energético.
    8. Seleccionar las variables necesarias para el modelo energético.
    9. Devolver el conjunto de datos preparado.

"""

import pandas as pd
import numpy as np

from tarifa_2_0TD import calcular_precio_energia, asignar_periodo



# CONFIGURACIÓN

RUTA_DATOS = "HEMS_dataset.csv"

# Vivienda sobre la que se trabaja (columna house_id del dataset)
HOUSE_ID = 26

# Columna de fecha/hora y su formato en el CSV original
COL_TIMESTAMP = "timestamp"
FORMATO_TIMESTAMP = "%d-%m-%Y %H:%M"

# Resolución temporal deseada para el estudio (frecuencia pandas)
# "15min" = igual que el dato original, "1H" = horaria, "1D" = diaria...
RESOLUCION_TEMPORAL = "15min"

# Periodo de estudio (None = usar todo el rango disponible para la vivienda)
FECHA_INICIO = None   # p.ej. "2025-01-01"
FECHA_FIN = None      # p.ej. "2025-01-31"

# Variables (columnas) que necesitamos conservar para el modelo energético.
# El resto se descarta al final del proceso.
VARIABLES_MODELO = [
    "timestamp",
    "outdoor_temp_C",            # temperatura exterior 
    "base_load_kW",              # carga base NO controlable 
    "hvac_power_kW",             # climatización (controlable)
    "washing_machine_on",        # electrodomésticos desplazables (controlables)
    "dishwasher_on",
    "water_heater_on",
    "demanda_sin_ev_kW",         # demanda real del piso: total_demand_kW - ev_charging_power_kW
    "periodo_tarifario",         # P1 / P2 / P3 (tarifa 2.0TD)
    "precio_energia_eur_kWh",    # precio por intervalo según tarifa 2.0TD
]

# Columnas que, por su naturaleza física, nunca pueden ser negativas.
# Se usan en la detección/tratamiento de valores anómalos.
COLUMNAS_NO_NEGATIVAS = [
    "base_load_kW", "hvac_power_kW", "total_demand_kW",
    "ev_charging_power_kW",
]

# Umbral (en desviaciones típicas) para considerar un valor un outlier estadístico
UMBRAL_OUTLIER_STD = 5

# Si True, los outliers estadísticos se sustituyen por NaN y se imputan (interpolación).
# Si False, solo se informa de ellos en comprobar_calidad() pero no se modifican:
# útil cuando los picos son consumo real (p.ej. electrodomésticos de alta potencia) y no errores.
TRATAR_OUTLIERS_ESTADISTICOS = False



# 1. CARGA DE DATOS

def cargar_datos(ruta: str, house_id: int) -> pd.DataFrame:
    """Carga el dataset y filtra los registros de la vivienda indicada."""
    df = pd.read_csv(ruta)

    if "house_id" not in df.columns:
        raise ValueError("El dataset no contiene la columna 'house_id'.")

    if house_id not in df["house_id"].unique():
        raise ValueError(f"No existe la vivienda house_id={house_id} en el dataset.")

    df = df[df["house_id"] == house_id].copy()

    print(f"[Carga] {len(df)} registros cargados para la vivienda {house_id}.")
    return df



# 2. ORDEN CRONOLÓGICO

def ordenar_cronologicamente(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la columna de fecha a datetime y ordena el DataFrame por ella."""
    df[COL_TIMESTAMP] = pd.to_datetime(df[COL_TIMESTAMP], format=FORMATO_TIMESTAMP)
    df = df.sort_values(COL_TIMESTAMP).reset_index(drop=True)
    print(f"[Orden] Datos ordenados de {df[COL_TIMESTAMP].min()} a {df[COL_TIMESTAMP].max()}.")
    return df



# 3. COMPROBACIÓN DE CALIDAD DE DATOS

def comprobar_calidad(df: pd.DataFrame) -> dict:
    """
    Analiza la calidad de los datos sin modificarlos todavía.
    Devuelve un diccionario con lo detectado, para poder informar y decidir el tratamiento.
    """
    informe = {}

    # --- Valores ausentes ---
    nulos = df.isnull().sum()
    informe["valores_ausentes"] = nulos[nulos > 0]

    # --- Huecos temporales (timestamps que faltan según la resolución esperada) ---
    rango_completo = pd.date_range(
        df[COL_TIMESTAMP].min(), df[COL_TIMESTAMP].max(), freq=RESOLUCION_TEMPORAL
    )
    huecos = rango_completo.difference(df[COL_TIMESTAMP])
    informe["huecos_temporales"] = huecos

    # --- Registros duplicados ---
    duplicados_exactos = df.duplicated().sum()
    duplicados_timestamp = df.duplicated(subset=[COL_TIMESTAMP]).sum()
    informe["duplicados_exactos"] = duplicados_exactos
    informe["duplicados_timestamp"] = duplicados_timestamp

    # --- Valores anómalos: negativos en columnas que no deberían serlo ---
    negativos = {}
    for col in COLUMNAS_NO_NEGATIVAS:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                negativos[col] = n_neg
    informe["valores_negativos"] = negativos

    # --- Valores anómalos: outliers estadísticos (z-score) ---
    outliers = {}
    for col in COLUMNAS_NO_NEGATIVAS:
        if col in df.columns:
            media, std = df[col].mean(), df[col].std()
            if std > 0:
                z = (df[col] - media).abs() / std
                n_outliers = (z > UMBRAL_OUTLIER_STD).sum()
                if n_outliers > 0:
                    outliers[col] = n_outliers
    informe["outliers_estadisticos"] = outliers

    # --- Resumen por consola ---
    print("[Calidad] Valores ausentes por columna:")
    print(informe["valores_ausentes"] if len(informe["valores_ausentes"]) else "  Ninguno.")
    print(f"[Calidad] Huecos temporales detectados: {len(huecos)}")
    print(f"[Calidad] Duplicados exactos: {duplicados_exactos} | Duplicados por timestamp: {duplicados_timestamp}")
    print(f"[Calidad] Columnas con valores negativos no válidos: {negativos if negativos else 'Ninguna'}")
    print(f"[Calidad] Columnas con outliers estadísticos (>{UMBRAL_OUTLIER_STD} std): {outliers if outliers else 'Ninguna'}")

    return informe



# 4. TRATAMIENTO DE LOS DATOS

def tratar_datos(df: pd.DataFrame, informe: dict) -> pd.DataFrame:
    """
    Corrige/elimina valores no válidos y completa valores ausentes,
    apoyándose en lo detectado en comprobar_calidad().
    """
    df = df.copy()

    # 1) Eliminar duplicados exactos y por timestamp (nos quedamos con el primero)
    antes = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[COL_TIMESTAMP], keep="first")
    print(f"[Tratamiento] Duplicados eliminados: {antes - len(df)}")

    # 2) Corregir valores negativos no físicos -> se llevan a 0
    for col in COLUMNAS_NO_NEGATIVAS:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                df.loc[df[col] < 0, col] = 0
                print(f"[Tratamiento] {n_neg} valores negativos corregidos a 0 en '{col}'.")

    # 3) Outliers estadísticos: por defecto no se tocan, porque en este dataset
    #    corresponden a consumo real (p.ej. picos de electrodomésticos de alta potencia)
    #    y no a errores de medida.
    if TRATAR_OUTLIERS_ESTADISTICOS:
        for col, _ in informe.get("outliers_estadisticos", {}).items():
            media, std = df[col].mean(), df[col].std()
            z = (df[col] - media).abs() / std
            n_outliers = (z > UMBRAL_OUTLIER_STD).sum()
            df.loc[z > UMBRAL_OUTLIER_STD, col] = np.nan
            print(f"[Tratamiento] {n_outliers} outliers marcados como ausentes en '{col}' para su posterior imputación.")
    else:
        n_informativo = sum(informe.get("outliers_estadisticos", {}).values())
        if n_informativo > 0:
            print(f"[Tratamiento] {n_informativo} valores estadísticamente extremos NO se tratan "
                  f"(se consideran consumo real, ej. carga de EV).")

    # 4) Reindexar sobre la rejilla temporal completa para materializar los huecos
    #    (registros de timestamp inexistentes) como filas con NaN.
    rango_completo = pd.date_range(
        df[COL_TIMESTAMP].min(), df[COL_TIMESTAMP].max(), freq=RESOLUCION_TEMPORAL
    )
    df = df.set_index(COL_TIMESTAMP).reindex(rango_completo)
    df.index.name = COL_TIMESTAMP
    n_reindexados = df.isnull().all(axis=1).sum()
    if n_reindexados > 0:
        print(f"[Tratamiento] {n_reindexados} timestamps ausentes añadidos a la rejilla temporal (a imputar).")

    # 5) Imputación de valores ausentes:
    #    - variables numéricas de consumo/entorno: interpolación temporal lineal
    #    - variables categóricas/temporales derivadas (day_of_week, hour_of_day, is_weekend):
    #      se recalculan a partir del propio índice de tiempo
    columnas_temporales_derivadas = ["day_of_week", "hour_of_day", "minute_of_hour", "is_weekend"]
    for col in columnas_temporales_derivadas:
        if col in df.columns:
            if col == "day_of_week":
                df[col] = df.index.dayofweek
            elif col == "hour_of_day":
                df[col] = df.index.hour
            elif col == "minute_of_hour":
                df[col] = df.index.minute
            elif col == "is_weekend":
                df[col] = (df.index.dayofweek >= 5).astype(int)

    columnas_binarias_estado = [c for c in df.columns if c.endswith("_on")]

    columnas_numericas = (
        df.select_dtypes(include=[np.number]).columns
        .difference(columnas_temporales_derivadas)
        .difference(columnas_binarias_estado)
    )
    df[columnas_numericas] = df[columnas_numericas].interpolate(method="time").ffill().bfill()
    if columnas_binarias_estado:
        df[columnas_binarias_estado] = df[columnas_binarias_estado].ffill().bfill()

    df = df.reset_index()
    print(f"[Tratamiento] Dataset tratado: {len(df)} registros, {df.isnull().sum().sum()} valores ausentes restantes.")
    return df


# 5. SELECCIÓN DEL PERIODO DE ESTUDIO

def seleccionar_periodo(df: pd.DataFrame, fecha_inicio=None, fecha_fin=None) -> pd.DataFrame:
    """Recorta el DataFrame al periodo de estudio indicado (o lo deja igual si no se especifica)."""
    if fecha_inicio is not None:
        df = df[df[COL_TIMESTAMP] >= pd.to_datetime(fecha_inicio)]
    if fecha_fin is not None:
        df = df[df[COL_TIMESTAMP] <= pd.to_datetime(fecha_fin)]
    df = df.reset_index(drop=True)
    print(f"[Periodo] Periodo de estudio: {df[COL_TIMESTAMP].min()} a {df[COL_TIMESTAMP].max()} "
          f"({len(df)} registros).")
    return df



# 6. RESOLUCIÓN TEMPORAL

def establecer_resolucion_temporal(df: pd.DataFrame, resolucion: str) -> pd.DataFrame:
    """
    Ajusta la resolución temporal de los datos mediante remuestreo (resample).
    Si 'resolucion' coincide con la resolución nativa de los datos, no cambia nada.
    Variables de potencia/energía se agregan por media; contadores on/off, por máximo.
    """
    df = df.set_index(COL_TIMESTAMP)

    columnas_binarias = [c for c in df.columns if c.endswith("_on")]
    columnas_media = [c for c in df.select_dtypes(include=[np.number]).columns if c not in columnas_binarias]

    agregaciones = {c: "mean" for c in columnas_media}
    agregaciones.update({c: "max" for c in columnas_binarias})

    df_resampled = df.resample(resolucion).agg(agregaciones)
    df_resampled = df_resampled.reset_index()

    print(f"[Resolución] Datos ajustados a resolución '{resolucion}': {len(df_resampled)} registros.")
    return df_resampled


# 6b. TARIFA 2.0TD Y DEMANDA DEL PISO (SIN VEHÍCULO ELÉCTRICO)

def anadir_tarifa_y_demanda_piso(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Genera el precio de cada intervalo a partir de su fecha y hora según la
       tarifa 2.0TD (tarifa_2_0TD.py), sustituyendo a la columna de precios del
       dataset original.
    2) Calcula la demanda real del piso SIN vehículo eléctrico: el dataset incluye la
       carga del EV dentro de total_demand_kW, y el escenario de estudio es un piso
       sin cargador de coche, así que se resta:
           demanda_sin_ev_kW = total_demand_kW - ev_charging_power_kW
    """
    df = df.copy()
    df["periodo_tarifario"] = asignar_periodo(df[COL_TIMESTAMP]).values
    df["precio_energia_eur_kWh"] = calcular_precio_energia(df[COL_TIMESTAMP]).values

    df["demanda_sin_ev_kW"] = (df["total_demand_kW"] - df["ev_charging_power_kW"]).clip(lower=0.0)

    reparto = df["periodo_tarifario"].value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"[Tarifa] Precio 2.0TD asignado por intervalo. Reparto de periodos: {reparto}")
    print("[Demanda] demanda_sin_ev_kW = total_demand_kW - ev_charging_power_kW (piso sin cargador de coche).")
    return df



# 7. CARACTERIZACIÓN DEL CONSUMO ENERGÉTICO

def caracterizar_consumo(df: pd.DataFrame, columna_consumo: str = "demanda_sin_ev_kW") -> dict:
    """
    Calcula estadísticos descriptivos y patrones de consumo:
    - consumo medio, máximo y mínimo
    - evolución temporal (serie diaria)
    - patrones de consumo (perfil medio por hora del día y por día de la semana)
    """
    caracterizacion = {}

    caracterizacion["consumo_medio"] = df[columna_consumo].mean()
    caracterizacion["consumo_maximo"] = df[columna_consumo].max()
    caracterizacion["consumo_minimo"] = df[columna_consumo].min()

    # Evolución temporal: consumo total diario
    evolucion_diaria = (
        df.set_index(COL_TIMESTAMP)[columna_consumo]
        .resample("1D")
        .sum()
    )
    caracterizacion["evolucion_diaria_kWh"] = evolucion_diaria

    # Patrón por hora del día (perfil de carga típico)
    if "hour_of_day" in df.columns:
        patron_horario = df.groupby("hour_of_day")[columna_consumo].mean()
        caracterizacion["patron_horario"] = patron_horario

    # Patrón por día de la semana
    if "day_of_week" in df.columns:
        patron_semanal = df.groupby("day_of_week")[columna_consumo].mean()
        caracterizacion["patron_semanal"] = patron_semanal

    print("[Caracterización] Consumo medio: {:.3f} kW | máximo: {:.3f} kW | mínimo: {:.3f} kW".format(
        caracterizacion["consumo_medio"], caracterizacion["consumo_maximo"], caracterizacion["consumo_minimo"]
    ))

    return caracterizacion



# 8. SELECCIÓN DE VARIABLES PARA EL MODELO

def seleccionar_variables(df: pd.DataFrame, variables: list) -> pd.DataFrame:
    """Conserva únicamente las columnas necesarias para el modelo energético."""
    variables_presentes = [v for v in variables if v in df.columns]
    faltantes = set(variables) - set(variables_presentes)
    if faltantes:
        print(f"[Variables] Aviso: no se encontraron en el dataset estas columnas: {faltantes}")
    df_modelo = df[variables_presentes].copy()
    print(f"[Variables] Dataset final con {df_modelo.shape[1]} columnas y {df_modelo.shape[0]} registros.")
    return df_modelo



# COMPILACIÓN DEL PROCESO COMPLETO

def preparar_datos(
    ruta: str = RUTA_DATOS,
    house_id: int = HOUSE_ID,
    fecha_inicio=FECHA_INICIO,
    fecha_fin=FECHA_FIN,
    resolucion: str = RESOLUCION_TEMPORAL,
    variables_modelo: list = VARIABLES_MODELO,
) -> pd.DataFrame:
    """
    Ejecuta el flujo completo de preparación de datos (Pseudocódigo 1)
    y devuelve el conjunto de datos preparado.
    """

    df = cargar_datos(ruta, house_id)
    df = ordenar_cronologicamente(df)

    informe_calidad = comprobar_calidad(df)
    df = tratar_datos(df, informe_calidad)

    df = seleccionar_periodo(df, fecha_inicio, fecha_fin)
    df = establecer_resolucion_temporal(df, resolucion)

  
    df = anadir_tarifa_y_demanda_piso(df)
    caracterizar_consumo(df)

    df_preparado = seleccionar_variables(df, variables_modelo)

    return df_preparado


if __name__ == "__main__":
    df_preparado = preparar_datos()
    print("\nVista previa del dataset preparado:")
    print(df_preparado.head())
