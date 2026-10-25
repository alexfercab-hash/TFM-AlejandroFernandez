"""
Tarifa eléctrica española 2.0TD (peajes de acceso, 3 periodos de energía).

Se definen los precios, los periodos horarios y los festivos.
El resto del proyecto solo consume el vector de precios que genera
'calcular_precio_energia()'.

Periodos de energía 2.0TD:
    P1 (punta): laborables 10-14 h y 18-22 h
    P2 (llano): laborables 8-10 h, 14-18 h y 22-24 h
    P3 (valle): laborables 0-8 h, y 24 h de sábados, domingos y festivos nacionales

"""

import pandas as pd



# CONFIGURACIÓN


# Precio del término de energía por periodo (€/kWh), sin IVA (21 %) ni impuesto
# eléctrico (5,11 %), que multiplican todos los precios por igual y no cambian
# ninguna decisión ni los porcentajes de ahorro.
#
# Escenario: piso en España (península), donde la 2.0TD es común a todo el territorio.
# Valores representativos del mercado libre, fijados de forma aproximada a partir de
# ofertas comerciales con discriminación horaria en tres periodos (punta 0,164-0,193;
# llano 0,094-0,125; valle 0,066-0,111 €/kWh) y del PVPC medio del invierno 2025-26
# (0,146 / 0,111 / 0,076 €/kWh). No corresponden a una oferta concreta.

PRECIOS_EUR_KWH = {
    "P1": 0.169,   # punta
    "P2": 0.111,   # llano
    "P3": 0.087,   # valle
}

# Festivos de ámbito NACIONAL dentro del periodo del dataset (enero-febrero 2025):
# 1 y 6 de enero. Los festivos autonómicos o locales (p. ej. el 28 de febrero, Día de
# Andalucía) no modifican los periodos de la 2.0TD, que solo distingue festivos nacionales.
FESTIVOS_NACIONALES = {
    pd.Timestamp("2025-01-01").date(),
    pd.Timestamp("2025-01-06").date(),
}

# Horas (de inicio del intervalo) de cada periodo en día laborable.
HORAS_P1 = {10, 11, 12, 13, 18, 19, 20, 21}
HORAS_P2 = {8, 9, 14, 15, 16, 17, 22, 23}
# El resto de horas laborables (0-7) es P3.



# FUNCIONES

def asignar_periodo(timestamps: pd.Series) -> pd.Series:
    """Devuelve 'P1' / 'P2' / 'P3' para cada instante, según su fecha y hora."""
    ts = pd.to_datetime(timestamps)
    hora = ts.dt.hour
    es_finde = ts.dt.dayofweek >= 5                      # 5 = sábado, 6 = domingo
    es_festivo = ts.dt.date.isin(FESTIVOS_NACIONALES)
    es_valle_todo_el_dia = es_finde | es_festivo

    periodo = pd.Series("P3", index=ts.index)            # por defecto, valle
    periodo[hora.isin(HORAS_P2)] = "P2"
    periodo[hora.isin(HORAS_P1)] = "P1"
    periodo[es_valle_todo_el_dia] = "P3"                 # fines de semana y festivos: valle 24 h
    return periodo


def calcular_precio_energia(timestamps: pd.Series) -> pd.Series:
    """Vector de precios (€/kWh) para cada instante, según el periodo 2.0TD."""
    return asignar_periodo(timestamps).map(PRECIOS_EUR_KWH)


if __name__ == "__main__":
    # Comprobación rápida de la asignación de periodos
    casos = pd.Series(pd.to_datetime([
        "2025-01-01 12:00",   # miércoles, festivo      -> P3
        "2025-01-06 19:00",   # lunes, festivo          -> P3
        "2025-01-02 10:00",   # jueves laborable        -> P1
        "2025-01-02 09:45",   # jueves laborable        -> P2
        "2025-01-02 14:00",   # jueves laborable        -> P2
        "2025-01-02 18:00",   # jueves laborable        -> P1
        "2025-01-02 22:00",   # jueves laborable        -> P2
        "2025-01-02 23:45",   # jueves laborable        -> P2
        "2025-01-03 00:00",   # viernes laborable       -> P3
        "2025-01-03 07:45",   # viernes laborable       -> P3
        "2025-01-04 12:00",   # sábado                  -> P3
        "2025-01-05 20:00",   # domingo                 -> P3
        "2025-01-07 11:00",   # martes laborable        -> P1
    ]))
    print(pd.DataFrame({"instante": casos, "periodo": asignar_periodo(casos),
                        "precio_eur_kWh": calcular_precio_energia(casos)}).to_string(index=False))
