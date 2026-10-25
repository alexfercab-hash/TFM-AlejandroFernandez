"""
Estima cuánto se ahorra al CAMBIAR LA CONSIGNA del aire acondicionado / calefacción
("sube 2 °C la consigna y ahorrarás X €"), una medida de comportamiento distinta del
desplazamiento horario que calculan los algoritmos de optimización.

"""

import numpy as np

DT_HORAS = 0.25


def ajustar_curva(temperatura: np.ndarray, potencia: np.ndarray) -> dict:
    """Ajusta P(T) = base + a*max(0,T-Tf) + b*max(0,Tc-T) por búsqueda de umbrales y mínimos cuadrados."""
    T = np.asarray(temperatura, dtype=float)
    P = np.asarray(potencia, dtype=float)
    mejor = None
    for t_frio in np.arange(22.0, 31.0, 0.5):
        for t_calor in np.arange(13.0, 23.0, 0.5):
            X = np.c_[np.ones_like(T), np.maximum(0.0, T - t_frio), np.maximum(0.0, t_calor - T)]
            coef, *_ = np.linalg.lstsq(X, P, rcond=None)
            sse = float(((P - X @ coef) ** 2).sum())
            if mejor is None or sse < mejor[0]:
                mejor = (sse, t_frio, t_calor, coef)
    sse, t_frio, t_calor, (base, pend_frio, pend_calor) = mejor
    r2 = 1.0 - sse / float(((P - P.mean()) ** 2).sum())
    return {"t_frio": float(t_frio), "t_calor": float(t_calor), "base_kW": float(base),
            "pendiente_frio_kW_por_C": float(pend_frio), "pendiente_calor_kW_por_C": float(pend_calor),
            "r2": float(r2)}


def reduccion_potencia(temperatura: np.ndarray, curva: dict, delta: float) -> tuple:
    """Potencia (kW) que se deja de usar al relajar la consigna 'delta' grados: (en frío, en calor)."""
    T = np.asarray(temperatura, dtype=float)
    frio = curva["pendiente_frio_kW_por_C"] * np.clip(T - curva["t_frio"], 0.0, delta)
    calor = curva["pendiente_calor_kW_por_C"] * np.clip(curva["t_calor"] - T, 0.0, delta)
    return frio, calor


def ahorro_por_consigna(temperatura_semana: np.ndarray, precio_semana: np.ndarray, curva: dict,
                        grados=(1, 2, 3)) -> list:
    """Ahorro semanal (kWh y €) de relajar la consigna 1, 2 y 3 grados, separado en modo frío y modo calor."""
    T = np.asarray(temperatura_semana, dtype=float)
    precio = np.asarray(precio_semana, dtype=float)
    resultados = []
    for g in grados:
        frio, calor = reduccion_potencia(T, curva, g)
        resultados.append({
            "grados": g,
            "kwh": round(float((frio + calor).sum() * DT_HORAS), 3),
            "eur": round(float(((frio + calor) * DT_HORAS * precio).sum()), 4),
            "eur_modo_frio": round(float((frio * DT_HORAS * precio).sum()), 4),
            "eur_modo_calor": round(float((calor * DT_HORAS * precio).sum()), 4),
        })
    return resultados


def analizar(df_historico, df_semana) -> dict:
    """Ajusta la curva con todo el histórico de la vivienda y calcula el ahorro en la semana indicada."""
    curva = ajustar_curva(df_historico["outdoor_temp_C"].to_numpy(), df_historico["hvac_power_kW"].to_numpy())
    T = df_semana["outdoor_temp_C"].to_numpy()
    return {
        "curva": {k: round(v, 4) for k, v in curva.items()},
        "porcentaje_tiempo_en_frio": round(100 * float((T > curva["t_frio"]).mean()), 1),
        "porcentaje_tiempo_en_calor": round(100 * float((T < curva["t_calor"]).mean()), 1),
        "ahorro_semanal": ahorro_por_consigna(T, df_semana["precio_energia_eur_kWh"].to_numpy(), curva),
    }


if __name__ == "__main__":
    import io
    import contextlib
    from procesado_datos import preparar_datos
    from Modelo_energetico import periodo_inicio_desde_fecha, FECHA_SEMANA_REFERENCIA, N_PERIODOS

    with contextlib.redirect_stdout(io.StringIO()):
        df = preparar_datos()
    ini = periodo_inicio_desde_fecha(df, FECHA_SEMANA_REFERENCIA)
    r = analizar(df, df.iloc[ini:ini + N_PERIODOS])
    print("Curva ajustada:", r["curva"])
    print(f"Tiempo en modo frío: {r['porcentaje_tiempo_en_frio']} % | en modo calor: {r['porcentaje_tiempo_en_calor']} %")
    for a in r["ahorro_semanal"]:
        print(f"  Relajar la consigna {a['grados']} °C -> {a['kwh']:.2f} kWh, {a['eur']:.2f} € a la semana "
              f"(frío {a['eur_modo_frio']:.2f} € + calor {a['eur_modo_calor']:.2f} €)")
