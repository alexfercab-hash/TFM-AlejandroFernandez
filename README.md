# Optimización de la programación de cargas en una vivienda (HEMS)

Código del Trabajo Fin de Máster *"Mecanismos de optimización del consumo energético"* (Máster Universitario en Robótica y Automatización, Universidad Europea de Madrid).

**Autor:** Alejandro Fernández Cabello
**Director:** Unai Armentia Sodupe

El sistema programa las cargas controlables de una vivienda (climatización, lavadora, lavavajillas y calentador de agua) para minimizar el coste de la energía con la tarifa 2.0TD, manteniendo el mismo servicio que el consumo histórico. El mismo problema se resuelve con un método exacto (MILP) y tres metaheurísticas (GA, PSO y SA), y los resultados se comparan en coste, pico de demanda y tiempo de cálculo.

## Requisitos

- Python 3.10 o superior.
- Librerías: `pip install -r requirements.txt`
- Solver **CBC**, que Pyomo utiliza para resolver el MILP (https://github.com/coin-or/Cbc). Debe estar instalado y accesible desde la línea de comandos (`cbc`).

## Datos

Se utiliza el conjunto de datos público *Home Energy Management Dataset* (Kaggle). Descárgalo y guarda el archivo `HEMS_dataset.csv` en esta misma carpeta. No se incluye en el repositorio.

## Ejecución

| Comando | Resultado |
|---|---|
| `python main.py` | Comparación de los cuatro métodos con presupuesto bajo |
| `python main_alto_presupuesto.py` | Comparación con presupuesto alto |
| `python convergencia_extendida.py` | Convergencia extendida de GA y PSO |
| `python analisis_semillas.py` | Robustez frente a la semilla aleatoria |
| `python analisis_generalizacion.py` | Generalización a otras viviendas y semanas |
| `python sensibilidad_ventana_confort.py` | Sensibilidad a la ventana de confort |
| `python generar_interfaz_usuario.py` | Genera la interfaz `interfaz_usuario.html` (ejecutar antes `main.py`) |

Los resultados se guardan en carpetas `resultados*/`.

## Estructura del código

| Archivo | Función |
|---|---|
| `procesado_datos.py` | Carga, limpieza y preparación de los datos |
| `tarifa_2_0TD.py` | Periodos y precios de la tarifa 2.0TD |
| `Modelo_energetico.py` | Definición de cargas, horizonte y variables |
| `Restricciones.py` | Restricciones del modelo |
| `Funcion_objetivo.py` | Función objetivo (coste energético) |
| `MILP.py` | Resolución mediante MILP (Pyomo + CBC) |
| `GA.py`, `PSO.py`, `SA.py` | Metaheurísticas con presupuesto bajo |
| `*_alto_presupuesto.py` | Metaheurísticas con presupuesto alto |
| `main.py`, `main_alto_presupuesto.py` | Programa principal y comparación de resultados |
| `consigna_climatizacion.py` | Análisis auxiliar de la climatización |
| `plan_usuario.py`, `generar_interfaz_usuario.py` | Generación de la interfaz para el usuario |

Los pseudocódigos citados en los comentarios corresponden al Anexo A del TFM.
