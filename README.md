# EDA Final Project – Uber 2024

Este proyecto corresponde al **Trabajo Final Integrador** del *Minor en Analítica de Datos* (Universidad Privada Boliviana).
El objetivo es realizar un análisis exploratorio y predictivo sobre datos reales de **Uber (2024)**, aplicando técnicas de limpieza, transformación, análisis estadístico, ingeniería de características y modelado predictivo.

Se utilizaron metodologías de **CRISP-DM** para garantizar un flujo estructurado: desde la comprensión del negocio, pasando por la preparación de datos, hasta el modelado y la evaluación de resultados.

---

## Contenido

El repositorio contiene múltiples scripts de Python que implementan cada fase del proyecto:

1. **01_view_null_values.py** – Exploración de valores nulos.
2. **02_transformation_and_cleaning.py** – Limpieza y transformación de datos.
3. **03_descriptive_statistics.py** – Estadística descriptiva y visualizaciones iniciales.
4. **04_feature_engineering_fixed.py** – Ingeniería de características (versión corregida).
5. **05_regression_models.py** – Modelos de regresión (Lineal Múltiple y Random Forest).
6. **06_ride_completion_prediction.py** – Predicción de finalización de viajes.
7. **07_visualizations.py** – Visualizaciones finales del modelo.

Además, se incluyen scripts descartados (`failed_*.py`) que muestran iteraciones que no resultaron óptimas.

---

## Pre-requisitos

* Python **3.13**
* [uv](https://docs.astral.sh/uv/) (gestor de entornos y dependencias)

Librerías necesarias:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

---

## Ejecución

Clonar el repositorio:

```bash
git clone https://github.com/joackagui/EDA_final_project.git
cd EDA_final_project
```

Crear y activar el entorno virtual con `uv`:

```bash
uv venv --python 3.13 --seed
.\.venv\Scripts\activate   # Windows
```

Instalar dependencias:

```bash
uv pip install pandas matplotlib seaborn scikit-learn numpy
```

Ejecutar los scripts en orden:

```bash
python 01_view_null_values.py
python 02_transformation_and_cleaning.py
python 03_descriptive_statistics.py
python 04_feature_engineering_fixed.py
python 05_regression_models.py
python 06_ride_completion_prediction.py
python 07_visualizations.py
```

También pueden probarse los scripts de prueba fallida:

```bash
python 04_failed_feature_engineering.py
python 06_failed_classification_models.py
python 06_failed_driver_rating_prediction.py
```

---

## Autor

**Joaquin Ignacio Aguilera Salinas**
Minor en Analítica de Datos – Universidad Privada Boliviana
La Paz, Bolivia – 2025
