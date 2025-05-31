# Aplicación Streamlit para el Monitoreo Térmico de EP-110

Este repositorio contiene los archivos para una aplicación Streamlit diseñada para monitorear el desempeño térmico de los enfriadores EP-110. Incluye el código de la aplicación y los archivos de datos utilizados por la aplicación.

## Descripción de los Archivos

* `app1.py`: Este es el código Python para la aplicación Streamlit. Proporciona un panel de control para visualizar y analizar los datos térmicos de los enfriadores EP-110, así como una interfaz de entrada de datos.
* `data.csv`: Este archivo CSV contiene los datos de los enfriadores EP-110. Incluye mediciones como lecturas de temperatura y el estado de los equipos.
* `temp_gcp_data.csv`: Este archivo CSV contiene datos de temperatura relacionados con los trenes GCP (Proceso de Limpieza de Gases).

## `app1.py` - Detalles de la Aplicación Streamlit

La aplicación Streamlit (`app1.py`) proporciona las siguientes funcionalidades:

* **Carga y Procesamiento de Datos:**
    * Lee datos de `data.csv` y `temp_gcp_data.csv`.
    * Realiza limpieza y preprocesamiento de datos, incluyendo conversiones de fecha/hora y cálculo de diferenciales de temperatura (ΔT Agua, ΔT Efluente).
* **Visualización de Datos:**
    * Crea gráficos interactivos utilizando Plotly para visualizar tendencias y distribuciones de temperatura.
    * Incluye funciones para filtrar datos por rango de fechas, planta y equipo.
* **Análisis de Datos:**
    * Calcula y muestra indicadores clave de rendimiento (KPI).
    * Realiza análisis de tendencias en los datos de temperatura.
    * Calcula las correlaciones entre las temperaturas de GCP y el rendimiento de EP-110.
* **Entrada de Datos:**
    * Proporciona una interfaz de usuario para ingresar nuevos datos para los enfriadores EP-110.
    * Incluye validación de datos para garantizar la integridad de los datos.
    * Guarda los datos ingresados en `data.csv`.

## Detalles de los Archivos de Datos

### `data.csv`

Este archivo contiene los datos operativos de los enfriadores EP-110. [cite: 1, 2, 3]

**Columnas:**

* `Fecha`: Fecha del registro. [cite: 1, 2, 3]
* `Hora`: Hora del registro. [cite: 1, 2, 3]
* `Planta`: Identificador de la planta (por ejemplo, GCP-2, GCP-4). [cite: 1, 2, 3]
* `Equipo`: Identificador del equipo (por ejemplo, EP-110 A, EP-110 B). [cite: 1, 2, 3]
* `Estado`: Estado del equipo (por ejemplo, "En Servicio", "Mantención"). [cite: 1, 2, 3]
* `Temp Entrada Agua Torre`: Temperatura del agua que entra a la torre. [cite: 1, 2, 3]
* `Temp Salida Agua Torre`: Temperatura del agua que sale de la torre. [cite: 1, 2, 3]
* `Delta Agua`: Diferencia entre `Temp Salida Agua Torre` y `Temp Entrada Agua Torre`. [cite: 1]
* `Temp Entrada Efluente`: Temperatura del efluente que entra. [cite: 1, 2, 3]
* `Temp Salida Efluente`: Temperatura del efluente que sale. [cite: 1, 2, 3]
* `Delta Efluente`: Diferencia entre `Temp Entrada Efluente` y `Temp Salida Efluente`. [cite: 1]
* `Comentarios`: Comentarios adicionales. [cite: 1, 2, 3]
* `Δ Temp Agua`: Igual que `Delta Agua`. [cite: 1]
* `Δ Temp Efluente`: Igual que `Delta Efluente`. [cite: 1]

### `temp_gcp_data.csv`

Este archivo contiene datos de temperatura para los trenes GCP. [cite: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

**Columnas:**

* `FechaHora`: Fecha y hora del registro. [cite: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
* `Tren`: Identificador del tren (por ejemplo, GCP-2A, GCP-4B). [cite: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
* `Temperatura`: Medición de temperatura. [cite: 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

## Dependencias

La aplicación `app1.py` requiere las siguientes bibliotecas de Python:

* streamlit
* pandas
* plotly
* datetime
* numpy
* scipy
* os
* shutil

## Instalación y Uso

1.  Asegúrate de tener Python instalado.
2.  Instala las bibliotecas de Python requeridas usando pip:

    ```bash
    pip install streamlit pandas plotly numpy scipy
    ```
3.  Coloca `app1.py`, `data.csv` y `temp_gcp_data.csv` en el mismo directorio.
4.  Ejecuta la aplicación Streamlit:

    ```bash
    streamlit run app1.py
    ```

Esto abrirá la aplicación en tu navegador web.
