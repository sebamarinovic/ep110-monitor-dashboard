# --- Streamlit App Mejorada para Monitoreo Térmico EP-110 ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import gaussian_kde
import os
import shutil

# Configuración general
st.set_page_config(page_title="🌡️ Monitoreo Térmico EP-110", layout="wide")

# Constantes
ARCHIVO = "data.csv"
EQUIPOS = ["EP-110 A", "EP-110 B", "EP-110 C", "EP-110 D", "EP-110 E", "EP-110 F"]
PLANTAS = ["GCP-2", "GCP-4"]
UMBRAL_AGUA = 4.0
UMBRAL_EFLUENTE = 3.0
TEMP_DISENO = 32
COLUMNAS_ESPERADAS = [
    "Fecha", "Hora", "Planta", "Equipo", "Estado",
    "Temp Entrada Agua Torre", "Temp Salida Agua Torre",
    "Temp Entrada Efluente", "Temp Salida Efluente"
]

# Definir trenes
TREN_A = ["EP-110 A", "EP-110 B", "EP-110 C"]
TREN_B = ["EP-110 D", "EP-110 E", "EP-110 F"]

# Funciones
@st.cache_data(ttl=3600)
def cargar_datos(archivo=ARCHIVO):
    """
    Carga y procesa los datos del archivo CSV de EP-110.
    
    Args:
        archivo (str): Ruta al archivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame procesado o vacío si hay errores.
    """
    try:
        df = pd.read_csv(archivo)
        if not all(col in df.columns for col in COLUMNAS_ESPERADAS):
            st.error(f"El archivo {archivo} no contiene todas las columnas esperadas.")
            return pd.DataFrame()
        df["FechaHora"] = pd.to_datetime(df["Fecha"] + " " + df["Hora"])
        if "Δ Temp Agua" in df.columns and "Δ Temp Efluente" in df.columns:
            df = df.drop(columns=["Δ Temp Agua", "Δ Temp Efluente"])
        df.rename(columns={"Delta Agua": "Δ Temp Agua", "Delta Efluente": "Δ Temp Efluente"}, inplace=True)
        # Calcular ΔT Agua y ΔT Efluente si no están presentes
        if "Δ Temp Agua" not in df.columns:
            df["Δ Temp Agua"] = df.apply(
                lambda row: round(row["Temp Salida Agua Torre"] - row["Temp Entrada Agua Torre"], 2)
                if pd.notna(row["Temp Salida Agua Torre"]) and pd.notna(row["Temp Entrada Agua Torre"])
                and row["Estado"] == "En Servicio" else np.nan, axis=1
            )
        if "Δ Temp Efluente" not in df.columns:
            df["Δ Temp Efluente"] = df.apply(
                lambda row: round(row["Temp Entrada Efluente"] - row["Temp Salida Efluente"], 2)
                if pd.notna(row["Temp Entrada Efluente"]) and pd.notna(row["Temp Salida Efluente"])
                and row["Estado"] == "En Servicio" else np.nan, axis=1
            )
        return df.sort_values("FechaHora")
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {archivo}. Verifique la ruta.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error(f"El archivo {archivo} está vacío.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos EP-110: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def cargar_datos_gcp(archivo="temp_gcp_data.csv"):
    """
    Carga y procesa los datos del archivo CSV de GCP.
    
    Args:
        archivo (str): Ruta al archivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame procesado o vacío si hay errores.
    """
    try:
        df_gcp = pd.read_csv(archivo)
        if 'FechaHora' not in df_gcp.columns or 'Tren' not in df_gcp.columns:
            st.error(f"El archivo {archivo} no contiene las columnas esperadas.")
            return pd.DataFrame()
        df_gcp['FechaHora'] = pd.to_datetime(df_gcp['FechaHora'])
        return df_gcp
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {archivo}. Verifique la ruta.")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error(f"El archivo {archivo} está vacío.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error al cargar datos GCP: {str(e)}")
        return pd.DataFrame()

def filtrar_datos_por_rango(df, fecha_inicio, fecha_fin, planta=None, equipo=None, incluir_mant=True):
    """
    Filtra el DataFrame por rango de fechas, planta, equipo y estado.
    
    Args:
        df (pd.DataFrame): DataFrame a filtrar.
        fecha_inicio (date): Fecha de inicio del rango.
        fecha_fin (date): Fecha de fin del rango.
        planta (str, optional): Planta a filtrar (o "Todas").
        equipo (str, optional): Equipo a filtrar (o "Todos").
        incluir_mant (bool): Incluir registros en mantención.
    
    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = pd.to_datetime(fecha_fin) + pd.Timedelta(days=1)
    df_filtrado = df[(df["FechaHora"] >= fecha_inicio_dt) & (df["FechaHora"] <= fecha_fin_dt)]
    if planta and planta != "Todas":
        df_filtrado = df_filtrado[df_filtrado["Planta"] == planta]
    if equipo and equipo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Equipo"] == equipo]
    if not incluir_mant:
        df_filtrado = df_filtrado[df_filtrado["Estado"] == "En Servicio"]
    return df_filtrado

def plot_evolucion_termica(df_filtrado, variable, umbral):
    """
    Gráfico de evolución temporal con análisis automático de tendencias.
    
    Args:
        df_filtrado (pd.DataFrame): Datos filtrados para el gráfico.
        variable (str): Variable a graficar (Δ Temp Agua o Efluente).
        umbral (float): Valor del umbral de referencia.
    """
    if df_filtrado.empty:
        st.warning("No hay datos disponibles para el período seleccionado")
        return
    
    fig = px.line(
        df_filtrado,
        x="FechaHora",
        y=variable,
        color="Equipo",
        markers=True,
        line_shape="linear",
        template="plotly_white",
        title=f"Evolución de {variable}",
        labels={variable: f"{variable} (°C)"},
        hover_data={"FechaHora": "|%d/%m/%Y %H:%M", variable: ":.2f", "Equipo": True}
    )
    
    fig.add_hline(
        y=umbral,
        line_dash="dot",
        line_color="red",
        annotation_text="Mínimo Recomendado",
        annotation_position="bottom right"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📈 Análisis de Tendencia Térmica", expanded=False):
        if len(df_filtrado) > 1:
            fecha_media = df_filtrado['FechaHora'].min() + (df_filtrado['FechaHora'].max() - df_filtrado['FechaHora'].min()) / 2
            primera_mitad = df_filtrado[df_filtrado['FechaHora'] <= fecha_media]
            segunda_mitad = df_filtrado[df_filtrado['FechaHora'] > fecha_media]
            
            if not primera_mitad.empty and not segunda_mitad.empty:
                tendencia = "mejorando ↗️" if segunda_mitad[variable].mean() > primera_mitad[variable].mean() else "empeorando ↘️"
            else:
                tendencia = "datos insuficientes"
        else:
            tendencia = "datos insuficientes"
            
        variabilidad = df_filtrado.groupby('Equipo')[variable].std().mean()
        max_val = df_filtrado[variable].max()
        min_val = df_filtrado[variable].min()
        
        st.markdown(f"""
        **Análisis de {variable.split(' ')[-1]}:**  
        
        - 📅 **Período analizado:** {df_filtrado['FechaHora'].dt.date.min()} al {df_filtrado['FechaHora'].dt.date.max()}  
        - 📉 **Tendencia:** {tendencia}  
        - 📊 **Variabilidad promedio:** {variabilidad:.2f}°C  
        - 🔥 **Valor máximo registrado:** {max_val:.2f}°C  
        - ❄️ **Valor mínimo registrado:** {min_val:.2f}°C  
        
        **Interpretación:**  
        - Variabilidad {'alta (>1.5°C)' if variabilidad > 1.5 else 'moderada' if variabilidad > 0.5 else 'baja'}  
        - Consistencia: {'pobre ❌' if variabilidad > 1.5 else 'aceptable ⚠️' if variabilidad > 1.0 else 'buena ✅'}  
        """)

def plot_comparacion_gcp(df, df_gcp, tren, equipos_relacionados, planta, fecha_inicio, fecha_fin):
    """
    Gráfico y análisis comparativo entre temperatura de gases y ΔT de equipos.
    
    Args:
        df (pd.DataFrame): Datos de EP-110.
        df_gcp (pd.DataFrame): Datos de GCP.
        tren (str): Tren a analizar (ej. GCP-2A).
        equipos_relacionados (list): Lista de equipos relacionados.
        planta (str): Planta a analizar.
        fecha_inicio (date): Fecha de inicio.
        fecha_fin (date): Fecha de fin.
    """
    if df.empty or df_gcp.empty:
        st.warning("⚠️ No hay datos disponibles para realizar el análisis")
        return
    
    # Filtrar datos de GCP
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = pd.to_datetime(fecha_fin) + pd.Timedelta(days=1)
    datos_gcp = df_gcp[(df_gcp['Tren'] == tren) & 
                       (df_gcp['FechaHora'] >= fecha_inicio_dt) & 
                       (df_gcp['FechaHora'] <= fecha_fin_dt)].copy()
    datos_gcp = datos_gcp.dropna(subset=['Temperatura', 'FechaHora'])
    
    if datos_gcp.empty:
        st.warning(f"⚠️ No se encontraron datos válidos para el tren {tren} en el rango de fechas seleccionado")
        return
    
    fig = go.Figure()
    
    # Temperatura de gases
    fig.add_trace(go.Scatter(
        x=datos_gcp["FechaHora"],
        y=datos_gcp["Temperatura"],
        name=f"Temp {tren}",
        line=dict(color="red", width=2),
        yaxis="y1",
        hovertemplate="%{x|%d/%m %H:%M}<br>Temp Gases: %{y:.1f}°C<extra></extra>"
    ))
    
    fig.add_hline(
        y=TEMP_DISENO,
        line=dict(color="red", dash="dash", width=1),
        annotation_text=f"Diseño: {TEMP_DISENO}°C",
        annotation_position="bottom right",
        yref="y1"
    )
    
    # ΔT de los EP-110
    colores_delta = ['blue', 'green', 'purple']
    equipos_con_datos = []
    for i, equipo in enumerate(equipos_relacionados):
        datos_equipo = filtrar_datos_por_rango(df, fecha_inicio, fecha_fin, planta, equipo)
        datos_equipo = datos_equipo.dropna(subset=['Δ Temp Agua', 'FechaHora'])
        
        if not datos_equipo.empty:
            fig.add_trace(go.Scatter(
                x=datos_equipo["FechaHora"],
                y=datos_equipo["Δ Temp Agua"],
                name=f"ΔT {equipo.split()[-1]}",
                line=dict(color=colores_delta[i], width=1.5, dash='dot'),
                yaxis="y2",
                hovertemplate="%{x|%d/%m %H:%M}<br>ΔT: %{y:.2f}°C<extra></extra>"
            ))
            equipos_con_datos.append(equipo)
    
    fig.update_layout(
        title=f"📊 {tren} vs EP-110's: Temperatura y ΔT",
        xaxis_title="Fecha y Hora",
        yaxis=dict(
            title=f"Temp Gases {tren} (°C)",
            side="left",
            color="red",
            range=[0, max(datos_gcp["Temperatura"].max()*1.1, 40)]
        ),
        yaxis2=dict(
            title="ΔT Agua (°C)",
            overlaying="y",
            side="right",
            color="blue",
            range=[0, max(df[(df["Planta"]==planta) & 
                           (df["FechaHora"] >= fecha_inicio_dt) & 
                           (df["FechaHora"] <= fecha_fin_dt)]["Δ Temp Agua"].max()*1.2, 10)]
        ),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📊 Análisis Operacional", expanded=True):
        try:
            temp_actual = datos_gcp["Temperatura"].iloc[-1] if not datos_gcp.empty else None
            avg_temp = datos_gcp["Temperatura"].mean()
            max_temp = datos_gcp["Temperatura"].max()
            min_temp = datos_gcp["Temperatura"].min()
            
            st.markdown(f"""
            **Estado Actual {tren}**
            - 📌 Última medición: **{temp_actual:.1f}°C** {'🔴 (ALTA)' if temp_actual and temp_actual > TEMP_DISENO*1.1 else '🟢 (NORMAL)' if temp_actual else '⚪ (N/A)'}
            - 📊 Promedio: {avg_temp:.1f}°C (Rango: {min_temp:.1f}°C a {max_temp:.1f}°C)
            - 🎯 Desviación diseño: {avg_temp-TEMP_DISENO:+.1f}°C
            """)
            
            if not equipos_con_datos:
                st.warning("⚠️ No hay datos válidos de EP-110's para análisis de correlación")
            else:
                st.markdown("**🔍 Relación con ΔT Agua:**")
                cols = st.columns(len(equipos_con_datos))
                correlaciones_inversas = []
                
                # Fusionar datos para correlación
                for i, equipo in enumerate(equipos_con_datos):
                    merged = pd.merge_asof(
                        datos_gcp.sort_values("FechaHora"),
                        df[(df["Equipo"] == equipo) & 
                           (df["Planta"] == planta) & 
                           (df["FechaHora"] >= fecha_inicio_dt) & 
                           (df["FechaHora"] <= fecha_fin_dt)].sort_values("FechaHora"),
                        on="FechaHora",
                        direction="nearest",
                        tolerance=pd.Timedelta("2h"),
                        suffixes=('_gcp', '_ep')
                    ).dropna(subset=['Temperatura', 'Δ Temp Agua'])
                    
                    if len(merged) >= 3:
                        corr = merged["Δ Temp Agua"].corr(merged["Temperatura"])
                        letra_equipo = equipo.split()[-1]
                        if corr < -0.5:
                            correlaciones_inversas.append((equipo, corr))
                        
                        with cols[i]:
                            if not pd.isna(corr):
                                st.metric(
                                    label=f"EP-110 {letra_equipo}",
                                    value=f"{corr:.2f}",
                                    delta=f"{'↑↑' if corr > 0.7 else '↑' if corr > 0.3 else '↔' if corr > -0.3 else '↓' if corr > -0.7 else '↓↓'} ({len(merged)} pts)"
                                )
                                st.caption(f"Período: {merged['FechaHora'].min().strftime('%d/%m')} a {merged['FechaHora'].max().strftime('%d/%m')}")
                            else:
                                st.metric(label=f"EP-110 {letra_equipo}", value="N/A", delta="Sin correlación")
                    else:
                        with cols[i]:
                            st.metric(
                                label=f"EP-110 {equipo.split()[-1]}",
                                value="N/A",
                                delta=f"Datos insuficientes ({len(merged)} pts)"
                            )
                            if len(merged) > 0:
                                st.caption(f"⏱️ Ajuste horario recomendado: ±{int(merged['FechaHora'].diff().mean().total_seconds()/3600)}h")
                
                if temp_actual and temp_actual > TEMP_DISENO*1.1:
                    st.warning("""
                    **🚨 Acciones Recomendadas:**
                    - Verificar carga térmica del sistema
                    - Revisar estado de intercambiadores
                    - Chequear flujo de agua de enfriamiento
                    """)
                
                if correlaciones_inversas:
                    st.info("""
                    **ℹ️ Observación:**
                    Relación inversa detectada (al subir temp gases, baja ΔT) en:
                    """ + ", ".join([f"{eq.split()[-1]} ({corr:.2f})" for eq, corr in correlaciones_inversas]))
                    
        except Exception as e:
            st.error(f"Error en análisis operacional: {str(e)}")

def plot_distribucion_delta(df_filtrado, planta):
    """
    Genera un gráfico de caja para la distribución de ΔT Agua por equipo con división por trenes.
    
    Args:
        df_filtrado (pd.DataFrame): Datos filtrados.
        planta (str): Planta seleccionada.
    """
    if df_filtrado.empty:
        st.warning("No hay registros disponibles para los filtros aplicados.")
        return

    # Filtrar registros para análisis
    df_historial = cargar_datos()
    if planta != "Todas":
        df_historial = df_historial[df_historial["Planta"] == planta]
    mantenciones = df_historial[df_historial["Estado"] == "Mantención"]
    mantenciones_ordenadas = mantenciones.sort_values("FechaHora", ascending=False)
    mantenciones_ultima = mantenciones_ordenadas.drop_duplicates("Planta")
    equipos_con_mantencion = mantenciones_ultima["Equipo"].tolist()

    # Preparar el dataframe incluyendo los equipos en mantención
    df_plot = df_filtrado[df_filtrado["Estado"] == "En Servicio"].copy()
    if planta != "Todas":
        df_plot = df_plot[df_plot["Planta"] == planta]
    df_plot["Mantencion"] = df_plot["Equipo"].apply(lambda x: "Mantención" if x in equipos_con_mantencion else "Operativo")

    st.subheader("📦 Distribución de Δ Temp Agua por Equipo")

    # Generar gráfico si hay datos para graficar
    if not df_plot.empty and df_plot['Δ Temp Agua'].notna().any():
        color_discrete_map = {"Operativo": "#636EFA", "Mantención": "gray"}
        opacity_map = {"Operativo": 1.0, "Mantención": 0.4}

        fig = px.box(
            df_plot,
            x="Equipo",
            y="Δ Temp Agua",
            color="Mantencion",
            color_discrete_map=color_discrete_map,
            category_orders={"Equipo": EQUIPOS},
            template="plotly_white",
            labels={"Δ Temp Agua": "ΔT Agua (°C)"},
            title="Distribución por Equipo"
        )

        for trace in fig.data:
            estado = trace.name
            trace.marker.opacity = opacity_map[estado]

        df_min_by_equipo = df_plot.groupby("Equipo")["Δ Temp Agua"].min().reset_index()

        # Punto crítico global (Tren A)
        df_min_tren_a = df_min_by_equipo[df_min_by_equipo["Equipo"].isin(TREN_A)]
        if not df_min_tren_a.empty:
            equipo_critico_tren_a = df_min_tren_a.loc[df_min_tren_a["Δ Temp Agua"].idxmin()]
            punto_critico_tren_a = df_plot[df_plot["Equipo"] == equipo_critico_tren_a["Equipo"]].loc[
                df_plot[df_plot["Equipo"] == equipo_critico_tren_a["Equipo"]]["Δ Temp Agua"].idxmin()
            ]
            fig.add_trace(go.Scatter(
                x=[punto_critico_tren_a["Equipo"]],
                y=[punto_critico_tren_a["Δ Temp Agua"]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="x"),
                name="Punto crítico Tren A",
                hovertemplate="<b>%{x}</b><br>ΔT: %{y:.2f}°C<extra></extra>"
            ))

        # Punto crítico Tren B
        df_min_tren_b = df_min_by_equipo[df_min_by_equipo["Equipo"].isin(TREN_B)]
        if not df_min_tren_b.empty:
            equipo_critico_tren_b = df_min_tren_b.loc[df_min_tren_b["Δ Temp Agua"].idxmin()]
            punto_critico_tren_b = df_plot[df_plot["Equipo"] == equipo_critico_tren_b["Equipo"]].loc[
                df_plot[df_plot["Equipo"] == equipo_critico_tren_b["Equipo"]]["Δ Temp Agua"].idxmin()
            ]
            fig.add_trace(go.Scatter(
                x=[punto_critico_tren_b["Equipo"]],
                y=[punto_critico_tren_b["Δ Temp Agua"]],
                mode="markers",
                marker=dict(color="red", size=12, symbol="triangle-up"),
                name="Punto crítico Tren B",
                hovertemplate="<b>%{x}</b><br>ΔT: %{y:.2f}°C<extra></extra>"
            ))

        fig.add_vline(x=2.5, line_dash="dash", line_color="gray", annotation_text="Tren A | Tren B", annotation_position="top")
        fig.add_hline(
            y=UMBRAL_AGUA,
            line_dash="dot",
            line_color="red",
            annotation_text="Mínimo Recomendado",
            annotation_position="top right"
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📊 Análisis de Prioridades", expanded=True):
            if df_plot['Δ Temp Agua'].notna().any():
                df_min_by_equipo = df_plot.groupby("Equipo")["Δ Temp Agua"].min().reset_index()

                df_min_tren_a = df_min_by_equipo[df_min_by_equipo["Equipo"].isin(TREN_A)]
                df_min_tren_b = df_min_by_equipo[df_min_by_equipo["Equipo"].isin(TREN_B)]

                equipo_critico_tren_a = df_min_tren_a.loc[df_min_tren_a["Δ Temp Agua"].idxmin()] if not df_min_tren_a.empty else None
                equipo_critico_tren_b = df_min_tren_b.loc[df_min_tren_b["Δ Temp Agua"].idxmin()] if not df_min_tren_b.empty else None

                equipo_a_nombre = equipo_critico_tren_a["Equipo"] if equipo_critico_tren_a is not None else "N/A"
                equipo_a_valor = f"{equipo_critico_tren_a['Δ Temp Agua']:.2f}" if equipo_critico_tren_a is not None else "N/A"
                equipo_b_nombre = equipo_critico_tren_b["Equipo"] if equipo_critico_tren_b is not None else "N/A"
                equipo_b_valor = f"{equipo_critico_tren_b['Δ Temp Agua']:.2f}" if equipo_critico_tren_b is not None else "N/A"

                st.markdown(f"""
                **Prioridad de Mantención por Tren:**

                - **Tren A (EP-110 A/B/C):**
                  - 🚨 **Equipo más crítico:** {equipo_a_nombre} (ΔT {equipo_a_valor}°C)

                - **Tren B (EP-110 D/E/F):**
                  - 🚨 **Equipo más crítico:** {equipo_b_nombre} (ΔT {equipo_b_valor}°C)
                """)
            else:
                st.warning("No hay datos válidos para determinar la prioridad de mantención.")
    else:
        st.error("No hay datos suficientes para generar el gráfico de distribución.")

    # Historial de mantenciones
    st.subheader("📅 Historial de Mantenciones")
    primeras_mantenciones = mantenciones.sort_values("FechaHora").groupby("Equipo").first().reset_index()

    if not primeras_mantenciones.empty:
        fig_tiempo = px.scatter(
            primeras_mantenciones,
            x="FechaHora",
            y="Equipo",
            text=primeras_mantenciones["FechaHora"].dt.strftime("%Y-%m-%d"),
            labels={"FechaHora": "Fecha"},
            color_discrete_sequence=["indianred"],
            template="plotly_white",
            title="Primera Mantención por Equipo"
        )
        fig_tiempo.update_traces(
            mode="markers+text",
            textposition="top center",
            marker=dict(size=12),
            hovertemplate="<b>%{y}</b><br>Primera mantención: %{x|%d/%m/%Y}"
        )
        fig_tiempo.update_layout(yaxis=dict(categoryorder="array", categoryarray=EQUIPOS))
        st.plotly_chart(fig_tiempo, use_container_width=True)

        with st.expander("📆 Frecuencia de Mantenciones", expanded=False):
            hoy = datetime.now()
            dias_desde_mantencion = [(hoy - m).days for m in primeras_mantenciones["FechaHora"]]
            promedio_dias = np.mean(dias_desde_mantencion) if dias_desde_mantencion else 0

            st.markdown(f"""
            **Estadísticas de Mantención:**

            - 📅 **Mantención más reciente:** {primeras_mantenciones["FechaHora"].max().strftime('%d/%m/%Y')}
            - ⏳ **Promedio desde mantención:** {promedio_dias:.0f} días
            - 🔄 **Equipos intervenidos:** {len(primeras_mantenciones)}/{len(EQUIPOS)}

            **Recomendación:**
            {"🔴 Revisar equipos no intervenidos" if len(primeras_mantenciones) < len(EQUIPOS) else "🟢 Ciclo completo"}
            """)

    # Ciclos de mantención
    st.subheader("🔁 Ciclos de Mantención Completos")
    df_mant = df_historial[df_historial["Estado"] == "Mantención"].copy()
    df_mant["FechaHora"] = pd.to_datetime(df_mant["Fecha"] + " " + df_mant["Hora"])
    df_mant = df_mant.sort_values("FechaHora")

    ciclos = []
    ciclo_actual = {}
    for idx, row in df_mant.iterrows():
        equipo = row["Equipo"]
        if equipo not in ciclo_actual:
            ciclo_actual[equipo] = row["FechaHora"]
        if len(ciclo_actual) == 6:
            inicio = min(ciclo_actual.values())
            fin = max(ciclo_actual.values())
            ciclos.append((inicio, fin, (fin - inicio).days))
            ciclo_actual = {}

    if ciclos:
        ciclo_df = pd.DataFrame(ciclos, columns=["Inicio", "Término", "Duración (días)"])
        fig = px.timeline(
            ciclo_df, 
            x_start="Inicio", 
            x_end="Término", 
            y=[f"Ciclo {i+1}" for i in range(len(ciclo_df))],
            color="Duración (días)",
            template="plotly_white",
            title="🔄 Línea de Tiempo de Ciclos de Mantención"
        )
        fig.update_layout(xaxis_title="Fecha", yaxis_title="Ciclo")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(ciclo_df, use_container_width=True)
    else:
        st.info("No hay ciclos de mantención completos registrados aún.")

@st.cache_data
def convert_df_to_csv(df):
    """
    Convierte un DataFrame a formato CSV para descarga.
    
    Args:
        df (pd.DataFrame): DataFrame a convertir.
    
    Returns:
        bytes: CSV codificado en UTF-8.
    """
    return df.to_csv(index=False).encode('utf-8')

# --- Inicio de la aplicación ---
df = cargar_datos()
df_gcp = cargar_datos_gcp()

if df.empty:
    st.error("⚠️ El archivo de datos EP-110 está vacío o no se pudo cargar correctamente.")
    st.stop()

st.sidebar.title("🌡️ Monitoreo EP-110")
opcion = st.sidebar.radio("Menú", ["📊 Dashboard", "📥 Ingreso de Datos"])

if opcion == "📊 Dashboard":
    st.title("📊 Dashboard Analítico EP-110")

    with st.sidebar:
        st.header("🔍 Filtros")
        planta = st.selectbox("Planta", ["Todas"] + PLANTAS)
        equipo = st.selectbox("Equipo", ["Todos"] + EQUIPOS)
        
        min_date = df["FechaHora"].min().date() if not df.empty else datetime.now().date()
        hoy = datetime.now().date()
        default_start = min_date if not df.empty else hoy - timedelta(days=30)
        default_end = hoy

        fecha_inicio, fecha_fin = st.date_input(
            "Rango de fechas",
            value=(default_start, default_end),
            min_value=min_date if not df.empty else hoy - timedelta(days=365),
            max_value=hoy,
            key="date_range"
        )
        
        if fecha_inicio > fecha_fin:
            st.error("Error: La fecha de inicio no puede ser posterior a la fecha final.")
            st.stop()
            
        incluir_mant = st.checkbox("Incluir mantenciones", False)

    # Filtrado de datos
    df_filtrado = filtrar_datos_por_rango(df, fecha_inicio, fecha_fin, planta, equipo, incluir_mant)

    # Descarga de datos filtrados
    st.download_button(
        label="📥 Descargar datos filtrados",
        data=convert_df_to_csv(df_filtrado),
        file_name="ep110_datos_filtrados.csv",
        mime="text/csv"
    )

    # Indicadores Clave
    st.subheader("📌 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)

    delta_agua_mean = df_filtrado['Δ Temp Agua'].mean() if not df_filtrado.empty else 0
    delta_efluente_mean = df_filtrado['Δ Temp Efluente'].mean() if not df_filtrado.empty else 0

    col1.metric("🌡️ ΔT Agua Prom", f"{delta_agua_mean:.2f} °C" if isinstance(delta_agua_mean, (int, float)) else "N/A")
    col2.metric("♨️ ΔT Efluente Prom", f"{delta_efluente_mean:.2f} °C" if isinstance(delta_efluente_mean, (int, float)) else "N/A")

    ultimos = df[df["Planta"] == planta] if planta != "Todas" else df
    ultimos = ultimos.sort_values("FechaHora").drop_duplicates("Equipo", keep="last")
    en_servicio = ultimos[ultimos["Estado"] == "En Servicio"]
    col3.metric("⚙️ Equipos Activos", f"{len(en_servicio)} / {len(EQUIPOS)}")

    ultimo_registro = df_filtrado["FechaHora"].max().strftime("%d/%m %H:%M") if not df_filtrado.empty else "N/A"
    col4.metric("🕒 Último Registro", ultimo_registro)

    fuera_de_servicio = [e for e in EQUIPOS if e not in en_servicio["Equipo"].tolist()]
    st.markdown(f"**🔧 Equipos fuera de servicio:** {', '.join(fuera_de_servicio) if fuera_de_servicio else 'Ninguno'}")

    # Evolución Térmica
    st.subheader("📈 Evolución Térmica")
    tabs = st.tabs(["ΔT Agua", "ΔT Efluente"])
    with tabs[0]:
        plot_evolucion_termica(df_filtrado, "Δ Temp Agua", UMBRAL_AGUA)
    with tabs[1]:
        plot_evolucion_termica(df_filtrado, "Δ Temp Efluente", UMBRAL_EFLUENTE)

    # Distribución por equipo
    plot_distribucion_delta(df_filtrado, planta)

    # Visualización GCP
    if planta in ["GCP-2", "Todas"]:
        st.markdown("## 📈 Análisis de Transferencia Térmica - GCP-2")
        gcp2_tabs = st.tabs(["GCP-2A vs EP-110 A/B/C", "GCP-2B vs EP-110 D/E/F"])
        with gcp2_tabs[0]:
            plot_comparacion_gcp(df, df_gcp, "GCP-2A", TREN_A, "GCP-2", fecha_inicio, fecha_fin)
        with gcp2_tabs[1]:
            plot_comparacion_gcp(df, df_gcp, "GCP-2B", TREN_B, "GCP-2", fecha_inicio, fecha_fin)

    if planta in ["GCP-4", "Todas"]:
        st.markdown("## 📈 Análisis de Transferencia Térmica - GCP-4")
        gcp4_tabs = st.tabs(["GCP-4A vs EP-110 A/B/C", "GCP-4B vs EP-110 D/E/F"])
        with gcp4_tabs[0]:
            plot_comparacion_gcp(df, df_gcp, "GCP-4A", TREN_A, "GCP-4", fecha_inicio, fecha_fin)
        with gcp4_tabs[1]:
            plot_comparacion_gcp(df, df_gcp, "GCP-4B", TREN_B, "GCP-4", fecha_inicio, fecha_fin)

elif opcion == "📥 Ingreso de Datos":
    st.title("📥 Registro de Datos EP-110")
    st.info("🔧 Ingrese los datos de temperatura y estado para cada equipo.")

    with st.form("formulario_registro"):
        col1, col2 = st.columns(2)
        with col1:
            planta = st.selectbox("🔧 Planta", PLANTAS)
            fecha = st.date_input("📅 Fecha", datetime.now())
        with col2:
            hora = st.text_input("⏰ Hora (HH:MM)", value=datetime.now().strftime("%H:%M"))

        st.subheader("📋 Registro de Datos por Equipo")
        columnas = ["Equipo", "Estado", "Temp Entrada Agua (°C)", "Temp Salida Agua (°C)", 
                    "Temp Entrada Efluente (°C)", "Temp Salida Efluente (°C)", "Comentario"]
        datos_form = [[equipo, "En Servicio", 0.0, 0.0, 0.0, 0.0, ""] for equipo in EQUIPOS]
        df_form = pd.DataFrame(datos_form, columns=columnas)
        df_editado = st.data_editor(df_form, use_container_width=True, num_rows="fixed")

        submitted = st.form_submit_button("💾 Guardar Datos")
        if submitted:
            errores = []
            try:
                datetime.strptime(hora, "%H:%M")
            except ValueError:
                errores.append("Formato de hora inválido. Use HH:MM")

            registros = []
            for _, row in df_editado.iterrows():
                equipo = row["Equipo"]
                estado = row["Estado"]
                t1 = row["Temp Entrada Agua (°C)"]
                t2 = row["Temp Salida Agua (°C)"]
                t3 = row["Temp Entrada Efluente (°C)"]
                t4 = row["Temp Salida Efluente (°C)"]
                comentario = row["Comentario"] if pd.notna(row["Comentario"]) else ""

                if estado == "Mantención" and not comentario.strip():
                    errores.append(f"Falta comentario de mantención para {equipo}")
                
                delta_agua = round(t2 - t1, 2) if estado == "En Servicio" and None not in [t1, t2] else 0.0
                delta_e = round(t3 - t4, 2) if estado == "En Servicio" and None not in [t3, t4] else 0.0
                
                if estado == "En Servicio":
                    if delta_agua < 0:
                        errores.append(f"ΔT Agua negativo para {equipo}. Verifique las temperaturas.")
                    if delta_e < 0:
                        errores.append(f"ΔT Efluente negativo para {equipo}. Verifique las temperaturas.")
                
                registros.append({
                    "Fecha": fecha.strftime("%Y-%m-%d"),
                    "Hora": hora,
                    "Planta": planta,
                    "Equipo": equipo,
                    "Estado": estado,
                    "Temp Entrada Agua Torre": t1,
                    "Temp Salida Agua Torre": t2,
                    "Δ Temp Agua": delta_agua,
                    "Temp Entrada Efluente": t3,
                    "Temp Salida Efluente": t4,
                    "Δ Temp Efluente": delta_e,
                    "Comentarios": comentario if comentario else "N/A"
                })

            if not errores:
                try:
                    df_nuevo = pd.DataFrame(registros)
                    if os.path.exists(ARCHIVO):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        shutil.copy2(ARCHIVO, f"backup_{ARCHIVO}_{timestamp}.csv")
                    df_existente = pd.read_csv(ARCHIVO) if os.path.exists(ARCHIVO) else pd.DataFrame()
                    df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
                    df_final = df_final.drop_duplicates(subset=["Fecha", "Hora", "Planta", "Equipo"], keep="last")
                    df_final.to_csv(ARCHIVO, index=False)
                    
                    st.session_state["datos_guardados"] = True
                    st.session_state["ultimo_registro"] = df_nuevo
                    st.success(f"✅ Datos guardados correctamente! Registrados {len(df_nuevo)} registros.")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error al guardar: {str(e)}")
            else:
                for error in errores:
                    st.error(f"⚠️ {error}")

    if st.session_state.get("datos_guardados") and st.session_state.get("ultimo_registro") is not None:
        st.markdown("---")
        st.subheader("Últimos datos registrados")
        st.dataframe(st.session_state["ultimo_registro"], hide_index=True)