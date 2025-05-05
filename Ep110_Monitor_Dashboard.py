# --- Streamlit App Mejorada para Monitoreo Térmico EP-110 ---
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import gaussian_kde

# Configuración general
st.set_page_config(page_title="🌡️ Monitoreo Térmico EP-110", layout="wide")

# Constantes
ARCHIVO = "data.csv"
EQUIPOS = ["EP-110 A", "EP-110 B", "EP-110 C", "EP-110 D", "EP-110 E", "EP-110 F"]
PLANTAS = ["GCP-2", "GCP-4"]
UMBRAL_AGUA = 4.0
UMBRAL_EFLUENTE = 3.0

# Funciones
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv(ARCHIVO)
        df["FechaHora"] = pd.to_datetime(df["Fecha"] + " " + df["Hora"])
        df.rename(columns={"Delta Agua": "Δ Temp Agua", "Delta Efluente": "Δ Temp Efluente"}, inplace=True)
        return df.sort_values("FechaHora")
    except Exception as e:
        st.error(f"Error al cargar datos EP-110: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def cargar_datos_gcp():
    try:
        df_gcp = pd.read_csv("temp_gcp_data.csv")
        df_gcp['FechaHora'] = pd.to_datetime(df_gcp['FechaHora'])
        return df_gcp
    except Exception as e:
        st.error(f"Error al cargar datos GCP: {str(e)}")
        return pd.DataFrame()

def plot_histograma(df_filtrado, columna, umbral, titulo):
    x_vals = df_filtrado[columna].dropna()
    fig = go.Figure(layout=dict(template="plotly_white"))
    
    if len(x_vals) == 0:
        st.warning(f"No hay datos disponibles para {columna}")
        return fig
    
    # Histograma principal
    fig.add_trace(go.Histogram(
        x=x_vals, 
        nbinsx=20, 
        name="Frecuencia",
        opacity=0.75,
        hovertemplate="Rango: %{x:.2f}°C<br>Cantidad: %{y}<extra></extra>"
    ))
    
    # Curva de densidad KDE
    if len(x_vals) > 1:
        kde = gaussian_kde(x_vals)
        x_range = np.linspace(min(x_vals), max(x_vals), 200)
        factor = max(np.histogram(x_vals, bins=20)[0])
        kde_scaled = kde(x_range) * factor
        fig.add_trace(go.Scatter(
            x=x_range,
            y=kde_scaled,
            mode="lines",
            name="Densidad",
            line=dict(color="orange", width=3),
            hoverinfo="x+y",
            hovertemplate="ΔT: %{x:.2f} °C<br>Densidad: %{y:.2f}<extra></extra>"
        ))
    
    # Línea de umbral
    fig.add_vline(
        x=umbral, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Mínimo Recomendado", 
        annotation_position="top"
    )
    
    fig.update_layout(
        barmode='overlay',
        xaxis_title=f"{columna} (°C)",
        yaxis_title="Frecuencia",
        title=titulo,
        hovermode="x unified"
    )
    
    # Análisis interpretativo automático
    with st.expander("🔍 Interpretación del Histograma", expanded=False):
        media = x_vals.mean()
        bajo_umbral = len(x_vals[x_vals < umbral])
        porcentaje_bajo = (bajo_umbral / len(x_vals)) * 100 if len(x_vals) > 0 else 0
        
        st.markdown(f"""
        **Distribución de {columna}:**  
        
        - 📊 **Media:** {media:.2f}°C  
        - ⚠️ **Valores bajo umbral ({umbral}°C):** {bajo_umbral} ({porcentaje_bajo:.1f}%)  
        - 📈 **Forma:** {"Asimétrica derecha" if x_vals.skew() > 0.5 else "Asimétrica izquierda" if x_vals.skew() < -0.5 else "Simétrica"}  
        
        **Recomendaciones:**  
        {f'🔴 Actuar urgentemente' if porcentaje_bajo > 20 else '🟢 Situación estable' if porcentaje_bajo < 5 else '🟡 Monitorear de cerca'}  
        {f'- {bajo_umbral} registros requieren atención' if bajo_umbral > 0 else '- Todos los valores dentro del rango recomendado'}
        """)
    
    return fig

def plot_evolucion_termica(df_filtrado, variable, umbral):
    """
    Gráfico de evolución temporal con análisis automático de tendencias
    """
    if df_filtrado.empty:
        st.warning("No hay datos disponibles para el período seleccionado")
        return
    
    # Crear figura
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
        hover_data={
            "FechaHora": "|%d/%m/%Y %H:%M",
            variable: ":.2f",
            "Equipo": True
        }
    )
    
    # Línea de umbral
    fig.add_hline(
        y=umbral,
        line_dash="dot",
        line_color="red",
        annotation_text="Mínimo Recomendado",
        annotation_position="bottom right"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de tendencia
    with st.expander("📈 Análisis de Tendencia Térmica", expanded=False):
        # Calcular métricas
        ultima_semana = df_filtrado[df_filtrado['FechaHora'] > (datetime.now() - timedelta(days=7))]
        
        # Calcular tendencia
        if not ultima_semana.empty:
            tendencia = "mejorando ↗️" if ultima_semana[variable].mean() > df_filtrado[variable].mean() else "empeorando ↘️"
        else:
            tendencia = "datos insuficientes"
            
        variabilidad = df_filtrado.groupby('Equipo')[variable].std().mean()
        max_val = df_filtrado[variable].max()
        min_val = df_filtrado[variable].min()
        
        st.markdown(f"""
        **Análisis de {variable.split(' ')[-1]}:**  
        
        - 📅 **Período analizado:** {df_filtrado['FechaHora'].dt.date.min()} al {df_filtrado['FechaHora'].dt.date.max()}  
        - 📉 **Tendencia reciente:** {tendencia}  
        - 📊 **Variabilidad promedio:** {variabilidad:.2f}°C  
        - 🔥 **Valor máximo registrado:** {max_val:.2f}°C  
        - ❄️ **Valor mínimo registrado:** {min_val:.2f}°C  
        
        **Interpretación:**  
        - Variabilidad {'alta (>1.5°C)' if variabilidad > 1.5 else 'moderada' if variabilidad > 0.5 else 'baja'}  
        - Consistencia: {'pobre ❌' if variabilidad > 1.5 else 'aceptable ⚠️' if variabilidad > 1.0 else 'buena ✅'}  
        """)

def plot_comparacion_gcp(df, df_gcp, tren, equipos_relacionados, planta):
    """
    Gráfico y análisis completo con dos ejes Y y manejo correcto de variables
    """
    # Verificación inicial de datos
    if df.empty or df_gcp.empty:
        st.warning("⚠️ No hay datos disponibles para realizar el análisis")
        return
    
    # Filtrar datos del tren específico
    try:
        datos_gcp = df_gcp[df_gcp['Tren'] == tren].copy()
        datos_gcp = datos_gcp.dropna(subset=['Temperatura', 'FechaHora'])
    except KeyError:
        st.error("Error: La columna 'Tren' no existe en los datos de GCP")
        return
    
    if datos_gcp.empty:
        st.warning(f"⚠️ No se encontraron datos válidos para el tren {tren}")
        return
    
    # Valor de diseño referencia
    TEMP_DISENO = 32  # °C
    
    # Lista para almacenar equipos con datos válidos (SOLUCIÓN AL ERROR)
    equipos_con_datos = []
    
    # --- GRÁFICO PRINCIPAL ---
    fig = go.Figure()
    
    # 1. Temperatura de gases (eje Y izquierdo)
    color_tren = "red"  # Color rojo para temperatura de gases
    fig.add_trace(go.Scatter(
        x=datos_gcp["FechaHora"],
        y=datos_gcp["Temperatura"],
        name=f"Temp {tren}",
        line=dict(color=color_tren, width=2),
        yaxis="y1",
        hovertemplate="%{x|%d/%m %H:%M}<br>Temp Gases: %{y:.1f}°C<extra></extra>"
    ))
    
    # 2. Línea de diseño (32°C)
    fig.add_hline(
        y=TEMP_DISENO,
        line=dict(color="red", dash="dash", width=1),
        annotation_text=f"Diseño: {TEMP_DISENO}°C",
        annotation_position="bottom right",
        yref="y1"
    )
    
    # 3. ΔT de los EP-110's (eje Y derecho)
    colores_delta = ['blue', 'green', 'purple']  # Colores para ΔT D, E, F
    for i, equipo in enumerate(equipos_relacionados):
        try:
            datos_equipo = df[(df["Equipo"] == equipo) & (df["Planta"] == planta)].copy()
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
                equipos_con_datos.append(equipo)  # Registrar equipos con datos válidos
        except Exception as e:
            st.warning(f"Error al graficar {equipo}: {str(e)}")
    
    # Configuración del layout
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
            range=[0, max(df[df["Planta"]==planta]["Δ Temp Agua"].max()*1.2, 10)]
        ),
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # --- ANÁLISIS OPERACIONAL ---
    with st.expander("📊 Análisis Operacional", expanded=True):
        # Resumen estadístico
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
            
            # Análisis de correlación (ahora con equipos_con_datos definido)
            if not equipos_con_datos:
                st.warning("⚠️ No hay datos válidos de EP-110's para análisis de correlación")
            else:
                st.markdown("**🔍 Relación con ΔT Agua:**")
                cols = st.columns(len(equipos_con_datos))
                
                correlaciones_inversas = []
                
                for i, equipo in enumerate(equipos_con_datos):
                    try:
                        merged = pd.merge_asof(
                            datos_gcp.sort_values("FechaHora"),
                            df[(df["Equipo"] == equipo) & (df["Planta"] == planta)].sort_values("FechaHora"),
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
                                    st.metric(
                                        label=f"EP-110 {letra_equipo}",
                                        value="N/A",
                                        delta="Sin correlación"
                                    )
                        else:
                            with cols[i]:
                                st.metric(
                                    label=f"EP-110 {equipo.split()[-1]}",
                                    value="N/A",
                                    delta=f"Datos insuficientes ({len(merged)} pts)"
                                )
                                if len(merged) > 0:
                                    st.caption(f"⏱️ Ajuste horario recomendado: ±{int(merged['FechaHora'].diff().mean().total_seconds()/3600)}h")
                    except Exception as e:
                        st.error(f"Error analizando {equipo}: {str(e)}")
                
                # Recomendaciones operacionales
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
    if df_filtrado.empty:
        st.warning("No hay registros disponibles para los filtros aplicados.")
        return

    # Obtener datos recientes
    recientes = df_filtrado[df_filtrado["Estado"] == "En Servicio"]
    if planta != "Todas":
        recientes = recientes[recientes["Planta"] == planta]
    recientes = recientes.sort_values("FechaHora").drop_duplicates("Equipo", keep="last")

    # Identificar equipos con mantención
    df_historial = cargar_datos()
    if planta != "Todas":
        df_historial = df_historial[df_historial["Planta"] == planta]
    mantenciones = df_historial[df_historial["Estado"] == "Mantención"]
    equipos_con_mantencion = mantenciones["Equipo"].unique().tolist()

    # Gráfico de caja con punto crítico
    st.subheader("📦 Distribución de Δ Temp Agua por Equipo")
    
    disponibles = recientes[~recientes["Equipo"].isin(equipos_con_mantencion)]
    if not disponibles.empty:
        # Crear figura
        fig = px.box(
            df_filtrado,
            x="Equipo",
            y="Δ Temp Agua",
            points="outliers",
            category_orders={"Equipo": EQUIPOS},
            template="plotly_white",
            labels={"Δ Temp Agua": "ΔT Agua (°C)"},
            title="Distribución por Equipo"
        )
        
        # Identificar punto crítico
        punto_critico = disponibles.nsmallest(1, "Δ Temp Agua").iloc[0]
        fig.add_trace(go.Scatter(
            x=[punto_critico["Equipo"]],
            y=[punto_critico["Δ Temp Agua"]],
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name="Punto crítico",
            hovertemplate="<b>%{x}</b><br>ΔT: %{y:.2f}°C<extra></extra>"
        ))
        
        # Línea de umbral
        fig.add_hline(
            y=UMBRAL_AGUA,
            line_dash="dot",
            line_color="red",
            annotation_text="Mínimo Recomendado",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Análisis interpretativo
        with st.expander("📊 Análisis de Prioridades", expanded=True):
            # Calcular estadísticas
            mejor_equipo = recientes.nlargest(1, "Δ Temp Agua").iloc[0]
            rango = recientes["Δ Temp Agua"].max() - recientes["Δ Temp Agua"].min()
            
            st.markdown(f"""
            **Prioridad de Mantención:**  
            
            - 🚨 **Equipo más crítico:** {punto_critico['Equipo']} (ΔT {punto_critico['Δ Temp Agua']:.2f}°C)  
            - ✅ **Mejor desempeño:** {mejor_equipo['Equipo']} (ΔT {mejor_equipo['Δ Temp Agua']:.2f}°C)  
            - 📏 **Variación entre equipos:** {rango:.2f}°C  
            
            **Diagnóstico:**  
            {"🔴 Diferencias significativas" if rango > 3 else "🟢 Variación normal"}  
            {"⚠️ Posible problema en " + punto_critico['Equipo'] if punto_critico['Δ Temp Agua'] < UMBRAL_AGUA else ""}
            """)
            
            st.markdown(f"### 🛠️ Prioridad 1: `{punto_critico['Equipo']}` (ΔT: `{punto_critico['Δ Temp Agua']:.2f}` °C)")
    else:
        st.info("Todos los equipos ya han sido intervenidos en esta planta.")

    # Estado del ciclo de mantención
    st.markdown("### 🔄 Estado del Mantenimiento")
    intervenidos = [e for e in EQUIPOS if e in equipos_con_mantencion]
    faltantes = [e for e in EQUIPOS if e not in equipos_con_mantencion]
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Equipos intervenidos:**\n{', '.join(intervenidos) if intervenidos else 'Ninguno'}")
    with col2:
        st.warning(f"**Equipos pendientes:**\n{', '.join(faltantes) if faltantes else 'Ninguno'}")

    # Línea de tiempo de mantenciones
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
        
        # Análisis de frecuencia
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
        mostrar_ciclos_mantencion(df_historial)


def mostrar_ciclos_mantencion(df):
    st.subheader("🔁 Ciclos de Mantención Completos")
    df_mant = df[df["Estado"] == "Mantención"].copy()
    df_mant["FechaHora"] = pd.to_datetime(df_mant["Fecha"] + " " + df_mant["Hora"])
    df_mant = df_mant.sort_values("FechaHora")

    ciclos = []
    ciclo_actual = {}
    for idx, row in df_mant.iterrows():
        equipo = row["Equipo"]
        if equipo not in ciclo_actual:
            ciclo_actual[equipo] = row["FechaHora"]
            if len(ciclo_actual) == 6:  # todos los equipos intervenidos
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

def mostrar_ciclos_mantencion(df):
    st.subheader("🔁 Ciclos de Mantención Completos")
    df_mant = df[df["Estado"] == "Mantención"].copy()
    df_mant["FechaHora"] = pd.to_datetime(df_mant["Fecha"] + " " + df_mant["Hora"])
    df_mant = df_mant.sort_values("FechaHora")

    ciclos = []
    ciclo_actual = {}
    for idx, row in df_mant.iterrows():
        equipo = row["Equipo"]
        if equipo not in ciclo_actual:
            ciclo_actual[equipo] = row["FechaHora"]
        if len(ciclo_actual) == 6:  # todos los equipos intervenidos
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


    # Tabla de datos filtrados
    st.subheader("📋 Datos Detallados")
    st.dataframe(df_filtrado.style.format({
        "Δ Temp Agua": "{:.2f}",
        "Δ Temp Efluente": "{:.2f}"
    }), use_container_width=True)

# --- Inicio de la aplicación ---
df = cargar_datos()
df_gcp = cargar_datos_gcp()

if df.empty:
    st.error("⚠️ El archivo de datos EP-110 está vacío o no se pudo cargar correctamente.")
    st.stop()

# Definir opción de menú primero
st.sidebar.title("🌡️ Monitoreo EP-110")
opcion = st.sidebar.radio("Menú", ["📊 Dashboard", "📥 Ingreso de Datos"])

if opcion == "📊 Dashboard":
    st.title("📊 Dashboard Analítico EP-110")

    # Filtros en el sidebar
    with st.sidebar:
        st.header("🔍 Filtros")
        planta = st.selectbox("Planta", ["Todas"] + PLANTAS)
        equipo = st.selectbox("Equipo", ["Todos"] + EQUIPOS)
        
        min_date = df["FechaHora"].min().date()
        hoy = datetime.now().date()
        default_start = min_date
        default_end = hoy

        fecha_inicio, fecha_fin = st.date_input(
            "Rango de fechas",
            value=(default_start, default_end),
            min_value=min_date,
            max_value=hoy,
            key="date_range"
        )
        
        if fecha_inicio > fecha_fin:
            st.error("Error: La fecha de inicio no puede ser posterior a la fecha final.")
            st.stop()
            
        incluir_mant = st.checkbox("Incluir mantenciones", False)

    # Filtrado de datos
    df_filtrado = df.copy()
    if planta != "Todas":
        df_filtrado = df_filtrado[df_filtrado["Planta"] == planta]
    if equipo != "Todos":
        df_filtrado = df_filtrado[df_filtrado["Equipo"] == equipo]
            
    df_filtrado = df_filtrado[
        (df_filtrado["FechaHora"] >= pd.to_datetime(fecha_inicio)) &
        (df_filtrado["FechaHora"] <= pd.to_datetime(fecha_fin) + pd.Timedelta(days=1))
    ]
    
    if not incluir_mant:
        df_filtrado = df_filtrado[df_filtrado["Estado"] == "En Servicio"]

    # Indicadores Clave
    st.subheader("📌 Indicadores Clave")
    col1, col2, col3, col4 = st.columns(4)
    
    delta_agua_mean = df_filtrado['Δ Temp Agua'].mean() if not df_filtrado.empty else 0
    delta_efluente_mean = df_filtrado['Δ Temp Efluente'].mean() if not df_filtrado.empty else 0
    
    col1.metric("🌡️ ΔT Agua Prom", f"{delta_agua_mean:.2f} °C")
    col2.metric("♨️ ΔT Efluente Prom", f"{delta_efluente_mean:.2f} °C")

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

    # Histogramas
    st.subheader("📊 Histogramas con curva de densidad")
    tabs_hist = st.tabs(["ΔT Agua", "ΔT Efluente"])
    with tabs_hist[0]:
        st.plotly_chart(plot_histograma(df_filtrado, "Δ Temp Agua", UMBRAL_AGUA, "Histograma ΔT Agua"), use_container_width=True)
    with tabs_hist[1]:
        st.plotly_chart(plot_histograma(df_filtrado, "Δ Temp Efluente", UMBRAL_EFLUENTE, "Histograma ΔT Efluente"), use_container_width=True)

    # Distribución por equipo
    plot_distribucion_delta(df_filtrado, planta)

    # Visualización GCP
    if planta in ["GCP-2", "Todas"]:
        st.markdown("## 📈 Análisis de Transferencia Térmica - GCP-2")
        gcp2_tabs = st.tabs(["GCP-2A vs EP-110 A/B/C", "GCP-2B vs EP-110 D/E/F"])

        with gcp2_tabs[0]:
            plot_comparacion_gcp(df, df_gcp, "GCP-2A", ["EP-110 A", "EP-110 B", "EP-110 C"], "GCP-2")

        with gcp2_tabs[1]:
            plot_comparacion_gcp(df, df_gcp, "GCP-2B", ["EP-110 D", "EP-110 E", "EP-110 F"], "GCP-2")

    if planta in ["GCP-4", "Todas"]:
        st.markdown("## 📈 Análisis de Transferencia Térmica - GCP-4")
        gcp4_tabs = st.tabs(["GCP-4A vs EP-110 A/B/C", "GCP-4B vs EP-110 D/E/F"])

        with gcp4_tabs[0]:
            plot_comparacion_gcp(df, df_gcp, "GCP-4A", ["EP-110 A", "EP-110 B", "EP-110 C"], "GCP-4")

        with gcp4_tabs[1]:
            plot_comparacion_gcp(df, df_gcp, "GCP-4B", ["EP-110 D", "EP-110 E", "EP-110 F"], "GCP-4")

elif opcion == "📥 Ingreso de Datos":
    st.title("📥 Registro de Datos EP-110")
    st.info("🔧 Módulo de registro actualmente disponible para ingreso manual de datos.")

    with st.form("formulario_registro"):
        col1, col2 = st.columns(2)
        with col1:
            planta = st.selectbox("🔧 Planta", PLANTAS)
            fecha = st.date_input("📅 Fecha", datetime.now())
        with col2:
            hora = st.text_input("⏰ Hora (HH:MM)", value=datetime.now().strftime("%H:%M"))

        registros = []
        for equipo in EQUIPOS:
            with st.expander(f"{equipo}", expanded=False):
                estado = st.radio("Estado", ["En Servicio", "Mantención"], 
                                horizontal=True, key=f"estado_{equipo}",
                                help="Seleccione 'Mantención' si el equipo está fuera de servicio")

                if estado == "Mantención":
                    # Sección de Mantención
                    st.warning("⚠️ Equipo en mantención - Datos térmicos deshabilitados")
                    
                    # Campos de temperatura deshabilitados
                    col1_, col2_ = st.columns(2)
                    with col1_:
                        st.number_input("🌊 Temp Entrada Agua", value=0.0, disabled=True, key=f"{equipo}_t1_disabled")
                        st.number_input("🌊 Temp Salida Agua", value=0.0, disabled=True, key=f"{equipo}_t2_disabled")
                    with col2_:
                        st.number_input("♨️ Temp Entrada Efluente", value=0.0, disabled=True, key=f"{equipo}_t3_disabled")
                        st.number_input("♨️ Temp Salida Efluente", value=0.0, disabled=True, key=f"{equipo}_t4_disabled")
                    
                    # Comentario obligatorio
                    comentario = st.text_area("📝 Comentario de Mantención*", 
                                            placeholder="Describa el trabajo realizado...",
                                            key=f"comentario_{equipo}",
                                            help="Campo obligatorio para equipos en mantención")
                    
                    # Valores por defecto para mantención
                    t1 = t2 = t3 = t4 = delta_agua = delta_efl = 0.0
                else:
                    # Sección En Servicio
                    col1_, col2_ = st.columns(2)
                    with col1_:
                        t1 = st.number_input("🌊 Temp Entrada Agua (°C)", 
                                            min_value=0.0, max_value=100.0, step=0.1,
                                            key=f"{equipo}_t1")
                        t2 = st.number_input("🌊 Temp Salida Agua (°C)", 
                                            min_value=0.0, max_value=100.0, step=0.1,
                                            key=f"{equipo}_t2")
                    with col2_:
                        t3 = st.number_input("♨️ Temp Entrada Efluente (°C)", 
                                            min_value=0.0, max_value=100.0, step=0.1,
                                            key=f"{equipo}_t3")
                        t4 = st.number_input("♨️ Temp Salida Efluente (°C)", 
                                            min_value=0.0, max_value=100.0, step=0.1,
                                            key=f"{equipo}_t4")
                    
                    # Cálculo automático de deltas
                    delta_agua = round(t2 - t1, 2) if None not in [t1, t2] else 0.0
                    delta_efl = round(t3 - t4, 2) if None not in [t3, t4] else 0.0
                    
                    # Comentario opcional
                    comentario = st.text_area("📝 Comentario (opcional)", 
                                            key=f"comentario_{equipo}",
                                            placeholder="Observaciones adicionales...")

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
                    "Δ Temp Efluente": delta_efl,
                    "Comentarios": comentario if comentario else "N/A"
                })

        submitted = st.form_submit_button("💾 Guardar Datos")
        if submitted:
            # Validación de campos
            errores = []
            
            # Validar formato de hora
            try:
                datetime.strptime(hora, "%H:%M")
            except ValueError:
                errores.append("Formato de hora inválido. Use HH:MM")
            
            # Validar comentarios en mantención
            for reg in registros:
                if reg["Estado"] == "Mantención" and not reg["Comentarios"].strip():
                    errores.append(f"Falta comentario de mantención para {reg['Equipo']}")
            
            if not errores:
                try:
                    # Guardar datos
                    df_nuevo = pd.DataFrame(registros)
                    try:
                        df_existente = pd.read_csv(ARCHIVO)
                        df_final = pd.concat([df_existente, df_nuevo], ignore_index=True)
                    except FileNotFoundError:
                        df_final = df_nuevo
                    
                    df_final.to_csv(ARCHIVO, index=False)
                    
                    # Actualizar estado y mostrar confirmación
                    st.session_state["datos_guardados"] = True
                    st.session_state["ultimo_registro"] = df_nuevo
                    st.success("✅ Datos guardados correctamente!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error al guardar: {str(e)}")
            else:
                for error in errores:
                    st.error(f"❌ {error}")

    # Mostrar últimos datos guardados
    if st.session_state.get("datos_guardados") and st.session_state.get("ultimo_registro") is not None:
        st.markdown("---")
        st.subheader("Últimos datos registrados")
        st.dataframe(st.session_state["ultimo_registro"], hide_index=True)