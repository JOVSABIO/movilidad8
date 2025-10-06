# insights.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, MarkerCluster
import matplotlib.pyplot as plt
from datetime import datetime

class AccidentInsights:
    def __init__(self, data):
        self.data = data
        self.setup_insights()
    
    def setup_insights(self):
        """Prepara los datos para el análisis"""
        if self.data.empty:
            return
            
        # Convertir fecha si es necesario
        if 'fecha' in self.data.columns:
            try:
                self.data['fecha_dt'] = pd.to_datetime(self.data['fecha'], errors='coerce')
                self.data['hora'] = self.data['fecha_dt'].dt.hour
                self.data['dia_semana'] = self.data['fecha_dt'].dt.day_name()
                self.data['mes'] = self.data['fecha_dt'].dt.month
                self.data['año'] = self.data['fecha_dt'].dt.year
            except Exception as e:
                st.warning(f"Error procesando fechas: {e}")
    
    def create_executive_dashboard(self):
        """Dashboard Ejecutivo - Métricas clave y visión general"""
        st.header("📊 Dashboard Ejecutivo")
        
        if self.data.empty:
            st.warning("No hay datos para mostrar")
            return
            
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_accidentes = len(self.data)
            st.metric("Total Accidentes", f"{total_accidentes:,}")
        
        with col2:
            if 'gravedad' in self.data.columns:
                try:
                    graves = self.data['gravedad'].str.contains('Grave|Fatal', case=False, na=False).sum()
                    st.metric("Accidentes Graves", f"{graves}")
                except:
                    st.metric("Accidentes Graves", "N/A")
        
        with col3:
            if 'comuna' in self.data.columns:
                try:
                    comunas_afectadas = self.data['comuna'].nunique()
                    st.metric("Comunas Afectadas", comunas_afectadas)
                except:
                    st.metric("Comunas Afectadas", "N/A")
        
        with col4:
            if 'año' in self.data.columns:
                try:
                    años = self.data['año'].nunique()
                    st.metric("Años Analizados", años)
                except:
                    st.metric("Años Analizados", "N/A")
        
        # Mapa de calor de puntos críticos
        st.subheader("🗺️ Mapa de Calor - Puntos Críticos")
        self.create_heatmap()
        
        # KPIs por comuna
        st.subheader("🏘️ KPIs por Comuna")
        self.comuna_analysis()
    
    def create_operational_dashboard(self):
        """Dashboard Operativo - Análisis temporal detallado"""
        st.header("⏰ Dashboard Operativo")
        
        if self.data.empty:
            st.warning("No hay datos para mostrar")
            return
            
        # Análisis temporal
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Accidentes por Hora del Día")
            self.hourly_analysis()
        
        with col2:
            st.subheader("📆 Accidentes por Día de la Semana")
            self.weekly_analysis()
        
        # Análisis por tipo de accidente
        st.subheader("🚗 Distribución por Tipo de Accidente")
        self.accident_type_analysis()
        
        # Evolución temporal
        st.subheader("📈 Evolución Temporal")
        self.temporal_evolution()
    
    def create_preventive_dashboard(self):
        """Dashboard Preventivo - Para acciones proactivas"""
        st.header("🚨 Dashboard Preventivo")
        
        if self.data.empty:
            st.warning("No hay datos para mostrar")
            return
            
        # Zonas de alto riesgo
        st.subheader("📍 Top 10 Zonas de Alto Riesgo")
        self.high_risk_zones()
        
        # Horarios críticos
        st.subheader("🕒 Horarios Críticos por Tipo de Accidente")
        self.critical_hours_analysis()
        
        # Recomendaciones preventivas
        st.subheader("💡 Recomendaciones Preventivas")
        self.preventive_recommendations()
    
    def create_heatmap(self):
        """Crea mapa de calor de puntos críticos"""
        if self.data.empty or 'lat' not in self.data.columns or 'lon' not in self.data.columns:
            st.warning("Datos insuficientes para mapa de calor")
            return
            
        try:
            # Centrar en Medellín
            medellin_center = [6.2442, -75.5812]
            m = folium.Map(location=medellin_center, zoom_start=12)
            
            # Datos para heatmap
            heat_data = [[row['lat'], row['lon']] for _, row in self.data.iterrows() 
                        if not pd.isna(row['lat']) and not pd.isna(row['lon'])]
            
            if heat_data:
                HeatMap(heat_data, radius=15, blur=10, gradient={
                    0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 
                    0.8: 'yellow', 1.0: 'red'
                }).add_to(m)
                
                st_folium(m, width=700, height=400)
        except Exception as e:
            st.error(f"Error creando mapa de calor: {e}")
    
    def comuna_analysis(self):
        """Análisis de KPIs por comuna"""
        if 'comuna' not in self.data.columns:
            st.warning("No hay datos de comuna para análisis")
            return
            
        try:
            comuna_stats = self.data.groupby('comuna').agg({
                'lat': 'count',  # Total accidentes
                'gravedad': lambda x: (x.str.contains('Grave|Fatal', case=False, na=False)).sum()
            }).rename(columns={'lat': 'total_accidentes', 'gravedad': 'accidentes_graves'})
            
            comuna_stats['tasa_gravedad'] = (comuna_stats['accidentes_graves'] / comuna_stats['total_accidentes'] * 100).round(1)
            comuna_stats = comuna_stats.sort_values('total_accidentes', ascending=False)
            
            # Mostrar tabla
            st.dataframe(comuna_stats.head(10), use_container_width=True)
            
            # Gráfico de barras
            top_comunas = comuna_stats.head(10)
            if not top_comunas.empty:
                fig = px.bar(top_comunas, 
                            x=top_comunas.index,
                            y='total_accidentes',
                            title="Top 10 Comunas con Más Accidentes")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error en análisis por comuna: {e}")
    
    def hourly_analysis(self):
        """Análisis de accidentes por hora del día - CORREGIDO"""
        if 'hora' not in self.data.columns:
            st.warning("No hay datos horarios para análisis")
            return
            
        try:
            # Filtrar horas válidas
            horas_validas = self.data['hora'].dropna()
            if horas_validas.empty:
                st.warning("No hay datos de hora válidos")
                return
                
            hourly_counts = horas_validas.value_counts().sort_index()
            
            if hourly_counts.empty:
                st.warning("No hay datos para análisis horario")
                return
            
            # Crear DataFrame para Plotly
            hourly_df = pd.DataFrame({
                'hora': hourly_counts.index,
                'accidentes': hourly_counts.values
            })
            
            fig = px.line(hourly_df, x='hora', y='accidentes',
                         labels={'hora': 'Hora del Día', 'accidentes': 'Número de Accidentes'},
                         title="Distribución Horaria de Accidentes")
            fig.update_traces(line=dict(color='red', width=3))
            st.plotly_chart(fig, use_container_width=True)
            
            # Identificar horas pico - CON VALIDACIÓN
            if not hourly_counts.empty:
                hora_pico = hourly_counts.idxmax()
                st.info(f"**Hora pico de accidentes:** {hora_pico}:00 - {hora_pico+1}:00")
            else:
                st.info("No se pudo identificar hora pico")
                
        except Exception as e:
            st.error(f"Error en análisis horario: {e}")
    
    def weekly_analysis(self):
        """Análisis de accidentes por día de la semana"""
        if 'dia_semana' not in self.data.columns:
            st.warning("No hay datos de días de semana para análisis")
            return
            
        try:
            # Filtrar días válidos
            dias_validos = self.data['dia_semana'].dropna()
            if dias_validos.empty:
                st.warning("No hay datos de días válidos")
                return
            
            # Ordenar días de la semana
            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dias_esp = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            
            weekly_counts = dias_validos.value_counts()
            weekly_counts = weekly_counts.reindex(dias_orden, fill_value=0)
            
            # Crear DataFrame para Plotly
            weekly_df = pd.DataFrame({
                'dia': dias_esp,
                'accidentes': weekly_counts.values
            })
            
            fig = px.bar(weekly_df, x='dia', y='accidentes',
                        labels={'dia': 'Día de la Semana', 'accidentes': 'Número de Accidentes'},
                        title="Accidentes por Día de la Semana")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error en análisis semanal: {e}")
    
    def accident_type_analysis(self):
        """Análisis por tipo de accidente"""
        if 'clase' not in self.data.columns:
            st.warning("No hay datos de tipo de accidente para análisis")
            return
            
        try:
            tipos_validos = self.data['clase'].dropna()
            if tipos_validos.empty:
                st.warning("No hay datos de tipos de accidente válidos")
                return
                
            tipo_counts = tipos_validos.value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de torta
                tipo_df = pd.DataFrame({
                    'tipo': tipo_counts.index,
                    'cantidad': tipo_counts.values
                })
                if not tipo_df.empty:
                    fig_pie = px.pie(tipo_df, values='cantidad', names='tipo',
                                   title="Distribución por Tipo de Accidente")
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Gráfico de barras horizontal
                if not tipo_df.empty:
                    fig_bar = px.bar(tipo_df, y='tipo', x='cantidad',
                                   orientation='h',
                                   title="Tipos de Accidentes Más Frecuentes",
                                   labels={'tipo': 'Tipo de Accidente', 'cantidad': 'Cantidad'})
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error en análisis de tipos: {e}")
    
    def temporal_evolution(self):
        """Evolución temporal de accidentes"""
        if 'mes' not in self.data.columns or 'año' not in self.data.columns:
            st.warning("No hay datos temporales para análisis de evolución")
            return
            
        try:
            # Filtrar datos temporales válidos
            temp_data = self.data[['año', 'mes']].dropna()
            if temp_data.empty:
                st.warning("No hay datos temporales válidos")
                return
                
            # Agrupar por mes y año
            temporal_data = temp_data.groupby(['año', 'mes']).size().reset_index(name='accidentes')
            temporal_data['periodo'] = temporal_data['año'].astype(str) + '-' + temporal_data['mes'].astype(str).str.zfill(2)
            
            if not temporal_data.empty:
                fig = px.line(temporal_data, x='periodo', y='accidentes',
                             title="Evolución Mensual de Accidentes",
                             labels={'periodo': 'Periodo', 'accidentes': 'Número de Accidentes'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos para mostrar evolución temporal")
                
        except Exception as e:
            st.error(f"Error en análisis temporal: {e}")
    
    def high_risk_zones(self):
        """Identifica las 10 zonas de mayor riesgo"""
        if self.data.empty or 'lat' not in self.data.columns or 'lon' not in self.data.columns:
            st.warning("Datos insuficientes para análisis de zonas de riesgo")
            return
            
        try:
            # Filtrar coordenadas válidas
            coords_validas = self.data[['lat', 'lon']].dropna()
            if coords_validas.empty:
                st.warning("No hay coordenadas válidas para análisis")
                return
            
            # Agrupar por ubicaciones similares (agrupamiento simple)
            coords_validas['lat_round'] = coords_validas['lat'].round(3)
            coords_validas['lon_round'] = coords_validas['lon'].round(3)
            
            zone_risk = coords_validas.groupby(['lat_round', 'lon_round']).size().reset_index(name='frecuencia')
            zone_risk = zone_risk.sort_values('frecuencia', ascending=False).head(10)
            
            if not zone_risk.empty:
                # Mostrar tabla
                st.dataframe(zone_risk, use_container_width=True)
                
                # Mapa con marcadores de zonas de riesgo
                medellin_center = [6.2442, -75.5812]
                m = folium.Map(location=medellin_center, zoom_start=12)
                
                for _, zone in zone_risk.iterrows():
                    folium.CircleMarker(
                        [zone['lat_round'], zone['lon_round']],
                        radius=min(zone['frecuencia']/2, 20),
                        popup=f"Accidentes: {zone['frecuencia']}",
                        color='red',
                        fill=True,
                        fillColor='red'
                    ).add_to(m)
                
                st_folium(m, width=700, height=400)
            else:
                st.warning("No se encontraron zonas de riesgo")
                
        except Exception as e:
            st.error(f"Error en análisis de zonas de riesgo: {e}")
    
    def critical_hours_analysis(self):
        """Análisis de horarios críticos por tipo de accidente"""
        if 'hora' not in self.data.columns or 'clase' not in self.data.columns:
            st.warning("Datos insuficientes para análisis de horarios críticos")
            return
            
        try:
            # Filtrar datos válidos
            datos_validos = self.data[['hora', 'clase']].dropna()
            if datos_validos.empty:
                st.warning("No hay datos válidos para análisis de horarios críticos")
                return
            
            # Crear tabla pivote de horas vs tipos de accidente
            hour_type_matrix = pd.crosstab(datos_validos['hora'], datos_validos['clase'])
            
            if not hour_type_matrix.empty:
                # Heatmap de horarios críticos
                fig = px.imshow(hour_type_matrix,
                               labels=dict(x="Tipo de Accidente", y="Hora del Día", color="Frecuencia"),
                               title="Frecuencia de Accidentes por Hora y Tipo",
                               aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos para el heatmap de horarios críticos")
                
        except Exception as e:
            st.error(f"Error en análisis de horarios críticos: {e}")
    
    def preventive_recommendations(self):
        """Genera recomendaciones preventivas basadas en los datos"""
        insights = []
        
        try:
            # Análisis de hora pico
            if 'hora' in self.data.columns:
                horas_validas = self.data['hora'].dropna()
                if not horas_validas.empty:
                    hora_pico = horas_validas.mode()[0] if not horas_validas.mode().empty else "N/A"
                    insights.append(f"**Reforzar vigilancia entre {hora_pico}:00 y {hora_pico+1}:00** - Hora pico identificada")
            
            # Análisis de días críticos
            if 'dia_semana' in self.data.columns:
                dias_validos = self.data['dia_semana'].dropna()
                if not dias_validos.empty:
                    dia_critico = dias_validos.mode()[0] if not dias_validos.mode().empty else "N/A"
                    dias_map = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                               'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
                    dia_esp = dias_map.get(dia_critico, dia_critico)
                    insights.append(f"**Atención especial los {dia_esp}** - Día con mayor siniestralidad")
            
            # Análisis de tipos frecuentes
            if 'clase' in self.data.columns:
                clases_validas = self.data['clase'].dropna()
                if not clases_validas.empty:
                    tipo_frecuente = clases_validas.mode()[0] if not clases_validas.mode().empty else "N/A"
                    insights.append(f"**Campañas preventivas para {tipo_frecuente}** - Tipo de accidente más frecuente")
            
            # Análisis de gravedad
            if 'gravedad' in self.data.columns:
                graves_mask = self.data['gravedad'].str.contains('Grave|Fatal', case=False, na=False)
                tasa_graves = (graves_mask.sum() / len(self.data)) * 100 if len(self.data) > 0 else 0
                insights.append(f"**{tasa_graves:.1f}% de accidentes son graves o fatales** - Enfoque en prevención de lesiones")
            
            # Mostrar recomendaciones
            if insights:
                for i, insight in enumerate(insights, 1):
                    st.write(f"{i}. {insight}")
            else:
                st.info("No hay suficientes datos para generar recomendaciones específicas")
                
        except Exception as e:
            st.error(f"Error generando recomendaciones: {e}")

def main():
    st.set_page_config(page_title="Insights de Accidentes", layout="wide")
    st.title("🔍 Insights Estratégicos - Accidentes de Tránsito Medellín")
    
    # Cargar datos desde session_state (asumiendo que vienen del integrador principal)
    if 'complete_data' in st.session_state and not st.session_state.complete_data.empty:
        data = st.session_state.complete_data
        insights = AccidentInsights(data)
        
        # Selector de dashboard
        dashboard_type = st.sidebar.selectbox(
            "Seleccionar Dashboard:",
            ["Ejecutivo", "Operativo", "Preventivo"]
        )
        
        if dashboard_type == "Ejecutivo":
            insights.create_executive_dashboard()
        elif dashboard_type == "Operativo":
            insights.create_operational_dashboard()
        else:
            insights.create_preventive_dashboard()
            
    else:
        st.warning("""
        **No se encontraron datos para análisis**
        
        Para usar los insights:
        1. Ve a la página del Integrador
        2. Carga y procesa los datos
        3. Regresa a esta página
        """)

if __name__ == "__main__":
    main()
