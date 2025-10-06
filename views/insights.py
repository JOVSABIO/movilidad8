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
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================================

@dataclass
class InsightsConfig:
    """Configuración centralizada para insights"""
    medellin_center: List[float] = None
    map_zoom: int = 12
    heatmap_radius: int = 15
    heatmap_blur: int = 10
    top_n_zones: int = 10
    coord_precision: int = 3
    
    def __post_init__(self):
        if self.medellin_center is None:
            self.medellin_center = [6.2442, -75.5812]

CONFIG = InsightsConfig()

# ============================================================================
# FUNCIONES UTILITARIAS
# ============================================================================

def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """
    Busca una columna por múltiples nombres posibles (case-insensitive).
    
    Args:
        df: DataFrame donde buscar
        possible_names: Lista de nombres posibles
    
    Returns:
        str: Nombre de la columna encontrada o None
    """
    cols_upper = {col.upper(): col for col in df.columns}
    for name in possible_names:
        if name.upper() in cols_upper:
            logger.info(f"Columna encontrada: {cols_upper[name.upper()]}")
            return cols_upper[name.upper()]
    
    logger.warning(f"No se encontró columna entre: {possible_names}")
    return None


def validate_data_not_empty(df: pd.DataFrame, column: str) -> bool:
    """
    Valida que una columna tenga datos válidos.
    
    Args:
        df: DataFrame a validar
        column: Nombre de la columna
    
    Returns:
        bool: True si tiene datos válidos
    """
    if column not in df.columns:
        return False
    
    valid_data = df[column].dropna()
    return len(valid_data) > 0


@st.cache_data
def prepare_temporal_data(_df: pd.DataFrame, fecha_col: str) -> pd.DataFrame:
    """
    Prepara datos temporales con caché para mejor performance.
    
    Args:
        _df: DataFrame (con _ para evitar hashing)
        fecha_col: Nombre de la columna de fecha
    
    Returns:
        DataFrame con columnas temporales procesadas
    """
    df_temp = _df.copy()
    
    try:
        df_temp['fecha_dt'] = pd.to_datetime(df_temp[fecha_col], errors='coerce')
        df_temp['hora'] = df_temp['fecha_dt'].dt.hour
        df_temp['dia_semana'] = df_temp['fecha_dt'].dt.day_name()
        df_temp['mes'] = df_temp['fecha_dt'].dt.month
        df_temp['año'] = df_temp['fecha_dt'].dt.year
        
        # Eliminar filas con fechas inválidas
        df_temp = df_temp.dropna(subset=['fecha_dt'])
        
        logger.info(f"Datos temporales preparados: {len(df_temp)} registros")
        return df_temp
        
    except Exception as e:
        logger.error(f"Error procesando fechas: {e}", exc_info=True)
        st.warning(f"⚠️ Error procesando fechas: {e}")
        return df_temp


# ============================================================================
# CLASE PRINCIPAL: AccidentInsights
# ============================================================================

class AccidentInsights:
    """Clase para análisis y visualización de insights de accidentes"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Inicializa la clase con datos de accidentes.
        
        Args:
            data: DataFrame con datos de accidentes
        """
        self.data = data.copy() if not data.empty else pd.DataFrame()
        self.setup_insights()
    
    def setup_insights(self):
        """Prepara los datos para el análisis"""
        if self.data.empty:
            logger.warning("DataFrame vacío en setup_insights")
            return
        
        # Buscar y preparar columna de fecha
        fecha_col = find_column(self.data, ['fecha', 'FECHA', 'fecha_accidente', 'FECHA_ACCIDENTE'])
        
        if fecha_col:
            self.data = prepare_temporal_data(self.data, fecha_col)
            logger.info("Datos preparados exitosamente")
        else:
            logger.warning("No se encontró columna de fecha")
    
    # ========================================================================
    # DASHBOARD EJECUTIVO
    # ========================================================================
    
    def create_executive_dashboard(self):
        """Dashboard Ejecutivo - Métricas clave y visión general"""
        st.header("📊 Dashboard Ejecutivo")
        
        if self.data.empty:
            st.warning("⚠️ No hay datos para mostrar")
            return
        
        # Métricas principales
        self._show_executive_metrics()
        
        # Mapa de calor de puntos críticos
        st.subheader("🗺️ Mapa de Calor - Puntos Críticos")
        self.create_heatmap()
        
        # KPIs por comuna
        st.subheader("🏙️ KPIs por Comuna")
        self.comuna_analysis()
    
    def _show_executive_metrics(self):
        """Muestra métricas ejecutivas principales"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_accidentes = len(self.data)
            st.metric("Total Accidentes", f"{total_accidentes:,}")
        
        with col2:
            gravedad_col = find_column(self.data, ['gravedad', 'GRAVEDAD', 'severidad'])
            if gravedad_col and validate_data_not_empty(self.data, gravedad_col):
                try:
                    graves = self.data[gravedad_col].str.contains(
                        'Grave|Fatal|grave|fatal', 
                        case=False, 
                        na=False
                    ).sum()
                    st.metric("Accidentes Graves", f"{graves:,}")
                    
                    # Porcentaje
                    porcentaje = (graves / len(self.data) * 100) if len(self.data) > 0 else 0
                    st.caption(f"({porcentaje:.1f}% del total)")
                except Exception as e:
                    logger.error(f"Error calculando accidentes graves: {e}")
                    st.metric("Accidentes Graves", "N/A")
            else:
                st.metric("Accidentes Graves", "N/A")
        
        with col3:
            comuna_col = find_column(self.data, ['comuna', 'COMUNA'])
            if comuna_col and validate_data_not_empty(self.data, comuna_col):
                try:
                    comunas_afectadas = self.data[comuna_col].nunique()
                    st.metric("Comunas Afectadas", comunas_afectadas)
                except:
                    st.metric("Comunas Afectadas", "N/A")
            else:
                st.metric("Comunas Afectadas", "N/A")
        
        with col4:
            if 'año' in self.data.columns and validate_data_not_empty(self.data, 'año'):
                try:
                    años = self.data['año'].nunique()
                    st.metric("Años Analizados", años)
                except:
                    st.metric("Años Analizados", "N/A")
            else:
                st.metric("Años Analizados", "N/A")
    
    # ========================================================================
    # DASHBOARD OPERATIVO
    # ========================================================================
    
    def create_operational_dashboard(self):
        """Dashboard Operativo - Análisis temporal detallado"""
        st.header("⏰ Dashboard Operativo")
        
        if self.data.empty:
            st.warning("⚠️ No hay datos para mostrar")
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
    
    # ========================================================================
    # DASHBOARD PREVENTIVO
    # ========================================================================
    
    def create_preventive_dashboard(self):
        """Dashboard Preventivo - Para acciones proactivas"""
        st.header("🚨 Dashboard Preventivo")
        
        if self.data.empty:
            st.warning("⚠️ No hay datos para mostrar")
            return
        
        # Zonas de alto riesgo
        st.subheader("🔍 Top 10 Zonas de Alto Riesgo")
        self.high_risk_zones()
        
        # Horarios críticos
        st.subheader("🕐 Horarios Críticos por Tipo de Accidente")
        self.critical_hours_analysis()
        
        # Recomendaciones preventivas
        st.subheader("💡 Recomendaciones Preventivas")
        self.preventive_recommendations()
    
    # ========================================================================
    # ANÁLISIS ESPECÍFICOS
    # ========================================================================
    
    def create_heatmap(self):
        """Crea mapa de calor de puntos críticos"""
        lat_col = find_column(self.data, ['lat', 'LAT', 'latitud', 'LATITUD'])
        lon_col = find_column(self.data, ['lon', 'LON', 'longitud', 'LONGITUD'])
        
        if not lat_col or not lon_col:
            st.warning("⚠️ No se encontraron columnas de coordenadas")
            return
        
        if not validate_data_not_empty(self.data, lat_col) or not validate_data_not_empty(self.data, lon_col):
            st.warning("⚠️ Datos de coordenadas insuficientes para mapa")
            return
        
        try:
            # Crear mapa
            m = folium.Map(location=CONFIG.medellin_center, zoom_start=CONFIG.map_zoom)
            
            # Datos para heatmap - filtrar coordenadas válidas
            heat_data = [
                [row[lat_col], row[lon_col]] 
                for _, row in self.data.iterrows() 
                if pd.notna(row[lat_col]) and pd.notna(row[lon_col])
                and -90 <= row[lat_col] <= 90  # Validar rango de latitud
                and -180 <= row[lon_col] <= 180  # Validar rango de longitud
            ]
            
            if heat_data:
                HeatMap(
                    heat_data,
                    radius=CONFIG.heatmap_radius,
                    blur=CONFIG.heatmap_blur,
                    gradient={
                        0.4: 'blue',      # Azul para baja concentración
                        0.6: 'cyan',      # Cyan para concentración baja
                        0.7: 'lime',      # Lima para concentración media
                        0.8: 'yellow',    # Amarillo para concentración alta
                        1.0: 'red'        # Rojo para alta concentración
                    }
                ).add_to(m)
                
                st_folium(m, width=700, height=400)
                st.caption(f"📍 {len(heat_data)} puntos visualizados en el mapa")
            else:
                st.warning("⚠️ No hay datos válidos para mostrar en el mapa")
                
        except Exception as e:
            logger.error(f"Error creando mapa de calor: {e}", exc_info=True)
            st.error(f"❌ Error creando mapa de calor: {e}")
    
    def comuna_analysis(self):
        """Análisis de KPIs por comuna"""
        comuna_col = find_column(self.data, ['comuna', 'COMUNA'])
        
        if not comuna_col:
            st.warning("⚠️ No se encontró columna de comuna")
            return
        
        if not validate_data_not_empty(self.data, comuna_col):
            st.warning("⚠️ No hay datos de comuna válidos")
            return
        
        try:
            gravedad_col = find_column(self.data, ['gravedad', 'GRAVEDAD', 'severidad'])
            
            # Agrupar por comuna
            if gravedad_col and validate_data_not_empty(self.data, gravedad_col):
                comuna_stats = self.data.groupby(comuna_col).agg({
                    comuna_col: 'count',  # Total accidentes
                }).rename(columns={comuna_col: 'total_accidentes'})
                
                # Calcular accidentes graves
                graves_por_comuna = self.data.groupby(comuna_col)[gravedad_col].apply(
                    lambda x: x.str.contains('Grave|Fatal|grave|fatal', case=False, na=False).sum()
                )
                
                comuna_stats['accidentes_graves'] = graves_por_comuna
                comuna_stats['tasa_gravedad'] = (
                    comuna_stats['accidentes_graves'] / comuna_stats['total_accidentes'] * 100
                ).round(1)
            else:
                # Sin datos de gravedad
                comuna_stats = self.data.groupby(comuna_col).size().reset_index(name='total_accidentes')
                comuna_stats = comuna_stats.set_index(comuna_col)
            
            comuna_stats = comuna_stats.sort_values('total_accidentes', ascending=False)
            
            # Mostrar tabla
            st.dataframe(comuna_stats.head(CONFIG.top_n_zones), use_container_width=True)
            
            # Gráfico de barras
            top_comunas = comuna_stats.head(CONFIG.top_n_zones)
            if not top_comunas.empty:
                fig = px.bar(
                    top_comunas.reset_index(),
                    x=comuna_col,
                    y='total_accidentes',
                    title=f"Top {CONFIG.top_n_zones} Comunas con Más Accidentes",
                    labels={comuna_col: 'Comuna', 'total_accidentes': 'Total Accidentes'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Botón de descarga
                csv = comuna_stats.to_csv()
                st.download_button(
                    label="📥 Descargar Análisis por Comuna",
                    data=csv,
                    file_name="analisis_por_comuna.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            logger.error(f"Error en análisis por comuna: {e}", exc_info=True)
            st.error(f"❌ Error en análisis por comuna: {e}")
    
    def hourly_analysis(self):
        """Análisis de accidentes por hora del día"""
        if not validate_data_not_empty(self.data, 'hora'):
            st.warning("⚠️ No hay datos horarios disponibles")
            return
        
        try:
            # Filtrar horas válidas
            horas_validas = self.data['hora'].dropna()
            horas_validas = horas_validas[(horas_validas >= 0) & (horas_validas <= 23)]
            
            if horas_validas.empty:
                st.warning("⚠️ No hay datos de hora válidos")
                return
            
            hourly_counts = horas_validas.value_counts().sort_index()
            
            if hourly_counts.empty:
                st.warning("⚠️ No hay datos para análisis horario")
                return
            
            # Crear DataFrame para Plotly
            hourly_df = pd.DataFrame({
                'hora': hourly_counts.index,
                'accidentes': hourly_counts.values
            })
            
            fig = px.line(
                hourly_df,
                x='hora',
                y='accidentes',
                labels={'hora': 'Hora del Día', 'accidentes': 'Número de Accidentes'},
                title="Distribución Horaria de Accidentes"
            )
            fig.update_traces(line=dict(color='red', width=3), mode='lines+markers')
            fig.update_xaxes(dtick=1)  # Corregido: update_xaxes en lugar de update_xaxis
            st.plotly_chart(fig, use_container_width=True)
            
            # Identificar horas pico
            if not hourly_counts.empty:
                hora_pico = hourly_counts.idxmax()
                max_accidentes = hourly_counts.max()
                st.info(f"📊 **Hora pico de accidentes:** {int(hora_pico)}:00 - {int(hora_pico)+1}:00 ({int(max_accidentes)} accidentes)")
            
        except Exception as e:
            logger.error(f"Error en análisis horario: {e}", exc_info=True)
            st.error(f"❌ Error en análisis horario: {e}")
    
    def weekly_analysis(self):
        """Análisis de accidentes por día de la semana"""
        if not validate_data_not_empty(self.data, 'dia_semana'):
            st.warning("⚠️ No hay datos de días de semana disponibles")
            return
        
        try:
            # Filtrar días válidos
            dias_validos = self.data['dia_semana'].dropna()
            
            if dias_validos.empty:
                st.warning("⚠️ No hay datos de días válidos")
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
            
            fig = px.bar(
                weekly_df,
                x='dia',
                y='accidentes',
                labels={'dia': 'Día de la Semana', 'accidentes': 'Número de Accidentes'},
                title="Accidentes por Día de la Semana",
                color='accidentes',
                color_continuous_scale='tempo'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Día con más accidentes
            dia_max_idx = weekly_df['accidentes'].idxmax()
            dia_max = weekly_df.loc[dia_max_idx, 'dia']
            max_accidentes = weekly_df.loc[dia_max_idx, 'accidentes']
            st.info(f"📊 **Día con más accidentes:** {dia_max} ({int(max_accidentes)} accidentes)")
            
        except Exception as e:
            logger.error(f"Error en análisis semanal: {e}", exc_info=True)
            st.error(f"❌ Error en análisis semanal: {e}")
    
    def accident_type_analysis(self):
        """Análisis por tipo de accidente"""
        clase_col = find_column(self.data, ['clase', 'CLASE', 'tipo_accidente', 'TIPO_ACCIDENTE'])
        
        if not clase_col:
            st.warning("⚠️ No se encontró columna de tipo de accidente")
            return
        
        if not validate_data_not_empty(self.data, clase_col):
            st.warning("⚠️ No hay datos de tipos de accidente válidos")
            return
        
        try:
            tipos_validos = self.data[clase_col].dropna()
            tipo_counts = tipos_validos.value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de torta
                tipo_df = pd.DataFrame({
                    'tipo': tipo_counts.index,
                    'cantidad': tipo_counts.values
                })
                
                if not tipo_df.empty:
                    fig_pie = px.pie(
                        tipo_df.head(10),  # Top 10 para mejor visualización
                        values='cantidad',
                        names='tipo',
                        title="Distribución por Tipo de Accidente (Top 10)"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Gráfico de barras horizontal
                if not tipo_df.empty:
                    fig_bar = px.bar(
                        tipo_df.head(10),
                        y='tipo',
                        x='cantidad',
                        orientation='h',
                        title="Tipos de Accidentes Más Frecuentes",
                        labels={'tipo': 'Tipo de Accidente', 'cantidad': 'Cantidad'},
                        color='cantidad',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # Mostrar tabla completa
            with st.expander("📋 Ver Todos los Tipos de Accidente"):
                st.dataframe(tipo_df, use_container_width=True)
                
                # Botón de descarga
                csv = tipo_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Análisis de Tipos",
                    data=csv,
                    file_name="tipos_accidente.csv",
                    mime="text/csv"
                )
            
        except Exception as e:
            logger.error(f"Error en análisis de tipos: {e}", exc_info=True)
            st.error(f"❌ Error en análisis de tipos: {e}")
    
    def temporal_evolution(self):
        """Evolución temporal de accidentes"""
        if not validate_data_not_empty(self.data, 'mes') or not validate_data_not_empty(self.data, 'año'):
            st.warning("⚠️ No hay datos temporales suficientes para evolución")
            return
        
        try:
            # Filtrar datos temporales válidos
            temp_data = self.data[['año', 'mes']].dropna()
            
            if temp_data.empty:
                st.warning("⚠️ No hay datos temporales válidos")
                return
            
            # Agrupar por mes y año
            temporal_data = temp_data.groupby(['año', 'mes']).size().reset_index(name='accidentes')
            temporal_data['periodo'] = temporal_data['año'].astype(int).astype(str) + '-' + temporal_data['mes'].astype(int).astype(str).str.zfill(2)
            temporal_data = temporal_data.sort_values(['año', 'mes'])
            
            if not temporal_data.empty:
                fig = px.line(
                    temporal_data,
                    x='periodo',
                    y='accidentes',
                    title="Evolución Mensual de Accidentes",
                    labels={'periodo': 'Periodo', 'accidentes': 'Número de Accidentes'},
                    markers=True
                )
                fig.update_traces(line=dict(width=3))
                fig.update_xaxes(tickangle=-45)  # Corregido: update_xaxes en lugar de update_xaxis
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas de tendencia
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    promedio = temporal_data['accidentes'].mean()
                    st.metric("Promedio Mensual", f"{promedio:.1f}")
                
                with col2:
                    maximo = temporal_data['accidentes'].max()
                    periodo_max = temporal_data.loc[temporal_data['accidentes'].idxmax(), 'periodo']
                    st.metric("Máximo Mensual", f"{int(maximo)}")
                    st.caption(f"en {periodo_max}")
                
                with col3:
                    minimo = temporal_data['accidentes'].min()
                    periodo_min = temporal_data.loc[temporal_data['accidentes'].idxmin(), 'periodo']
                    st.metric("Mínimo Mensual", f"{int(minimo)}")
                    st.caption(f"en {periodo_min}")
            else:
                st.warning("⚠️ No hay datos para mostrar evolución temporal")
            
        except Exception as e:
            logger.error(f"Error en análisis temporal: {e}", exc_info=True)
            st.error(f"❌ Error en análisis temporal: {e}")
    
    def high_risk_zones(self):
        """Identifica las zonas de mayor riesgo"""
        lat_col = find_column(self.data, ['lat', 'LAT', 'latitud', 'LATITUD'])
        lon_col = find_column(self.data, ['lon', 'LON', 'longitud', 'LONGITUD'])
        
        if not lat_col or not lon_col:
            st.warning("⚠️ No se encontraron columnas de coordenadas")
            return
        
        if not validate_data_not_empty(self.data, lat_col) or not validate_data_not_empty(self.data, lon_col):
            st.warning("⚠️ Datos de coordenadas insuficientes")
            return
        
        try:
            # Filtrar coordenadas válidas
            coords_validas = self.data[[lat_col, lon_col]].dropna()
            coords_validas = coords_validas[
                (coords_validas[lat_col].between(-90, 90)) &
                (coords_validas[lon_col].between(-180, 180))
            ]
            
            if coords_validas.empty:
                st.warning("⚠️ No hay coordenadas válidas para análisis")
                return
            
            # Agrupar por ubicaciones similares
            coords_validas['lat_round'] = coords_validas[lat_col].round(CONFIG.coord_precision)
            coords_validas['lon_round'] = coords_validas[lon_col].round(CONFIG.coord_precision)
            
            zone_risk = coords_validas.groupby(
                ['lat_round', 'lon_round']
            ).size().reset_index(name='frecuencia')
            zone_risk = zone_risk.sort_values('frecuencia', ascending=False).head(CONFIG.top_n_zones)
            
            if not zone_risk.empty:
                # Mostrar tabla
                st.dataframe(
                    zone_risk.rename(columns={
                        'lat_round': 'Latitud',
                        'lon_round': 'Longitud',
                        'frecuencia': 'Accidentes'
                    }),
                    use_container_width=True
                )
                
                # Mapa con marcadores de zonas de riesgo
                m = folium.Map(location=CONFIG.medellin_center, zoom_start=CONFIG.map_zoom)
                
                for _, zone in zone_risk.iterrows():
                    folium.CircleMarker(
                        [zone['lat_round'], zone['lon_round']],
                        radius=min(zone['frecuencia']/2, 20),
                        popup=f"Accidentes: {zone['frecuencia']}",
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.6
                    ).add_to(m)
                
                st_folium(m, width=700, height=400)
                
                # Botón de descarga
                csv = zone_risk.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Zonas de Riesgo",
                    data=csv,
                    file_name="zonas_alto_riesgo.csv",
                    mime="text/csv"
                )
            else:
                st.warning("⚠️ No se encontraron zonas de riesgo")
            
        except Exception as e:
            logger.error(f"Error en análisis de zonas de riesgo: {e}", exc_info=True)
            st.error(f"❌ Error en análisis de zonas de riesgo: {e}")
    
    def critical_hours_analysis(self):
        """Análisis de horarios críticos por tipo de accidente"""
        clase_col = find_column(self.data, ['clase', 'CLASE', 'tipo_accidente', 'TIPO_ACCIDENTE'])
        
        if not validate_data_not_empty(self.data, 'hora') or not clase_col:
            st.warning("⚠️ Datos insuficientes para análisis de horarios críticos")
            return
        
        if not validate_data_not_empty(self.data, clase_col):
            st.warning("⚠️ No hay datos de tipos de accidente válidos")
            return
        
        try:
            # Filtrar datos válidos
            datos_validos = self.data[['hora', clase_col]].dropna()
            datos_validos = datos_validos[
                (datos_validos['hora'] >= 0) & (datos_validos['hora'] <= 23)
            ]
            
            if datos_validos.empty:
                st.warning("⚠️ No hay datos válidos para análisis de horarios críticos")
                return
            
            # Crear tabla pivote de horas vs tipos de accidente
            hour_type_matrix = pd.crosstab(datos_validos['hora'], datos_validos[clase_col])
            
            if not hour_type_matrix.empty:
                # Limitar a top tipos para mejor visualización
                top_types = hour_type_matrix.sum().nlargest(10).index
                hour_type_matrix_filtered = hour_type_matrix[top_types]
                
                # Heatmap de horarios críticos
                fig = px.imshow(
                    hour_type_matrix_filtered.T,
                    labels=dict(x="Hora del Día", y="Tipo de Accidente", color="Frecuencia"),
                    title="Frecuencia de Accidentes por Hora y Tipo (Top 10 Tipos)",
                    aspect="auto",
                    color_continuous_scale='tempo'
                )
                fig.update_xaxes(dtick=1)  # Corregido: update_xaxes en lugar de update_xaxis
                st.plotly_chart(fig, use_container_width=True)
                
                # Identificar hora-tipo más crítico
                max_val = hour_type_matrix_filtered.max().max()
                max_loc = hour_type_matrix_filtered.stack().idxmax()
                hora_critica, tipo_critico = max_loc
                st.info(f"📊 **Combinación más crítica:** {tipo_critico} a las {int(hora_critica)}:00 ({int(max_val)} accidentes)")
            else:
                st.warning("⚠️ No hay datos para el heatmap de horarios críticos")
            
        except Exception as e:
            logger.error(f"Error en análisis de horarios críticos: {e}", exc_info=True)
            st.error(f"❌ Error en análisis de horarios críticos: {e}")
    
    def preventive_recommendations(self):
        """Genera recomendaciones preventivas basadas en los datos"""
        insights = []
        
        try:
            # Análisis de hora pico
            if validate_data_not_empty(self.data, 'hora'):
                horas_validas = self.data['hora'].dropna()
                horas_validas = horas_validas[(horas_validas >= 0) & (horas_validas <= 23)]
                
                if not horas_validas.empty:
                    hora_pico = int(horas_validas.mode()[0]) if not horas_validas.mode().empty else None
                    if hora_pico is not None:
                        insights.append(
                            f"**Reforzar vigilancia entre {hora_pico}:00 y {hora_pico+1}:00** - Hora pico identificada"
                        )
            
            # Análisis de días críticos
            if validate_data_not_empty(self.data, 'dia_semana'):
                dias_validos = self.data['dia_semana'].dropna()
                
                if not dias_validos.empty:
                    dia_critico = dias_validos.mode()[0] if not dias_validos.mode().empty else None
                    
                    if dia_critico:
                        dias_map = {
                            'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles',
                            'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado',
                            'Sunday': 'Domingo'
                        }
                        dia_esp = dias_map.get(dia_critico, dia_critico)
                        insights.append(
                            f"**Atención especial los {dia_esp}** - Día con mayor siniestralidad"
                        )
            
            # Análisis de tipos frecuentes
            clase_col = find_column(self.data, ['clase', 'CLASE', 'tipo_accidente'])
            if clase_col and validate_data_not_empty(self.data, clase_col):
                clases_validas = self.data[clase_col].dropna()
                
                if not clases_validas.empty:
                    tipo_frecuente = clases_validas.mode()[0] if not clases_validas.mode().empty else None
                    
                    if tipo_frecuente:
                        insights.append(
                            f"**Campañas preventivas para {tipo_frecuente}** - Tipo de accidente más frecuente"
                        )
            
            # Análisis de gravedad
            gravedad_col = find_column(self.data, ['gravedad', 'GRAVEDAD', 'severidad'])
            if gravedad_col and validate_data_not_empty(self.data, gravedad_col):
                graves_mask = self.data[gravedad_col].str.contains(
                    'Grave|Fatal|grave|fatal',
                    case=False,
                    na=False
                )
                tasa_graves = (graves_mask.sum() / len(self.data)) * 100 if len(self.data) > 0 else 0
                
                if tasa_graves > 0:
                    insights.append(
                        f"**{tasa_graves:.1f}% de accidentes son graves o fatales** - Enfoque en prevención de lesiones"
                    )
            
            # Análisis de zonas críticas
            comuna_col = find_column(self.data, ['comuna', 'COMUNA'])
            if comuna_col and validate_data_not_empty(self.data, comuna_col):
                comunas_validas = self.data[comuna_col].dropna()
                
                if not comunas_validas.empty:
                    comuna_critica = comunas_validas.mode()[0] if not comunas_validas.mode().empty else None
                    
                    if comuna_critica:
                        n_accidentes = (comunas_validas == comuna_critica).sum()
                        insights.append(
                            f"**Priorizar intervenciones en {comuna_critica}** - {n_accidentes} accidentes registrados"
                        )
            
            # Mostrar recomendaciones
            if insights:
                for i, insight in enumerate(insights, 1):
                    st.write(f"{i}. {insight}")
                
                # Resumen ejecutivo
                st.markdown("---")
                st.write("**📋 Resumen de Acciones Prioritarias:**")
                
                if len(insights) >= 3:
                    st.success("✅ Se identificaron múltiples áreas de intervención")
                elif len(insights) >= 1:
                    st.info("ℹ️ Se identificaron algunas áreas de intervención")
                
                # Exportar recomendaciones
                recommendations_text = "\n".join([f"{i}. {rec}" for i, rec in enumerate(insights, 1)])
                st.download_button(
                    label="📥 Descargar Recomendaciones",
                    data=recommendations_text,
                    file_name="recomendaciones_preventivas.txt",
                    mime="text/plain"
                )
            else:
                st.info("ℹ️ No hay suficientes datos para generar recomendaciones específicas")
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {e}", exc_info=True)
            st.error(f"❌ Error generando recomendaciones: {e}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    st.set_page_config(page_title="Insights de Accidentes", layout="wide")
    st.title("🔍 Insights Estratégicos - Accidentes de Tránsito Medellín")
    
    st.markdown("""
    Esta aplicación proporciona análisis estratégicos de accidentes de tránsito a través de tres dashboards especializados:
    - **Ejecutivo**: Visión general y métricas clave
    - **Operativo**: Análisis temporal y patrones de operación
    - **Preventivo**: Identificación de riesgos y recomendaciones
    """)
    st.markdown("---")
    
    # Cargar datos desde session_state
    if 'complete_data' in st.session_state and not st.session_state.complete_data.empty:
        data = st.session_state.complete_data
        
        # Mostrar información básica del dataset
        with st.sidebar:
            st.header("📊 Información del Dataset")
            st.metric("Total Registros", f"{len(data):,}")
            st.metric("Total Columnas", len(data.columns))
            
            # Completitud
            completeness = 1 - data.isnull().sum().sum() / (len(data) * len(data.columns))
            st.metric("Completitud", f"{completeness:.1%}")
            
            st.markdown("---")
        
        # Inicializar clase de insights
        try:
            insights = AccidentInsights(data)
            
            # Selector de dashboard
            dashboard_type = st.sidebar.selectbox(
                "🎯 Seleccionar Dashboard:",
                ["Ejecutivo", "Operativo", "Preventivo"],
                help="Selecciona el tipo de análisis que deseas visualizar"
            )
            
            # Renderizar dashboard seleccionado
            if dashboard_type == "Ejecutivo":
                insights.create_executive_dashboard()
            elif dashboard_type == "Operativo":
                insights.create_operational_dashboard()
            else:
                insights.create_preventive_dashboard()
            
            logger.info(f"Dashboard {dashboard_type} renderizado exitosamente")
            
        except Exception as e:
            logger.error(f"Error inicializando insights: {e}", exc_info=True)
            st.error(f"❌ Error al procesar los datos: {e}")
            st.info("💡 Verifica que los datos estén en el formato correcto")
    
    else:
        st.warning("""
        ⚠️ **No se encontraron datos para análisis**
        
        Para usar los insights:
        1. Ve a la página del **Integrador**
        2. Carga y procesa los datos
        3. Regresa a esta página para ver los análisis
        """)
        
        # Mostrar información sobre los datos esperados
        with st.expander("ℹ️ Información sobre los datos esperados"):
            st.markdown("""
            **Columnas esperadas (opcionales):**
            - `fecha` o `FECHA`: Fecha del accidente
            - `hora` o `HORA`: Hora del accidente
            - `lat` o `LAT` o `latitud`: Coordenada de latitud
            - `lon` o `LON` o `longitud`: Coordenada de longitud
            - `comuna` o `COMUNA`: Comuna donde ocurrió el accidente
            - `clase` o `CLASE`: Tipo de accidente
            - `gravedad` o `GRAVEDAD`: Gravedad del accidente
            
            **Nota:** La aplicación es flexible y buscará estas columnas automáticamente,
            independientemente de si están en mayúsculas o minúsculas.
            """)


if __name__ == "__main__":
    main()
