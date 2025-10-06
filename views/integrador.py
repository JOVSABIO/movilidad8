import streamlit as st
import pandas as pd
import requests
from io import StringIO

try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import MarkerCluster, HeatMap
    folium_installed = True
except ImportError:
    folium_installed = False

def load_csv_from_drive(drive_link):
    # Carga un archivo CSV desde un enlace de Google Drive
    try:
        if 'id=' in drive_link:
            file_id = drive_link.split('id=')[1].split('&')[0]
        elif '/d/' in drive_link:
            file_id = drive_link.split('/d/')[1].split('/')[0]
        else:
            return None, "Formato de enlace no válido"
        
        download_url = f'https://drive.google.com/uc?id=13f0cf70-OBPbZGlBxRsQGVXNzhbROrgy&export=download'
        response = requests.get(download_url)
        response.raise_for_status()
        
        content = response.text
        try:
            df = pd.read_csv(StringIO(content))
        except Exception as e:
            print(f"Error leyendo CSV con utf-8: {e}")
            content = response.content.decode('latin-1')
            df = pd.read_csv(StringIO(content))
        
        return df, "success"
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def process_data(df):
    # Procesa los datos del DataFrame, extrayendo coordenadas y registros completos
    try:
        coordinates_data = []
        complete_data = []
        skipped_records = 0
        # Optimización: buscar la columna de ubicación una sola vez
        location_col = next((col for col in df.columns if 'LOCATION' in col.upper()), None)
        if location_col is None:
            print("No se encontró columna de ubicación")
            return pd.DataFrame(), pd.DataFrame(), len(df)
        # Procesar solo filas válidas
        for i, row in df.iterrows():
            try:
                location_str = str(row[location_col]).strip('[] ')
                if location_str in ['', 'NaN', 'nan']:
                    skipped_records += 1
                    continue
                parts = location_str.split(',')
                if len(parts) != 2:
                    skipped_records += 1
                    continue
                lon = float(parts[0].strip())
                lat = float(parts[1].strip())
                if not ((6.1 <= lat <= 6.4) and (-75.7 <= lon <= -75.5)):
                    skipped_records += 1
                    continue
                coordinates_data.append({'lat': lat, 'lon': lon})
                record = {
                    'lat': lat,
                    'lon': lon,
                    'clase': row.get('CLASE_ACCIDENTE', row.get('CLASE', '')),
                    'gravedad': row.get('GRAVEDAD_ACCIDENTE', row.get('GRAVEDAD', '')),
                    'barrio': row.get('BARRIO', ''),
                    'comuna': row.get('COMUNA', ''),
                    'año': row.get('AÑO', row.get('YEAR', '')),
                    'fecha': row.get('FECHA', row.get('DATE', '')),
                    'direccion': row.get('DIRECCION', row.get('ADDRESS', ''))
                }
                complete_data.append(record)
            except Exception as e:
                print(f"Error procesando fila {i}: {e}")
                skipped_records += 1
                continue
        return pd.DataFrame(coordinates_data), pd.DataFrame(complete_data), skipped_records
    except Exception as e:
        print(f"Error general en process_data: {e}")
        return pd.DataFrame(), pd.DataFrame(), 0

def create_folium_map(data, map_type='markers', zoom_start=12, center=None):
    """
    Crea un mapa interactivo con Folium usando los datos proporcionados.
    Args:
        data (DataFrame): Datos con columnas 'lat' y 'lon'.
        map_type (str): Tipo de visualización ('markers', 'heatmap', 'circle').
        zoom_start (int): Nivel de zoom inicial.
        center (list): Coordenadas [lat, lon] para centrar el mapa.
    Returns:
        folium.Map: Mapa generado.
    """
    if center is None:
        center = [6.245053207648148, -75.57931461508907]  # Centro de Medellín por defecto
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    if map_type == 'markers':
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in data.iterrows():
            gravedad_lower = str(row.get('gravedad', '')).lower()
            color = 'gray'

            if 'con muertos' in gravedad_lower:
                color = 'black'
            elif 'con heridos' in gravedad_lower:
                color = 'red'
            elif 'solo daños' in gravedad_lower:
                color = 'orange'
            
            popup_content = f"""
            <div style="width: 250px;">
                <h4 style="margin: 0; color: {color};">🚗 Accidente</h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 2px 0;"><b>Tipo:</b> {row.get('clase', 'N/A')}</p>
                <p style="margin: 2px 0;"><b>Gravedad:</b> {row.get('gravedad', 'N/A')}</p>
                <p style="margin: 2px 0;"><b>Fecha:</b> {row.get('fecha', 'N/A')}</p>
                <p style="margin: 2px 0;"><b>Dirección:</b> {row.get('direccion', 'N/A')}</p>
                <p style="margin: 2px 0;"><b>Barrio:</b> {row.get('barrio', 'N/A')}</p>
                <p style="margin: 2px 0;"><b>Comuna:</b> {row.get('comuna', 'N/A')}</p>
            </div>
            """

            folium.Marker(
                [row['lat'], row['lon']],
                popup=folium.Popup(popup_content, max_width=700),
                tooltip=f"{row.get('clase', 'N/A')} - {row.get('gravedad', 'N/A')}",
                icon=folium.Icon(color=color, icon='car', prefix='fa')
            ).add_to(marker_cluster)
    
    elif map_type == 'heatmap':
        heat_data = [[row['lat'], row['lon']] for _, row in data.iterrows()]
        HeatMap(heat_data, radius=15, blur=10).add_to(m)
    
    elif map_type == 'circle':
        for _, row in data.iterrows():
            folium.CircleMarker(
                [row['lat'], row['lon']],
                radius=3,
                popup=f"Lat: {row['lat']:.6f}, Lon: {row['lon']:.6f}",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.6
            ).add_to(m)

    return m

def create_advanced_map(complete_data, center=None):
    """
    Crea un mapa avanzado con información detallada en los popups de cada marcador.
    Args:
        complete_data (DataFrame): Datos completos de accidentes.
        center (list): Coordenadas [lat, lon] para centrar el mapa.
    Returns:
        folium.Map: Mapa avanzado con popups personalizados.
    """
    if center is None:
        center = [6.245053207648148, -75.57931461508907]  # Centro de Medellín por defecto
    
    m = folium.Map(
        location=center,
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    color_map = {
        'con muertos': 'black',  # Negro para "Con Muertos"
        'con heridos': 'red',    # Rojo para "Con Heridos"
        'solo daños': 'orange'   # Naranja para "Solo Daños"
    }
    
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in complete_data.iterrows():
        gravedad_lower = str(row['gravedad']).lower()
        color = 'gray'  # Color por defecto

        if 'con muertos' in gravedad_lower:
            color = 'black'
        elif 'con heridos' in gravedad_lower:
            color = 'red'
        elif 'solo daños' in gravedad_lower:
            color = 'orange'

        popup_content = f"""
        <div style="width: 250px;">
            <h4 style="margin: 0; color: {color};">🚗 Accidente</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 2px 0;"><b>Tipo:</b> {row['clase']}</p>
            <p style="margin: 2px 0;"><b>Gravedad:</b> {row['gravedad']}</p>
            <p style="margin: 2px 0;"><b>Fecha:</b> {row['fecha']}</p>
            <p style="margin: 2px 0;"><b>Dirección:</b> {row['direccion']}</p>
            <p style="margin: 2px 0;"><b>Barrio:</b> {row['barrio']}</p>
            <p style="margin: 2px 0;"><b>Comuna:</b> {row['comuna']}</p>
        </div>
        """
        
        folium.Marker(
            [row['lat'], row['lon']],
            popup=folium.Popup(popup_content, max_width=700),
            tooltip=f"{row['clase']} - {row['gravedad']}",
            icon=folium.Icon(color=color, icon='car', prefix='fa')
        ).add_to(marker_cluster)
    
    # Leyenda eliminada para evitar el recuadro en el mapa
    
    return m

def get_unique_values_safe(series):
    """
    Obtiene valores únicos de una serie de pandas de forma segura, sin ordenar.
    Args:
        series (pd.Series): Serie de pandas.
    Returns:
        list: Lista de valores únicos.
    """
    try:
        unique_vals = series.astype(str).dropna().unique()
        return list(unique_vals)
    except Exception as e:
        print(f"Error obteniendo valores únicos: {e}")
        return []

def create_metric_card(icon, value, label, trend=None, card_type="default"):
    """
    Crea una tarjeta de métrica moderna para mostrar estadísticas en la interfaz.
    Args:
        icon (str): Emoji o icono para la tarjeta.
        value (str/int): Valor a mostrar.
        label (str): Etiqueta descriptiva.
        trend (float, optional): Tendencia porcentual.
        card_type (str): Tipo de tarjeta para estilos.
    Returns:
        str: HTML de la tarjeta de métrica.
    """
    card_class = "metric-card"
    if card_type == "danger":
        card_class += " metric-card-danger"
    elif card_type == "success":
        card_class += " metric-card-success"
    elif card_type == "warning":
        card_class += " metric-card-warning"
    elif card_type == "info":
        card_class += " metric-card-info"
    
    trend_html = ""
    if trend:
        trend_class = "trend-up" if trend > 0 else "trend-down"
        trend_icon = "↗️" if trend > 0 else "↘️"
        trend_html = f'<div class="metric-trend {trend_class}">{trend_icon} {abs(trend)}%</div>'
    
    return f"""
    <div class="{card_class}">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {trend_html}
    </div>
    """

def main():
    # Función principal de la aplicación Streamlit (vista avanzada)
    st.markdown('<h1 class="main-header">🗺️ Mapa Interactivo de Accidentes de Tránsito en Medellín</h1>', unsafe_allow_html=True)

    if not folium_installed:
        st.error(
            """**Error de dependencia: Faltan las librerías `folium` y `streamlit-folium`.**

            Por favor, asegúrate de que tu archivo `requirements.txt` en la raíz de tu repositorio de GitHub contiene las siguientes líneas:

            ```
            folium
            streamlit-folium
            ```

            Después de añadir estas líneas, reinicia la aplicación en Streamlit Cloud ('Manage app' -> 'Reboot')."""
        )
        return
    
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    if 'map_loaded' not in st.session_state:
        st.session_state.map_loaded = False
    
    if 'map_data' not in st.session_state or 'complete_data' not in st.session_state:
        with st.spinner("📂 Cargando datos iniciales..."):
            default_drive_link = "https://drive.google.com/uc?id=13f0cf70-OBPbZGlBxRsQGVXNzhbROrgy&export=download"
            df, status = load_csv_from_drive(default_drive_link)
            if df is not None:
                map_data, complete_data, skipped_records = process_data(df)
                st.session_state.map_data = map_data
                st.session_state.complete_data = complete_data
                st.session_state.skipped_records = skipped_records
                st.session_state.used_encoding = "utf-8"
                # Liberar memoria de df
                del df
            else:
                st.session_state.map_data = pd.DataFrame()
                st.session_state.complete_data = pd.DataFrame()
                st.session_state.skipped_records = 0
                st.session_state.used_encoding = "utf-8"
    
    with st.sidebar:
        st.header("🎛️ Controles del Mapa")
        
        map_type = st.radio(
            "Tipo de Visualización:",
            ["Marcadores", "Mapa de Calor", "Círculos", "Avanzado"],
            index=0
        )
        
        zoom_level = st.slider("Nivel de Zoom:", 10, 18, 12)
        
        st.markdown("---")
        st.header("🔍 Filtros de Datos")
        
        complete_data = st.session_state.complete_data
        
        if not complete_data.empty:
            años = get_unique_values_safe(complete_data['año'])
            clases = get_unique_values_safe(complete_data['clase'])
            gravedades = get_unique_values_safe(complete_data['gravedad'])
            comunas = get_unique_values_safe(complete_data['comuna'])
            
            año_seleccionado = st.multiselect('Año', options=años, default=años)
            clase_seleccionada = st.multiselect('Tipo de accidente', options=clases, default=clases)
            gravedad_seleccionada = st.multiselect('Gravedad', options=gravedades, default=gravedades)
            comuna_seleccionada = st.multiselect('Comuna', options=comunas, default=comunas)
            
            if st.button("🚀 Aplicar Filtros y Mostrar Mapa", type="primary"):
                st.session_state.filters_applied = True
                st.session_state.map_loaded = True
                st.session_state.año_seleccionado = año_seleccionado
                st.session_state.clase_seleccionada = clase_seleccionada
                st.session_state.gravedad_seleccionada = gravedad_seleccionada
                st.session_state.comuna_seleccionada = comuna_seleccionada
                st.session_state.map_type = map_type
                st.session_state.zoom_level = zoom_level
                st.rerun()
        
        st.markdown("---")
        st.info(f"📝 Codificación usada: {st.session_state.used_encoding.upper()}")
        if st.session_state.skipped_records > 0:
            st.warning(f"Se omitieron {st.session_state.skipped_records} registros con coordenadas inválidas")
        
    # Mostrar métricas modernas
    if not st.session_state.filters_applied:
        st.info("🎯 Configure los filtros en la barra lateral y haga clic en 'Aplicar Filtros' para visualizar el mapa interactivo")
        
        # Grid de métricas modernas
        st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_registros = len(st.session_state.map_data)
            st.markdown(create_metric_card(
                "📊", 
                f"{total_registros:,}", 
                "Total de Registros", 
                None, 
                "info"
            ), unsafe_allow_html=True)
        
        with col2:
            columnas_count = len(st.session_state.complete_data.columns) if not st.session_state.complete_data.empty else 0
            st.markdown(create_metric_card(
                "📋", 
                columnas_count, 
                "Columnas Disponibles", 
                None, 
                "success"
            ), unsafe_allow_html=True)
        
        with col3:
            skipped = st.session_state.skipped_records
            st.markdown(create_metric_card(
                "⚠️", 
                skipped, 
                "Registros Omitidos", 
                None, 
                "warning" if skipped > 0 else "success"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card(
                "📍", 
                "Medellín", 
                "Área de Cobertura", 
                None, 
                "default"
            ), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🗺️ Mapa Interactivo")
        st.markdown('<div class="loading-placeholder">', unsafe_allow_html=True)
        st.write("👈 Configure los filtros en la barra lateral y haga clic en 'Aplicar Filtros' para visualizar el mapa")
        st.write("🚀 El mapa se cargará instantáneamente después de aplicar los filtros")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if not st.session_state.complete_data.empty:
            st.markdown("### 📋 Vista Previa de Datos")
            st.dataframe(
                st.session_state.complete_data[['fecha', 'clase', 'gravedad', 'barrio', 'comuna']].head(10),
                use_container_width=True
            )
    
    else:
        filtered_complete = st.session_state.complete_data.copy()
        # Optimización: aplicar filtros en un bucle
        filtros = {
            'año': st.session_state.get('año_seleccionado', []),
            'clase': st.session_state.get('clase_seleccionada', []),
            'gravedad': st.session_state.get('gravedad_seleccionada', []),
            'comuna': st.session_state.get('comuna_seleccionada', [])
        }
        for campo, valores in filtros.items():
            if valores:
                filtered_complete = filtered_complete[filtered_complete[campo].astype(str).isin(valores)]
        
        filtered_map = st.session_state.map_data.iloc[filtered_complete.index]
        
        # Métricas modernas para datos filtrados
        st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            incidentes_count = len(filtered_map)
            porcentaje = (incidentes_count / len(st.session_state.map_data)) * 100 if len(st.session_state.map_data) > 0 else 0
            st.markdown(create_metric_card(
                "📊", 
                f"{incidentes_count:,}", 
                "Incidentes Mostrados", 
                None, 
                "info"
            ), unsafe_allow_html=True)
        
        with col2:
            if not filtered_map.empty:
                lat_media = filtered_map['lat'].mean()
                st.markdown(create_metric_card(
                    "📍", 
                    f"{lat_media:.6f}", 
                    "Latitud Media", 
                    None, 
                    "success"
                ), unsafe_allow_html=True)
            else:
                st.markdown(create_metric_card(
                    "📍", 
                    "N/A", 
                    "Latitud Media", 
                    None, 
                    "warning"
                ), unsafe_allow_html=True)
        
        with col3:
            if not filtered_map.empty:
                lon_media = filtered_map['lon'].mean()
                st.markdown(create_metric_card(
                    "🌐", 
                    f"{lon_media:.6f}", 
                    "Longitud Media", 
                    None, 
                    "success"
                ), unsafe_allow_html=True)
            else:
                st.markdown(create_metric_card(
                    "🌐", 
                    "N/A", 
                    "Longitud Media", 
                    None, 
                    "warning"
                ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card(
                "📈", 
                f"{porcentaje:.1f}%", 
                "Porcentaje Total", 
                None, 
                "default"
            ), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🗺️ Mapa Interactivo")
        
        if not filtered_map.empty:
            # Centra el mapa en la zona centro de Medellín y elimina controles
            medellin_center = [6.245053207648148, -75.57931461508907]  # Centro solicitado
            if st.session_state.map_type == "Marcadores":
                folium_map = create_folium_map(filtered_complete, 'markers', st.session_state.zoom_level, center=medellin_center)
            elif st.session_state.map_type == "Mapa de Calor":
                folium_map = create_folium_map(filtered_map, 'heatmap', st.session_state.zoom_level, center=medellin_center)
            elif st.session_state.map_type == "Círculos":
                folium_map = create_folium_map(filtered_map, 'circle', st.session_state.zoom_level, center=medellin_center)
            elif st.session_state.map_type == "Avanzado" and not filtered_complete.empty:
                folium_map = create_advanced_map(filtered_complete, center=medellin_center)
            else:
                folium_map = create_folium_map(filtered_map, 'markers', st.session_state.zoom_level, center=medellin_center)
            st_folium(folium_map, height=700, returned_objects=[])
        else:
            st.warning("No hay datos para mostrar en el mapa con los filtros seleccionados.")
            if st.button("🗑️ Limpiar Filtros"):
                st.session_state.filters_applied = False
                st.session_state.map_loaded = False
                st.rerun()
        
        if not filtered_complete.empty:
            st.markdown("---")
            st.markdown(f"### 📊 Análisis de Datos ({len(filtered_complete)} registros)")
            tab1, tab2, tab3 = st.tabs(["Distribución", "Temporal", "Datos Crudos"])
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Distribución por Tipo de Accidente**")
                    if 'clase' in filtered_complete.columns:
                        clase_counts = filtered_complete['clase'].value_counts()
                        st.bar_chart(clase_counts)
                with col2:
                    st.markdown("**Distribución por Gravedad**")
                    if 'gravedad' in filtered_complete.columns:
                        gravedad_counts = filtered_complete['gravedad'].value_counts()
                        st.bar_chart(gravedad_counts)
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Distribución por Año**")
                    if 'año' in filtered_complete.columns:
                        año_counts = filtered_complete['año'].value_counts()
                        st.bar_chart(año_counts)
                with col2:
                    st.markdown("**Distribución por Comuna**")
                    if 'comuna' in filtered_complete.columns:
                        comuna_counts = filtered_complete['comuna'].value_counts()
                        st.bar_chart(comuna_counts)
            with tab3:
                st.markdown(f"**Datos Filtrados ({len(filtered_complete)} registros)**")
                st.dataframe(
                    filtered_complete[
                        ['fecha', 'clase', 'gravedad', 'direccion', 'barrio', 'comuna', 'año']
                    ].head(50),
                    use_container_width=True,
                    height=400
                )
        else:
            st.info("No hay datos filtrados para mostrar.")

if __name__ == "__main__":
    main()
