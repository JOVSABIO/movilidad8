import streamlit as st
import pandas as pd
import requests
from io import StringIO

try:
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import MarkerCluster, HeatMap
    folium_installed = True
except ImportError as e:
    folium_installed = False
    st.error(f"Error importing folium: {e}")

# AÑADIR CSS AL INICIO
st.markdown("""
<style>
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
.metric-card-danger { border-left-color: #f44336; }
.metric-card-success { border-left-color: #4CAF50; }
.metric-card-warning { border-left-color: #ff9800; }
.metric-card-info { border-left-color: #2196F3; }
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.metric-value { font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0; }
.metric-label { color: #666; font-size: 0.9rem; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

def load_csv_from_drive(drive_link):
    # Carga un archivo CSV desde un enlace de Google Drive
    try:
        if 'id=' in drive_link:
            file_id = drive_link.split('id=')[1].split('&')[0]
        elif '/d/' in drive_link:
            file_id = drive_link.split('/d/')[1].split('/')[0]
        else:
            # Si no es un enlace, asumimos que es el ID directo
            file_id = drive_link
            
        # CORREGIDO: Usar el file_id dinámicamente
        download_url = f'https://drive.google.com/uc?id={file_id}&export=download'
        response = requests.get(download_url)
        response.raise_for_status()
        
        # Intentar diferentes codificaciones
        try:
            df = pd.read_csv(StringIO(response.text))
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(response.content.decode('latin-1')))
        
        return df, "success"
        
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return None, f"Error: {str(e)}"

def process_data(df):
    # Procesa los datos del DataFrame, extrayendo coordenadas y registros completos
    try:
        coordinates_data = []
        complete_data = []
        skipped_records = 0
        
        # Buscar columnas relevantes
        location_col = next((col for col in df.columns if 'LOCATION' in col.upper()), None)
        clase_col = next((col for col in df.columns if 'CLASE' in col.upper()), None)
        gravedad_col = next((col for col in df.columns if 'GRAVEDAD' in col.upper()), None)
        barrio_col = next((col for col in df.columns if 'BARRIO' in col.upper()), None)
        comuna_col = next((col for col in df.columns if 'COMUNA' in col.upper()), None)
        año_col = next((col for col in df.columns if 'AÑO' in col.upper() or 'YEAR' in col.upper()), None)
        fecha_col = next((col for col in df.columns if 'FECHA' in col.upper() or 'DATE' in col.upper()), None)
        direccion_col = next((col for col in df.columns if 'DIRECCION' in col.upper() or 'ADDRESS' in col.upper()), None)
        
        if location_col is None:
            st.warning("No se encontró columna de ubicación. Columnas disponibles: " + ", ".join(df.columns))
            return pd.DataFrame(), pd.DataFrame(), len(df)
            
        # Procesar datos
        for i, row in df.iterrows():
            try:
                location_str = str(row[location_col]).strip('[] ')
                if location_str in ['', 'NaN', 'nan', 'None']:
                    skipped_records += 1
                    continue
                    
                parts = location_str.split(',')
                if len(parts) != 2:
                    skipped_records += 1
                    continue
                    
                lon = float(parts[0].strip())
                lat = float(parts[1].strip())
                
                # Validar coordenadas de Medellín
                if not ((6.1 <= lat <= 6.4) and (-75.7 <= lon <= -75.5)):
                    skipped_records += 1
                    continue
                    
                coordinates_data.append({'lat': lat, 'lon': lon})
                
                record = {
                    'lat': lat,
                    'lon': lon,
                    'clase': row[clase_col] if clase_col in row else '',
                    'gravedad': row[gravedad_col] if gravedad_col in row else '',
                    'barrio': row[barrio_col] if barrio_col in row else '',
                    'comuna': row[comuna_col] if comuna_col in row else '',
                    'año': row[año_col] if año_col in row else '',
                    'fecha': row[fecha_col] if fecha_col in row else '',
                    'direccion': row[direccion_col] if direccion_col in row else ''
                }
                complete_data.append(record)
                
            except Exception as e:
                skipped_records += 1
                continue
                
        return pd.DataFrame(coordinates_data), pd.DataFrame(complete_data), skipped_records
        
    except Exception as e:
        st.error(f"Error procesando datos: {e}")
        return pd.DataFrame(), pd.DataFrame(), 0

# [Mantener las funciones create_folium_map, create_advanced_map, etc. iguales...]

def main():
    st.set_page_config(page_title="Mapa de Accidentes Medellín", layout="wide")
    
    st.markdown('<h1 class="main-header">🗺️ Mapa Interactivo de Accidentes de Tránsito en Medellín</h1>', unsafe_allow_html=True)

    # Verificar dependencias
    if not folium_installed:
        st.error("""
        **Error de dependencia: Faltan las librerías `folium` y `streamlit-folium`.**
        
        Las dependencias se están instalando pero hay un problema de importación.
        Por favor, reinicia la aplicación en Streamlit Cloud ('Manage app' -> 'Reboot').
        """)
        return

    # Inicializar session state
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    # Cargar datos solo una vez
    if not st.session_state.data_loaded:
        with st.spinner("📂 Cargando datos desde Google Drive..."):
            # CORREGIDO: Usar el ID correcto
            file_id = "1R5JxWJZK_OvFYdGmE2mG3wUhFRb7StdD"
            df, status = load_csv_from_drive(file_id)
            
            if df is not None and not df.empty:
                map_data, complete_data, skipped_records = process_data(df)
                st.session_state.map_data = map_data
                st.session_state.complete_data = complete_data
                st.session_state.skipped_records = skipped_records
                st.session_state.data_loaded = True
                st.success(f"✅ Datos cargados: {len(complete_data)} registros válidos")
            else:
                st.error("❌ No se pudieron cargar los datos. Verifica el enlace de Google Drive.")
                return

    # [El resto del código permanece igual...]

if __name__ == "__main__":
    main()
