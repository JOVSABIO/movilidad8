from pathlib import Path
import pandas as pd
import streamlit as st

# Title of the app
st.title("Análisis de accidentes de tránsito en Medellín")
st.write("\n")

# Subheader
st.subheader("COMPONENTES DEL PROYECTO")

st.write("\n")

# --- SECTION INTRODUCCIÓN---
st.subheader("Introducción")
st.write("""
El análisis de accidentes de tránsito es crucial para mejorar la seguridad vial y reducir el número de incidentes en las carreteras. En este proyecto, nos enfocamos en el análisis de datos de accidentes de tránsito en Medellín, Colombia, utilizando herramientas de ciencia de datos y visualización para identificar patrones y tendencias que puedan informar políticas y estrategias de prevención.
""")
st.write("\n")  

# --- SECTION INTRODUCCIÓN---
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Dataset", "Diagrama Entidad-Relación", "Scripts","Herramientas","Conclusiones"])

with tab1:
    st.write("\n") 
    # Introducción
    st.subheader("DATASET")
    st.write("\n")

    # --- SECTION DATASET---
    st.subheader("Dataset")
    st.write("""
            El conjunto de datos utilizado en este análisis proviene de una BD de datos abiertos de la Secretaría de Movilidad de Medellín y contiene información detallada sobre los accidentes de tránsito ocurridos en la ciudad, entre el año 2014 hasta el año 2020. El dataset incluye variables como la fecha y hora del accidente, la ubicación y la gravedad del accidente, entre otros.
            """)
    st.write("\n")

    # Load dataset
    current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
    data_path = current_dir / "incidentes_viales_limpio.csv"    
    df = pd.read_csv(data_path, parse_dates=['FECHA_ACCIDENTE'])
    st.write("A continuación, se muestra una vista previa del conjunto de datos:")
    st.dataframe(df.head(10))
    st.write("\n")
    st.write(f"El conjunto de datos contiene un total de {df.shape[0]} registros y {df.shape[1]} columnas.")
    st.write("\n")
    st.write("Las columnas del dataset son las siguientes:")
    st.write(df.columns.tolist())
    st.write("\n")


with tab2:
    st.write("\n") 
    # Diagrama Entidad-Relación
    st.subheader("DIAGRAMA ENTIDAD-RELACIÓN")
    st.write("\n")

    st.subheader("Diagrama Entidad-Relación")
    st.write("\n")

    profile_pic = current_dir / "er_sql.png"

    st.image(profile_pic, width=1800)
    st.write("\n")

with tab3:
    st.write("\n") 
    # Dataset
    st.subheader("DATASET")
    st.write("\n")

    st.subheader("Script SQL para la creación de la base de datos") 

    st.write("\n")

    code = '''# Tabla para los datos de tiempo

        CREATE TABLE Tiempo (
        tiempo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha_accidente DATE,
        año INTEGER,
        mes INTEGER
    );
    '''
    st.code(code, language="sql")
    st.write("\n")

    code = '''# Tabla para los datos de ubicación

        CREATE TABLE Ubicacion (
        ubicacion_id INTEGER PRIMARY KEY AUTOINCREMENT,
        direccion VARCHAR,
        comuna VARCHAR,
        barrio VARCHAR,
        x FLOAT,
        y FLOAT,
        diseno VARCHAR
    );
    '''

    st.code(code, language="sql")
    st.write("\n")

    code = '''# Tabla para los datos de descripcion

        CREATE TABLE Descripcion (
        descripcion_id INTEGER PRIMARY KEY AUTOINCREMENT,
        cbml INTEGER,
        clase_accidente VARCHAR,
        nro_radicado INTEGER
    );
    '''
    
    st.code(code, language="sql")
    st.write("\n")

    code = '''# Tabla principal o maestra: inccidentes

        CREATE TABLE Incidentes (
        incidente_id INTEGER PRIMARY KEY AUTOINCREMENT,
        expediente VARCHAR,
        gravedad_accidente VARCHAR,
        tiempo_id INTEGER,
        ubicacion_id INTEGER,
        descripcion_id INTEGER,
        FOREIGN KEY (tiempo_id) REFERENCES Tiempo(tiempo_id),
        FOREIGN KEY (ubicacion_id) REFERENCES Ubicacion(ubicacion_id),
        FOREIGN KEY (descripcion_id) REFERENCES Descripcion(descripcion_id)
    );
    '''

    st.code(code, language="sql")
    st.write("\n")

with tab4:
    st.write("\n") 
    # Herramientas
    st.subheader("HERRAMIENTAS")
    st.write("\n")
    
    # --- SECTION HERRAMIENTAS---
    st.subheader("Herramientas")
    st.write("""
            Para llevar a cabo este análisis, se utilizaron las siguientes herramientas y bibliotecas de Python:
            """)
    st.write("\n")
    st.write("""
    - **Pandas**: Para la manipulación y análisis de datos.
    - **Streamlit**: Para desarrollar la aplicación web interactiva.
    - **Folium**: Para la visualización de datos geoespaciales en mapas.
    - **requests**: Para realizar solicitudes HTTP.
    - **io**: Para manejar flujos de datos.
    - **sqlite3**: Para la gestión de bases de datos SQLite.
             """)
    st.write("\n")

with tab5:
     # --- SECTION CONCLUSIONES---
    st.write("\n") 
    # Conclusiones
    st.subheader("CONCLUSIONES")
    st.write("""
            El análisis de accidentes de tránsito en Medellín ha permitido identificar patrones y tendencias que pueden ser útiles para mejorar la seguridad vial en la ciudad. A través de la visualización de datos y el uso de herramientas de ciencia de datos, se han destacado áreas críticas y factores contribuyentes a los accidentes. Este proyecto sirve como base para futuras investigaciones y para la implementación de políticas de prevención más efectivas.
            """)
    st.write("\n")

st.write("¡Gracias por explorar este análisis de accidentes de tránsito en Medellín!")