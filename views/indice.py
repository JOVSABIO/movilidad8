import pandas as pd
import streamlit as st

# Title of the app
st.title("Análisis de accidentes de tránsito en Medellín")
st.write("\n")

# Subheader
st.subheader("INTEGRANTES DEL EQUIPO")

# The team members
st.write("\n")
st.write('''
         - Diana Castaño,
         - Jovani Guerrero
         - Natalia Villa
         - Néstor Correa
         - Oliver Bohórquez
         ''')

st.write("\n")

# --- SECTION INTRODUCCIÓN---
tab1, tab2,tab3 = st.tabs(["Introducción", "Contexto General", "Objetivo General del Proyecto"])
with tab1:
    st.write("\n") 
    # Introducción
    st.subheader("INTRODUCCIÓN")
    st.write("\n")
    st.write('''
            La accidentalidad vial es una problemática crítica en Medellín, afectando la movilidad, la seguridad y la salud pública de sus habitantes. Con un notable impacto en motociclistas y peatones, este fenómeno demanda un análisis detallado para su correcta gestión.
            
            El presente estudio se enfoca en identificar los patrones de accidentes viales en la ciudad, analizando la correlación entre la hora, el tipo de vehículo involucrado y las comunas más afectadas. Al examinar los datos disponibles, buscamos descubrir las tendencias subyacentes que influyen en estos incidentes. El objetivo final es proporcionar información esencial que sirva como base para la creación de estrategias de prevención efectivas y la optimización de la seguridad vial en Medellín.
            ''')

with tab2:
    st.write("\n") 
    # Contexto General
    st.write("\n") 
    st.subheader("CONTEXTO GENERAL")
    st.write("\n")
    st.write('''
            La ciudad de Medellín presenta altos índices de accidentalidad vial, especialmente en motociclistas y peatones, lo que impacta la movilidad, la seguridad y la salud pública. Analizar estos datos permite identificar patrones en el tiempo, lugares y tipos de vehículos involucrados, aportando información clave para diseñar estrategias de prevención y mejorar la gestión de la seguridad vial en la ciudad.
            ''')

with tab3:
    st.write("\n") 
    #Objetivo Generaldel Proyecto
    st.subheader("OBJETIVO GENERAL DEL PROYECTO")
    st.write("\n")
    st.write('''             
            Identificar patrones de accidentes viales en Medellín según lugar, comuna y gravedad.
            ''')
    st.write("\n")