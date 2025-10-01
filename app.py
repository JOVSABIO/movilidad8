from pathlib import Path
import streamlit as st


# --- PATH SETTINGS ---
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"
profile_pic = current_dir / "assets" / "m8vilidad.png"

# --- PAGE SETUP ---
Settings_page = st.Page(
    page="views/indice.py",
    title="Indice",
    icon="📄",
    default=True,
)

Components = st.Page(
    page="views/componentes.py",
    title="Componentes",
    icon="🛠️"
)

Integrador = st.Page(
    page="views/integrador.py",
    title="Mapa Interactivo",
    icon="🌎"
)

# --- NAVIGATION SETUP [WITH SECTIONS] ---
pg = st.navigation(
    {
        "Introducción": [Settings_page],
        "Temas": [Components, Integrador],
    }
)

# --- SHARED ON ALL PAGES ---
st.image(str(profile_pic), width=1800)

st.sidebar.text("Made with ❤️ by GRUPO 8")

# --- RUN NAVIGATION ---
pg.run()

