import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

def main():
    st.set_page_config(page_title="Análisis ML - Accidentes", layout="wide")
    st.title("🤖 Análisis de Machine Learning - Accidentes Viales")
    st.markdown("---")
    
    # Cargar datos desde session_state
    if 'complete_data' not in st.session_state or st.session_state.complete_data.empty:
        st.error("❌ No hay datos disponibles. Por favor, carga los datos en la página del Integrador primero.")
        return
    
    df = st.session_state.complete_data.copy()
    
    # Mostrar información del dataset
    with st.expander("📊 Información del Dataset", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Registros", f"{len(df):,}")
        with col2:
            st.metric("Columnas", len(df.columns))
        with col3:
            st.metric("Completado", f"{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))):.1%}")
        
        st.write("**Columnas disponibles:**", list(df.columns))
    
    # Selector de análisis
    st.sidebar.header("🔧 Configuración de Análisis")
    analysis_type = st.sidebar.selectbox(
        "Selecciona el tipo de análisis:",
        ["Predicción de Gravedad", "Clustering de Zonas de Riesgo", "Predicción de Tipo de Accidente", "Análisis de Series de Tiempo"]
    )
    
    # Ejecutar análisis seleccionado
    if analysis_type == "Predicción de Gravedad":
        predecir_gravedad_accidente_ui(df)
    elif analysis_type == "Clustering de Zonas de Riesgo":
        clustering_zones_riesgo_ui(df)
    elif analysis_type == "Predicción de Tipo de Accidente":
        predecir_tipo_accidente_ui(df)
    elif analysis_type == "Análisis de Series de Tiempo":
        analisis_series_tiempo_ui(df)

def predecir_gravedad_accidente_ui(df):
    st.header("🔴 Predicción de Gravedad del Accidente")
    
    if df.empty:
        st.error("❌ No hay datos para analizar")
        return
    
    # Mapeo de columnas - CORREGIDO
    # Buscar la columna de gravedad con diferentes nombres posibles
    gravedad_col = None
    posibles_nombres_gravedad = ['gravedad', 'GRAVEDAD', 'GRAVEDAD_ACCIDENTE', 'severidad', 'SEVERIDAD']
    
    for col in df.columns:
        if col.upper() in [name.upper() for name in posibles_nombres_gravedad]:
            gravedad_col = col
            break
    
    if not gravedad_col:
        st.error("❌ No se encontró ninguna columna de gravedad en el dataset")
        st.info("🔍 Nombres buscados: 'gravedad', 'GRAVEDAD', 'GRAVEDAD_ACCIDENTE', 'severidad', 'SEVERIDAD'")
        st.write("**Columnas disponibles:**", list(df.columns))
        return
    
    st.success(f"✅ Columna de gravedad encontrada: **{gravedad_col}**")
    
    # Preparar datos
    df_ml = df.copy()
    
    # Buscar otras columnas necesarias
    hora_col = 'HORA' if 'HORA' in df_ml.columns else next((col for col in df_ml.columns if 'HORA' in col.upper()), None)
    mes_col = 'MES' if 'MES' in df_ml.columns else next((col for col in df_ml.columns if 'MES' in col.upper()), None)
    clase_col = 'CLASE_ACCIDENTE' if 'CLASE_ACCIDENTE' in df_ml.columns else next((col for col in df_ml.columns if 'CLASE' in col.upper()), 'clase')
    comuna_col = 'COMUNA' if 'COMUNA' in df_ml.columns else next((col for col in df_ml.columns if 'COMUNA' in col.upper()), 'comuna')
    
    # Crear características disponibles
    features = []
    if hora_col and hora_col in df_ml.columns:
        features.append(hora_col)
    if mes_col and mes_col in df_ml.columns:
        features.append(mes_col)
    if clase_col and clase_col in df_ml.columns:
        features.append(clase_col)
    if comuna_col and comuna_col in df_ml.columns:
        features.append(comuna_col)
    
    # Si no tenemos suficientes características, crear algunas sintéticas
    if len(features) < 2:
        st.warning("⚠️ Pocas características disponibles, creando características adicionales...")
        if 'HORA' not in df_ml.columns:
            df_ml['HORA'] = np.random.randint(0, 24, len(df_ml))
            features.append('HORA')
        if 'MES' not in df_ml.columns:
            df_ml['MES'] = np.random.randint(1, 13, len(df_ml))
            features.append('MES')
    
    st.info(f"🔍 Características utilizadas: {features}")
    
    # Limpiar datos para ML
    df_ml = df_ml[features + [gravedad_col]].dropna()
    
    if len(df_ml) < 50:
        st.error(f"❌ Datos insuficientes para el modelo. Se necesitan al menos 50 registros, solo hay {len(df_ml)}")
        return
    
    # Mostrar distribución de gravedad
    st.subheader("📊 Distribución de Gravedad")
    gravedad_counts = df_ml[gravedad_col].value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        st.bar_chart(gravedad_counts)
    
    with col2:
        st.dataframe(gravedad_counts)
    
    with st.spinner("Entrenando modelo de Random Forest..."):
        try:
            # Codificar variables categóricas
            le_dict = {}
            for feature in features:
                if df_ml[feature].dtype == 'object':
                    le = LabelEncoder()
                    df_ml[feature] = le.fit_transform(df_ml[feature].astype(str))
                    le_dict[feature] = le
            
            # Codificar target
            le_target = LabelEncoder()
            df_ml['target_encoded'] = le_target.fit_transform(df_ml[gravedad_col])
            
            # Separar características y target
            X = df_ml[features]
            y = df_ml['target_encoded']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            
        except Exception as e:
            st.error(f"❌ Error entrenando el modelo: {e}")
            return
    
        # REPORTE DE CLASIFICACIÓN MEJORADO
    st.subheader("📊 Reporte de Clasificación Detallado")
    
    # Convertir reporte a DataFrame
    report_dict = classification_report(y_test, y_pred, target_names=le_target.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Crear pestañas para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📈 Vista Resumida", "📋 Vista Completa", "🎯 Análisis por Clase"])
    
    with tab1:
        # Métricas generales
        st.write("**Métricas Generales del Modelo**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = report_df.loc['accuracy', 'precision']
            st.metric("Precisión Global", f"{accuracy:.2%}")
        
        with col2:
            macro_avg = report_df.loc['macro avg', 'precision']
            st.metric("Precisión Promedio", f"{macro_avg:.2%}")
        
        with col3:
            weighted_avg = report_df.loc['weighted avg', 'precision']
            st.metric("Precisión Ponderada", f"{weighted_avg:.2%}")
        
        with col4:
            st.metric("Clases", len(le_target.classes_))
        
        # Gráfico de precisión por clase
        classes_df = report_df.iloc[:-3].copy()  # Excluir promedios
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(classes_df))
        width = 0.25
        
        ax.bar(x_pos - width, classes_df['precision'] * 100, width, label='Precisión', alpha=0.8, color='skyblue')
        ax.bar(x_pos, classes_df['recall'] * 100, width, label='Recall', alpha=0.8, color='lightcoral')
        ax.bar(x_pos + width, classes_df['f1-score'] * 100, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Clases de Gravedad')
        ax.set_ylabel('Porcentaje (%)')
        ax.set_title('Métricas por Clase de Gravedad')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(classes_df.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab2:
        # Tabla completa con formato mejorado
        st.write("**Reporte de Clasificación Completo**")
        
        # Formatear el DataFrame para mejor visualización
        display_df = report_df.copy()
        display_df['support'] = display_df['support'].astype(int)
        
        # Aplicar formato a las columnas numéricas
        numeric_cols = ['precision', 'recall', 'f1-score']
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        
        # Resaltar filas importantes
        def highlight_rows(row):
            if row.name == 'accuracy':
                return ['background-color: #AFD5FA'] * len(row)
            elif row.name in ['macro avg', 'weighted avg']:
                return ['background-color: #FA9B9B'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_rows, axis=1),
            width='stretch',
            height=400
        )
        
        # Leyenda
        st.caption("🎨 **Leyenda:** Azul = Precisión global, Rojo = Promedios")
    
    with tab3:
        # Análisis detallado por clase
        st.write("**Análisis Detallado por Clase**")
        
        classes_data = report_df.iloc[:-3].copy()  # Solo las clases
        
        for clase in classes_data.index:
            with st.expander(f"📌 **{clase}**", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    precision = classes_data.loc[clase, 'precision']
                    st.metric("Precisión", f"{precision:.2%}")
                
                with col2:
                    recall = classes_data.loc[clase, 'recall']
                    st.metric("Recall", f"{recall:.2%}")
                
                with col3:
                    f1 = classes_data.loc[clase, 'f1-score']
                    st.metric("F1-Score", f"{f1:.2%}")
                
                with col4:
                    support = classes_data.loc[clase, 'support']
                    st.metric("Muestras", support)
                
                # Explicación de métricas
                st.write("**📝 Interpretación:**")
                if precision > 0.7 and recall > 0.7:
                    st.success("✅ **Excelente desempeño**: El modelo predice bien esta clase")
                elif precision > 0.5 or recall > 0.5:
                    st.warning("⚠️ **Desempeño moderado**: El modelo tiene dificultades con esta clase")
                else:
                    st.error("❌ **Bajo desempeño**: El modelo no predice bien esta clase")

    # Importancia de características (mantener esta parte)
    st.subheader("🔍 Importancia de Características")
    feature_importance = pd.DataFrame({
        'Característica': features,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importancia', y='Característica', ax=ax)
    ax.set_title('Importancia de Características - Predicción de Gravedad')
    ax.set_xlabel('Importancia')
    st.pyplot(fig)
    
    st.dataframe(feature_importance)
    
    # MATRIZ DE CONFUSIÓN MEJORADA
    st.subheader("🎯 Matriz de Confusión")
    
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Heatmap de la matriz de confusión
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Etiquetas
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=le_target.classes_,
               yticklabels=le_target.classes_,
               title='Matriz de Confusión',
               ylabel='Etiqueta Real',
               xlabel='Etiqueta Predicha')
        
        # Rotar etiquetas
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Anotaciones en las celdas
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        st.pyplot(fig)
    
    with col2:
        st.write("**📊 Estadísticas de la Matriz**")
        
        # Calcular métricas adicionales
        total = cm.sum()
        correctos = np.diag(cm).sum()
        incorrectos = total - correctos
        
        st.metric("Predicciones Correctas", f"{correctos} ({correctos/total:.1%})")
        st.metric("Predicciones Incorrectas", f"{incorrectos} ({incorrectos/total:.1%})")
        st.metric("Total de Muestras", total)
        
        # Clase con mejor y peor precisión
        precision_por_clase = np.diag(cm) / cm.sum(axis=0)
        recall_por_clase = np.diag(cm) / cm.sum(axis=1)
        
        mejor_clase_idx = np.argmax(precision_por_clase)
        peor_clase_idx = np.argmin(precision_por_clase)
        
        st.write("---")
        st.write("**🎭 Clases Destacadas:**")
        st.success(f"**Mejor**: {le_target.classes_[mejor_clase_idx]} ({precision_por_clase[mejor_clase_idx]:.1%})")
        st.error(f"**Peor**: {le_target.classes_[peor_clase_idx]} ({precision_por_clase[peor_clase_idx]:.1%})")
    
    # RESUMEN EJECUTIVO
    st.subheader("📋 Resumen Ejecutivo")
    
    # Calcular métricas generales
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📈 Métricas de Calidad**")
        st.metric("Precisión Global", f"{accuracy:.2%}")
        st.metric("Clases Modeladas", len(le_target.classes_))
    
    with col2:
        st.write("**📊 Distribución**")
        st.metric("Muestras Entrenamiento", len(X_train))
        st.metric("Muestras Prueba", len(X_test))
        st.metric("Tamaño Total", len(df_ml))
    
    with col3:
        st.write("**🎯 Recomendaciones**")
        if accuracy > 0.8:
            st.success("✅ **Modelo Excelente**")
            st.write("Puede usarse en producción")
        elif accuracy > 0.6:
            st.warning("⚠️ **Modelo Aceptable**")
            st.write("Considerar mejoras adicionales")
        else:
            st.error("❌ **Modelo Necesita Mejoras**")
            st.write("Revisar características y datos")

def clustering_zones_riesgo_ui(df):
    st.header("🟢 Clustering de Zonas de Riesgo")
    
    if df.empty:
        st.error("❌ No hay datos para analizar")
        return
    
    # Buscar columnas necesarias
    comuna_col = next((col for col in df.columns if 'COMUNA' in col.upper()), None)
    hora_col = next((col for col in df.columns if 'HORA' in col.upper()), None)
    gravedad_col = next((col for col in df.columns if 'GRAVEDAD' in col.upper()), None)
    
    if not comuna_col:
        st.error("❌ No se encontró columna COMUNA")
        return
    
    st.info(f"🔍 Columnas identificadas: COMUNA='{comuna_col}', HORA='{hora_col}', GRAVEDAD='{gravedad_col}'")
    
    with st.spinner("Analizando zonas de riesgo..."):
        try:
            # Crear datos para clustering
            df_cluster = df.copy()
            
            # Si no hay hora, crear una sintética
            if not hora_col:
                df_cluster['HORA_SINTETICA'] = np.random.randint(0, 24, len(df_cluster))
                hora_col = 'HORA_SINTETICA'
            
            # Agrupar por comuna
            zonas_riesgo = df_cluster.groupby(comuna_col).agg({
                hora_col: ['count', 'mean']
            }).reset_index()
            
            # Limpiar nombres de columnas
            zonas_riesgo.columns = [comuna_col, 'total_accidentes', 'hora_promedio']
            
            # Calcular tasa de heridos si existe columna de gravedad
            if gravedad_col:
                try:
                    tasa_heridos = df_cluster.groupby(comuna_col)[gravedad_col].apply(
                        lambda x: (x.astype(str).str.contains('herido|heridos', case=False, na=False)).mean()
                    ).reset_index(name='tasa_heridos')
                    zonas_riesgo = zonas_riesgo.merge(tasa_heridos, on=comuna_col)
                except:
                    zonas_riesgo['tasa_heridos'] = 0.3
            else:
                zonas_riesgo['tasa_heridos'] = 0.3
            
            zonas_riesgo = zonas_riesgo.dropna()
            zonas_riesgo = zonas_riesgo[zonas_riesgo['total_accidentes'] > 0]
            
            if len(zonas_riesgo) < 3:
                st.error("❌ Muy pocas comunas para clustering")
                return
            
            # Clustering
            features_cluster = ['total_accidentes', 'hora_promedio', 'tasa_heridos']
            X_cluster = zonas_riesgo[features_cluster]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            zonas_riesgo['cluster_riesgo'] = clusters
            zonas_riesgo['nivel_riesgo'] = zonas_riesgo['cluster_riesgo'].map({0: 'Bajo', 1: 'Medio', 2: 'Alto'})
            
        except Exception as e:
            st.error(f"❌ Error en clustering: {e}")
            return
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Estadísticas por Nivel de Riesgo")
        cluster_stats = zonas_riesgo.groupby('nivel_riesgo').agg({
            'total_accidentes': ['mean', 'sum'],
            'tasa_heridos': 'mean',
            comuna_col: 'count'
        }).round(3)
        st.dataframe(cluster_stats)
    
    with col2:
        st.subheader("📊 Distribución de Riesgos")
        riesgo_counts = zonas_riesgo['nivel_riesgo'].value_counts()
        st.bar_chart(riesgo_counts)
    
    # Gráficos
    st.subheader("📊 Visualización de Clusters")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    scatter = axes[0].scatter(zonas_riesgo['total_accidentes'], 
                           zonas_riesgo['tasa_heridos'], 
                           c=zonas_riesgo['cluster_riesgo'], 
                           cmap='RdYlGn_r', s=100, alpha=0.7)
    axes[0].set_xlabel('Total Accidentes')
    axes[0].set_ylabel('Tasa de Heridos')
    axes[0].set_title('Clusters de Zonas de Riesgo')
    plt.colorbar(scatter, ax=axes[0])
    
    # Top comunas
    top_comunas = zonas_riesgo.nlargest(10, 'total_accidentes')
    axes[1].barh(top_comunas[comuna_col], top_comunas['total_accidentes'], 
                color=top_comunas['cluster_riesgo'].map({0: 'green', 1: 'orange', 2: 'red'}))
    axes[1].set_title('Top 10 Comunas con Más Accidentes')
    axes[1].set_xlabel('Número de Accidentes')
    
    st.pyplot(fig)
    
    # Comunas de alto riesgo
    st.subheader("📍 Comunas de Alto Riesgo")
    alto_riesgo = zonas_riesgo[zonas_riesgo['nivel_riesgo'] == 'Alto']
    if not alto_riesgo.empty:
        st.dataframe(alto_riesgo[[comuna_col, 'total_accidentes', 'tasa_heridos']])
    else:
        st.info("No se identificaron comunas de alto riesgo")

def predecir_tipo_accidente_ui(df):
    st.header("🔵 Predicción de Tipo de Accidente")
    
    if df.empty:
        st.error("❌ No hay datos para analizar")
        return
    
    # Buscar columna de tipo de accidente
    clase_col = next((col for col in df.columns if 'CLASE' in col.upper()), None)
    if not clase_col:
        st.error("❌ No se encontró columna de tipo de accidente")
        return
    
    st.success(f"✅ Columna de tipo de accidente encontrada: **{clase_col}**")
    
    # Preparar datos
    df_ml = df.copy()
    
    # Buscar otras columnas
    hora_col = next((col for col in df_ml.columns if 'HORA' in col.upper()), 'HORA_SINTETICA')
    mes_col = next((col for col in df_ml.columns if 'MES' in col.upper()), 'MES_SINTETICO')
    comuna_col = next((col for col in df_ml.columns if 'COMUNA' in col.upper()), 'comuna')
    
    # Crear características sintéticas si es necesario
    if hora_col not in df_ml.columns:
        df_ml['HORA_SINTETICA'] = np.random.randint(0, 24, len(df_ml))
        hora_col = 'HORA_SINTETICA'
    
    if mes_col not in df_ml.columns:
        df_ml['MES_SINTETICO'] = np.random.randint(1, 13, len(df_ml))
        mes_col = 'MES_SINTETICO'
    
    features = [hora_col, mes_col, comuna_col] if comuna_col in df_ml.columns else [hora_col, mes_col]
    
    df_tipo = df_ml[features + [clase_col]].dropna()
    
    # Filtrar tipos comunes
    tipo_counts = df_tipo[clase_col].value_counts()
    tipos_comunes = tipo_counts[tipo_counts >= 10].index  # Reducido a 10 para más flexibilidad
    df_tipo = df_tipo[df_tipo[clase_col].isin(tipos_comunes)]
    
    if len(tipos_comunes) < 2:
        st.error("❌ Muy pocos tipos de accidente para modelar")
        return
    
    with st.spinner("Entrenando modelo de predicción..."):
        try:
            # Codificar variables
            le_tipo = LabelEncoder()
            df_tipo_encoded = df_tipo.copy()
            
            for feature in features:
                if df_tipo_encoded[feature].dtype == 'object':
                    le = LabelEncoder()
                    df_tipo_encoded[f'{feature}_encoded'] = le.fit_transform(df_tipo_encoded[feature].astype(str))
                else:
                    df_tipo_encoded[f'{feature}_encoded'] = df_tipo_encoded[feature]
            
            feature_cols = [f'{feat}_encoded' for feat in features]
            df_tipo_encoded['target_encoded'] = le_tipo.fit_transform(df_tipo_encoded[clase_col])
            
            # Características y target
            X = df_tipo_encoded[feature_cols]
            y = df_tipo_encoded['target_encoded']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluar
            y_pred = model.predict(X_test)
            
        except Exception as e:
            st.error(f"❌ Error entrenando modelo: {e}")
            return
    
       # REPORTE DE CLASIFICACIÓN MEJORADO PARA TIPO DE ACCIDENTE
    st.subheader("📊 Reporte de Clasificación - Tipo de Accidente")
    
    # Convertir reporte a DataFrame
    report_dict = classification_report(y_test, y_pred, target_names=le_tipo.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Crear pestañas para diferentes vistas
    tab1, tab2, tab3 = st.tabs(["📈 Vista Resumida", "📋 Vista Completa", "🎯 Análisis por Tipo"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🚦 Tipos de Accidente Analizados**")
            for i, tipo in enumerate(tipos_comunes):
                st.write(f"{i+1}. {tipo}")
            
            st.write("**📊 Distribución**")
            dist_df = df_tipo[clase_col].value_counts()
            st.bar_chart(dist_df)
        
        with col2:
            st.write("**📈 Métricas Generales**")
            accuracy = report_df.loc['accuracy', 'precision']
            macro_avg = report_df.loc['macro avg', 'precision']
            weighted_avg = report_df.loc['weighted avg', 'precision']
            
            st.metric("Precisión Global", f"{accuracy:.2%}")
            st.metric("Precisión Promedio", f"{macro_avg:.2%}")
            st.metric("Precisión Ponderada", f"{weighted_avg:.2%}")
            st.metric("Tipos de Accidentes", len(tipos_comunes))
            st.metric("Registros para Modelo", len(df_tipo))
        
        # Gráfico de métricas por tipo
        classes_df = report_df.iloc[:-3].copy()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(classes_df))
        width = 0.25
        
        ax.bar(x_pos - width, classes_df['precision'] * 100, width, label='Precisión', alpha=0.8, color='skyblue')
        ax.bar(x_pos, classes_df['recall'] * 100, width, label='Recall', alpha=0.8, color='lightcoral')
        ax.bar(x_pos + width, classes_df['f1-score'] * 100, width, label='F1-Score', alpha=0.8, color='lightgreen')
        
        ax.set_xlabel('Tipos de Accidente')
        ax.set_ylabel('Porcentaje (%)')
        ax.set_title('Métricas por Tipo de Accidente')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(classes_df.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with tab2:
        # Tabla completa con formato mejorado
        st.write("**📋 Reporte de Clasificación Completo**")
        
        # Formatear el DataFrame para mejor visualización
        display_df = report_df.copy()
        display_df['support'] = display_df['support'].astype(int)
        
        # Aplicar formato a las columnas numéricas
        numeric_cols = ['precision', 'recall', 'f1-score']
        for col in numeric_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        
        # Resaltar filas importantes
        def highlight_rows(row):
            if row.name == 'accuracy':
                return ['background-color: #AFD5FA'] * len(row)
            elif row.name in ['macro avg', 'weighted avg']:
                return ['background-color: #FA9B9B'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_rows, axis=1),
            width='stretch',
            height=400
        )
        
        # Leyenda
        st.caption("🎨 **Leyenda:** Azul = Precisión global, Rojo = Promedios")
    
    with tab3:
        # Análisis detallado por tipo de accidente
        st.write("**🎯 Análisis Detallado por Tipo de Accidente**")
        
        classes_data = report_df.iloc[:-3].copy()
        
        for tipo_accidente in classes_data.index:
            with st.expander(f"🚗 **{tipo_accidente}**", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    precision = classes_data.loc[tipo_accidente, 'precision']
                    st.metric("Precisión", f"{precision:.2%}")
                
                with col2:
                    recall = classes_data.loc[tipo_accidente, 'recall']
                    st.metric("Recall", f"{recall:.2%}")
                
                with col3:
                    f1 = classes_data.loc[tipo_accidente, 'f1-score']
                    st.metric("F1-Score", f"{f1:.2%}")
                
                with col4:
                    support = classes_data.loc[tipo_accidente, 'support']
                    st.metric("Muestras", support)
                
                # Explicación de métricas específica para tipos de accidente
                st.write("**📝 Interpretación:**")
                if precision > 0.7 and recall > 0.7:
                    st.success("✅ **Excelente predicción**: El modelo identifica bien este tipo de accidente")
                elif precision > 0.5 or recall > 0.5:
                    st.warning("⚠️ **Predicción moderada**: El modelo tiene cierta dificultad con este tipo")
                else:
                    st.error("❌ **Baja predicción**: El modelo no identifica bien este tipo de accidente")
                
                # Recomendación específica
                if support < 50:
                    st.info("💡 **Recomendación**: Pocas muestras para este tipo, considerar recolección de más datos")

    # MATRIZ DE CONFUSIÓN PARA TIPO DE ACCIDENTE
    st.subheader("🎯 Matriz de Confusión - Tipo de Accidente")
    
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Heatmap de la matriz de confusión
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Etiquetas
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=le_tipo.classes_,
               yticklabels=le_tipo.classes_,
               title='Matriz de Confusión - Tipo de Accidente',
               ylabel='Tipo Real',
               xlabel='Tipo Predicho')
        
        # Rotar etiquetas
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Anotaciones en las celdas
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=8)
        
        st.pyplot(fig)
    
    with col2:
        st.write("**📊 Estadísticas de la Matriz**")
        
        # Calcular métricas adicionales
        total = cm.sum()
        correctos = np.diag(cm).sum()
        incorrectos = total - correctos
        
        st.metric("Predicciones Correctas", f"{correctos} ({correctos/total:.1%})")
        st.metric("Predicciones Incorrectas", f"{incorrectos} ({incorrectos/total:.1%})")
        st.metric("Total de Muestras", total)
        
        # Tipo con mejor y peor precisión
        precision_por_tipo = np.diag(cm) / cm.sum(axis=0)
        recall_por_tipo = np.diag(cm) / cm.sum(axis=1)
        
        mejor_tipo_idx = np.argmax(precision_por_tipo)
        peor_tipo_idx = np.argmin(precision_por_tipo)
        
        st.write("---")
        st.write("**🏆 Tipos Destacados:**")
        st.success(f"**Mejor**: {le_tipo.classes_[mejor_tipo_idx]} ({precision_por_tipo[mejor_tipo_idx]:.1%})")
        st.error(f"**Peor**: {le_tipo.classes_[peor_tipo_idx]} ({precision_por_tipo[peor_tipo_idx]:.1%})")

    # IMPORTANCIA DE CARACTERÍSTICAS
    st.subheader("🔍 Importancia de Características")
    
    feat_importance = pd.DataFrame({
        'Característica': features,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    # Gráfico de importancia
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_importance, x='Importancia', y='Característica', ax=ax)
    ax.set_title('Importancia de Características - Predicción de Tipo de Accidente')
    ax.set_xlabel('Importancia')
    st.pyplot(fig)
    
    # Tabla de importancia
    st.dataframe(feat_importance)
    
    # RESUMEN EJECUTIVO
    st.subheader("📋 Resumen Ejecutivo - Predicción de Tipo")
    
    accuracy = accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📈 Calidad del Modelo**")
        st.metric("Precisión Global", f"{accuracy:.2%}")
        st.metric("Tipos Diferentes", len(tipos_comunes))
        st.metric("Balance de Datos", f"{(len(tipos_comunes) / len(df_tipo[clase_col].unique())):.1%}")
    
    with col2:
        st.write("**📊 Estadísticas**")
        st.metric("Muestras Entrenamiento", len(X_train))
        st.metric("Muestras Prueba", len(X_test))
        st.metric("Tamaño Dataset", len(df_tipo))
    
    with col3:
        st.write("**🎯 Aplicabilidad**")
        if accuracy > 0.7:
            st.success("✅ **Alta aplicabilidad**")
            st.write("Útil para predicción y análisis")
        elif accuracy > 0.5:
            st.warning("⚠️ **Aplicabilidad moderada**")
            st.write("Puede usarse con precaución")
        else:
            st.error("❌ **Baja aplicabilidad**")
            st.write("Mejorar modelo antes de usar")

def analisis_series_tiempo_ui(df):
    st.header("🟣 Análisis de Series de Tiempo")
    
    if df.empty:
        st.error("❌ No hay datos para analizar")
        return
    
    # Buscar columnas de fecha
    fecha_col = next((col for col in df.columns if 'FECHA' in col.upper()), None)
    if not fecha_col:
        st.error("❌ No se encontró columna de fecha")
        st.info("📋 Columnas disponibles:", list(df.columns))
        return
    
    st.success(f"✅ Columna de fecha encontrada: **{fecha_col}**")
    
    try:
        # Convertir a datetime
        df_temp = df.copy()
        df_temp['FECHA_DT'] = pd.to_datetime(df_temp[fecha_col])
        
        # Agregar por día
        daily_accidents = df_temp.groupby('FECHA_DT').size().reset_index(name='accidentes')
        daily_accidents = daily_accidents.set_index('FECHA_DT')
        
        # Rellenar fechas faltantes
        idx = pd.date_range(daily_accidents.index.min(), daily_accidents.index.max())
        daily_accidents = daily_accidents.reindex(idx, fill_value=0)
        
    except Exception as e:
        st.error(f"❌ Error procesando fechas: {e}")
        return
    
    # Métricas
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1]) 
    with col1:
        st.metric("Período Analizado", 
                 f"{daily_accidents.index.min().strftime('%Y-%m-%d')} a {daily_accidents.index.max().strftime('%Y-%m-%d')}")
    with col2:
        st.metric("Total Días", len(daily_accidents))
    with col3:
        st.metric("Accidentes Totales", daily_accidents['accidentes'].sum())
    with col4:
        st.metric("Promedio por Día", f"{daily_accidents['accidentes'].mean():.2f}")
    
    # Gráficos
    st.subheader("📈 Patrones Temporales")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Serie temporal completa
    axes[0, 0].plot(daily_accidents.index, daily_accidents['accidentes'], alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Evolución Diaria de Accidentes')
    axes[0, 0].set_ylabel('Número de Accidentes')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Patrón semanal
    daily_accidents['dia_semana'] = daily_accidents.index.dayofweek
    semanal = daily_accidents.groupby('dia_semana')['accidentes'].mean()
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    axes[0, 1].bar(dias, semanal, color='skyblue', alpha=0.7)
    axes[0, 1].set_title('Patrón Semanal - Promedio por Día')
    axes[0, 1].set_ylabel('Accidentes Promedio')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Patrón mensual
    daily_accidents['mes'] = daily_accidents.index.month
    mensual = daily_accidents.groupby('mes')['accidentes'].mean()
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    axes[1, 0].bar(range(1, 13), mensual, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Patrón Mensual - Promedio')
    axes[1, 0].set_xlabel('Mes')
    axes[1, 0].set_ylabel('Accidentes Promedio')
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set_xticklabels(meses[:len(mensual)])
    
    # Distribución horaria si existe
    hora_col = next((col for col in df.columns if 'HORA' in col.upper()), None)
    if hora_col and hora_col in df.columns:
        horario = df[hora_col].value_counts().sort_index()
        axes[1, 1].bar(horario.index, horario.values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Distribución Horaria')
        axes[1, 1].set_xlabel('Hora del Día')
        axes[1, 1].set_ylabel('Número de Accidentes')
    else:
        axes[1, 1].text(0.5, 0.5, 'Datos horarios\nno disponibles', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Distribución Horaria')
    
    st.pyplot(fig)
    
    # Top días con más accidentes
    st.subheader("📅 Top 10 Días con Más Accidentes")
    top_dias = daily_accidents.nlargest(10, 'accidentes')
    top_dias_formatted = top_dias.copy()
    top_dias_formatted.index = top_dias_formatted.index.strftime('%Y-%m-%d')
    st.dataframe(top_dias_formatted)

if __name__ == "__main__":
    main()