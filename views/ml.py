import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.naive_bayes import MultinomialNB
import warnings
import streamlit as st
import logging
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración visual
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================================
@dataclass
class ModelConfig:
    """Configuración centralizada para modelos ML"""
    n_estimators: int = 100
    test_size: float = 0.3
    random_state: int = 42
    min_samples: int = 50
    n_clusters: int = 3
    cv_folds: int = 5

CONFIG = ModelConfig()

# ============================================================================
# FUNCIONES UTILITARIAS MEJORADAS
# ============================================================================

def find_column(df, possible_names, case_sensitive=False):
    """
    Busca una columna por múltiples nombres posibles.
    
    Args:
        df: DataFrame
        possible_names: Lista de nombres posibles
        case_sensitive: Si la búsqueda distingue mayúsculas
    
    Returns:
        str: Nombre de la columna encontrada o None
    """
    if not case_sensitive:
        cols_upper = {col.upper(): col for col in df.columns}
        for name in possible_names:
            if name.upper() in cols_upper:
                logger.info(f"Columna encontrada: {cols_upper[name.upper()]}")
                return cols_upper[name.upper()]
    else:
        for name in possible_names:
            if name in df.columns:
                logger.info(f"Columna encontrada: {name}")
                return name
    
    logger.warning(f"No se encontró columna entre: {possible_names}")
    return None


def validate_dataframe(df, min_rows=50):
    """
    Valida que el DataFrame tenga datos suficientes.
    
    Args:
        df: DataFrame a validar
        min_rows: Mínimo de filas requeridas
    
    Returns:
        tuple: (bool, str) - (es_valido, mensaje)
    """
    if df is None or df.empty:
        return False, "DataFrame vacío o None"
    
    if len(df) < min_rows:
        return False, f"Datos insuficientes: {len(df)} < {min_rows} registros"
    
    return True, "DataFrame válido"


def analizar_desbalanceo(y, threshold=0.3):
    """
    Analiza desbalanceo en las clases del target.
    
    Args:
        y: Series o array con clases
        threshold: Umbral de ratio mínimo aceptable
    
    Returns:
        tuple: (desbalanceado, ratio)
    """
    counts = pd.Series(y).value_counts()
    ratio = counts.min() / counts.max()
    
    if ratio < threshold:
        st.warning(f"⚠️ Datos desbalanceados detectados (ratio: {ratio:.2%})")
        st.info("""
        **Recomendaciones:**
        - El modelo puede tener sesgo hacia la clase mayoritaria
        - Se usarán métricas balanceadas para mejor evaluación
        - Considerar técnicas de balanceo en versiones futuras
        """)
        return True, ratio
    
    return False, ratio


@st.cache_data
def preparar_datos_ml(_df, features, target_col):
    """
    Prepara datos para ML con caché para mejor performance.
    
    Args:
        _df: DataFrame (con _ para evitar hashing en Streamlit)
        features: Lista de características
        target_col: Columna objetivo
    
    Returns:
        DataFrame: Datos limpios listos para ML
    """
    df_ml = _df[features + [target_col]].dropna().copy()
    logger.info(f"Datos preparados: {len(df_ml)} registros, {len(features)} features")
    return df_ml


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    st.set_page_config(page_title="Análisis ML - Accidentes", layout="wide")
    st.title("🤖 Análisis de Machine Learning - Accidentes Viales")
    st.markdown("---")
    
    # Cargar datos desde session_state
    if 'complete_data' not in st.session_state or st.session_state.complete_data.empty:
        st.error("❌ No hay datos disponibles. Por favor, carga los datos en la página del Integrador primero.")
        return
    
    df = st.session_state.complete_data.copy()
    
    # Validar DataFrame
    is_valid, msg = validate_dataframe(df)
    if not is_valid:
        st.error(f"❌ {msg}")
        return
    
    # Mostrar información del dataset
    with st.expander("📊 Información del Dataset", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Registros", f"{len(df):,}")
        with col2:
            st.metric("Columnas", len(df.columns))
        with col3:
            completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
            st.metric("Completado", f"{completeness:.1%}")
        
        st.write("**Columnas disponibles:**", list(df.columns))
    
    # Selector de análisis
    st.sidebar.header("🔧 Configuración de Análisis")
    analysis_type = st.sidebar.selectbox(
        "Selecciona el tipo de análisis:",
        [
            "Predicción de Gravedad",
            "Clustering de Zonas de Riesgo",
            "Predicción de Tipo de Accidente",
            "Análisis de Series de Tiempo"
        ]
    )
    
    # Ejecutar análisis seleccionado
    try:
        if analysis_type == "Predicción de Gravedad":
            predecir_gravedad_accidente_ui(df)
        elif analysis_type == "Clustering de Zonas de Riesgo":
            clustering_zones_riesgo_ui(df)
        elif analysis_type == "Predicción de Tipo de Accidente":
            predecir_tipo_accidente_ui(df)
        elif analysis_type == "Análisis de Series de Tiempo":
            analisis_series_tiempo_ui(df)
    except Exception as e:
        logger.error(f"Error en análisis: {e}", exc_info=True)
        st.error(f"❌ Error ejecutando análisis: {e}")
        st.info("💡 Revisa los logs para más detalles")


# ============================================================================
# FUNCIÓN REUTILIZABLE: REPORTE DE CLASIFICACIÓN
# ============================================================================

def mostrar_reporte_clasificacion(y_test, y_pred, class_names, titulo="Reporte de Clasificación"):
    """
    Genera y muestra un reporte de clasificación completo y reutilizable.
    
    Args:
        y_test: Etiquetas reales
        y_pred: Predicciones
        class_names: Nombres de las clases
        titulo: Título del reporte
    """
    st.subheader(f"📊 {titulo}")
    
    # Generar reporte
    report_dict = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs([
        "📈 Vista Resumida",
        "📋 Vista Completa",
        "🎯 Análisis por Clase"
    ])
    
    with tab1:
        _mostrar_vista_resumida(report_df, class_names)
    
    with tab2:
        _mostrar_vista_completa(report_df)
    
    with tab3:
        _mostrar_analisis_detallado(report_df)


def _mostrar_vista_resumida(report_df, class_names):
    """Muestra vista resumida del reporte"""
    # Métricas generales
    st.write("**📊 Métricas Generales del Modelo**")
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
        st.metric("Clases", len(class_names))
    
    # Gráfico de métricas por clase
    classes_df = report_df.iloc[:-3].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(classes_df))
    width = 0.25
    
    ax.bar(x_pos - width, classes_df['precision'] * 100, width, 
           label='Precisión', alpha=0.8, color='skyblue')
    ax.bar(x_pos, classes_df['recall'] * 100, width,
           label='Recall', alpha=0.8, color='lightcoral')
    ax.bar(x_pos + width, classes_df['f1-score'] * 100, width,
           label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax.set_xlabel('Clases')
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title('Métricas por Clase')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # Importante: liberar memoria


def _mostrar_vista_completa(report_df):
    """Muestra vista completa del reporte"""
    st.write("**📋 Reporte Completo**")
    
    display_df = report_df.copy()
    display_df['support'] = display_df['support'].astype(int)
    
    # Formatear columnas numéricas
    numeric_cols = ['precision', 'recall', 'f1-score']
    for col in numeric_cols:
        display_df[col] = display_df[col].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A"
        )
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Botón de descarga
    csv = report_df.to_csv(index=True)
    st.download_button(
        label="📥 Descargar Reporte CSV",
        data=csv,
        file_name="reporte_clasificacion.csv",
        mime="text/csv"
    )


def _mostrar_analisis_detallado(report_df):
    """Muestra análisis detallado por clase"""
    st.write("**🎯 Análisis Detallado por Clase**")
    
    classes_data = report_df.iloc[:-3].copy()
    
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
                st.metric("Muestras", int(support))
            
            # Interpretación
            st.write("**🔍 Interpretación:**")
            if precision > 0.7 and recall > 0.7:
                st.success("✅ **Excelente desempeño**: El modelo predice bien esta clase")
            elif precision > 0.5 or recall > 0.5:
                st.warning("⚠️ **Desempeño moderado**: El modelo tiene dificultades con esta clase")
            else:
                st.error("❌ **Bajo desempeño**: El modelo no predice bien esta clase")
            
            if support < 50:
                st.info("💡 **Recomendación**: Pocas muestras, considerar recolección de más datos")


# ============================================================================
# FUNCIÓN REUTILIZABLE: MATRIZ DE CONFUSIÓN
# ============================================================================

def mostrar_matriz_confusion(y_test, y_pred, class_names, titulo="Matriz de Confusión"):
    """
    Genera y muestra matriz de confusión con estadísticas.
    
    Args:
        y_test: Etiquetas reales
        y_pred: Predicciones
        class_names: Nombres de las clases
        titulo: Título de la matriz
    """
    st.subheader(f"🎯 {titulo}")
    
    cm = confusion_matrix(y_test, y_pred)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=titulo,
            ylabel='Etiqueta Real',
            xlabel='Etiqueta Predicha'
        )
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Anotaciones
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.write("**📊 Estadísticas**")
        
        total = cm.sum()
        correctos = np.diag(cm).sum()
        incorrectos = total - correctos
        
        st.metric("Predicciones Correctas", f"{correctos} ({correctos/total:.1%})")
        st.metric("Predicciones Incorrectas", f"{incorrectos} ({incorrectos/total:.1%})")
        st.metric("Total de Muestras", int(total))
        
        # Mejor y peor clase
        precision_por_clase = np.diag(cm) / (cm.sum(axis=0) + 1e-10)
        mejor_idx = np.argmax(precision_por_clase)
        peor_idx = np.argmin(precision_por_clase)
        
        st.write("---")
        st.write("**🎭 Clases Destacadas:**")
        st.success(f"**Mejor**: {class_names[mejor_idx]} ({precision_por_clase[mejor_idx]:.1%})")
        st.error(f"**Peor**: {class_names[peor_idx]} ({precision_por_clase[peor_idx]:.1%})")


# ============================================================================
# ANÁLISIS 1: PREDICCIÓN DE GRAVEDAD
# ============================================================================

def predecir_gravedad_accidente_ui(df):
    """Análisis de predicción de gravedad de accidentes"""
    st.header("🔴 Predicción de Gravedad del Accidente")
    
    # Validar datos
    is_valid, msg = validate_dataframe(df)
    if not is_valid:
        st.error(f"❌ {msg}")
        return
    
    # Buscar columna de gravedad
    gravedad_col = find_column(
        df,
        ['gravedad', 'GRAVEDAD', 'GRAVEDAD_ACCIDENTE', 'severidad', 'SEVERIDAD']
    )
    
    if not gravedad_col:
        st.error("❌ No se encontró columna de gravedad en el dataset")
        st.info("🔍 Columnas disponibles: " + ", ".join(df.columns))
        return
    
    st.success(f"✅ Columna de gravedad encontrada: **{gravedad_col}**")
    
    # Buscar otras columnas
    hora_col = find_column(df, ['HORA', 'hora'])
    mes_col = find_column(df, ['MES', 'mes'])
    clase_col = find_column(df, ['CLASE_ACCIDENTE', 'CLASE', 'clase'])
    comuna_col = find_column(df, ['COMUNA', 'comuna'])
    
    # Crear features disponibles
    df_work = df.copy()
    features = []
    
    if hora_col:
        features.append(hora_col)
    else:
        df_work['HORA_SINTETICA'] = np.random.randint(0, 24, len(df_work))
        features.append('HORA_SINTETICA')
        st.info("ℹ️ Columna HORA no encontrada, usando valores sintéticos")
    
    if mes_col:
        features.append(mes_col)
    else:
        df_work['MES_SINTETICO'] = np.random.randint(1, 13, len(df_work))
        features.append('MES_SINTETICO')
        st.info("ℹ️ Columna MES no encontrada, usando valores sintéticos")
    
    if clase_col:
        features.append(clase_col)
    if comuna_col:
        features.append(comuna_col)
    
    st.info(f"🔍 Características utilizadas: {', '.join(features)}")
    
    # Preparar datos con caché
    df_ml = preparar_datos_ml(df_work, features, gravedad_col)
    
    if len(df_ml) < CONFIG.min_samples:
        st.error(f"❌ Datos insuficientes: {len(df_ml)} < {CONFIG.min_samples}")
        return
    
    # Mostrar distribución de gravedad
    st.subheader("📊 Distribución de Gravedad")
    gravedad_counts = df_ml[gravedad_col].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(gravedad_counts)
    with col2:
        st.dataframe(gravedad_counts, use_container_width=True)
    
    # Entrenar modelo
    with st.spinner("🔄 Entrenando modelo Random Forest..."):
        try:
            # Codificar variables
            le_dict = {}
            df_encoded = df_ml.copy()
            
            for feature in features:
                if df_encoded[feature].dtype == 'object':
                    le = LabelEncoder()
                    df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                    le_dict[feature] = le
            
            # Codificar target
            le_target = LabelEncoder()
            df_encoded['target_encoded'] = le_target.fit_transform(df_ml[gravedad_col])
            
            # Preparar X, y
            X = df_encoded[features]
            y = df_encoded['target_encoded']
            
            # Analizar desbalanceo
            analizar_desbalanceo(y)
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=CONFIG.test_size,
                random_state=CONFIG.random_state,
                stratify=y
            )
            
            # Entrenar
            model = RandomForestClassifier(
                n_estimators=CONFIG.n_estimators,
                random_state=CONFIG.random_state,
                n_jobs=-1  # Usar todos los cores
            )
            model.fit(X_train, y_train)
            
            # Predecir
            y_pred = model.predict(X_test)
            
            logger.info(f"Modelo entrenado: accuracy={accuracy_score(y_test, y_pred):.2%}")
            st.success("✅ Modelo entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}", exc_info=True)
            st.error(f"❌ Error entrenando modelo: {e}")
            return
    
    # Mostrar reportes usando funciones reutilizables
    mostrar_reporte_clasificacion(
        y_test, y_pred,
        le_target.classes_,
        "Reporte de Clasificación - Gravedad"
    )
    
    # Importancia de características
    st.subheader("🔍 Importancia de Características")
    feature_importance = pd.DataFrame({
        'Característica': features,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importancia', y='Característica', ax=ax)
    ax.set_title('Importancia de Características - Predicción de Gravedad')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.dataframe(feature_importance, use_container_width=True)
    
    # Matriz de confusión
    mostrar_matriz_confusion(
        y_test, y_pred,
        le_target.classes_,
        "Matriz de Confusión - Gravedad"
    )
    
    # Resumen ejecutivo
    _mostrar_resumen_ejecutivo(
        y_test, y_pred,
        len(X_train), len(X_test), len(df_ml),
        len(le_target.classes_)
    )


def _mostrar_resumen_ejecutivo(y_test, y_pred, n_train, n_test, n_total, n_classes):
    """Muestra resumen ejecutivo del modelo"""
    st.subheader("📋 Resumen Ejecutivo")
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📈 Métricas de Calidad**")
        st.metric("Precisión Global", f"{accuracy:.2%}")
        st.metric("Precisión Balanceada", f"{balanced_acc:.2%}")
        st.metric("Clases Modeladas", n_classes)
    
    with col2:
        st.write("**📊 Distribución de Datos**")
        st.metric("Muestras Entrenamiento", f"{n_train:,}")
        st.metric("Muestras Prueba", f"{n_test:,}")
        st.metric("Tamaño Total", f"{n_total:,}")
    
    with col3:
        st.write("**🎯 Evaluación**")
        if accuracy > 0.8:
            st.success("✅ **Modelo Excelente**")
            st.write("Puede usarse en producción")
        elif accuracy > 0.6:
            st.warning("⚠️ **Modelo Aceptable**")
            st.write("Considerar mejoras adicionales")
        else:
            st.error("❌ **Modelo Necesita Mejoras**")
            st.write("Revisar características y datos")


# ============================================================================
# ANÁLISIS 2: CLUSTERING DE ZONAS DE RIESGO
# ============================================================================

def clustering_zones_riesgo_ui(df):
    """Análisis de clustering de zonas de riesgo"""
    st.header("🟢 Clustering de Zonas de Riesgo")
    
    is_valid, msg = validate_dataframe(df)
    if not is_valid:
        st.error(f"❌ {msg}")
        return
    
    # Buscar columnas
    comuna_col = find_column(df, ['COMUNA', 'comuna'])
    hora_col = find_column(df, ['HORA', 'hora'])
    gravedad_col = find_column(df, ['GRAVEDAD', 'gravedad', 'GRAVEDAD_ACCIDENTE'])
    
    if not comuna_col:
        st.error("❌ No se encontró columna COMUNA")
        return
    
    st.info(f"🔍 Columnas identificadas: COMUNA='{comuna_col}', HORA='{hora_col}', GRAVEDAD='{gravedad_col}'")
    
    with st.spinner("🔄 Analizando zonas de riesgo..."):
        try:
            df_cluster = df.copy()
            
            # Crear hora sintética si no existe
            if not hora_col:
                df_cluster['HORA_SINTETICA'] = np.random.randint(0, 24, len(df_cluster))
                hora_col = 'HORA_SINTETICA'
            
            # Agrupar por comuna
            zonas_riesgo = df_cluster.groupby(comuna_col).agg({
                hora_col: ['count', 'mean']
            }).reset_index()
            
            zonas_riesgo.columns = [comuna_col, 'total_accidentes', 'hora_promedio']
            
            # Calcular tasa de heridos si existe gravedad
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
            
            kmeans = KMeans(n_clusters=CONFIG.n_clusters, random_state=CONFIG.random_state)
            clusters = kmeans.fit_predict(X_scaled)
            
            zonas_riesgo['cluster_riesgo'] = clusters
            zonas_riesgo['nivel_riesgo'] = zonas_riesgo['cluster_riesgo'].map({
                0: 'Bajo', 1: 'Medio', 2: 'Alto'
            })
            
            st.success("✅ Clustering completado")
            
        except Exception as e:
            logger.error(f"Error en clustering: {e}", exc_info=True)
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
        st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribución de Riesgos")
        riesgo_counts = zonas_riesgo['nivel_riesgo'].value_counts()
        st.bar_chart(riesgo_counts)
    
    # Visualización
    st.subheader("📊 Visualización de Clusters")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    scatter = axes[0].scatter(
        zonas_riesgo['total_accidentes'],
        zonas_riesgo['tasa_heridos'],
        c=zonas_riesgo['cluster_riesgo'],
        cmap='RdYlGn_r', s=100, alpha=0.7
    )
    axes[0].set_xlabel('Total Accidentes')
    axes[0].set_ylabel('Tasa de Heridos')
    axes[0].set_title('Clusters de Zonas de Riesgo')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0])
    
    # Top comunas
    top_comunas = zonas_riesgo.nlargest(10, 'total_accidentes')
    colors = top_comunas['cluster_riesgo'].map({0: 'green', 1: 'orange', 2: 'red'})
    axes[1].barh(top_comunas[comuna_col], top_comunas['total_accidentes'], color=colors)
    axes[1].set_title('Top 10 Comunas con Más Accidentes')
    axes[1].set_xlabel('Número de Accidentes')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Comunas de alto riesgo
    st.subheader("🔍 Comunas de Alto Riesgo")
    alto_riesgo = zonas_riesgo[zonas_riesgo['nivel_riesgo'] == 'Alto']
    if not alto_riesgo.empty:
        st.dataframe(
            alto_riesgo[[comuna_col, 'total_accidentes', 'tasa_heridos']].sort_values(
                'total_accidentes', ascending=False
            ),
            use_container_width=True
        )
        
        # Descargar datos
        csv = alto_riesgo.to_csv(index=False)
        st.download_button(
            label="📥 Descargar Comunas de Alto Riesgo",
            data=csv,
            file_name="comunas_alto_riesgo.csv",
            mime="text/csv"
        )
    else:
        st.info("✅ No se identificaron comunas de alto riesgo")


# ============================================================================
# ANÁLISIS 3: PREDICCIÓN DE TIPO DE ACCIDENTE
# ============================================================================

def predecir_tipo_accidente_ui(df):
    """Análisis de predicción de tipo de accidente"""
    st.header("🔵 Predicción de Tipo de Accidente")
    
    is_valid, msg = validate_dataframe(df)
    if not is_valid:
        st.error(f"❌ {msg}")
        return
    
    # Buscar columna de tipo de accidente
    clase_col = find_column(df, ['CLASE_ACCIDENTE', 'CLASE', 'clase', 'tipo_accidente'])
    
    if not clase_col:
        st.error("❌ No se encontró columna de tipo de accidente")
        st.info("🔍 Columnas disponibles: " + ", ".join(df.columns))
        return
    
    st.success(f"✅ Columna de tipo de accidente: **{clase_col}**")
    
    # Buscar otras columnas
    df_work = df.copy()
    features = []
    
    hora_col = find_column(df, ['HORA', 'hora'])
    if hora_col:
        features.append(hora_col)
    else:
        df_work['HORA_SINTETICA'] = np.random.randint(0, 24, len(df_work))
        features.append('HORA_SINTETICA')
        st.info("ℹ️ Columna HORA no encontrada, usando valores sintéticos")
    
    mes_col = find_column(df, ['MES', 'mes'])
    if mes_col:
        features.append(mes_col)
    else:
        df_work['MES_SINTETICO'] = np.random.randint(1, 13, len(df_work))
        features.append('MES_SINTETICO')
        st.info("ℹ️ Columna MES no encontrada, usando valores sintéticos")
    
    comuna_col = find_column(df, ['COMUNA', 'comuna'])
    if comuna_col:
        features.append(comuna_col)
    
    st.info(f"🔍 Características utilizadas: {', '.join(features)}")
    
    # Preparar datos
    df_tipo = preparar_datos_ml(df_work, features, clase_col)
    
    # Filtrar tipos comunes
    tipo_counts = df_tipo[clase_col].value_counts()
    tipos_comunes = tipo_counts[tipo_counts >= 10].index
    df_tipo = df_tipo[df_tipo[clase_col].isin(tipos_comunes)]
    
    if len(tipos_comunes) < 2:
        st.error("❌ Muy pocos tipos de accidente para modelar")
        st.info(f"Se requieren al menos 2 tipos con 10+ muestras cada uno")
        return
    
    st.info(f"📊 Tipos de accidente a modelar: {len(tipos_comunes)}")
    
    # Entrenar modelo
    with st.spinner("🔄 Entrenando modelo de predicción..."):
        try:
            # Codificar variables
            df_encoded = df_tipo.copy()
            le_dict = {}
            
            for feature in features:
                if df_encoded[feature].dtype == 'object':
                    le = LabelEncoder()
                    df_encoded[f'{feature}_encoded'] = le.fit_transform(
                        df_encoded[feature].astype(str)
                    )
                    le_dict[feature] = le
                else:
                    df_encoded[f'{feature}_encoded'] = df_encoded[feature]
            
            feature_cols = [f'{feat}_encoded' for feat in features]
            
            # Codificar target
            le_tipo = LabelEncoder()
            df_encoded['target_encoded'] = le_tipo.fit_transform(df_tipo[clase_col])
            
            # Preparar X, y
            X = df_encoded[feature_cols]
            y = df_encoded['target_encoded']
            
            # Analizar desbalanceo
            analizar_desbalanceo(y)
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=CONFIG.test_size,
                random_state=CONFIG.random_state,
                stratify=y
            )
            
            # Entrenar
            model = RandomForestClassifier(
                n_estimators=CONFIG.n_estimators,
                random_state=CONFIG.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predecir
            y_pred = model.predict(X_test)
            
            logger.info(f"Modelo tipo accidente entrenado: accuracy={accuracy_score(y_test, y_pred):.2%}")
            st.success("✅ Modelo entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}", exc_info=True)
            st.error(f"❌ Error entrenando modelo: {e}")
            return
    
    # Mostrar distribución de tipos
    st.subheader("📊 Distribución de Tipos de Accidente")
    col1, col2 = st.columns(2)
    
    with col1:
        dist_df = df_tipo[clase_col].value_counts()
        st.bar_chart(dist_df)
    
    with col2:
        st.write("**🚦 Tipos Analizados**")
        for i, tipo in enumerate(tipos_comunes[:10], 1):
            st.write(f"{i}. {tipo}")
        if len(tipos_comunes) > 10:
            st.write(f"... y {len(tipos_comunes) - 10} más")
    
    # Reportes usando funciones reutilizables
    mostrar_reporte_clasificacion(
        y_test, y_pred,
        le_tipo.classes_,
        "Reporte de Clasificación - Tipo de Accidente"
    )
    
    # Importancia de características
    st.subheader("🔍 Importancia de Características")
    feature_importance = pd.DataFrame({
        'Característica': features,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importancia', y='Característica', ax=ax)
    ax.set_title('Importancia de Características - Predicción de Tipo')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    st.dataframe(feature_importance, use_container_width=True)
    
    # Matriz de confusión
    mostrar_matriz_confusion(
        y_test, y_pred,
        le_tipo.classes_,
        "Matriz de Confusión - Tipo de Accidente"
    )
    
    # Resumen ejecutivo
    _mostrar_resumen_ejecutivo(
        y_test, y_pred,
        len(X_train), len(X_test), len(df_tipo),
        len(le_tipo.classes_)
    )


# ============================================================================
# ANÁLISIS 4: SERIES DE TIEMPO
# ============================================================================

def analisis_series_tiempo_ui(df):
    """Análisis de series de tiempo de accidentes"""
    st.header("🟣 Análisis de Series de Tiempo")
    
    is_valid, msg = validate_dataframe(df, min_rows=30)
    if not is_valid:
        st.error(f"❌ {msg}")
        return
    
    # Buscar columna de fecha
    fecha_col = find_column(df, ['FECHA', 'fecha', 'FECHA_ACCIDENTE', 'date'])
    
    if not fecha_col:
        st.error("❌ No se encontró columna de fecha")
        st.info("🔍 Columnas disponibles: " + ", ".join(df.columns))
        return
    
    st.success(f"✅ Columna de fecha encontrada: **{fecha_col}**")
    
    try:
        # Convertir a datetime
        df_temp = df.copy()
        df_temp['FECHA_DT'] = pd.to_datetime(df_temp[fecha_col], errors='coerce')
        
        # Eliminar fechas inválidas
        df_temp = df_temp.dropna(subset=['FECHA_DT'])
        
        if len(df_temp) == 0:
            st.error("❌ No se pudieron convertir las fechas")
            return
        
        # Agregar por día
        daily_accidents = df_temp.groupby('FECHA_DT').size().reset_index(name='accidentes')
        daily_accidents = daily_accidents.set_index('FECHA_DT')
        
        # Rellenar fechas faltantes
        idx = pd.date_range(daily_accidents.index.min(), daily_accidents.index.max())
        daily_accidents = daily_accidents.reindex(idx, fill_value=0)
        daily_accidents.index.name = 'FECHA_DT'
        
        logger.info(f"Serie temporal creada: {len(daily_accidents)} días")
        
    except Exception as e:
        logger.error(f"Error procesando fechas: {e}", exc_info=True)
        st.error(f"❌ Error procesando fechas: {e}")
        return
    
    # Métricas
    st.subheader("📊 Métricas Generales")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fecha_inicio = daily_accidents.index.min().strftime('%Y-%m-%d')
        fecha_fin = daily_accidents.index.max().strftime('%Y-%m-%d')
        st.metric("Período Analizado", f"{fecha_inicio}")
        st.caption(f"hasta {fecha_fin}")
    
    with col2:
        st.metric("Total Días", f"{len(daily_accidents):,}")
    
    with col3:
        st.metric("Accidentes Totales", f"{daily_accidents['accidentes'].sum():,}")
    
    with col4:
        promedio = daily_accidents['accidentes'].mean()
        st.metric("Promedio por Día", f"{promedio:.2f}")
    
    # Gráficos de patrones temporales
    st.subheader("📈 Patrones Temporales")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Serie temporal completa
    axes[0, 0].plot(daily_accidents.index, daily_accidents['accidentes'], 
                    alpha=0.7, linewidth=1, color='steelblue')
    axes[0, 0].set_title('Evolución Diaria de Accidentes')
    axes[0, 0].set_ylabel('Número de Accidentes')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Patrón semanal
    daily_accidents_copy = daily_accidents.copy()
    daily_accidents_copy['dia_semana'] = daily_accidents_copy.index.dayofweek
    semanal = daily_accidents_copy.groupby('dia_semana')['accidentes'].mean()
    
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    axes[0, 1].bar(range(len(semanal)), semanal.values, color='skyblue', alpha=0.7)
    axes[0, 1].set_title('Patrón Semanal - Promedio por Día')
    axes[0, 1].set_ylabel('Accidentes Promedio')
    axes[0, 1].set_xticks(range(len(semanal)))
    axes[0, 1].set_xticklabels(dias[:len(semanal)], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Patrón mensual
    daily_accidents_copy['mes'] = daily_accidents_copy.index.month
    mensual = daily_accidents_copy.groupby('mes')['accidentes'].mean()
    
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
             'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    axes[1, 0].bar(mensual.index, mensual.values, color='lightcoral', alpha=0.7)
    axes[1, 0].set_title('Patrón Mensual - Promedio')
    axes[1, 0].set_xlabel('Mes')
    axes[1, 0].set_ylabel('Accidentes Promedio')
    axes[1, 0].set_xticks(mensual.index)
    axes[1, 0].set_xticklabels([meses[i-1] for i in mensual.index], rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Distribución horaria (si existe)
    hora_col = find_column(df, ['HORA', 'hora'])
    if hora_col and hora_col in df.columns:
        horario = df[hora_col].value_counts().sort_index()
        axes[1, 1].bar(horario.index, horario.values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Distribución Horaria')
        axes[1, 1].set_xlabel('Hora del Día')
        axes[1, 1].set_ylabel('Número de Accidentes')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'Datos horarios\nno disponibles',
                       ha='center', va='center',
                       transform=axes[1, 1].transAxes,
                       fontsize=12, color='gray')
        axes[1, 1].set_title('Distribución Horaria')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # Top días con más accidentes
    st.subheader("📅 Top 10 Días con Más Accidentes")
    top_dias = daily_accidents.nlargest(10, 'accidentes')
    top_dias_formatted = top_dias.copy()
    top_dias_formatted.index = top_dias_formatted.index.strftime('%Y-%m-%d (%A)')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(top_dias_formatted, use_container_width=True)
    
    with col2:
        st.bar_chart(top_dias)
    
    # Estadísticas adicionales
    st.subheader("📊 Estadísticas Adicionales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📈 Tendencia**")
        st.metric("Máximo en un día", int(daily_accidents['accidentes'].max()))
        st.metric("Mínimo en un día", int(daily_accidents['accidentes'].min()))
        st.metric("Mediana", f"{daily_accidents['accidentes'].median():.1f}")
    
    with col2:
        st.write("**📉 Dispersión**")
        st.metric("Desviación Estándar", f"{daily_accidents['accidentes'].std():.2f}")
        st.metric("Varianza", f"{daily_accidents['accidentes'].var():.2f}")
    
    with col3:
        st.write("**🗓️ Periodo**")
        dias_totales = (daily_accidents.index.max() - daily_accidents.index.min()).days
        st.metric("Días en el dataset", dias_totales)
        st.metric("Meses aprox.", f"{dias_totales / 30:.1f}")
    
    # Descargar datos
    csv = daily_accidents.to_csv()
    st.download_button(
        label="📥 Descargar Serie Temporal",
        data=csv,
        file_name="serie_temporal_accidentes.csv",
        mime="text/csv"
    )


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    main()
