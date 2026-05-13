"""
🔍 App de Detección de Fraude en Seguros Vehiculares
Construida con Streamlit + Plotly + Scikit-Learn / XGBoost
Ejecutar:  streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------
# Configuración de la página
# --------------------------------------------------------------
st.set_page_config(page_title="Fraud Detection", page_icon="🔍", layout="wide")
st.title("🔍 Detección de Fraude en Seguros Vehiculares")
st.markdown(
    "Aplicación académica — Universidad de Medellín — Proyecto 1 | "
    "Profesor: David Palacio Jimenez | "
    "Estudiantes: Andres Felipe Londoño Ocampo · Jhonatan Caro Atehortúa · "
    "Paulina Perez Ramirez · Victor Manuel Galeano Alvarez"
)

# --------------------------------------------------------------
# Mapeos ordinales
# --------------------------------------------------------------
ORDINAL_MAPPINGS = {
    'VehiclePrice': {
        'less than 20000': 0, '20000 to 29000': 1, '30000 to 39000': 2,
        '40000 to 59000': 3, '60000 to 69000': 4, 'more than 69000': 5
    },
    'Days_Policy_Accident': {
        'none': 0, '1 to 7': 1, '8 to 15': 2, '15 to 30': 3, 'more than 30': 4
    },
    'Days_Policy_Claim': {
        'none': 0, '8 to 15': 1, '15 to 30': 2, 'more than 30': 3
    },
    'PastNumberOfClaims': {
        'none': 0, '1': 1, '2 to 4': 2, 'more than 4': 3
    },
    'AgeOfVehicle': {
        'new': 0, '2 years': 1, '3 years': 2, '4 years': 3,
        '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7
    },
    'NumberOfSuppliments': {
        'none': 0, '1 to 2': 1, '3 to 5': 2, 'more than 5': 3
    },
    'AddressChange_Claim': {
        'no change': 0, 'under 6 months': 1, '1 year': 2,
        '2 to 3 years': 3, '4 to 8 years': 4
    },
    'NumberOfCars': {
        '1 vehicle': 1, '2 vehicles': 2, '3 to 4': 3,
        '5 to 8': 4, 'more than 8': 5
    }
}

FILTROS_EDA = {
    'Month':              'Mes del accidente',
    'DayOfWeek':          'Día de la semana del accidente',
    'MonthClaimed':       'Mes de la reclamación',
    'DayOfWeekClaimed':   'Día semana reclamación',
    'Make':               'Marca del vehículo',
    'AccidentArea':       'Zona del accidente',
    'Sex':                'Sexo del asegurado',
    'MaritalStatus':      'Estado civil',
    'Fault':              'Responsable del siniestro',
    'PolicyType':         'Tipo de póliza',
    'VehicleCategory':    'Categoría del vehículo',
    'PoliceReportFiled':  'Reporte policial',
    'WitnessPresent':     'Testigos presentes',
    'AgentType':          'Tipo de agente',
    'BasePolicy':         'Póliza base',
}

# --------------------------------------------------------------
# Carga cacheada
# --------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    import os
    # 1. Buscar la metadata (intenta en raíz y luego en models/)
    meta_path = "model_metadata.pkl" if os.path.exists("model_metadata.pkl") else "models/model_metadata.pkl"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No se encontró model_metadata.pkl en la raíz ni en models/")
    
    meta = joblib.load(meta_path)
    models_dict = {}
    
    # 2. Definir dónde buscar los modelos individuales
    folder = "models" if os.path.isdir("models") else "."
    
    for name in meta.get('all_model_names', []):
        safe_name = name.replace(' ', '_').replace('+', 'plus')
        path = os.path.join(folder, f"{safe_name}.pkl")
        if os.path.exists(path):
            models_dict[name] = joblib.load(path)
    
    # 3. Fallback: Si no encontró los 4, cargar al menos el "mejor modelo"
    if not models_dict:
        best_path = "best_fraud_model.pkl" if os.path.exists("best_fraud_model.pkl") else "models/best_fraud_model.pkl"
        if os.path.exists(best_path):
            models_dict[meta['best_model_name']] = joblib.load(best_path)
    
    return models_dict, meta

@st.cache_data
def load_data():
    return pd.read_csv("fraud_clean.csv")

@st.cache_data
def compute_summary_metrics(_models_dict):
    """Calcula métricas de los modelos sobre el set de prueba."""
    df_local = pd.read_csv("fraud_clean.csv")
    X = df_local.drop(columns=['FraudFound_P'])
    y = df_local['FraudFound_P']
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    rows = []
    for name, m in _models_dict.items():
        y_pred  = m.predict(X_te)
        y_proba = m.predict_proba(X_te)[:, 1]
        rows.append({
            'Modelo':    name,
            'Accuracy':  accuracy_score(y_te, y_pred),
            'Precision': precision_score(y_te, y_pred, zero_division=0),
            'Recall':    recall_score(y_te, y_pred),
            'F1':        f1_score(y_te, y_pred),
            'ROC-AUC':   roc_auc_score(y_te, y_proba),
            'PR-AUC':    average_precision_score(y_te, y_proba),
        })
    return pd.DataFrame(rows)

models_dict, meta = load_artifacts()
df = load_data()
all_model_names = list(models_dict.keys())
best_model_name = meta['best_model_name']

# --------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------
st.sidebar.title("📋 Menú")
page = st.sidebar.radio("Navegar a:", [
    "🏠 Inicio",
    "📊 EDA Interactivo",
    "🎯 Predicción",
    "📈 Desempeño del modelo"
])
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Modelo activo")
selected_model_name = st.sidebar.selectbox(
    "Selecciona un modelo:",
    all_model_names,
    index=all_model_names.index(best_model_name) if best_model_name in all_model_names else 0
)
selected_model = models_dict[selected_model_name]
if selected_model_name == best_model_name:
    st.sidebar.success(f"⭐ Mejor modelo: {selected_model_name}")
else:
    st.sidebar.info(f"Modelo: {selected_model_name}")

# --------------------------------------------------------------
# Página 1 – Inicio (RESUMEN EJECUTIVO)
# --------------------------------------------------------------
if page == "🏠 Inicio":
    # Banner principal
    st.header("📋 Resumen Ejecutivo del Proyecto")

    # ============ KPIs principales ============
    metrics_df = compute_summary_metrics(models_dict)
    best_pr_auc = metrics_df.loc[metrics_df['Modelo'] == best_model_name, 'PR-AUC'].values[0] if best_model_name in metrics_df['Modelo'].values else metrics_df['PR-AUC'].max()
    best_roc_auc = metrics_df.loc[metrics_df['Modelo'] == best_model_name, 'ROC-AUC'].values[0] if best_model_name in metrics_df['Modelo'].values else metrics_df['ROC-AUC'].max()
    best_recall = metrics_df.loc[metrics_df['Modelo'] == best_model_name, 'Recall'].values[0] if best_model_name in metrics_df['Modelo'].values else metrics_df['Recall'].max()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Reclamaciones", f"{len(df):,}")
    k2.metric("Tasa de fraude", f"{df['FraudFound_P'].mean()*100:.2f}%")
    k3.metric("Variables", df.shape[1] - 1)
    k4.metric("Modelos", len(all_model_names))
    k5.metric("Mejor PR-AUC", f"{best_pr_auc:.3f}")
    k6.metric("Mejor Recall", f"{best_recall:.3f}")

    st.markdown("---")

    # ============ CONTEXTO Y PROBLEMÁTICA ============
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🎯 Resumen del Proyecto")
        st.markdown("""
        Análisis y modelado predictivo para la **detección automática de fraude**
        en reclamaciones de seguros vehiculares, utilizando el dataset
        *Vehicle Insurance Claim Fraud Detection* (Oracle/Kaggle) con
        15,420 registros del período 1994-1996.

        Se aplicaron técnicas de **EDA**, **preprocesamiento con
        Pipelines de Scikit-Learn** y se entrenaron **4 modelos de clasificación**
        comparándolos sobre métricas adecuadas para el desbalance de clases.
        """)

        st.subheader("⚠️ Problemática")
        st.markdown("""
        El fraude en seguros vehiculares representa **pérdidas multimillonarias**
        para la industria aseguradora a nivel global y encarece indirectamente
        las primas de los asegurados honestos.

        La detección manual es **lenta, costosa y propensa a errores humanos**,
        lo que hace inviable revisar el 100% de las reclamaciones que recibe
        una compañía aseguradora.
        """)

    with col_b:
        st.subheader("💼 Contexto de Negocio")
        st.markdown("""
        La aseguradora necesita un **sistema de soporte a la decisión** que asigne
        una *probabilidad de fraude* a cada reclamación nueva, permitiendo a los
        analistas **priorizar la revisión** de los casos de mayor riesgo.

        Un modelo predictivo bien calibrado:
        - **Reduce costos operativos** (menos casos a revisar manualmente)
        - **Aumenta la tasa de detección** sin generar fricción con clientes legítimos
        - **Genera trazabilidad** para auditorías y cumplimiento regulatorio
        """)

        # Gráfico de torta del desbalance
        fraud_counts = df['FraudFound_P'].value_counts()
        fig_pie = px.pie(
            values=fraud_counts.values,
            names=['Legítimo', 'Fraude'],
            title='Distribución de la variable objetivo',
            color_discrete_sequence=['#2E86AB', '#E63946'],
            hole=0.5
        )
        fig_pie.update_traces(textinfo='percent+label+value')
        fig_pie.update_layout(
            height=350,
            margin=dict(t=80, b=20, l=20, r=20),
            title_y=0.98,
            legend=dict(orientation='v', yanchor='middle', y=0.5, xanchor='left', x=1.05)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ============ METODOLOGÍA ============
    st.subheader("🔬 Metodología aplicada")
    met_cols = st.columns(5)
    with met_cols[0]:
        st.markdown("**1️⃣ Comprensión**")
        st.caption("Definición del problema, exploración inicial del dataset y diccionario de variables.")
    with met_cols[1]:
        st.markdown("**2️⃣ Preprocesamiento**")
        st.caption("Tratamiento de datos faltantes, codificación ordinal y exclusión de variables (Year, AgeOfPolicyHolder, PolicyNumber).")
    with met_cols[2]:
        st.markdown("**3️⃣ EDA**")
        st.caption("Análisis univariado y bivariado, correlaciones, tasas de fraude por categoría.")
    with met_cols[3]:
        st.markdown("**4️⃣ Modelado**")
        st.caption("Pipelines con StandardScaler, OneHotEncoder y 4 modelos comparados.")
    with met_cols[4]:
        st.markdown("**5️⃣ Interpretación**")
        st.caption("Coeficientes, importancias, permutation importance y SHAP values.")


    st.markdown("---")

    # ============ MEJOR MODELO ============
    st.subheader("🏆 Mejor modelo seleccionado")

    # Métricas completas del mejor modelo
    best_row = metrics_df[metrics_df['Modelo'] == best_model_name].iloc[0]

    mod_col1, mod_col2 = st.columns([1, 2])

    with mod_col1:
        st.markdown(f"""
        ### 🥇 {best_model_name}

        **Modelo seleccionado** tras comparar 4 modelos
        candidatos sobre el conjunto de prueba.

        **Justificación técnica:**
        El modelo fue seleccionado tras una comparación
        sistemática contra Regresión Logística, Random Forest
        y Logística + SMOTE, evaluados sobre el mismo conjunto
        de prueba con métricas apropiadas para datos
        desbalanceados. El modelo obtuvo el mayor PR-AUC,
        métrica más fiable que accuracy o ROC-AUC en
        escenarios con clases asimétricas como el nuestro.

        **Métricas en el set de prueba:**
        """)

        c1, c2 = st.columns(2)
        c1.metric("PR-AUC", f"{best_row['PR-AUC']:.4f}")
        c2.metric("ROC-AUC", f"{best_row['ROC-AUC']:.4f}")
        c3, c4 = st.columns(2)
        c3.metric("Recall", f"{best_row['Recall']:.4f}")
        c4.metric("F1-Score", f"{best_row['F1']:.4f}")

    with mod_col2:
        # Gráfico de métricas SOLO del mejor modelo
        best_metrics = pd.DataFrame({
            'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'],
            'Valor': [best_row['Accuracy'], best_row['Precision'], best_row['Recall'],
                      best_row['F1'], best_row['ROC-AUC'], best_row['PR-AUC']]
        })
        fig = px.bar(
            best_metrics, x='Métrica', y='Valor',
            text_auto='.3f',
            title=f'Métricas del modelo ganador — {best_model_name}',
            color='Valor',
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=380, yaxis_range=[0, 1], coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.info("💡 Para ver la comparativa con los otros 3 modelos candidatos, ve a la página **📈 Desempeño del modelo**.")

    st.markdown("---")

    # ============ TOP VARIABLES PREDICTORAS ============
    st.subheader("⭐ Top variables predictoras (ranking del modelo ganador)")

    # Tabla descriptiva de top variables
    top_vars_data = pd.DataFrame({
        'Variable': ['Fault', 'BasePolicy', 'PastNumberOfClaims', 'AddressChange_Claim',
                     'PolicyType', 'VehiclePrice', 'AgeOfVehicle', 'Deductible'],
        'Tipo': ['Categórica', 'Categórica', 'Ordinal', 'Ordinal',
                 'Categórica', 'Ordinal', 'Ordinal', 'Continua'],
        'Importancia': ['Muy alta', 'Muy alta', 'Alta', 'Alta',
                        'Media-alta', 'Media', 'Media', 'Media'],
        'Interpretación': [
            'Responsable del siniestro (Policy Holder = mayor riesgo)',
            'Tipo de póliza base (All Perils/Collision = mayor riesgo)',
            'Reclamaciones previas del asegurado',
            'Cambio de dirección reciente antes de la reclamación',
            'Combinación de tipo de póliza y categoría',
            'Rango de precio del vehículo',
            'Antigüedad del vehículo',
            'Monto del deducible de la póliza'
        ]
    })
    st.dataframe(top_vars_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ============ APLICACIÓN PRÁCTICA ============
    st.subheader("💡 Aplicación práctica del modelo")

    app_cols = st.columns(3)
    with app_cols[0]:
        st.info("""
        **🟢 Probabilidad < 30%**

        **Riesgo bajo.** Procesamiento automático
        sin revisión adicional. Reduce carga
        operativa al equipo.
        """)
    with app_cols[1]:
        st.warning("""
        **🟡 Probabilidad 30% – 60%**

        **Riesgo medio.** Revisión estándar por
        analista. Solicitar documentación
        complementaria si aplica.
        """)
    with app_cols[2]:
        st.error("""
        **🔴 Probabilidad > 60%**

        **Riesgo alto.** Investigación
        profunda. Asignar a analista senior y
        considerar peritaje adicional.
        """)

    st.markdown("---")

    # ============ NAVEGACIÓN ============
    st.subheader("🧭 Cómo navegar esta aplicación")
    nav_cols = st.columns(3)
    with nav_cols[0]:
        st.markdown("""
        **📊 EDA Interactivo**

        Explora el dataset filtrando por
        las 15 variables categóricas. Visualiza
        tasas de fraude por cualquier segmento.
        """)
    with nav_cols[1]:
        st.markdown("""
        **🎯 Predicción**

        Ingresa los datos de una reclamación
        y obtén la probabilidad de fraude
        comparada entre los 4 modelos.
        """)
    with nav_cols[2]:
        st.markdown("""
        **📈 Desempeño**

        Tabla comparativa de métricas con el
        mejor modelo resaltado, curvas ROC/PR
        y matrices de confusión.
        """)


# --------------------------------------------------------------
# Página 2 – EDA Interactivo
# --------------------------------------------------------------
elif page == "📊 EDA Interactivo":
    st.header("📊 Análisis exploratorio interactivo")
    st.markdown(f"**{len(FILTROS_EDA)} filtros disponibles** — selecciona los valores que quieras incluir.")

    btn_col1, btn_col2, _ = st.columns([1, 1, 6])
    select_all = btn_col1.button("✅ Seleccionar todo")
    clear_all  = btn_col2.button("❌ Limpiar todo")
    st.markdown("---")

    selections = {}
    cols = st.columns(3)
    for i, (col_name, label) in enumerate(FILTROS_EDA.items()):
        opts = sorted(df[col_name].dropna().unique().tolist())
        if select_all:
            default = opts
        elif clear_all:
            default = []
        else:
            default = opts
        with cols[i % 3]:
            selections[col_name] = st.multiselect(
                label, opts, default=default, key=f"filter_{col_name}"
            )

    st.markdown("---")

    mask = pd.Series([True] * len(df), index=df.index)
    for col_name, selected in selections.items():
        if selected:
            mask = mask & df[col_name].isin(selected)
    df_f = df[mask]

    if len(df_f) == 0:
        st.warning("No hay datos con los filtros seleccionados. Ajusta los filtros.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Filas filtradas", f"{len(df_f):,}")
        m2.metric("Tasa de fraude", f"{df_f['FraudFound_P'].mean()*100:.2f}%")
        m3.metric("% del total", f"{len(df_f)/len(df)*100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df_f, x='Age', color='FraudFound_P',
                barmode='overlay', nbins=40, opacity=0.7,
                color_discrete_map={0: '#2E86AB', 1: '#E63946'},
                title='Distribución de edad por clase'
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            rate = (
                df_f.groupby('Make')['FraudFound_P']
                .mean()
                .sort_values(ascending=False) * 100
            )
            fig = px.bar(
                x=rate.index, y=rate.values,
                title='Tasa de fraude (%) por marca',
                labels={'x': 'Marca', 'y': 'Tasa fraude (%)'}
            )
            fig.add_hline(
                y=df['FraudFound_P'].mean() * 100,
                line_dash='dash', line_color='red',
                annotation_text='Promedio global'
            )
            st.plotly_chart(fig, use_container_width=True)

        if df_f['BasePolicy'].nunique() > 0 and df_f['Fault'].nunique() > 0:
            st.subheader("Tasa de fraude por BasePolicy × Fault")
            pivot = df_f.pivot_table(
                index='BasePolicy', columns='Fault',
                values='FraudFound_P', aggfunc='mean'
            ).round(4) * 100
            fig = px.imshow(
                pivot, text_auto='.2f', color_continuous_scale='Reds',
                title='Tasa de fraude (%) — BasePolicy × Fault'
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Tasa de fraude por variable")
        vars_to_plot = [c for c in FILTROS_EDA if df_f[c].nunique() > 1]
        for i in range(0, len(vars_to_plot), 2):
            row_cols = st.columns(2)
            for j, col_name in enumerate(vars_to_plot[i:i+2]):
                label = FILTROS_EDA[col_name]
                tasa = (
                    df_f.groupby(col_name)['FraudFound_P']
                    .mean()
                    .sort_values(ascending=False) * 100
                ).reset_index()
                tasa.columns = [col_name, 'Tasa fraude (%)']
                fig = px.bar(
                    tasa, x=col_name, y='Tasa fraude (%)',
                    title=f'Tasa por {label}',
                    color='Tasa fraude (%)',
                    color_continuous_scale='Reds'
                )
                fig.add_hline(
                    y=df['FraudFound_P'].mean() * 100,
                    line_dash='dash', line_color='gray',
                    annotation_text='Promedio'
                )
                with row_cols[j]:
                    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------
# Página 3 – Predicción individual
# --------------------------------------------------------------
elif page == "🎯 Predicción":
    st.header(f"🎯 Predicción individual — Modelo: {selected_model_name}")
    st.markdown("Ingresa los datos de la reclamación:")

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            Month = st.selectbox("Mes accidente", [
                'Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec'
            ])
            DayOfWeek = st.selectbox("Día semana accidente", [
                'Monday','Tuesday','Wednesday','Thursday',
                'Friday','Saturday','Sunday'
            ])
            Make = st.selectbox("Marca", sorted(df['Make'].unique()))
            AccidentArea = st.selectbox("Zona accidente", ['Urban', 'Rural'])
            Sex = st.selectbox("Sexo", ['Male', 'Female'])
            MaritalStatus = st.selectbox("Estado civil", [
                'Single', 'Married', 'Widow', 'Divorced'
            ])
            Age = st.slider("Edad del asegurado", 16, 80, 35)

        with c2:
            Fault = st.selectbox("Responsable", ['Policy Holder', 'Third Party'])
            PolicyType = st.selectbox("Tipo de póliza", sorted(df['PolicyType'].unique()))
            VehicleCategory = st.selectbox("Categoría vehículo", ['Sport', 'Sedan', 'Utility'])
            VehiclePrice = st.selectbox(
                "Rango precio vehículo",
                list(ORDINAL_MAPPINGS['VehiclePrice'].keys())
            )
            BasePolicy = st.selectbox("BasePolicy", ['Liability', 'Collision', 'All Perils'])
            Deductible = st.select_slider(
                "Deducible ($)", options=[300, 400, 500, 700], value=400
            )
            DriverRating = st.slider("DriverRating (1-4)", 1, 4, 2)

        with c3:
            Days_Policy_Accident = st.selectbox(
                "Días póliza-accidente",
                list(ORDINAL_MAPPINGS['Days_Policy_Accident'].keys())
            )
            Days_Policy_Claim = st.selectbox(
                "Días póliza-reclamación",
                list(ORDINAL_MAPPINGS['Days_Policy_Claim'].keys())
            )
            PastNumberOfClaims = st.selectbox(
                "Reclamaciones previas",
                list(ORDINAL_MAPPINGS['PastNumberOfClaims'].keys())
            )
            AgeOfVehicle = st.selectbox(
                "Antigüedad vehículo",
                list(ORDINAL_MAPPINGS['AgeOfVehicle'].keys())
            )
            PoliceReportFiled = st.selectbox("Reporte policial", ['Yes', 'No'])
            WitnessPresent = st.selectbox("Testigos presentes", ['Yes', 'No'])

        c4, c5 = st.columns(2)
        with c4:
            AgentType = st.selectbox("Tipo de agente", ['Internal', 'External'])
            NumberOfSuppliments = st.selectbox(
                "Suplementos a la reclamación",
                list(ORDINAL_MAPPINGS['NumberOfSuppliments'].keys())
            )
        with c5:
            AddressChange_Claim = st.selectbox(
                "Cambio de dirección",
                list(ORDINAL_MAPPINGS['AddressChange_Claim'].keys())
            )
            NumberOfCars = st.selectbox(
                "Número de vehículos",
                list(ORDINAL_MAPPINGS['NumberOfCars'].keys())
            )

        submitted = st.form_submit_button("🚀 Predecir", type="primary")

    if submitted:
        new = {
            'Month': Month, 'WeekOfMonth': 3, 'DayOfWeek': DayOfWeek,
            'Make': Make, 'AccidentArea': AccidentArea,
            'DayOfWeekClaimed': DayOfWeek, 'MonthClaimed': Month,
            'WeekOfMonthClaimed': 3, 'Sex': Sex,
            'MaritalStatus': MaritalStatus, 'Age': Age,
            'Fault': Fault, 'PolicyType': PolicyType,
            'VehicleCategory': VehicleCategory,
            'VehiclePrice': ORDINAL_MAPPINGS['VehiclePrice'][VehiclePrice],
            'RepNumber': 12,
            'Deductible': Deductible,
            'DriverRating': DriverRating,
            'Days_Policy_Accident': ORDINAL_MAPPINGS['Days_Policy_Accident'][Days_Policy_Accident],
            'Days_Policy_Claim': ORDINAL_MAPPINGS['Days_Policy_Claim'][Days_Policy_Claim],
            'PastNumberOfClaims': ORDINAL_MAPPINGS['PastNumberOfClaims'][PastNumberOfClaims],
            'AgeOfVehicle': ORDINAL_MAPPINGS['AgeOfVehicle'][AgeOfVehicle],
            'PoliceReportFiled': PoliceReportFiled,
            'WitnessPresent': WitnessPresent,
            'AgentType': AgentType,
            'NumberOfSuppliments': ORDINAL_MAPPINGS['NumberOfSuppliments'][NumberOfSuppliments],
            'AddressChange_Claim': ORDINAL_MAPPINGS['AddressChange_Claim'][AddressChange_Claim],
            'NumberOfCars': ORDINAL_MAPPINGS['NumberOfCars'][NumberOfCars],
            'BasePolicy': BasePolicy
        }
        X_new = pd.DataFrame([new])

        try:
            prob = selected_model.predict_proba(X_new)[0, 1]
            pred = int(prob >= 0.5)

            cA, cB = st.columns(2)
            cA.metric("Probabilidad de fraude", f"{prob*100:.2f}%")
            cB.metric("Predicción", "🚨 FRAUDE" if pred == 1 else "✅ Legítima")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': f"Riesgo de fraude (%) — {selected_model_name}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#E63946" if prob > 0.5 else "#2E86AB"},
                    'steps': [
                        {'range': [0, 30],   'color': '#A8E6A1'},
                        {'range': [30, 60],  'color': '#FFE066'},
                        {'range': [60, 100], 'color': '#FFB3B3'}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔍 Comparativa con los 4 modelos")
            comp_data = []
            for name, m in models_dict.items():
                p = m.predict_proba(X_new)[0, 1]
                comp_data.append({
                    'Modelo': name + (' ⭐' if name == best_model_name else ''),
                    'Probabilidad fraude (%)': round(p * 100, 2),
                    'Predicción': 'FRAUDE' if p >= 0.5 else 'Legítima'
                })
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            risk_factors = []
            if Fault == 'Policy Holder':
                risk_factors.append("El asegurado es el responsable del siniestro")
            if BasePolicy in ['Collision', 'All Perils']:
                risk_factors.append(f"Póliza tipo {BasePolicy} con alta incidencia de fraude")
            if PastNumberOfClaims in ['2 to 4', 'more than 4']:
                risk_factors.append("Historial de múltiples reclamaciones previas")
            if AddressChange_Claim in ['under 6 months', '1 year']:
                risk_factors.append("Cambio de dirección reciente antes de la reclamación")
            if risk_factors:
                msg = "**Factores de riesgo detectados:**\n- " + "\n- ".join(risk_factors)
                st.warning(msg)

        except Exception as e:
            st.error(f"Error en la predicción: {e}")

# --------------------------------------------------------------
# Página 4 – Desempeño del modelo
# --------------------------------------------------------------
else:
    st.header("📈 Desempeño y comparación de los 4 modelos")
    st.markdown("Métricas calculadas sobre el mismo conjunto de prueba (20% del dataset, estratificado).")

    X = df.drop(columns=['FraudFound_P'])
    y = df['FraudFound_P']
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    @st.cache_data
    def compute_all_metrics(_models_dict, _X_te, _y_te):
        rows = []
        curves = {}
        for name, m in _models_dict.items():
            y_pred  = m.predict(_X_te)
            y_proba = m.predict_proba(_X_te)[:, 1]
            rows.append({
                'Modelo':    name,
                'Accuracy':  accuracy_score(_y_te, y_pred),
                'Precision': precision_score(_y_te, y_pred, zero_division=0),
                'Recall':    recall_score(_y_te, y_pred),
                'F1':        f1_score(_y_te, y_pred),
                'ROC-AUC':   roc_auc_score(_y_te, y_proba),
                'PR-AUC':    average_precision_score(_y_te, y_proba),
            })
            fpr, tpr, _ = roc_curve(_y_te, y_proba)
            prec, rec, _ = precision_recall_curve(_y_te, y_proba)
            curves[name] = {
                'fpr': fpr, 'tpr': tpr, 'precision': prec, 'recall': rec,
                'roc_auc': roc_auc_score(_y_te, y_proba),
                'pr_auc': average_precision_score(_y_te, y_proba),
                'cm': confusion_matrix(_y_te, y_pred)
            }
        return pd.DataFrame(rows), curves

    metrics_df, curves = compute_all_metrics(models_dict, X_te, y_te)

    best_idx = metrics_df['PR-AUC'].idxmax()
    best_name_auto = metrics_df.loc[best_idx, 'Modelo']

    st.success(
        f"⭐ **Mejor modelo (por PR-AUC):** {best_name_auto} "
        f"con PR-AUC = {metrics_df.loc[best_idx, 'PR-AUC']:.4f}"
    )

    st.subheader("📊 Tabla comparativa de métricas")

    def highlight_best(row):
        if row['Modelo'] == best_name_auto:
            return ['background-color: #C6EFCE; color: #006100; font-weight: bold'] * len(row)
        return [''] * len(row)

    def highlight_max(s):
        if s.dtype.kind in 'biufc':
            is_max = s == s.max()
            return ['background-color: #FFEB9C; font-weight: bold' if v else '' for v in is_max]
        return [''] * len(s)

    styled = (
        metrics_df.style
        .apply(highlight_best, axis=1)
        .apply(highlight_max, subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'])
        .format({
            'Accuracy':  '{:.4f}',
            'Precision': '{:.4f}',
            'Recall':    '{:.4f}',
            'F1':        '{:.4f}',
            'ROC-AUC':   '{:.4f}',
            'PR-AUC':    '{:.4f}',
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption(
        "🟩 Verde = mejor modelo global (por PR-AUC)  |  "
        "🟨 Amarillo = mejor valor por columna"
    )

    st.subheader("📊 Comparación visual de indicadores")
    metrics_long = metrics_df.melt(
        id_vars='Modelo', var_name='Métrica', value_name='Valor'
    )
    fig = px.bar(
        metrics_long, x='Métrica', y='Valor', color='Modelo',
        barmode='group', text_auto='.3f',
        title='Métricas comparadas — los 4 modelos',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(height=500, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🎯 Vista radar — perfil de cada modelo")
    radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    fig = go.Figure()
    colors = ['#2E86AB', '#E63946', '#06A77D', '#F4A261']
    for i, row in metrics_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in radar_metrics],
            theta=radar_metrics,
            fill='toself',
            name=row['Modelo'] + (' ⭐' if row['Modelo'] == best_name_auto else ''),
            line=dict(color=colors[i % len(colors)])
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, height=500,
        title='Perfil de desempeño por modelo'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Curvas ROC y Precision-Recall")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Curva ROC', 'Curva Precision-Recall'))
    for i, (name, c) in enumerate(curves.items()):
        suffix = ' ⭐' if name == best_name_auto else ''
        fig.add_trace(
            go.Scatter(x=c['fpr'], y=c['tpr'], mode='lines',
                       name=f"{name}{suffix} (AUC={c['roc_auc']:.3f})",
                       legendgroup=name,
                       line=dict(color=colors[i % len(colors)], width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=c['recall'], y=c['precision'], mode='lines',
                       name=f"{name} (AP={c['pr_auc']:.3f})",
                       legendgroup=name, showlegend=False,
                       line=dict(color=colors[i % len(colors)], width=3)),
            row=1, col=2
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                   line=dict(dash='dash', color='gray'), showlegend=False),
        row=1, col=1
    )
    fig.update_xaxes(title_text='FPR', row=1, col=1)
    fig.update_yaxes(title_text='TPR', row=1, col=1)
    fig.update_xaxes(title_text='Recall', row=1, col=2)
    fig.update_yaxes(title_text='Precision', row=1, col=2)
    fig.update_layout(height=500, title_text='ROC y PR — los 4 modelos')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧮 Matrices de confusión")
    st.caption(
        "**Convenciones:**  "
        "**VP** = Verdadero Positivo · "
        "**VN** = Verdadero Negativo · "
        "**FP** = Falso Positivo · "
        "**FN** = Falso Negativo"
    )
    cm_cols = st.columns(len(curves))
    for i, (name, c) in enumerate(curves.items()):
        with cm_cols[i]:
            suffix = ' ⭐' if name == best_name_auto else ''
            cm_arr = c['cm']
            # Construir etiquetas con abreviaciones VP/VN/FP/FN
            labels_text = [
                [f"VN: {cm_arr[0,0]}", f"FP: {cm_arr[0,1]}"],
                [f"FN: {cm_arr[1,0]}", f"VP: {cm_arr[1,1]}"]
            ]
            fig = go.Figure(data=go.Heatmap(
                z=cm_arr,
                x=['Pred. Legítimo', 'Pred. Fraude'],
                y=['Real Legítimo', 'Real Fraude'],
                text=labels_text,
                texttemplate='%{text}',
                textfont={'size': 14, 'color': 'black'},
                colorscale='Blues',
                showscale=False
            ))
            fig.update_layout(
                title=f"{name}{suffix}",
                height=380,
                yaxis=dict(autorange='reversed')
            )
            st.plotly_chart(fig, use_container_width=True)

