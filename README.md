# 🔍 Detección de Fraude en Seguros Vehiculares

> Proyecto académico — **Universidad de Medellín**
> Curso: Ciencia de Datos e Inteligencia Artificial
> Profesor: David Palacio Jimenez

[![Tests](https://github.com/<tu-usuario>/fraud-detection/actions/workflows/tests.yml/badge.svg)](https://github.com/<tu-usuario>/fraud-detection/actions)

## 📌 Descripción

Pipeline completo de Ciencia de Datos para identificar reclamaciones fraudulentas en seguros vehiculares, basado en el dataset *Vehicle Insurance Claim Fraud Detection* (Oracle/Kaggle, 15,420 registros, período 1994–1996).

## ✨ Características

- ✅ Análisis Exploratorio (EDA) interactivo con Plotly y 15 filtros dinámicos
- ✅ Preprocesamiento profesional con `Pipeline` y `ColumnTransformer` de Scikit-Learn
- ✅ 4 modelos comparados: Logistic Regression, Random Forest, XGBoost, Logistic Regression + SMOTE
- ✅ Manejo del desbalance de clases (5.99% fraude)
- ✅ Interpretabilidad: coeficientes, odds ratios, importancia de variables, permutation importance, SHAP values
- ✅ **Aplicación interactiva en Streamlit** con 4 páginas (Inicio, EDA, Predicción, Desempeño)
- ✅ **Lista para desplegar** en Streamlit Cloud, Heroku, Railway, Render

## 🏆 Resultados principales

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6433 | 0.131 | **0.881** | 0.229 | 0.804 | 0.170 |
| Random Forest | 0.7354 | 0.151 | 0.735 | 0.250 | 0.826 | 0.192 |
| **XGBoost** ⭐ | **0.8320** | **0.193** | 0.568 | **0.288** | **0.831** | **0.241** |
| Logistic Regression + SMOTE | 0.6592 | 0.132 | 0.838 | 0.228 | 0.801 | 0.164 |

🏆 **Modelo seleccionado: XGBoost** (mejor PR-AUC, métrica clave en datasets desbalanceados).

## 📁 Estructura del proyecto

```
fraud-detection/
├── app.py                              # Punto de entrada de Streamlit
├── pyproject.toml                      # Configuración del paquete
├── requirements.txt                    # Dependencias de Python
├── runtime.txt                         # Versión de Python para deploy
├── Procfile                            # Comando para Heroku/Railway
├── uv.lock                             # Lock file de uv
├── .python-version                     # Python 3.11
├── .env.example                        # Plantilla de variables de entorno
├── .gitignore                          # Archivos excluidos de Git
├── .streamlit/config.toml              # Configuración de Streamlit
├── README.md
│
├── src/
│   └── fraud_detection/                # Paquete principal
│       ├── __init__.py
│       ├── config.py                   # Constantes y rutas
│       ├── mappings.py                 # Mapeos ordinales y filtros
│       ├── data_loader.py              # Carga de datos y modelos
│       ├── metrics.py                  # Cálculo de métricas
│       ├── plotting.py                 # Funciones de gráficos
│       ├── prediction.py               # Lógica de predicción
│       └── pages/                      # Páginas de la app
│           ├── home.py                 # Resumen ejecutivo
│           ├── eda.py                  # EDA interactivo
│           ├── prediction.py           # Predicción individual
│           └── performance.py          # Comparación de modelos
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb  # Notebook completo de análisis
│
├── data/
│   ├── raw/                            # Dataset original (NO en git)
│   └── processed/
│       └── fraud_clean.csv             # Dataset preprocesado
│
├── models/                             # Modelos entrenados
│   ├── best_fraud_model.pkl
│   ├── model_metadata.pkl
│   ├── Logistic_Regression.pkl
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   └── Logistic_Regression_plus_SMOTE.pkl
│
├── tests/
│   └── test_fraud_detection.py         # Tests unitarios (8 tests)
│
└── .github/workflows/
    └── tests.yml                       # CI: corre tests en cada push
```

## 🚀 Instalación local

### Opción A — Con `uv` (recomendado, más rápido)

```bash
git clone https://github.com/<tu-usuario>/fraud-detection.git
cd fraud-detection
uv venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/macOS
uv pip install -e .
```

### Opción B — Con `pip` tradicional

```bash
git clone https://github.com/<tu-usuario>/fraud-detection.git
cd fraud-detection
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # Linux/macOS
pip install -r requirements.txt
```

### Variables de entorno

```bash
cp .env.example .env                # Linux/macOS
copy .env.example .env              # Windows
```

## ▶️ Lanzar la aplicación localmente

```bash
streamlit run app.py
```

Disponible en `http://localhost:8501`.

## 🌐 Despliegue como app web

Esta app está lista para desplegarse en cualquiera de estas plataformas:

### 🅰️ Streamlit Community Cloud (recomendado, gratis)

1. Sube el repositorio a GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta de GitHub.
3. Selecciona el repositorio y el archivo `app.py`.
4. Click en **Deploy**.
5. ¡Listo! Recibirás un link público tipo `https://tu-usuario-fraud-detection.streamlit.app`.

### 🅱️ Hugging Face Spaces

1. Crea un Space en [huggingface.co/spaces](https://huggingface.co/spaces) con SDK = Streamlit.
2. Sube los archivos del repositorio.
3. Hugging Face desplegará automáticamente.

### 🅲 Heroku / Railway / Render

El proyecto incluye `Procfile` y `runtime.txt`, listos para deploy directo:

```bash
# Heroku
heroku create
git push heroku main

# Railway / Render: solo conectar el repo de GitHub
```

## 📊 Funcionalidades de la app

| Página | Descripción |
|---|---|
| 🏠 **Inicio** | Resumen ejecutivo con KPIs, contexto, metodología y mejor modelo |
| 📊 **EDA Interactivo** | 15 filtros dinámicos sobre todas las variables categóricas |
| 🎯 **Predicción** | Formulario para predecir fraude en una nueva reclamación |
| 📈 **Desempeño del modelo** | Comparativa de los 4 modelos con tabla, gráficos y matrices de confusión |

## 🧪 Ejecutar tests

```bash
pytest tests/ -v
```

Los tests también se ejecutan automáticamente en cada `push` o `pull request` mediante GitHub Actions.

## 🔄 Reentrenar los modelos

Si quieres reentrenar los modelos desde cero:

1. Coloca el dataset original en `data/raw/fraud_oracle.csv`.
2. Abre `notebooks/fraud_detection_analysis.ipynb` y ejecuta todas las celdas.
3. Los nuevos modelos se guardarán en `models/` automáticamente.

## 👥 Equipo

- Andres Felipe Londoño Ocampo
- Jhonatan Caro Atehortúa
- Paulina Perez Ramirez
- Victor Manuel Galeano Alvarez

## 🛠️ Stack tecnológico

Python 3.11 · Pandas · NumPy · Scikit-Learn · XGBoost · imbalanced-learn · Plotly · SHAP · Streamlit · pytest · uv

## 📄 Licencia

Proyecto académico — Universidad de Medellín, 2026.
