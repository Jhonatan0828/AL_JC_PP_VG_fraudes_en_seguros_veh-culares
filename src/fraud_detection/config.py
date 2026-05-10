from pathlib import Path
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# --------------------------------------------------------------
# Rutas del proyecto (Usando pathlib + .env)
# --------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

# Si el .env tiene una ruta, la usamos; si no, usamos la ruta por defecto
RAW_DATASET = Path(os.getenv("RAW_DATA_PATH", ROOT_DIR / "data/raw/fraud_oracle.csv"))
CLEAN_DATASET = Path(os.getenv("PROCESSED_DATA_PATH", ROOT_DIR / "data/processed/fraud_clean.csv"))

MODELS_DIR = ROOT_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_fraud_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.pkl"

# --------------------------------------------------------------
# Constantes del modelado (Esto está perfecto como lo tenías)
# --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "FraudFound_P"

# Variables a excluir del modelado
COLUMNS_TO_DROP = ["PolicyNumber", "Year", "AgeOfPolicyHolder"]

# --------------------------------------------------------------
# Colores y temas
# --------------------------------------------------------------
COLOR_LEGITIMATE = "#2E86AB"
COLOR_FRAUD = "#E63946"
MODEL_COLORS = ["#2E86AB", "#E63946", "#06A77D", "#F4A261"]