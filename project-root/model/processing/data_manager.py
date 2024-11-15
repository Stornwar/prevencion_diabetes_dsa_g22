import os
import pandas as pd
import joblib
from model.config.core import app_config, model_config
from model.pipeline import agrupa_edades, agrupa_ingreso

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Carga un conjunto de datos en formato CSV."""
    data = pd.read_csv(f"{app_config.data_folder}/{file_name}")
    
    # Solo aplica el preprocesamiento si se trata del archivo de características
    if file_name == app_config.train_data_file:
        # Agrupación de edad e ingreso
        data['Age'] = data['Age'].apply(agrupa_edades)
        data['Income'] = data['Income'].apply(agrupa_ingreso)
    
    return data

def save_pipeline(*, pipeline_to_persist) -> None:
    """Guarda el pipeline entrenado con el nombre de la versión."""
    save_dir = os.path.dirname(f"{app_config.data_folder}/{app_config.pipeline_save_file}_{_version}.pkl")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = f"{app_config.data_folder}/{app_config.pipeline_save_file}_{_version}.pkl"
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str):
    """Carga un pipeline entrenado."""
    file_path = f"data/artifacts/{file_name}"
    return joblib.load(file_path)

# Función para cargar la versión del archivo VERSION
def get_version() -> str:
    version_path = os.path.join(os.path.dirname(__file__), "..", "VERSION")
    with open(version_path) as version_file:
        return version_file.read().strip()

_version = get_version()  # Obtiene la versión actual