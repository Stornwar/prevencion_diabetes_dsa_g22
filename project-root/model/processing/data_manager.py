from pipeline import agrupa_edades, agrupa_ingreso
import pandas as pd
import joblib
from config.core import config

def load_dataset(*, file_name: str) -> pd.DataFrame:
    """Carga un conjunto de datos en formato CSV."""
    data = pd.read_csv(f"{config.app_config.data_folder}/{file_name}")
    
    # Agrupación de edad e ingreso
    data['Age'] = data['Age'].apply(agrupa_edades)
    data['Income'] = data['Income'].apply(agrupa_ingreso)
    
    return data

def save_pipeline(*, pipeline_to_persist) -> None:
    """Guarda el pipeline entrenado."""
    save_path = f"{config.app_config.pipeline_save_file}{pipeline_to_persist.__version__}.pkl"
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str):
    """Carga un pipeline entrenado."""
    return joblib.load(file_name)
