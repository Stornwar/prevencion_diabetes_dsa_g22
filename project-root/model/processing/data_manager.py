import pandas as pd
import joblib
from config.core import app_config, model_config
from model.pipeline import agrupa_edades, agrupa_ingreso

def load_dataset(*, file_name: str, target_file: str = None) -> pd.DataFrame:
    """Carga un conjunto de datos en formato CSV."""
    data = pd.read_csv(f"{app_config.data_folder}/{file_name}")
    
    # Cargar el archivo de la variable objetivo si está separado
    if target_file:
        target = pd.read_csv(f"{app_config.data_folder}/{target_file}", names=[model_config.target])
        data = pd.concat([data, target], axis=1)
    
    # Agrupación de edad e ingreso
    data['Age'] = data['Age'].apply(agrupa_edades)
    data['Income'] = data['Income'].apply(agrupa_ingreso)
    
    return data

def save_pipeline(*, pipeline_to_persist) -> None:
    """Guarda el pipeline entrenado."""
    save_path = f"{app_config.data_folder}/{app_config.pipeline_save_file}.pkl"
    joblib.dump(pipeline_to_persist, save_path)

def load_pipeline(*, file_name: str):
    """Carga un pipeline entrenado."""
    return joblib.load(file_name)
