import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from config.core import config
from pipeline import diabetes_pipe
from processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    """Entrena y guarda el modelo."""

    # Carga de datos de entrenamiento
    data = load_dataset(file_name=config.app_config.train_data_file)
    
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=config.model_config.target),
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    # Balanceo con SMOTE
    sm = SMOTE(random_state=config.model_config.random_state)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    # Entrena el modelo
    diabetes_pipe.fit(X_train_balanced, y_train_balanced)

    # Guarda el modelo entrenado
    save_pipeline(pipeline_to_persist=diabetes_pipe)

if __name__ == "__main__":
    run_training()
