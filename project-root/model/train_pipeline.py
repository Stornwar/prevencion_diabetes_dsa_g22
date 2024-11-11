import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from model.config.core import app_config, model_config
from model.pipeline import diabetes_pipe
from model.processing.data_manager import load_dataset, save_pipeline
from collections import Counter

def run_training() -> None:
    """Entrena y guarda el modelo."""

    # Carga de datos de entrenamiento
    data = load_dataset(
        file_name=app_config.train_data_file,
        target_file=app_config.train_target_file
    )
    
    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=model_config.target),
        data[model_config.target],
        test_size=model_config.test_size,
        random_state=model_config.random_state,
    )

    print("Distribución de clases en y_train antes de SMOTE:")
    print(y_train.value_counts())

    # Balanceo con SMOTE
    min_class_count = min(Counter(y_train).values())
    sm = SMOTE(k_neighbors=min(min_class_count - 1, 1), random_state=model_config.random_state)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)
    
    print("Distribución de clases en y_train después de la agrupación:")
    print(y_train.value_counts())

    # Entrena el modelo
    diabetes_pipe.fit(X_train_balanced, y_train_balanced)

    # Guarda el modelo entrenado
    save_pipeline(pipeline_to_persist=diabetes_pipe)

if __name__ == "__main__":
    run_training()
