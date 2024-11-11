import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from model.config.core import app_config, model_config
from model.pipeline import diabetes_pipe
from model.processing.data_manager import load_dataset, save_pipeline

def run_training() -> None:
    """Entrena y guarda el modelo con XGBoost."""

    # Carga de datos de entrenamiento
    data = load_dataset(file_name=app_config.train_data_file)
    target = load_dataset(file_name="cdc_diabetes_health_indicators_target.csv")

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=model_config.test_size, random_state=model_config.random_state
    )

    # Balanceo con SMOTE
    sm = SMOTE(random_state=model_config.random_state)
    X_train_balanced, y_train_balanced = sm.fit_resample(X_train, y_train)

    # Actualización del pipeline con XGBoost
    xgb_model = XGBClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        random_state=model_config.random_state,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    diabetes_pipe.set_params(classifier=xgb_model)

    # Entrena el modelo
    diabetes_pipe.fit(X_train_balanced, y_train_balanced)

    # Guarda el modelo entrenado
    save_pipeline(pipeline_to_persist=diabetes_pipe)

if __name__ == "__main__":
    run_training()
