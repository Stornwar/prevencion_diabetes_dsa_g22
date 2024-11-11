from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    data_folder: str = "data"  # Ruta relativa a la carpeta `data` en la raíz del proyecto
    train_data_file: str = "cdc_diabetes_health_indicators_features.csv"
    train_target_file: str = "cdc_diabetes_health_indicators_target.csv"  # Archivo de la variable objetivo
    pipeline_save_file: str = "artifacts/diabetes_prediction_pipeline"

# Configuración para el modelo
class ModelConfig(BaseSettings):
    # Parámetros comunes
    features: list[str] = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Education"
    ]
    target: str = "Diabetes_binary"
    random_state: int = 42

    # Parámetros para RandomForest
    n_estimators: int = 100
    max_depth: int = 10
    
    test_size: float = 0.3

    # Parámetros para XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1

app_config = AppConfig()
model_config = ModelConfig()