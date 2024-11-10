from pydantic_settings import BaseSettings

class AppConfig(BaseSettings):
    data_folder: str = "data"  # Ruta relativa a la carpeta `data` en la raíz del proyecto
    train_data_file: str = "cdc_diabetes_health_indicators_features.csv"
    pipeline_save_file: str = "artifacts/diabetes_prediction_pipeline"

# Configuración para el modelo
class ModelConfig(BaseSettings):
    features: list[str] = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Education"
    ]
    target: str = "Diabetes_binary"
    n_estimators: int = 100
    max_depth: int = 10
    test_size: float = 0.2
    random_state: int = 42

app_config = AppConfig()
model_config = ModelConfig()
