import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from model.processing.data_manager import load_dataset
from models import models  # Asegúrate de que models.py esté correctamente configurado
from sklearn.model_selection import train_test_split

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    Función para entrenar un modelo, hacer predicciones y registrar los resultados en MLflow.
    """
    with mlflow.start_run(run_name=model_name):
        # Entrenamiento del modelo
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Registro de métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        # Log de métricas en MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        if auc is not None:
            mlflow.log_metric("auc", auc)

        # Log del modelo
        mlflow.sklearn.log_model(model, model_name)

        print(f"Modelo {model_name} registrado en MLflow con accuracy: {accuracy:.4f}")

# Carga de datos
data = load_dataset(file_name="cdc_diabetes_health_indicators_features.csv")
target = load_dataset(file_name="cdc_diabetes_health_indicators_target.csv")

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Experimentación con cada modelo
for model_name, model in models.items():
    train_and_log_model(model, model_name, X_train, y_train, X_test, y_test)
