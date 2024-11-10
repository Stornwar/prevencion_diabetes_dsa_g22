from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# Guardar en CSV para el uso en el pipeline
X.to_csv("data/cdc_diabetes_health_indicators_features.csv", index=False)
y.to_csv("data/cdc_diabetes_health_indicators_target.csv", index=False)

print("Datos descargados y guardados en la carpeta 'data'.")