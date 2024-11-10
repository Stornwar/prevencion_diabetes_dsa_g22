from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from config.core import model_config

# Funciones de agrupación para Edad e Ingreso
def agrupa_edades(val):
    agrupaciones = [(1, 3), (4, 6), (7, 8), (9, 13)]
    for i, (ini, fin) in enumerate(agrupaciones):
        if ini <= val <= fin:
            return i + 1
    return val

def agrupa_ingreso(val):
    agrupaciones = [(1, 4), (5, 7), (8, 9)]
    for i, (ini, fin) in enumerate(agrupaciones):
        if ini <= val <= fin:
            return i + 1
    return val

# Definición de preprocesamiento personalizado
preprocessor = ColumnTransformer(
    transformers=[
        ('age_group', FunctionTransformer(lambda x: x.applymap(agrupa_edades)), ['Age']),
        ('income_group', FunctionTransformer(lambda x: x.applymap(agrupa_ingreso)), ['Income']),
        ('scaler', StandardScaler(), model_config.features),  # Estandarizar otras características
    ],
    remainder='passthrough'  # Deja las demás columnas sin cambios
)

# Pipeline con preprocesamiento y el modelo
diabetes_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        random_state=model_config.random_state
    ))
])
