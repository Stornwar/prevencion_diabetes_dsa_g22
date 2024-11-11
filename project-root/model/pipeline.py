from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from config.core import model_config

# Definición de las funciones explícitas para la agrupación de edades e ingresos
def aplicar_agrupa_edades(df):
    return df.applymap(agrupa_edades)

def aplicar_agrupa_ingreso(df):
    return df.applymap(agrupa_ingreso)

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
        ('age_group', FunctionTransformer(aplicar_agrupa_edades), ['Age']),
        ('income_group', FunctionTransformer(aplicar_agrupa_ingreso), ['Income']),
        ('scaler', StandardScaler(), model_config.features),
    ],
    remainder='passthrough'
)

# Pipeline con preprocesamiento y el modelo XGBClassifier
diabetes_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        random_state=model_config.random_state,
        use_label_encoder=False,
        eval_metric="logloss"
    ))
])
