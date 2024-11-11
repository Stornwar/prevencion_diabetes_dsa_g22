import typing as t
import numpy as np
import pandas as pd
from model.config.core import app_config, model_config
from model.processing.data_manager import load_pipeline
from model.processing.validation import validate_inputs

# Lee la versión desde el archivo VERSION
try:
    with open("model/VERSION") as version_file:
        _version = version_file.read().strip()
except FileNotFoundError:
    _version = "0.1"  # Coloca un valor predeterminado si no se encuentra el archivo
    
# Carga del pipeline entrenado usando la versión actual del modelo
pipeline_file_name = f"{app_config.pipeline_save_file}{_version}.pkl"
_diabetes_pipe = load_pipeline(file_name=pipeline_file_name)

def make_prediction(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Realiza una predicción usando un modelo guardado."""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _diabetes_pipe.predict(
            X=validated_data[model_config.features]
        )
        results = {
            "predictions": [pred for pred in predictions],
            "version": _version,
            "errors": errors,
        }

    return results
