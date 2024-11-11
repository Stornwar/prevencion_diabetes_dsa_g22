import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from model.predict import make_prediction
from app.schemas.inputs import MultipleDataInputs
from app.schemas.outputs import PredictionResults

router = APIRouter()

@router.post("/predict", response_model=PredictionResults)
async def predict(input_data: MultipleDataInputs) -> dict:
    """Realiza una predicción usando el modelo de prevención de diabetes"""

    # Convierte cada registro en un diccionario
    input_df = pd.DataFrame([record.dict() for record in input_data.inputs])
    
    logger.info(f"Datos de entrada para predicción: {input_data.inputs}")
    
    # Pasa el DataFrame a la función de predicción
    results = make_prediction(input_data=input_df)

    if results["errors"]:
        logger.warning(f"Error en predicción: {results['errors']}")
        raise HTTPException(status_code=400, detail=results["errors"])

    logger.info(f"Resultados de predicción: {results['predictions']}")
    return results
