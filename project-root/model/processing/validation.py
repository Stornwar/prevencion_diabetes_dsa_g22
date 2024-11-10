import pandas as pd
from pydantic import BaseModel, ValidationError
from typing import Tuple, Optional, List

class DataInputSchema(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI: float
    Smoker: int
    Stroke: int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Valida los datos de entrada usando Pydantic."""
    errors = []
    try:
        # Validación de cada entrada
        for record in input_data.to_dict(orient="records"):
            DataInputSchema(**record)
    except ValidationError as e:
        errors.append(str(e))

    return input_data, errors if errors else None
