from fastapi import APIRouter
from app.api.schemas.outputs import Health

router = APIRouter()

@router.get("/health", response_model=Health)
def health() -> dict:
    """Verifica el estado de la API"""
    return {"name": "API Prevención Diabetes", "api_version": "0.1", "model_version": "0.0.1"}
