from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import health, predict
from app.api.v1.config import settings
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Monta la carpeta 'dashboard' para servir archivos estáticos
app.mount("/dashboard", StaticFiles(directory="dashboard"), name="dashboard")

# Ruta para servir el archivo dash.html en la raíz
@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    html_path = Path("dashboard/dash.html")
    return HTMLResponse(content=html_path.read_text(), status_code=200)

# Configuración de CORS
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Incluye las rutas
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Your Project Name"
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: list[str] = ["*"]  # or specific origins as a list

settings = Settings()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
