# server.py

from fastapi import FastAPI
from api.routes.openrouter import router as openrouter_router
from api.routes.openrouter_models import router as openrouter_models_router

app = FastAPI()

# Include routers
app.include_router(openrouter_router)
app.include_router(openrouter_models_router, prefix="/api/openrouter", tags=["openrouter"])

# Other app setup code...