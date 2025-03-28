from fastapi import APIRouter, Request, HTTPException, Depends, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from scripts.llm.open_router_client import OpenRouterClient
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/openrouter", tags=["openrouter"])

# Pydantic models for request/response
class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    context_k: Optional[int] = None
    max_tokens: Optional[int] = None

class GenerateResponse(BaseModel):
    text: str
    model: Optional[str] = None

class ModelInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = ""
    context_length: Optional[int] = 0
    pricing: Optional[Dict[str, Any]] = {}

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

class SetAPIKeyRequest(BaseModel):
    api_key: str

class ConfigResponse(BaseModel):
    status: str
    message: str
    default_model: Optional[str] = None

# Client instantiation
def get_client():
    """Create or reuse the OpenRouter client"""
    # Check if there's an existing client in app state
    from app import app
    
    if not hasattr(app.state, "openrouter_client"):
        logger.info("Creating new OpenRouter client")
        app.state.openrouter_client = OpenRouterClient()
    
    return app.state.openrouter_client

@router.get("/status", response_model=ConfigResponse)
async def check_status():
    """Check connection status to OpenRouter"""
    client = get_client()
    ready, message = await client.check_connection()
    
    status = "connected" if ready else "disconnected"
    
    return ConfigResponse(
        status=status,
        message=message,
        default_model=client.default_model if ready else None
    )

@router.post("/generate", response_model=GenerateResponse)
async def generate_response(request: GenerateRequest):
    """Generate a response using OpenRouter"""
    client = get_client()
    
    # Check if API key is set
    if not client.api_key:
        raise HTTPException(
            status_code=401,
            detail="OpenRouter API key not set. Use /api/openrouter/config endpoint to set it."
        )
    
    # Generate response
    response_text = await client.generate_response(
        prompt=request.prompt,
        model=request.model,
        context_k=request.context_k,
        max_tokens=request.max_tokens
    )
    
    return GenerateResponse(
        text=response_text,
        model=request.model or client.default_model
    )

@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available models from OpenRouter"""
    client = get_client()
    
    # Check if API key is set
    if not client.api_key:
        raise HTTPException(
            status_code=401,
            detail="OpenRouter API key not set. Use /api/openrouter/config endpoint to set it."
        )
    
    models = await client.list_available_models()
    
    # Convert to Pydantic models
    model_infos = [
        ModelInfo(
            id=model["id"],
            name=model["name"],
            description=model.get("description", ""),
            context_length=model.get("context_length", 0),
            pricing=model.get("pricing", {})
        )
        for model in models
    ]
    
    return ModelsResponse(models=model_infos)

@router.post("/config", response_model=ConfigResponse)
async def set_api_key(request: SetAPIKeyRequest):
    """Set OpenRouter API key"""
    client = get_client()
    
    # Update API key
    client.api_key = request.api_key
    
    # Test connection
    ready, message = await client.check_connection()
    
    # Store in environment variable for persistence
    os.environ["OPENROUTER_API_KEY"] = request.api_key
    
    # Update status message
    status = "connected" if ready else "disconnected"
    if ready:
        message = "API key set and connection successful"
    else:
        message = f"API key set but connection failed: {message}"
    
    return ConfigResponse(
        status=status,
        message=message,
        default_model=client.default_model if ready else None
    )

@router.post("/default-model", response_model=ConfigResponse)
async def set_default_model(model_id: str = Body(..., embed=True)):
    """Set default model for OpenRouter"""
    client = get_client()
    
    # Check if API key is set
    if not client.api_key:
        raise HTTPException(
            status_code=401,
            detail="OpenRouter API key not set. Use /api/openrouter/config endpoint to set it."
        )
    
    # Update default model
    client.default_model = model_id
    
    return ConfigResponse(
        status="updated",
        message=f"Default model set to {model_id}",
        default_model=model_id
    )