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

@router.get("/models")
async def list_models():
    """List available models from OpenRouter"""
    client = get_client()
    
    # Check if API key is set
    if not client.api_key:
        return {
            "success": False,
            "message": "OpenRouter API key not set. Use /api/openrouter/config endpoint to set it.",
            "models": [],
            "current_model": None
        }
    
    try:
        models = await client.list_available_models()
        
        # Get current model from settings
        settings = client.settings_manager.load_settings()
        current_model = settings.openrouter_model if hasattr(settings, 'openrouter_model') else None
        logger.info(f"Current OpenRouter model from settings: {current_model}")
        
        # Convert to compatible format
        model_infos = [
            {
                "id": model.get("id"),
                "name": model.get("name"),
                "description": model.get("description", ""),
                "context_length": model.get("context_length", 0),
                "pricing": model.get("pricing", {}),
                "is_current": model.get("is_current", False),
                "is_free": model.get("is_free", False)
            }
            for model in models
        ]
        
        return {
            "success": True,
            "models": model_infos,
            "current_model": current_model
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {
            "success": False,
            "message": str(e),
            "models": [],
            "current_model": None
        }

@router.post("/config", response_model=ConfigResponse)
async def set_api_key(request: SetAPIKeyRequest):
    """Set OpenRouter API key"""
    client = get_client()
    
    # Update API key
    client.api_key = request.api_key
    
    # Test connection
    ready, message = await client.check_connection()
    
    # Save the token to ~/.blob/.env for persistence
    from pathlib import Path
    from dotenv import set_key, find_dotenv
    
    # Create ~/.blob directory if it doesn't exist
    blob_dir = Path.home() / ".blob"
    os.makedirs(blob_dir, exist_ok=True)
    
    # Path to the .env file
    env_file = blob_dir / ".env"
    
    # Check if file exists, create if not
    if not env_file.exists():
        env_file.touch()
    
    # Save the token to ~/.blob/.env
    set_key(str(env_file), "OPENROUTER_API_KEY", request.api_key)
    logger.info(f"Saved OpenRouter API token to {env_file}")
    
    # Also save to the project's .env file for backwards compatibility
    project_env = find_dotenv()
    if project_env:
        set_key(project_env, "OPENROUTER_API_KEY", request.api_key)
        logger.info(f"Also saved token to project .env: {project_env}")
    
    # Store in environment variable for current session
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