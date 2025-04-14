from fastapi import APIRouter, HTTPException
import httpx
import json
import os
from typing import List, Optional
import logging
import time

logger = logging.getLogger(__name__)

# Create a FastAPI router with prefix 
router = APIRouter(
    prefix="/api/ollama",
    tags=["ollama"],
    responses={404: {"description": "Not found"}},
)

# Define the base Ollama API URL
OLLAMA_API_BASE = "http://localhost:11434/api"

# Cache for Ollama models
_models_cache = {
    "last_check_time": 0,
    "check_interval": 30,  # 30 seconds between model list refreshes
    "models": [],
    "current_model": None
}

class OllamaModel:
    def __init__(self, name: str, modified_at: Optional[str] = None, size: Optional[int] = None):
        self.name = name
        self.modified_at = modified_at
        self.size = size

@router.get("/models", response_model=dict)
async def get_models():
    """
    Fetch available models from Ollama service with caching
    """
    global _models_cache
    
    try:
        # Check if we can use cached models
        current_time = time.time()
        if (current_time - _models_cache["last_check_time"] < _models_cache["check_interval"] and 
            _models_cache["models"]):
            logger.debug("Using cached Ollama models")
            return {
                "success": True,
                "models": _models_cache["models"],
                "current_model": _models_cache["current_model"],
                "cached": True
            }
            
        # If cache is expired or empty, request models from Ollama API
        logger.debug("Fetching fresh Ollama models list")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_API_BASE}/tags")
            
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Error from Ollama API: {response.text}"
            )
            
        # Parse response
        data = response.json()
        models = data.get('models', [])
        
        # Get current model from model_settings.json
        current_model = None
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        settings_dir = os.path.join(base_dir, "scripts", "settings")
        settings_file = os.path.join(settings_dir, "model_settings.json")
        
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                    # Check if provider is ollama and get the model_name
                    if settings.get("provider") == "ollama":
                        current_model = settings.get("model_name")
                        logger.info(f"Loaded current model from settings: {current_model}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading model settings: {str(e)}")
        
        # Update cache
        _models_cache["last_check_time"] = current_time
        _models_cache["models"] = models
        _models_cache["current_model"] = current_model
        
        return {
            "success": True,
            "models": models,
            "current_model": current_model,
            "cached": False
        }
        
    except httpx.RequestError as e:
        logger.error(f"Error connecting to Ollama service: {str(e)}")
        raise HTTPException(status_code=503, detail="Could not connect to Ollama service")
    except Exception as e:
        logger.error(f"Unexpected error in get_models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/set-model", response_model=dict)
async def set_model(model_data: dict):
    """
    Set the current Ollama model
    """
    global _models_cache
    
    try:
        if "model" not in model_data:
            raise HTTPException(status_code=400, detail="Missing model parameter")
            
        model_name = model_data["model"]
        
        # Get path to model_settings.json
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        settings_dir = os.path.join(base_dir, "scripts", "settings")
        settings_file = os.path.join(settings_dir, "model_settings.json")
        
        # Ensure settings directory exists
        os.makedirs(settings_dir, exist_ok=True)
        
        # Load existing settings if available
        settings = {}
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r") as f:
                    settings = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in settings file, starting with empty settings")
        
        # Update settings with new model selection
        settings["provider"] = "ollama"  # Ensure provider is set to ollama
        settings["model_name"] = model_name
        
        # Update cache immediately
        _models_cache["current_model"] = model_name
        
        # Log all information for debugging
        logger.info(f"Saving model selection: {model_name}")
        
        # Save updated settings
        with open(settings_file, "w") as f:
            json.dump(settings, f)
            
        logger.info(f"Set Ollama model to: {model_name}")
        return {
            "success": True,
            "message": f"Model set to {model_name}"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in set_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/status", response_model=dict)
async def get_ollama_status():
    """
    Check if Ollama service is running and responding
    """
    try:
        # Try to connect to the Ollama API
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_API_BASE}/version")
            
        if response.status_code == 200:
            version_data = response.json()
            return {
                "success": True,
                "status": "running",
                "version": version_data.get("version", "unknown")
            }
        else:
            return {
                "success": False,
                "status": "error",
                "message": f"Ollama returned status code {response.status_code}"
            }
            
    except httpx.ConnectError:
        return {
            "success": False,
            "status": "not_running",
            "message": "Unable to connect to Ollama service"
        }
    except Exception as e:
        logger.error(f"Error checking Ollama status: {str(e)}")
        return {
            "success": False,
            "status": "error",
            "message": f"Error: {str(e)}"
        }
