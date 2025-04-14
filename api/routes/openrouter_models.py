import os
from pathlib import Path
from dotenv import load_dotenv, set_key, find_dotenv
from scripts.llm.open_router_client import OpenRouterClient
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/set-token", response_model=Dict[str, Any])
async def set_token(token_data: dict):
    """
    Set the OpenRouter API token in the ~/.blob/.env file
    """
    try:
        if "token" not in token_data:
            raise HTTPException(status_code=400, detail="Missing token parameter")
            
        token = token_data["token"]
        
        # Create ~/.blob directory if it doesn't exist
        blob_dir = Path.home() / ".blob"
        os.makedirs(blob_dir, exist_ok=True)
        
        # Path to the .env file
        env_file = blob_dir / ".env"
        
        # Check if file exists, create if not
        if not env_file.exists():
            env_file.touch()
        
        # Save the token to ~/.blob/.env
        set_key(str(env_file), "OPENROUTER_API_KEY", token)
        logger.info(f"Saved OpenRouter API token to {env_file}")
        
        # Also save to the project's .env file for backwards compatibility
        project_env = find_dotenv()
        if project_env:
            set_key(project_env, "OPENROUTER_API_KEY", token)
            logger.info(f"Also saved token to project .env: {project_env}")
        
        # Set in current environment
        os.environ["OPENROUTER_API_KEY"] = token
        
        # Test the token
        client = OpenRouterClient(api_key=token)
        success, message = await client.check_connection()
        
        if not success:
            logger.warning(f"Token saved but validation failed: {message}")
            
        return {
            "success": success,
            "message": message if success else f"Token saved but validation failed: {message}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting API token: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/token-status")
async def get_token_status():
    """
    Check if OpenRouter API token is set and valid
    """
    try:
        # First check ~/.blob/.env
        blob_env = Path.home() / ".blob" / ".env"
        api_key = None
        
        # Load from ~/.blob/.env if exists
        if blob_env.exists():
            try:
                # Load env vars from this file without affecting others
                with open(blob_env) as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if key == "OPENROUTER_API_KEY":
                                api_key = value.strip().strip('"\'')
                                break
            except Exception as e:
                logger.warning(f"Error reading from {blob_env}: {e}")
        
        # If not found in ~/.blob/.env, check environment
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        
        # If we found a token, test it
        if api_key:
            client = OpenRouterClient(api_key=api_key)
            success, message = await client.check_connection()
            
            # Mask the API key for security (show only last 4 chars)
            masked_key = "•••••••••" + api_key[-4:] if len(api_key) > 4 else "•••••••"
            
            return {
                "has_token": True,
                "valid": success,
                "message": message,
                "masked_token": masked_key
            }
        
        # No token found
        return {
            "has_token": False,
            "valid": False,
            "message": "No API token found"
        }
        
    except Exception as e:
        logger.error(f"Error checking API token status: {str(e)}")
        return {
            "has_token": False,
            "valid": False,
            "message": f"Error: {str(e)}"
        }

@router.get("/models", response_model=Dict[str, Any])
async def get_models():
    """
    Get available OpenRouter models and which one is currently selected
    """
    try:
        # Create client to get list of available models from API
        client = OpenRouterClient()
        models = await client.list_available_models()
        
        # Get current model from model_settings.json
        current_model = None
        try:
            # Get path to model_settings.json
            settings_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "scripts" / "settings"
            settings_file = settings_dir / "model_settings.json"
            
            if settings_file.exists():
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                    # Get the openrouter_model
                    current_model = settings.get("openrouter_model")
                    logger.info(f"Current OpenRouter model from settings: {current_model}")
        except Exception as e:
            logger.warning(f"Error reading model settings: {e}")
        
        # If we have a API client for models but no setting present, set a default if needed
        if models and not current_model:
            # Try to set a default model
            free_models = [m for m in models if m.get("is_free", False)]
            if free_models:
                current_model = free_models[0].get("id")
                logger.info(f"No model setting found, using default: {current_model}")
        
        return {
            "success": True,
            "models": models,
            "current_model": current_model
        }
        
    except Exception as e:
        logger.error(f"Error getting models: {str(e)}")
        return {
            "success": False,
            "message": str(e),
            "models": [],
            "current_model": None
        }

@router.post("/set-model", response_model=Dict[str, Any])
async def set_model(model_data: dict):
    """
    Set the current OpenRouter model in model_settings.json
    """
    try:
        if "model" not in model_data:
            raise HTTPException(status_code=400, detail="Missing model parameter")
            
        model_name = model_data["model"]
        
        # Get path to model_settings.json
        settings_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / "scripts" / "settings"
        settings_file = settings_dir / "model_settings.json"
        
        # Ensure settings directory exists
        os.makedirs(settings_dir, exist_ok=True)
        
        # Load existing settings if available
        settings = {}
        if settings_file.exists():
            try:
                with open(settings_file, "r") as f:
                    settings = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in settings file, starting with empty settings")
        
        # Update settings with new model selection
        settings["openrouter_model"] = model_name
        
        # Save updated settings
        with open(settings_file, "w") as f:
            json.dump(settings, f)
            
        logger.info(f"Set OpenRouter model to: {model_name}")
        return {
            "success": True,
            "message": f"OpenRouter model set to {model_name}"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in set_openrouter_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")