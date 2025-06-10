from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, Optional
from scripts.analysis.universal_analysis_llm import UniversalAnalysisLLM

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analysis", tags=["analysis"])

class AnalysisSettings(BaseModel):
    provider: str
    model_name: Optional[str] = None
    openrouter_model: Optional[str] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_ctx: Optional[int] = None
    num_thread: Optional[int] = None

@router.get("/settings", response_model=Dict[str, Any])
async def get_analysis_settings():
    """Get current analysis model settings"""
    try:
        # Path to analysis model settings
        settings_path = Path(__file__).parent.parent.parent / "scripts" / "settings" / "analysis-model-settings.json"
        
        if not settings_path.exists():
            # If file doesn't exist, create it with default settings by initializing the class
            temp_instance = UniversalAnalysisLLM()
            settings = temp_instance.settings
        else:
            # Load existing settings
            with open(settings_path, "r") as f:
                settings = json.load(f)
        
        return {
            "success": True,
            "settings": settings
        }
    except Exception as e:
        logger.error(f"Error getting analysis settings: {e}")
        return {
            "success": False,
            "message": str(e)
        }

@router.post("/settings", response_model=Dict[str, Any])
async def update_analysis_settings(settings: AnalysisSettings):
    """Update analysis model settings"""
    try:
        # Update only non-None fields
        update_data = {k: v for k, v in settings.dict().items() if v is not None}
        
        # Call the class method to update settings
        success = UniversalAnalysisLLM.update_analysis_settings(update_data)
        
        if success:
            return {
                "success": True,
                "message": "Analysis settings updated successfully"
            }
        else:
            return {
                "success": False,
                "message": "Failed to update analysis settings"
            }
    except Exception as e:
        logger.error(f"Error updating analysis settings: {e}")
        return {
            "success": False,
            "message": str(e)
        }

@router.post("/test-connection", response_model=Dict[str, Any])
async def test_analysis_connection():
    """Test connection with current analysis model settings"""
    try:
        # Create an instance with current settings
        llm = UniversalAnalysisLLM()
        
        # Try a simple test prompt
        test_prompt = "Generate a simple JSON with one key called 'status' and value 'success'"
        result = llm.generate_structured_response(test_prompt, format_type="json")
        
        if result and isinstance(result, dict):
            return {
                "success": True,
                "message": f"Successfully connected to {llm.settings['provider']} model",
                "test_result": result
            }
        else:
            return {
                "success": False,
                "message": f"Failed to get valid response from {llm.settings['provider']} model"
            }
    except Exception as e:
        logger.error(f"Error testing analysis model connection: {e}")
        return {
            "success": False,
            "message": str(e)
        }