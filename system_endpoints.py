# Add these classes and endpoints to the existing system.py file

import os
import logging
from pathlib import Path
from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import Dict, List

from utils.config import save_to_env

# Get logger instance
logger = logging.getLogger(__name__)

# Create router instance - this would already exist in system.py
router = APIRouter()

class EnvVariablesRequest(BaseModel):
    variables: List[str]

class UpdateEnvRequest(BaseModel):
    variables: Dict[str, str]
    
class UpdateEnvVarRequest(BaseModel):
    key: str
    value: str

@router.post("/env-variables")
async def get_env_variables(request: EnvVariablesRequest):
    """Get values of specified environment variables from .env file"""
    try:
        env_path = Path(__file__).parent.parent.parent / '.env'
        values = {}
        
        if env_path.exists():
            with open(env_path, 'r') as f:
                lines = f.readlines()
                
            for var_name in request.variables:
                for line in lines:
                    if line.strip() and not line.strip().startswith('#') and '=' in line:
                        key, value = line.strip().split('=', 1)
                        if key == var_name:
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            values[var_name] = value
                            break
        
        return {"values": values}
    except Exception as e:
        logger.error(f"Error getting env variables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get env variables: {str(e)}")

@router.post("/update-env")
async def update_env_variables(request: UpdateEnvRequest):
    """Update multiple environment variables in .env file"""
    try:
        for key, value in request.variables.items():
            save_to_env(key, value)
            # Also update the current environment
            os.environ[key] = value
        
        return {"success": True}
    except Exception as e:
        logger.error(f"Error updating env variables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update env variables: {str(e)}")

@router.post("/update-env-var")
async def update_env_var(request: UpdateEnvVarRequest):
    """Update a single environment variable in .env file"""
    try:
        save_to_env(request.key, request.value)
        # Also update the current environment
        os.environ[request.key] = request.value
        
        return {"success": True}
    except Exception as e:
        logger.error(f"Error updating env variable: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update env variable: {str(e)}")
