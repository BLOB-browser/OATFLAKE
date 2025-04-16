from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel  # Add this import
from typing import List, Optional
from pathlib import Path
from scripts.models.schemas import Definition, Project, Method, Resource, JsonRequest  # Add missing imports
from scripts.data.data_processor import DataProcessor
from utils.config import BACKEND_CONFIG
import json
import logging
import shutil
import os
from datetime import datetime
from scripts.services import storage

# Add JsonRequest model definition
class JsonRequest(BaseModel):
    path: str

from fastapi import APIRouter
from .projects import router as projects_router
from .tables import router as tables_router
from .training import router as training_router
from .definitions import router as definitions_router  # Add this import
from .materials import router as materials_router  # Add this import
from .resources import router as resources_router  # Add this import
from .methods import router as methods_router  # Add this import
from .stats import router as stats_router  # Add this import

router = APIRouter(tags=["data"], prefix="/api/data")

# Include sub-routers
router.include_router(projects_router)
router.include_router(tables_router)
router.include_router(training_router)
router.include_router(definitions_router)  # Add this line
router.include_router(materials_router)  # Add this line
router.include_router(resources_router)  # Add this line
router.include_router(methods_router)  # Add this line
router.include_router(stats_router)  # Add this line

logger = logging.getLogger(__name__)

@router.post("/scan")
async def scan_data():
    """Scan and process all data"""
    try:
        data_path = Path(BACKEND_CONFIG['data_path'])
        if not data_path.exists():
            raise HTTPException(status_code=400, detail="Data path not configured")
        
        # Get group ID from config
        config_path = Path.home() / '.blob' / 'config.json'
        group_id = "default"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                group_id = config.get('group_id', 'default')
            
        processor = DataProcessor(data_path, group_id=group_id)
        
        # Scan directory and create default topic-based stores
        result = await processor.scan_directory()
        
        # Optionally create type-based stores
        type_results = await processor.process_by_type()
        
        # Create custom store for specific documents
        custom_docs = [doc for doc in result["chunks"] 
                      if "important" in doc.metadata.get("tags", [])]
        if custom_docs:
            await processor.create_custom_vector_store(custom_docs, "important_docs")
        
        return {
            "status": "success",
            "data": {
                **result,
                "type_stores": type_results
            }
        }
        
    except Exception as e:
        logger.error(f"Error scanning data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/storage/set")
async def set_storage_path(request: JsonRequest):
    """Set storage path"""
    try:
        logger.info(f"Setting storage path to: {request.path}")
        
        # Handle relative paths properly
        path = request.path
        path_obj = Path(path)
        
        # If it's not an absolute path, resolve it properly
        if not path_obj.is_absolute():
            # Try to resolve relative to Documents folder first
            documents_path = Path.home() / "Documents" / path
            if not documents_path.exists():
                # Create it in Documents folder
                documents_path.mkdir(parents=True, exist_ok=True)
            path = str(documents_path)
            logger.info(f"Converted relative path to absolute: {path}")
        
        # Get both config paths - project and user
        project_config_path = Path(__file__).parent.parent.parent / 'config.json'
        user_config_path = Path.home() / '.blob' / 'config.json'
        
        # Load existing project config or create new
        config = {}
        if project_config_path.exists():
            with open(project_config_path, 'r') as f:
                config = json.load(f)
        
        # Update data path with resolved path
        config['data_path'] = path
        
        # Update BACKEND_CONFIG in memory so it takes effect immediately
        from utils.config import BACKEND_CONFIG
        BACKEND_CONFIG['data_path'] = path
        
        # Save updated config to project file first
        with open(project_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Also update user config for consistency
        user_config = {}
        if user_config_path.exists():
            try:
                with open(user_config_path, 'r') as f:
                    user_config = json.load(f)
            except:
                pass
                
        user_config['data_path'] = path
        
        # Ensure user config directory exists
        user_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save updated user config
        with open(user_config_path, 'w') as f:
            json.dump(user_config, f, indent=2)
            
        logger.info(f"Updated config with new data path: {path}")
        return {"status": "success", "data_path": path}
    
    except Exception as e:
        logger.error(f"Error setting storage path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/storage/space")
async def get_storage_space(path: str):
    """Get storage space information for a path"""
    try:
        logger.info(f"Getting storage space info for path: {path}")
        
        # Validate path exists
        data_path = Path(path)
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        
        # Get disk usage information
        total, used, free = shutil.disk_usage(data_path)
        
        # Convert to human-readable format
        def format_bytes(bytes_value):
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if bytes_value < 1024 or unit == 'TB':
                    return f"{bytes_value:.2f} {unit}"
                bytes_value /= 1024
        
        return {
            "status": "success",
            "path": str(data_path),
            "total": total,
            "used": used,
            "free": free,
            "total_formatted": format_bytes(total),
            "used_formatted": format_bytes(used),
            "free_formatted": format_bytes(free),
            "usage_percent": round((used / total) * 100, 2)
        }
    
    except Exception as e:
        logger.error(f"Error getting storage space info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

