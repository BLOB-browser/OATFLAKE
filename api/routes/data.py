from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel  # Add this import
from typing import List
from pathlib import Path
from scripts.models.schemas import Definition, Project, Method, Resource, JsonRequest  # Add missing imports
from scripts.data.data_processor import DataProcessor
from utils.config import BACKEND_CONFIG
import json
import logging
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

router = APIRouter(tags=["data"])

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
        config_path = Path.home() / '.blob' / 'config.json'
        
        # Load existing config or create new
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Update data path
        config['data_path'] = str(request.path)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Updated config with new data path: {request.path}")
        return {"status": "success", "data_path": request.path}
    
    except Exception as e:
        logger.error(f"Error setting storage path: {e}")
        raise HTTPException(status_code=500, detail=str(e))

