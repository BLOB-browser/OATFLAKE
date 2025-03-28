from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
from utils.config import BACKEND_CONFIG
from langchain.schema import Document
from scripts.data.data_processor import DataProcessor
from pydantic import BaseModel
from scripts.services import training_scheduler  # Update this import
from scripts.services import question_generator  # Add this import
import json
import logging
import asyncio
from datetime import datetime

router = APIRouter(prefix="/api/data/train", tags=["training"])
logger = logging.getLogger(__name__)

# This file maintains backward compatibility with old endpoints
# All knowledge processing is now done through stats.py

class ProcessingOptions(BaseModel):
    force_update: bool = False
    
@router.post("/process-all")
async def process_all(request: Request, options: ProcessingOptions = None):
    """
    Process knowledge base and generate new questions by redirecting to stats/knowledge/process
    This is maintained for backward compatibility only - use /api/data/stats/knowledge/process instead
    """
    try:
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(status_code=401, detail="Authentication required")
            
        # Default options if none provided
        force_update = False
        if options:
            force_update = options.force_update
            
        # Import and redirect to the more comprehensive knowledge processing endpoint
        logger.info(f"Redirecting to stats/knowledge/process endpoint with force_update={force_update}")
        
        from api.routes.stats import process_knowledge_base
        
        # Forward the request to the stats endpoint with appropriate parameters
        result = await process_knowledge_base(
            force_update=force_update,
            analyze_resources=True,
            analyze_all_resources=force_update,  # Only analyze all if force_update is true
            batch_size=5,
            resource_limit=None  # No limit
        )
        
        # Return the result directly (it's already properly formatted)
        return result
        
    except Exception as e:
        logger.error(f"Error redirecting to knowledge processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/process-status")
async def get_process_status():
    """
    Get the status of knowledge processing
    This is maintained for backward compatibility - processing is now handled by stats/knowledge/process
    """
    # Get the vector stats file to check last processing time
    try:
        data_path = Path(BACKEND_CONFIG['data_path'])
        stats_path = data_path / "stats" / "vector_stats.json"
        
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    
                # Extract relevant information
                return {
                    "is_running": False,  # We don't track this anymore since processing is synchronous
                    "last_result": "success",
                    "last_error": None,
                    "start_time": None,
                    "end_time": stats.get("last_updated"),
                    "processed_data": {
                        "total_documents": stats.get("total", 0),
                        "vectorized": stats.get("vectorized", 0),
                        "last_update": stats.get("last_updated")
                    }
                }
            except Exception as e:
                logger.error(f"Error reading vector stats: {e}")
        
        # Return default status if no stats file
        return {
            "is_running": False,
            "last_result": None,
            "last_error": None,
            "start_time": None,
            "end_time": None,
            "message": "No processing history found"
        }
    except Exception as e:
        logger.error(f"Error getting process status: {e}")
        return {
            "error": str(e)
        }

@router.post("/{data_type}")
async def train_data_type(data_type: str):
    """Train/process specific data type"""
    try:
        data_path = Path(BACKEND_CONFIG['data_path'])
        file_path = data_path / f"{data_type}.json"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"No data found for type: {data_type}")

        with open(file_path, 'r') as f:
            items = json.load(f)

        documents = []
        for item in items:
            content = create_document_content(data_type, item)
            if content:
                doc = Document(
                    page_content=content,
                    metadata={
                        "type": data_type,
                        "title": item.get("title", ""),
                        "created_at": item.get("created_at", ""),
                        "source": f"{data_type}/{item.get('id', '')}"
                    }
                )
                documents.append(doc)

        processor = DataProcessor(data_path)
        result = await processor.create_custom_vector_store(
            documents,
            f"{data_type}_store"
        )

        return {
            "status": "success",
            "message": f"Processed {len(documents)} {data_type} documents",
            "result": result
        }

    except Exception as e:
        logger.error(f"Error training {data_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def create_document_content(data_type: str, item: dict) -> str:
    """Create document content based on data type"""
    if data_type == "projects":
        return (f"Project: {item['title']}\n{item['description']}\n"
                f"Goals: {', '.join(item['goals'])}\n"
                f"Outcomes: {', '.join(item['outcomes'])}")
    elif data_type == "methods":
        return (f"Method: {item['title']}\n{item['usecase']}\n"
                f"Steps: {', '.join(item['steps'])}")
    return ""

class TrainingTime(BaseModel):
    hour: int
    minute: int = 0

@router.get("/schedule")
async def get_training_schedule(request: Request):
    """Get current training schedule"""
    try:
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(status_code=401, detail="Authentication required")
        
        return training_scheduler.get_status()
    except Exception as e:
        logger.error(f"Error getting training schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schedule")
async def set_training_schedule(time: TrainingTime, request: Request):
    """Set training schedule"""
    try:
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if not (0 <= time.hour < 24 and 0 <= time.minute < 60):
            raise HTTPException(status_code=400, detail="Invalid time format")
            
        training_scheduler.set_training_time(time.hour, time.minute)
        return {"success": True, "message": f"Training scheduled for {time.hour:02d}:{time.minute:02d}"}
    except Exception as e:
        logger.error(f"Error setting training schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))
