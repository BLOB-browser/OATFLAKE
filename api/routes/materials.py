from fastapi import APIRouter, HTTPException, File, UploadFile, Form, Request
from typing import List
from scripts.services import storage
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import json
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/api/data/materials", tags=["materials"])
logger = logging.getLogger(__name__)

@router.post("")
async def add_reading_material(
    title: str = Form(...),
    description: str = Form(None),
    fields: List[str] = Form([]),
    file: UploadFile = File(...),
):
    """Add reading material with file"""
    try:
        if not file.content_type == "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files accepted")
            
        material_data = {
            "title": title,
            "description": description,
            "fields": fields,
            "created_at": datetime.now().isoformat()
        }
        
        if storage.save_reading_material(material_data, file):
            return {
                "status": "success",
                "data": material_data
            }
        raise HTTPException(status_code=500, detail="Error saving material")
    except Exception as e:
        logger.error(f"Error adding material: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_materials(request: Request):
    """List all materials"""
    try:
        if not getattr(request.app.state, "supabase_client", None):
            logger.warning("No authenticated client")
            raise HTTPException(status_code=401, detail="Authentication required")

        data_path = Path(BACKEND_CONFIG['data_path'])
        materials_path = data_path / "materials.csv"
        
        logger.info(f"Looking for materials CSV at: {materials_path}")
        
        if not materials_path.exists():
            # If no materials file exists yet, return empty data
            return {
                "status": "success",
                "data": {
                    "rows": [],
                    "columns": [
                        'title', 'description', 'file_path', 'fields',
                        'creator_id', 'created_at'
                    ],
                    "stats": {
                        "total_count": 0,
                        "last_updated": datetime.now().isoformat()
                    }
                }
            }
        
        # Read existing materials CSV
        df = pd.read_csv(materials_path)
        
        # Replace NaN with None in the entire DataFrame
        df = df.replace({np.nan: None})
        
        # Additionally handle object columns
        string_columns = df.select_dtypes(include=['object']).columns
        df[string_columns] = df[string_columns].fillna('')
        
        # Convert timestamps to ISO format strings
        if 'created_at' in df.columns:
            df['created_at'] = df['created_at'].apply(lambda x: x if pd.isna(x) else pd.to_datetime(x).isoformat())
        
        return {
            "status": "success",
            "data": {
                "rows": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "stats": {
                    "total_count": len(df),
                    "last_updated": datetime.fromtimestamp(
                        materials_path.stat().st_mtime
                    ).isoformat() if materials_path.exists() else datetime.now().isoformat()
                }
            }
        }
    except Exception as e:
        logger.error(f"Error listing materials: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
