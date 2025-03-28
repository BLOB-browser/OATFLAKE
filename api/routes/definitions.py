from fastapi import APIRouter, HTTPException
from scripts.models.schemas import Definition
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/api/data/definitions", tags=["definitions"])
logger = logging.getLogger(__name__)

@router.post("")
async def add_definition(definition: Definition):
    """Add a new definition"""
    try:
        if not definition.created_at:
            definition.created_at = datetime.now()

        # Get data path from config
        data_path = Path(BACKEND_CONFIG['data_path'])
        definitions_path = data_path / "definitions.csv"
        
        # Ensure directory exists
        definitions_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame row
        definition_data = definition.model_dump()
        df = pd.DataFrame([definition_data])
        
        if definitions_path.exists():
            existing_df = pd.read_csv(definitions_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_csv(definitions_path, index=False)
        
        return {
            "status": "success",
            "message": "Definition saved successfully",
            "data": definition_data
        }

    except Exception as e:
        logger.error(f"Error saving definition: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_definitions():
    """List all definitions"""
    try:
        data_path = Path(BACKEND_CONFIG['data_path'])
        definitions_path = data_path / "definitions.csv"
        
        if not definitions_path.exists():
            return {
                "status": "success",
                "data": {
                    "rows": [],
                    "columns": ["term", "definition", "tags", "source", "created_at"],
                    "stats": {"total_count": 0}
                }
            }
            
        df = pd.read_csv(definitions_path)
        df = df.replace({pd.NA: None})
        
        return {
            "status": "success",
            "data": {
                "rows": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "stats": {
                    "total_count": len(df),
                    "last_updated": datetime.fromtimestamp(
                        definitions_path.stat().st_mtime
                    ).isoformat()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing definitions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
