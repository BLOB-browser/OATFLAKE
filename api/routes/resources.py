from fastapi import APIRouter, HTTPException, Request
from scripts.models.schemas import Resource
from pathlib import Path
import pandas as pd
import json
import logging
from datetime import datetime
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/resources", tags=["resources"])
logger = logging.getLogger(__name__)

@router.post("")
async def create_resource(resource: Resource, request: Request):
    """Create a new resource"""
    try:
        # Verify authentication
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(
                status_code=401,
                detail="Authentication required"
            )

        # Set timestamp if not provided
        if not resource.created_at:
            resource.created_at = datetime.now()

        # Get data path from config
        data_path = Path(BACKEND_CONFIG['data_path'])
        resources_path = data_path / "resources.csv"
        
        # Ensure directory exists
        resources_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame row
        resource_data = resource.model_dump()
        df = pd.DataFrame([resource_data])
        
        # Convert lists to comma-separated strings for CSV storage
        df['tags'] = df['tags'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        
        if resources_path.exists():
            existing_df = pd.read_csv(resources_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        df.to_csv(resources_path, index=False)
        
        return {
            "status": "success",
            "message": "Resource created successfully",
            "data": resource_data
        }

    except Exception as e:
        logger.error(f"Error creating resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_resources(request: Request):
    """List all resources"""
    try:
        if not getattr(request.app.state, "supabase_client", None):
            raise HTTPException(status_code=401, detail="Authentication required")

        data_path = Path(BACKEND_CONFIG['data_path'])
        resources_path = data_path / "resources.csv"
        
        if not resources_path.exists():
            return {
                "status": "success",
                "data": {
                    "rows": [],
                    "columns": ["title", "url", "description", "type", 
                              "category", "tags", "created_at"],
                    "stats": {"total_count": 0}
                }
            }
            
        df = pd.read_csv(resources_path)
        df = df.replace({pd.NA: None})
        
        # Create a copy of the tags column before processing to avoid modifying the original
        if 'tags' in df.columns:
            # Process tags column - handle strings, lists, and empty values
            df['tags'] = df['tags'].apply(lambda x: 
                ([] if pd.isna(x) or x is None or x == '' else
                 x.split(',') if isinstance(x, str) else
                 x if isinstance(x, list) else [])
            )
        else:
            # If no tags column, add an empty one
            df['tags'] = [[]] * len(df)
        
        return {
            "status": "success",
            "data": {
                "rows": df.to_dict('records'),
                "columns": df.columns.tolist(),
                "stats": {
                    "total_count": len(df),
                    "last_updated": datetime.fromtimestamp(
                        resources_path.stat().st_mtime
                    ).isoformat()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        raise HTTPException(status_code=500, detail=str(e))
