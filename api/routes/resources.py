from fastapi import APIRouter, HTTPException, Request
from scripts.models.schemas import Resource
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/data/resources", tags=["resources"])
logger = logging.getLogger(__name__)

@router.post("")
async def create_resource(resource: Resource, request: Request):
    """Create a new resource"""
    try:
        # Authentication check removed
        
        # Log received data for debugging
        logger.info(f"Received resource data: title='{resource.title}', url='{resource.origin_url}'")
        logger.debug(f"Full resource data: {resource.model_dump()}")

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
            
        # Handle any potential NaN values before saving
        df = df.replace({np.nan: None, float('nan'): None, pd.NA: None})
        df.to_csv(resources_path, index=False)
        
        # Ensure resource_data is JSON serializable
        clean_data = {}
        for key, value in resource_data.items():
            if pd.isna(value):
                clean_data[key] = None
            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                clean_data[key] = None
            else:
                clean_data[key] = value
        
        return {
            "status": "success",
            "message": "Resource created successfully",
            "data": clean_data
        }

    except Exception as e:
        logger.error(f"Error creating resource: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_resources(request: Request):
    """List all resources"""
    try:
        # Authentication check removed
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
        df = df.replace({pd.NA: None, float('nan'): None, np.nan: None})
        
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
        
        # Convert to records and ensure all values are JSON serializable
        records = []
        for record in df.to_dict('records'):
            # Ensure all values in the record are JSON serializable
            clean_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    clean_record[key] = None
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    clean_record[key] = None
                else:
                    clean_record[key] = value
            records.append(clean_record)
        
        return {
            "status": "success",
            "data": {
                "rows": records,
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

@router.post("/debug")
async def debug_resource_request(request: Request):
    """Debug endpoint to help diagnose resource upload issues"""
    try:
        # Get the raw request body
        body = await request.body()
        body_str = body.decode()
        
        # Try to parse as JSON
        try:
            body_json = json.loads(body_str)
        except json.JSONDecodeError:
            body_json = {"error": "Not valid JSON"}
        
        # Return all info about the request
        return {
            "status": "debug",
            "headers": dict(request.headers),
            "raw_body": body_str[:500] + "..." if len(body_str) > 500 else body_str,
            "parsed_body": body_json,
            "content_type": request.headers.get("content-type"),
            "expected_fields": {
                "required": ["title", "origin_url or url"],
                "optional": ["description", "tags", "purpose", "related_url", "status", 
                            "creator_id", "collaborators", "group_id", "visibility", "category"]
            }
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        return {
            "status": "error",
            "message": str(e),
            "expected_fields": {
                "required": ["title", "origin_url or url"],
                "optional": ["description", "tags", "purpose", "related_url", "status", 
                            "creator_id", "collaborators", "group_id", "visibility", "category"]
            }
        }
