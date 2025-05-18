from fastapi import APIRouter, HTTPException, Request
from scripts.models.schemas import Resource
from pathlib import Path
import pandas as pd
import numpy as np
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
        # Add more debug logging
        logger.info("Processing resources/list request")
        
        # Authentication check removed
        data_path = Path(BACKEND_CONFIG['data_path'])
        resources_path = data_path / "resources.csv"
        
        logger.info(f"Looking for resources file at {resources_path}")
        
        if not resources_path.exists():
            logger.warning(f"Resources file not found at {resources_path}")
            return {
                "status": "success",
                "data": {
                    "rows": [],
                    "columns": ["title", "url", "description", "type", 
                              "category", "tags", "created_at"],
                    "stats": {"total_count": 0}
                }
            }
        
        logger.info(f"Found resources file, loading data")
            
        # Use a much simpler approach - read everything as strings
        df = pd.read_csv(resources_path, dtype=str)
        
        # Convert NaN to None
        df = df.replace({np.nan: None})
        
        # Very simple approach to process tags
        if 'tags' in df.columns:
            # Process the tags column manually with a list comprehension
            tags_list = []
            for tag in df['tags']:
                if tag is None or str(tag).strip() == '':
                    tags_list.append([])
                else:
                    # Split by comma and strip whitespace
                    tags_list.append([t.strip() for t in str(tag).split(',') if t.strip()])
            
            # Replace the column
            df['tags'] = tags_list
        else:
            df['tags'] = [[] for _ in range(len(df))]
        
        # Convert to records - simple approach
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                # Handle special list columns that might be string representations of lists
                if col not in ['tags'] and isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                    try:
                        record[col] = eval(val)
                    except:
                        record[col] = val
                else:
                    record[col] = val
            records.append(record)
        
        logger.info(f"Successfully processed {len(records)} resources")
        
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
        logger.exception("Stack trace:")  # Log stack trace for debugging
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
