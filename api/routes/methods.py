from fastapi import APIRouter, HTTPException, Request
from scripts.models.schemas import Method
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/methods", tags=["methods"])
logger = logging.getLogger(__name__)

@router.post("")
async def create_method(method: Method, request: Request):
    """Create a new method"""
    try:
        # Authentication check removed
        
        logger.info(f"Received method data: {method.model_dump()}")
        
        # Set default values and timestamp
        if not method.created_at:
            method.created_at = datetime.now()
        method.tags = method.tags or []
        method.steps = method.steps or []

        # Get data path from config
        data_path = Path(BACKEND_CONFIG['data_path'])
        methods_path = data_path / "methods.csv"
        
        # Ensure directory exists
        methods_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame row
        method_data = method.model_dump()
        
        try:
            # Ensure tags and steps are lists before converting to JSON
            if not isinstance(method_data.get('tags', []), list):
                method_data['tags'] = []
            if not isinstance(method_data.get('steps', []), list):
                method_data['steps'] = []
                
            # Convert lists to strings for CSV storage
            method_data['steps'] = json.dumps(method_data['steps'])
            method_data['tags'] = json.dumps(method_data['tags'])
        except Exception as e:
            logger.error(f"Error converting lists to JSON: {e}")
            method_data['steps'] = json.dumps([])
            method_data['tags'] = json.dumps([])
        
        df = pd.DataFrame([method_data])
        
        if methods_path.exists():
            existing_df = pd.read_csv(methods_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            
        # Handle any potential NaN values before saving
        df = df.replace({np.nan: None, float('nan'): None, pd.NA: None})
        df.to_csv(methods_path, index=False)
        
        # Convert back to original format for response
        method_data['steps'] = method.steps
        method_data['tags'] = method.tags
        method_data['id'] = str(len(df) - 1)
        
        # Ensure method_data is JSON serializable
        clean_data = {}
        for key, value in method_data.items():
            try:
                # Handle scalar values
                if hasattr(pd, 'isna') and callable(pd.isna) and pd.isna(value):
                    clean_data[key] = None
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    clean_data[key] = None
                else:
                    clean_data[key] = value
            except ValueError:
                # If pd.isna fails on an array, handle each element
                if isinstance(value, (list, np.ndarray)):
                    clean_data[key] = value  # Lists are already JSON serializable
                else:
                    clean_data[key] = None
        
        return {
            "status": "success",
            "message": "Method created successfully",
            "data": clean_data
        }

    except Exception as e:
        logger.error(f"Error creating method: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_methods(request: Request):
    """List all methods"""
    try:
        # Authentication check removed
        
        data_path = Path(BACKEND_CONFIG['data_path'])
        methods_path = data_path / "methods.csv"
        
        if not methods_path.exists():
            return {
                "status": "success",
                "data": {
                    "rows": [],
                    "columns": ["title", "description", "steps", "tags", 
                              "group_id", "creator_id", "created_at"],
                    "stats": {"total_count": 0}
                }
            }
            
        df = pd.read_csv(methods_path)
        df = df.replace({pd.NA: None, float('nan'): None, np.nan: None})
        
        # Ensure all required columns exist
        required_columns = ['title', 'description', 'steps', 'tags', 'group_id', 'creator_id', 'created_at']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
                
        # Handle different formats in steps and tags
        def safe_process_list(list_value):
            # Check for NaN or None values
            if pd.isna(list_value) or list_value is None:
                return []
                
            # If it's already a list, return it
            if isinstance(list_value, list):
                return list_value
                
            try:
                # Try to parse as JSON first
                parsed = json.loads(list_value)
                # Ensure the result is a list
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [parsed]
            except (json.JSONDecodeError, TypeError):
                # If not valid JSON, try splitting by pipe or comma
                if isinstance(list_value, str):
                    if '|' in list_value:
                        return [item.strip() for item in list_value.split('|') if item.strip()]
                    else:
                        # For steps that are paragraphs, return as a single item in the list
                        if '\n' in list_value or len(list_value) > 100:
                            return [list_value]
                        # Otherwise attempt comma split
                        return [item.strip() for item in list_value.split(',') if item.strip()]
                return []
                
        # Convert stored strings back to lists with error handling
        df['steps'] = df['steps'].apply(lambda x: safe_process_list(x) if not pd.isna(x) else [])
        df['tags'] = df['tags'].apply(lambda x: safe_process_list(x) if not pd.isna(x) else [])
        
        # Convert to records and ensure all values are JSON serializable
        records = []
        for record in df.to_dict('records'):
            # Ensure all values in the record are JSON serializable
            clean_record = {}
            for key, value in record.items():
                try:
                    # Handle scalar values
                    if hasattr(pd, 'isna') and callable(pd.isna) and pd.isna(value):
                        clean_record[key] = None
                    elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                        clean_record[key] = None
                    else:
                        clean_record[key] = value
                except ValueError:
                    # If pd.isna fails on an array, handle each element
                    if isinstance(value, (list, np.ndarray)):
                        clean_record[key] = value  # Lists are already JSON serializable
                    else:
                        clean_record[key] = None
            records.append(clean_record)
        
        return {
            "status": "success",
            "data": {
                "rows": records,
                "columns": df.columns.tolist(),
                "stats": {
                    "total_count": len(df),
                    "last_updated": datetime.fromtimestamp(
                        methods_path.stat().st_mtime
                    ).isoformat()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error listing methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))
