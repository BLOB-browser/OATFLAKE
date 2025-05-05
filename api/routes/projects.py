from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
from scripts.models.schemas import Project
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/api/data/projects", tags=["projects"])
logger = logging.getLogger(__name__)

@router.get("/list")
async def list_projects(request: Request):
    """List all projects from projects.csv"""
    try:
        # Authentication check removed
        data_path = Path(BACKEND_CONFIG['data_path'])
        projects_path = data_path / "projects.csv"
        
        logger.info(f"Looking for projects CSV at: {projects_path}")
        
        if not projects_path.exists():
            # Create new DataFrame with empty columns - using only tags (no fields)
            df = pd.DataFrame(columns=[
                'title', 'description', 'goals', 'achievement',
                'documentation_url', 'tags', 'privacy', 'status',
                'creator_id', 'created_at', 'modified_at'
            ])
        else:
            # Read existing CSV and handle NaN values
            df = pd.read_csv(projects_path)
            
            # Replace NaN with None in the entire DataFrame
            df = df.replace({np.nan: None, float('nan'): None, pd.NA: None})
            
            # Additionally handle object columns
            string_columns = df.select_dtypes(include=['object']).columns
            df[string_columns] = df[string_columns].fillna('')
            
            # Handle numeric columns
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Convert timestamps to ISO format strings
        if 'created_at' in df.columns:
            df['created_at'] = df['created_at'].apply(lambda x: x if pd.isna(x) else pd.to_datetime(x).isoformat())
        if 'modified_at' in df.columns:
            df['modified_at'] = df['modified_at'].apply(lambda x: x if pd.isna(x) else pd.to_datetime(x).isoformat())
        
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
                        projects_path.stat().st_mtime
                    ).isoformat() if projects_path.exists() else datetime.now().isoformat()
                }
            }
        }
    except Exception as e:
        logger.error(f"Error listing projects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
async def create_project(project: Project):
    """Create a new project"""
    try:
        # Set timestamps if not provided
        if not project.created_at:
            project.created_at = datetime.now()
        project.modified_at = datetime.now()

        # Convert to CSV row format
        project_data = project.model_dump()
        
        # Read existing CSV
        data_path = Path(BACKEND_CONFIG['data_path'])
        projects_path = data_path / "projects.csv"
        
        # Ensure mdef directory exists
        projects_path.parent.mkdir(parents=True, exist_ok=True)

        # Create or append to CSV
        df = pd.DataFrame([project_data])
        if projects_path.exists():
            existing_df = pd.read_csv(projects_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(projects_path, index=False)
        
        return {
            "status": "success",
            "message": "Project created successfully",
            "data": project_data
        }

    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update/{project_id}")
async def update_project(project_id: int, project: Project):
    """Update an existing project"""
    try:
        project.modified_at = datetime.now()

        # Read existing CSV
        data_path = Path(BACKEND_CONFIG['data_path'])
        projects_path = data_path / "projects.csv"

        if not projects_path.exists():
            raise HTTPException(status_code=404, detail="Projects file not found")

        df = pd.read_csv(projects_path)
        if project_id >= len(df):
            raise HTTPException(status_code=404, detail="Project not found")

        # Update project data
        project_data = project.model_dump()
        for key, value in project_data.items():
            df.at[project_id, key] = value

        # Handle any potential NaN values that could cause JSON serialization issues
        df = df.replace({np.nan: None, float('nan'): None, pd.NA: None})
        
        # Save back to CSV
        df.to_csv(projects_path, index=False)

        # Ensure project_data is JSON serializable
        clean_data = {}
        for key, value in project_data.items():
            if pd.isna(value):
                clean_data[key] = None
            elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                clean_data[key] = None
            else:
                clean_data[key] = value

        return {
            "status": "success",
            "message": "Project updated successfully",
            "data": clean_data
        }

    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))
