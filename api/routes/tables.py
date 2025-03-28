from fastapi import APIRouter, HTTPException
from pathlib import Path
from utils.config import BACKEND_CONFIG
import pandas as pd
import logging
from datetime import datetime

router = APIRouter(prefix="/api/data/tables", tags=["tables"])
logger = logging.getLogger(__name__)

@router.get("/{table_name}")
async def get_table_data(table_name: str, page: int = 1, page_size: int = 50):
    """Get paginated data for a specific table"""
    try:
        data_path = Path(BACKEND_CONFIG['data_path'])
        table_path = data_path / "mdef" / f"{table_name}.csv"
        
        logger.info(f"Looking for table at: {table_path}")
        
        if not table_path.exists():
            raise HTTPException(status_code=404, detail=f"Table {table_name} not found")

        df = pd.read_csv(table_path)
        
        total_rows = len(df)
        total_pages = (total_rows + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return {
            "status": "success",
            "data": {
                "name": table_name,
                "columns": df.columns.tolist(),
                "rows": df.iloc[start_idx:end_idx].to_dict('records'),
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                    "total_rows": total_rows
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting table data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
