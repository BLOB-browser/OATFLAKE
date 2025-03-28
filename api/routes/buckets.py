from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
from pathlib import Path
from scripts.storage.store_types import StoreType, BucketMetadata
from scripts.storage.bucket_manager import BucketManager
from langchain.schema import Document
from utils.config import BACKEND_CONFIG
import json
import logging

router = APIRouter(prefix="/api/bucket", tags=["buckets"])
logger = logging.getLogger(__name__)

def get_current_group_id() -> str:
    """Helper to get current group ID from config"""
    config_path = Path.home() / '.blob' / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            return config.get('group_id', 'default')
    return "default"

@router.post("/{bucket_type}")
async def create_bucket(
    bucket_type: StoreType,
    metadata: BucketMetadata,
    documents: List[Dict]
):
    """Create or update a bucket"""
    try:
        document_objects = [
            Document(
                page_content=doc["content"],
                metadata=doc.get("metadata", {})
            ) for doc in documents
        ]
        
        bucket_manager = BucketManager(
            group_id=get_current_group_id(),
            base_path=Path(BACKEND_CONFIG['data_path'])
        )
        
        success = await bucket_manager.create_or_update_bucket(
            bucket_type=bucket_type,
            documents=document_objects,
            metadata=metadata
        )
        
        return {
            "status": "success" if success else "error",
            "message": f"Bucket {bucket_type.value} {'updated' if success else 'failed'}"
        }
        
    except Exception as e:
        logger.error(f"Error creating bucket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/search")  # Update route to match app.py
async def search_buckets(
    query: str,
    bucket_type: Optional[StoreType] = None,
    top_k: int = 5
):
    """Search across buckets"""
    try:
        bucket_manager = BucketManager(
            group_id=get_current_group_id(),
            base_path=Path(BACKEND_CONFIG['data_path'])
        )
        
        results = await bucket_manager.search_bucket(  # Updated to match app.py
            bucket_type=bucket_type,
            query=query,
            top_k=top_k
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching buckets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/info")
async def get_bucket_info(bucket_type: Optional[StoreType] = None):
    """Get bucket information"""
    try:
        bucket_manager = BucketManager(
            group_id=get_current_group_id(),
            base_path=Path(BACKEND_CONFIG['data_path'])
        )
        
        return bucket_manager.get_bucket_info(bucket_type)
        
    except Exception as e:
        logger.error(f"Error getting bucket info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{bucket_type}/{name}")
async def delete_bucket(bucket_type: StoreType, name: str):
    """Delete a specific bucket"""
    try:
        bucket_manager = BucketManager(
            group_id=get_current_group_id(),
            base_path=Path(BACKEND_CONFIG['data_path'])
        )
        success = await bucket_manager.delete_bucket(bucket_type, name)
        return {"status": "success" if success else "error"}
    except Exception as e:
        logger.error(f"Error deleting bucket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{bucket_type}/list")
async def list_buckets(bucket_type: StoreType):
    """List all buckets of a specific type"""
    try:
        bucket_manager = BucketManager(
            group_id=get_current_group_id(),
            base_path=Path(BACKEND_CONFIG['data_path'])
        )
        return bucket_manager.get_bucket_info(bucket_type)
    except Exception as e:
        logger.error(f"Error listing buckets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{bucket_type}/{name}/update")
async def update_bucket_metadata(
    bucket_type: StoreType,
    name: str,
    metadata: BucketMetadata
):
    """Update bucket metadata"""
    try:
        bucket_manager = BucketManager(
            group_id=get_current_group_id(),
            base_path=Path(BACKEND_CONFIG['data_path'])
        )
        success = await bucket_manager.update_bucket_metadata(
            bucket_type,
            name,
            metadata
        )
        return {"status": "success" if success else "error"}
    except Exception as e:
        logger.error(f"Error updating bucket metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))
