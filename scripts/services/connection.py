import json
from pathlib import Path
from datetime import datetime
import logging
from scripts.storage.supabase import SupabaseClient, DEFAULT_GROUP_IMAGE  # Import the constant
from fastapi import HTTPException
import os
import httpx

logger = logging.getLogger(__name__)

async def save_connection(uuid: str, group_info: dict) -> dict:
    """Save connection details to config"""
    try:
        logger.info(f"Saving connection for group: {uuid}")
        
        # Process cover image URL
        cover_image = group_info.get('cover_image')
        supabase_url = os.getenv('SUPABASE_URL')
        
        if cover_image:
            if not cover_image.startswith('http'):
                # Construct full Supabase storage URL
                cover_image = f"{supabase_url}/storage/v1/object/public/group-covers/{cover_image}"
                logger.info(f"Using Supabase storage URL: {cover_image}")
            else:
                logger.info("Using provided full URL")
        else:
            cover_image = DEFAULT_GROUP_IMAGE
            logger.info("Using local default image")

        # Generate a display name from available fields
        group_name = (
            group_info.get('institution_type') or 
            group_info.get('type') or 
            'Unknown Group'
        )

        config_path = Path.home() / '.blob' / 'config.json'
        
        # Load existing config
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        
        # Update config with complete group info
        config.update({
            'group_id': uuid,
            'group_name': group_name,
            'group_image': cover_image,
            'group_type': group_info.get('type'),
            'group_description': group_info.get('description'),
            'group_institution': group_info.get('institution_type'),
            'group_backend_url': group_info.get('backend_url'),
            'group_fields': group_info.get('fields', []),
            'last_connected': datetime.now().isoformat()
        })
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Connection saved for group: {group_name}")
        
        return {
            "status": "success",
            "message": f"Connected to {group_name}",
            "server_version": "0.1.0",
            "group_name": group_name,
            "group_image": cover_image,
            "group_backend_url": group_info.get('backend_url'),
            "group_type": group_info.get('type', 'unknown')
        }
        
    except Exception as e:
        logger.error(f"Error saving connection: {e}")
        raise ValueError(f"Failed to save connection: {str(e)}")

def get_connection_info() -> dict:
    """Get current connection information"""
    try:
        config_path = Path.home() / '.blob' / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return {
                    "group_id": config.get('group_id'),
                    "last_connected": config.get('last_connected')
                }
        return {"group_id": None, "last_connected": None}
    except Exception as e:
        logger.error(f"Error getting connection info: {e}")
        raise
