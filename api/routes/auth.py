from fastapi import APIRouter, Request, HTTPException
from scripts.storage.supabase import SupabaseClient
from scripts.models.schemas import ConnectionRequest, ConnectionResponse
from scripts.services import connection
from pathlib import Path
import json
import logging
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)

@router.post("/login")
async def login(request: Request):
    """Handle user login"""
    try:
        data = await request.json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
            
        try:
            supabase = SupabaseClient()
            auth_result = await supabase.login(email, password)
            
            # Store the authenticated client globally in app state
            request.app.state.supabase_client = supabase  # Add this line
            
            # Update config
            config_path = Path.home() / '.blob' / 'config.json'
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    
            config.update({
                'supabase_email': email,
                'supabase_token': auth_result['token'],
                'supabase_refresh_token': auth_result['refresh_token']
            })
            
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Login successful for: {email}")
            return auth_result
            
        except ValueError as ve:
            logger.error(f"Login failed: {ve}")
            raise HTTPException(status_code=401, detail=str(ve))
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/connect")
async def connect(request: ConnectionRequest, current_request: Request):
    """Handle group connection"""
    try:
        logger.info(f"Connection request received for group: {request.group_id}")
        
        if not current_request.app.state.supabase_client:
            logger.error("No authenticated Supabase client")
            raise HTTPException(status_code=401, detail="Please login first")

        # Use the stored authenticated client from app state
        supabase_client = current_request.app.state.supabase_client
        group_info = await supabase_client.get_group_info(request.group_id)
        
        if not group_info:
            logger.error(f"Group not found: {request.group_id}")
            raise HTTPException(status_code=404, detail="Group not found")
        
        result = await connection.save_connection(request.group_id, group_info)
        logger.info(f"Connection successful: {result}")
        return ConnectionResponse(**result)
        
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def auth_status():
    """Get current auth status"""
    try:
        config_path = Path.home() / '.blob' / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                return {
                    "authenticated": bool(config.get('supabase_token')),
                    "user_email": config.get('supabase_email'),
                    "group_id": config.get('group_id'),
                    "group_name": config.get('group_name')
                }
        return {"authenticated": False}
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/logout")
async def logout():
    """Handle user logout"""
    try:
        config_path = Path.home() / '.blob' / 'config.json'
        if config_path.exists():
            config = {}
            with open(config_path) as f:
                config = json.load(f)
            # Clear auth tokens but keep other settings
            config.pop('supabase_token', None)
            config.pop('supabase_refresh_token', None)
            config.pop('supabase_email', None)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
