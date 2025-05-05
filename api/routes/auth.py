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
    """Handle group connection - no authentication required"""
    try:
        logger.info(f"Connection request received for group: {request.group_id}")
        
        # Check if we have group info included in the request
        if hasattr(request, 'group_info') and request.group_info:
            # Directly use provided group info (from frontend)
            logger.info("Using group info provided by frontend")
            group_info = request.group_info
        else:
            # Try to get group info from Supabase if client exists
            supabase_client = None
            if hasattr(current_request.app.state, 'supabase_client') and current_request.app.state.supabase_client:
                supabase_client = current_request.app.state.supabase_client
                
                try:
                    # If we have a client, try to get group info
                    group_info = await supabase_client.get_group_info(request.group_id)
                    if not group_info:
                        logger.warning(f"Group not found in Supabase: {request.group_id}")
                        # Use simplified group info with provided ID
                        group_info = {
                            "id": request.group_id,
                            "name": f"Group {request.group_id[:6]}",
                            "description": "Local group connection"
                        }
                except Exception as e:
                    logger.warning(f"Error fetching group from Supabase: {e}")
                    # Use simplified group info with provided ID
                    group_info = {
                        "id": request.group_id,
                        "name": f"Group {request.group_id[:6]}",
                        "description": "Local group connection"
                    }
            else:
                # No Supabase client, just use basic info
                logger.info("No Supabase client, using basic group info")
                group_info = {
                    "id": request.group_id,
                    "name": f"Group {request.group_id[:6]}",
                    "description": "Local group connection"
                }
        
        # Save the connection information regardless of auth
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
