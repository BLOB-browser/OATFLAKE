from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
import os
from pathlib import Path

# Create router with proper configuration
router = APIRouter(tags=["ngrok"])

logger = logging.getLogger(__name__)

@router.get("/api/system/get-ngrok-status")
async def get_ngrok_status(request: Request):
    """Get status of Ngrok configuration"""
    try:
        # Check if ngrok token is set in environment
        has_token = os.getenv("NGROK_AUTH_TOKEN") is not None
        
        # Check if there's an active tunnel
        has_tunnel = hasattr(request.app.state, 'ngrok_url') and request.app.state.ngrok_url is not None
        
        # Get the tunnel URL if it exists
        tunnel_url = getattr(request.app.state, 'ngrok_url', None)
        
        # Check if token is saved in config
        token_saved_in_config = False
        try:
            # Check user's home directory for .env file
            env_path = Path.home() / '.blob' / '.env'
            if env_path.exists():
                with open(env_path, 'r') as f:
                    content = f.read()
                    token_saved_in_config = 'NGROK_AUTH_TOKEN=' in content
        except Exception as e:
            logger.error(f"Error checking for saved token: {e}")
        
        return {
            "hasToken": has_token or token_saved_in_config,
            "hasTunnel": has_tunnel,
            "tunnelUrl": tunnel_url,
            "tokenSource": "environment" if has_token else "config" if token_saved_in_config else None
        }
    except Exception as e:
        logger.error(f"Error checking Ngrok status: {e}")
        return {
            "hasToken": False,
            "hasTunnel": False,
            "error": str(e)
        }

class NgrokTokenRequest(BaseModel):
    token: str

@router.post("/api/system/set-ngrok-token")
async def set_ngrok_token(request: NgrokTokenRequest):
    """Save Ngrok auth token to configuration"""
    try:
        if not request.token:
            return {
                "success": False,
                "message": "Token cannot be empty"
            }
        
        # Create .blob directory in user's home if it doesn't exist
        blob_dir = Path.home() / '.blob'
        blob_dir.mkdir(exist_ok=True)
        
        # Create or update .env file with the token
        env_path = blob_dir / '.env'
        
        env_content = ""
        if env_path.exists():
            # Read existing content to preserve other variables
            with open(env_path, 'r') as f:
                lines = f.readlines()
                # Filter out any existing NGROK_AUTH_TOKEN line
                env_content = "".join([line for line in lines if not line.startswith('NGROK_AUTH_TOKEN=')])
        
        # Add the token
        with open(env_path, 'w') as f:
            f.write(f"{env_content}\nNGROK_AUTH_TOKEN={request.token}\n")
        
        # Set in current environment
        os.environ["NGROK_AUTH_TOKEN"] = request.token
        
        logger.info("Ngrok auth token saved successfully")
        
        return {
            "success": True,
            "message": "Ngrok token saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving Ngrok token: {e}")
        return {
            "success": False,
            "message": str(e)
        }
