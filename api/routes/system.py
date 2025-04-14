from fastapi import APIRouter, HTTPException, Request
from scripts.models.settings import ModelSettings, TrainingSchedule, LLMProvider
from scripts.services import training_scheduler
from scripts.services.settings_manager import SettingsManager
from utils.config import BACKEND_CONFIG
from scripts.llm.ollama_client import OllamaClient
from scripts.llm.open_router_client import OpenRouterClient
import httpx
import os
import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime, time
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Create router with proper configuration
router = APIRouter(tags=["system"])

logger = logging.getLogger(__name__)

@router.get("/api/status")  # Full path
async def status(request: Request):  # Add request parameter
    """Get system status"""
    try:
        # Read both config files
        user_config_path = Path.home() / '.blob' / 'config.json'
        project_config_path = Path(__file__).parent.parent.parent / 'config.json'
        
        # Get data path directly from project config.json
        data_path = BACKEND_CONFIG['data_path']  # Default
        if project_config_path.exists():
            try:
                with open(project_config_path) as f:
                    project_config = json.load(f)
                    if 'data_path' in project_config:
                        data_path = project_config['data_path']
            except Exception as e:
                logger.error(f"Error reading project config: {e}")
        
        # Get user credentials from user config
        user_config = {}
        if user_config_path.exists():
            try:
                with open(user_config_path) as f:
                    user_config = json.load(f)
            except Exception as e:
                logger.error(f"Error reading user config: {e}")
                
        # Check OpenRouter connection
        openrouter_status = "disconnected"
        if hasattr(request.app.state, 'openrouter_client') and request.app.state.openrouter_client.api_key:
            # Check if OpenRouter client has a valid connection
            try:
                is_connected, _ = await request.app.state.openrouter_client.check_connection()
                openrouter_status = "connected" if is_connected else "disconnected"
            except:
                pass

        return {
            "server": "running",
            "ollama": await check_ollama_status(),
            "openrouter": openrouter_status,
            "tunnel": "connected" if hasattr(request.app.state, 'ngrok_url') else "disconnected",
            "ngrok_url": getattr(request.app.state, 'ngrok_url', ''),
            "data_path": str(data_path),  # Use freshly loaded path 
            "group_id": user_config.get('group_id'),
            "group_image": user_config.get('group_image'),
            "group_name": user_config.get('group_name'),
            "last_connected": user_config.get('last_connected')
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {
            "server": "running",
            "ollama": "disconnected",
            "openrouter": "disconnected",
            "tunnel": "disconnected",
            "ngrok_url": "",
            "data_path": str(BACKEND_CONFIG['data_path'])
        }

@router.get("/health")  # Keep as is
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "local",
        "version": "0.1.0",
        "port": BACKEND_CONFIG['PORT']
    }

@router.get("/api/check-update")  # Full path
async def check_update():
    """Check for system updates"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/repos/yourusername/blob/releases/latest"
            )
            if response.status_code == 200:
                latest = response.json()
                current_version = "0.1.0"
                latest_version = latest['tag_name'].strip('v')
                return {
                    "update_available": latest_version > current_version,
                    "current_version": current_version,
                    "latest_version": latest_version
                }
            return {"update_available": False, "current_version": "0.1.0"}
    except Exception as e:
        logger.error(f"Update check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check for updates")

@router.get("/api/system-settings")  # Full path
async def get_system_settings():
    """Get system settings"""
    try:
        settings_manager = SettingsManager()
        settings = settings_manager.load_settings()
        
        # Add current training schedule
        try:
            # Import directly
            from scripts.services.training_scheduler import get_status
            scheduler_status = get_status()
            
            # Set training schedule in settings if not already set
            if not settings.training and scheduler_status.get("schedule"):
                from scripts.models.settings import TrainingSchedule
                settings.training = TrainingSchedule(
                    start=scheduler_status["schedule"]["start"],
                    stop=scheduler_status["schedule"]["stop"]
                )
            
            # Add scheduler status to the response
            return {
                **settings.model_dump(),
                "scheduler_status": scheduler_status
            }
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return {
                **settings.model_dump(),
                "scheduler_status": {
                    "active": False,
                    "error": str(e)
                }
            }
    except Exception as e:
        logger.error(f"Settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/system-settings")  # Full path
async def update_system_settings(settings: ModelSettings):
    """Update system settings"""
    try:
        settings_manager = SettingsManager()
        
        # Update training schedule if provided
        if settings.training:
            try:
                # Log the received values
                logger.info(f"Received training schedule: start={settings.training.start}, stop={settings.training.stop}")
                
                # Ensure we have valid strings
                if not isinstance(settings.training.start, str) or not isinstance(settings.training.stop, str):
                    raise ValueError(f"Invalid types: start={type(settings.training.start)}, stop={type(settings.training.stop)}")
                
                # Parse start and stop times (HH:MM format)
                try:
                    start_parts = settings.training.start.strip().split(':')
                    stop_parts = settings.training.stop.strip().split(':')
                    
                    if len(start_parts) != 2 or len(stop_parts) != 2:
                        raise ValueError(f"Invalid format: start={settings.training.start}, stop={settings.training.stop}")
                    
                    start_hour = int(start_parts[0])
                    start_minute = int(start_parts[1])
                    stop_hour = int(stop_parts[0])
                    stop_minute = int(stop_parts[1])
                    
                except ValueError as e:
                    logger.error(f"Failed to parse time components: {e}")
                    raise ValueError(f"Invalid time format: {str(e)}")
                
                # Validate times
                if not (0 <= start_hour < 24 and 0 <= start_minute < 60 and
                       0 <= stop_hour < 24 and 0 <= stop_minute < 60):
                    raise ValueError(f"Time values out of range: start={start_hour}:{start_minute}, stop={stop_hour}:{stop_minute}")

                # Import directly and use the function
                from scripts.services.training_scheduler import set_training_time
                set_training_time(
                    start_hour=start_hour,
                    start_minute=start_minute,
                    stop_hour=stop_hour,
                    stop_minute=stop_minute
                )
                logger.info(f"Updated training schedule: {start_hour:02d}:{start_minute:02d} - {stop_hour:02d}:{stop_minute:02d}")
            except Exception as e:
                logger.error(f"Error setting training schedule: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid training schedule format: {str(e)}")

        # Save all settings
        if settings_manager.save_settings(settings):
            # Import directly
            from scripts.services.training_scheduler import get_status
            return {
                "status": "success",
                "message": "Settings updated",
                "training_schedule": get_status() if settings.training else None
            }
        raise HTTPException(status_code=500, detail="Failed to save settings")
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class OpenRouterKeyRequest(BaseModel):
    api_key: str
    
class OpenRouterModelRequest(BaseModel):
    model_id: str
    
class OpenRouterResponse(BaseModel):
    status: str
    message: str
    
class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]

# Helper function for checking Ollama status
async def check_ollama_status():
    """Check Ollama availability"""
    try:
        ollama_url = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}/api/version"
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(ollama_url)
            return "connected" if resp.status_code == 200 else "disconnected"
    except:
        return "disconnected"

async def check_ollama_status():
    """Check Ollama availability"""
    try:
        ollama_url = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}/api/version"
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(ollama_url)
            return "connected" if resp.status_code == 200 else "disconnected"
    except:
        return "disconnected"

@router.get("/api/vector-store-status")
async def vector_store_status(request: Request):
    """Get detailed status of vector stores for debugging"""
    try:
        # Use local Ollama client to check vector stores
        client = getattr(request.app.state, 'ollama_client', None)
        if not client:
            client = OllamaClient()
            request.app.state.ollama_client = client
            
        # Get config info
        config_path = client.get_config_path() if hasattr(client, 'get_config_path') else None
        config_data = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
        # Get vector store paths
        data_path = Path(config_data.get('data_path', ''))
        vector_path = data_path / "vector_stores" / "default"
        reference_path = vector_path / "reference_store"
        content_path = vector_path / "content_store"
        
        # Check for FAISS index files
        reference_index_exists = (reference_path / "index.faiss").exists() if reference_path.exists() else False
        content_index_exists = (content_path / "index.faiss").exists() if content_path.exists() else False
        
        # Get document counts
        reference_count = len(client.reference_store.docstore._dict) if client.reference_store and hasattr(client.reference_store, 'docstore') else 0
        content_count = len(client.content_store.docstore._dict) if client.content_store and hasattr(client.content_store, 'docstore') else 0
        
        # Count topic stores
        topic_stores = []
        if vector_path.exists():
            topic_store_paths = list(vector_path.glob("topic_*"))
            for topic_path in topic_store_paths:
                if (topic_path / "index.faiss").exists():
                    topic_stores.append(topic_path.name)
        
        # Get FAISS index size if possible
        reference_index_size = 0
        content_index_size = 0
        try:
            if reference_index_exists:
                reference_index_size = (reference_path / "index.faiss").stat().st_size
            if content_index_exists:
                content_index_size = (content_path / "index.faiss").stat().st_size
        except Exception as e:
            logger.error(f"Error getting index sizes: {e}")
            
        return {
            "status": "success",
            "vector_stores": {
                "data_path": str(data_path),
                "vector_path": str(vector_path),
                "reference_store": {
                    "path": str(reference_path),
                    "index_exists": reference_index_exists,
                    "index_size_bytes": reference_index_size,
                    "document_count": reference_count,
                    "loaded": client.reference_store is not None
                },
                "content_store": {
                    "path": str(content_path),
                    "index_exists": content_index_exists, 
                    "index_size_bytes": content_index_size,
                    "document_count": content_count,
                    "loaded": client.content_store is not None
                },
                "topic_stores": topic_stores,
                "topic_store_count": len(topic_stores)
            },
            "action_tips": [
                "To rebuild all indexes, run python scripts/tools/rebuild_faiss_indexes.py",
                "For full reprocessing, run python run_complete_processing.py to regenerate all indexes",
                "For incremental processing, use python scripts/tools/run_incremental_processing.py"
            ]
        }
    except Exception as e:
        logger.error(f"Vector store status error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Try running python run_complete_processing.py to regenerate all vector stores"
        }

@router.post("/api/rebuild-faiss-indexes")
async def rebuild_faiss_indexes(request: Request):
    """
    Rebuild all FAISS indexes from existing document stores.
    This ensures consistency between document content and vector indexes.
    """
    try:
        # Use local Ollama client to access vector stores
        client = getattr(request.app.state, 'ollama_client', None)
        if not client:
            client = OllamaClient()
            request.app.state.ollama_client = client
            
        # Get config info to identify data_path
        config_path = client.get_config_path() if hasattr(client, 'get_config_path') else None
        config_data = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        
        data_path = Path(config_data.get('data_path', ''))
        
        # Start time for performance tracking
        start_time = datetime.now()
        logger.info("Starting FAISS index rebuild for all stores")
        
        # Use the new modular code for rebuilding if it exists, otherwise fall back to old method
        try:
            from scripts.data.processing_manager import ProcessingManager
            processing_manager = ProcessingManager(data_path)
            result = await processing_manager.rebuild_all_indexes()
            logger.info("Rebuilding indexes using new modular code")
        except ImportError:
            # Fall back to old implementation
            from scripts.data.data_processor import DataProcessor
            data_processor = DataProcessor(data_path)
            result = await data_processor.rebuild_all_vector_stores()
            logger.info("Rebuilding indexes using legacy code")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Reload the vector stores in the client to use the new indexes
        client.load_vector_stores()
        request.app.state.ollama_client = client
        
        return {
            "status": "success",
            "message": "All FAISS indexes have been rebuilt successfully",
            "processing_time_seconds": processing_time,
            "stores_rebuilt": result.get("stores_rebuilt", []),
            "document_counts": result.get("document_counts", {}),
            "total_documents": result.get("total_documents", 0)
        }
    except Exception as e:
        logger.error(f"Error rebuilding FAISS indexes: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Check server logs for details"
        }

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
