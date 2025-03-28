import httpx
import os
import json
from pathlib import Path
import logging
from utils.config import BACKEND_CONFIG

logger = logging.getLogger(__name__)

async def check_ollama_status() -> str:
    """Check if Ollama is running and responding"""
    try:
        ollama_url = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}/api/version"
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(ollama_url)
            return "connected" if resp.status_code == 200 else "disconnected"
    except:
        return "disconnected"

def get_system_status() -> dict:
    """Get complete system status"""
    try:
        config_path = Path.home() / '.blob' / 'config.json'
        data_path = BACKEND_CONFIG['data_path']
        config = {}
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    data_path = config.get('data_path', data_path)
            except Exception as e:
                logger.error(f"Error reading config: {e}")
        
        return {
            "server": "running",
            "data_path": str(data_path),
            "group_id": config.get('group_id'),
            "last_connected": config.get('last_connected')
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise
