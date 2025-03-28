import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def get_data_path():
    """Get configured data path from project config.json"""
    # First try project root config.json
    project_root = Path(__file__).parent.parent
    project_config_path = project_root / 'config.json'
    
    # Default path if no config is found
    default_path = str(Path.home() / 'blob' / 'data')
    
    if project_config_path.exists():
        try:
            with open(project_config_path, 'r') as f:
                config = json.load(f)
            return config.get('data_path', default_path)
        except Exception as e:
            print(f"Error reading config.json: {e}")
            pass
            
    # Fallback to user home directory config
    user_config_path = Path.home() / '.blob' / 'config.json'
    if user_config_path.exists():
        try:
            with open(user_config_path, 'r') as f:
                config = json.load(f)
            return config.get('data_path', default_path)
        except:
            pass
            
    return default_path

# Basic configuration
BACKEND_CONFIG = {
    'PORT': int(os.getenv('PORT', '8999')),
    'HOST': os.getenv('HOST', '127.0.0.1'),
    'data_path': get_data_path()
}
