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

def load_config():
    """Load the main project config.json as a dict."""
    project_root = Path(__file__).parent.parent
    project_config_path = project_root / 'config.json'
    user_config_path = Path.home() / '.blob' / 'config.json'

    if project_config_path.exists():
        try:
            with open(project_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading config.json: {e}")
            pass

    if user_config_path.exists():
        try:
            with open(user_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading user config.json: {e}")
            pass

    return {}

def save_to_env(key, value):
    """Save a key-value pair to the .env file."""
    env_path = Path(__file__).parent.parent / '.env'
    try:
        # Read existing .env content
        if env_path.exists():
            with open(env_path, 'r') as f:
                lines = f.readlines()
        else:
            lines = []

        # Update or add the key-value pair
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}={value}\n"
                updated = True
                break
        if not updated:
            lines.append(f"{key}={value}\n")

        # Write back to the .env file
        with open(env_path, 'w') as f:
            f.writelines(lines)
    except Exception as e:
        print(f"Error saving to .env: {e}")

def save_config(config_data):
    """Save config data to the main project config.json."""
    project_root = Path(__file__).parent.parent
    project_config_path = project_root / 'config.json'
    
    try:
        with open(project_config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config.json: {e}")
        return False

# Basic configuration
BACKEND_CONFIG = {
    'PORT': int(os.getenv('PORT', '8999')),
    'HOST': os.getenv('HOST', '127.0.0.1'),
    'data_path': get_data_path()
}
