import os
import json
from pathlib import Path

def setup_environment():
    """Setup necessary directories and files"""
    # Create .blob directory
    blob_dir = Path.home() / ".blob"
    blob_dir.mkdir(exist_ok=True)
    
    # Load or create config
    config_file = blob_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}
    
    # Set default data path if not exists
    if 'data_path' not in config:
        config['data_path'] = str(blob_dir / "data")
    
    # Create data directory
    data_dir = Path(config['data_path'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(config_file, "w") as f:
        json.dump(config, f)
    
    return True

if __name__ == "__main__":
    setup_environment()