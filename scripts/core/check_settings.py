from pathlib import Path
import json
import os

def check_settings():
    project_root = Path(os.path.dirname(os.path.dirname(__file__)))
    settings_file = project_root / 'settings' / 'model_settings.json'
    
    if settings_file.exists():
        print(f"Settings file found at: {settings_file}")
        print("\nCurrent settings:")
        print(json.dumps(json.loads(settings_file.read_text()), indent=2))
    else:
        print(f"No settings file found at: {settings_file}")

if __name__ == "__main__":
    check_settings()