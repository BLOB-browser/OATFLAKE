import json
from pathlib import Path
from typing import Optional
from scripts.models.settings import ModelSettings
import logging
import os

logger = logging.getLogger(__name__)

class SettingsManager:
    def __init__(self):
        # Get the project root directory (where the app.py is located)
        self.project_root = Path(os.path.dirname(os.path.dirname(__file__)))
        self.settings_dir = self.project_root / 'settings'
        self.settings_file = self.settings_dir / 'model_settings.json'
        self._ensure_settings_dir()
        self._load_default_settings()

    def _ensure_settings_dir(self):
        self.settings_dir.mkdir(parents=True, exist_ok=True)

    def _load_default_settings(self):
        if not self.settings_file.exists():
            default_settings = ModelSettings(
                system_prompt="""You are an AI assistant for a Local Knwledgebase.""",
                model_name="llama3.2:1b"
            )
            self.save_settings(default_settings)

    def load_settings(self) -> ModelSettings:
        try:
            if self.settings_file.exists():
                data = json.loads(self.settings_file.read_text())
                return ModelSettings(**data)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return self._load_default_settings()

    def save_settings(self, settings: ModelSettings) -> bool:
        try:
            self.settings_file.write_text(settings.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
