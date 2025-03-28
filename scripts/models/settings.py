from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TrainingSchedule(BaseModel):
    start: str  # Time in HH:MM format
    stop: str   # Time in HH:MM format

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"

class ModelSettings(BaseModel):
    # LLM provider settings
    provider: LLMProvider = LLMProvider.OLLAMA  # Default to Ollama
    model_name: str = "llama3.2:1b"  # Default model for Ollama
    openrouter_model: str = "openai/gpt-3.5-turbo"  # Default model for OpenRouter
    
    # Common settings
    system_prompt: str = "You are a Knowledge Base Assistant that helps with retrieving and explaining information."
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    
    # Ollama specific settings
    top_k: int = 40
    num_ctx: int = 256
    num_thread: int = 4
    
    # Other settings
    stop_sequences: Optional[List[str]] = None
    custom_parameters: Optional[Dict[str, Any]] = None
    training: Optional[TrainingSchedule] = None
