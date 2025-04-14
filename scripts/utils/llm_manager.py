"""
Manager for LLM clients to ensure we only create one instance where needed.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global instance cache for singleton-like behavior
_client_instances = {}

def get_openrouter_client():
    """Get or create a shared OpenRouterClient instance"""
    global _client_instances
    
    if "openrouter" not in _client_instances:
        from scripts.llm.open_router_client import OpenRouterClient
        logger.info("Creating new OpenRouterClient instance")
        _client_instances["openrouter"] = OpenRouterClient()
    else:
        logger.debug("Using existing OpenRouterClient instance")
        
    return _client_instances["openrouter"]

def get_ollama_client():
    """Get or create a shared OllamaClient instance"""
    global _client_instances
    
    if "ollama" not in _client_instances:
        from scripts.llm.ollama_client import OllamaClient
        logger.info("Creating new OllamaClient instance")
        _client_instances["ollama"] = OllamaClient()
    else:
        logger.debug("Using existing OllamaClient instance")
        
    return _client_instances["ollama"]

def get_resource_llm():
    """Get or create a shared ResourceLLM instance"""
    global _client_instances
    
    if "resource_llm" not in _client_instances:
        from scripts.analysis.resource_llm import ResourceLLM
        from scripts.llm.processor_config_utils import get_adaptive_model_config, get_best_available_model
        
        # Get optimal configuration and model
        model = get_best_available_model()
        config = get_adaptive_model_config()
        
        # Create ResourceLLM instance with optimal settings
        logger.info(f"Creating new ResourceLLM instance with model {model}")
        _client_instances["resource_llm"] = ResourceLLM(model=model)
        _client_instances["resource_llm"].model_config = config
    else:
        logger.debug("Using existing ResourceLLM instance")
        
    return _client_instances["resource_llm"]
