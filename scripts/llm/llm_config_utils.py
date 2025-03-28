#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for LLM configuration to ensure consistency across different LLM interfaces.
"""

import logging
import multiprocessing
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Default model to use across all LLM classes
DEFAULT_MODEL = "mistral:7b-instruct-v0.2-q4_0"

def get_adaptive_model_config() -> Dict[str, Any]:
    """
    Get a model configuration that adapts to the available hardware resources.
    
    This ensures consistent LLM settings across different classes while
    allowing each machine to utilize its available resources.
    
    Returns:
        Dictionary with model configuration parameters
    """
    # Get available CPU cores for threading
    available_cores = multiprocessing.cpu_count()
    recommended_threads = max(4, min(available_cores, 16))  # Min 4, max 16 threads
    
    # Get memory info if available
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        # Adjust batch size based on available memory
        if total_memory_gb > 32:
            batch_size = 4096  # High memory machine
        elif total_memory_gb > 16:
            batch_size = 2048  # Medium memory machine
        elif total_memory_gb > 8:
            batch_size = 1024  # Low memory machine
        else:
            batch_size = 512   # Very low memory machine
    except ImportError:
        # Default if psutil not available
        batch_size = 1024
    
    # Adaptive configuration based on available hardware
    config = {
        "threads": recommended_threads,  # Use detected CPU cores with reasonable limits
        "num_ctx": 16384,                # Keep large context size consistent
        "num_gpu": 0,                    # Let Ollama decide GPU usage
        "batch_size": batch_size,        # Adjust based on available memory
        "num_keep": 64,
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "parallel": recommended_threads  # Match threads for consistent parallelism
    }
    
    logger.info(f"Created adaptive LLM config: {recommended_threads} threads, {batch_size} batch size")
    return config

def get_best_available_model(base_url: str = "http://localhost:11434") -> str:
    """
    Gets the best available Mistral model from Ollama.
    Falls back to standard model if preferred one isn't available.
    
    Args:
        base_url: Ollama API base URL
        
    Returns:
        Name of the best available model
    """
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags", timeout=3.0)
        
        if response.status_code == 200:
            data = response.json()
            available_models = [m.get("name") for m in data.get("models", [])]
            
            # Check for preferred model first
            if DEFAULT_MODEL in available_models:
                return DEFAULT_MODEL
                
            # Otherwise look for any instruct model in order of preference
            instruct_models = [m for m in available_models if 'instruct' in m.lower()]
            if instruct_models:
                # Prioritize Mistral or similar modern models
                for prefix in ['mistral', 'llama3', 'llama2', 'codellama', 'gemma', 'phi']:
                    for model in instruct_models:
                        if prefix in model.lower():
                            logger.info(f"Selected alternate model: {model}")
                            return model
                
                # If no specific model found, use first available instruct model
                logger.info(f"Using alternative instruct model: {instruct_models[0]}")
                return instruct_models[0]
            
            # If no instruct models, use any available model
            if available_models:
                logger.warning(f"No instruct models found, using: {available_models[0]}")
                return available_models[0]
    
    except Exception as e:
        logger.warning(f"Error checking available models: {e}")
    
    # Return the default model if we couldn't check or found nothing
    return DEFAULT_MODEL
