#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adaptive configuration system that automatically detects hardware capabilities
and configures optimal settings for LLM processing.
"""

import logging
import multiprocessing
import platform
import os
import json
import requests
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AdaptiveSystemConfig:
    """
    Automatically detects system capabilities and generates optimal configurations
    for LLM processing based on available hardware resources.
    """
    
    def __init__(self):
        self.system_info = self._detect_system_capabilities()
        
    def _detect_system_capabilities(self) -> Dict[str, Any]:
        """Detect available system resources and capabilities."""
        capabilities = {
            'cpu_cores': multiprocessing.cpu_count(),
            'platform': platform.system(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'memory_gb': self._get_memory_info(),
            'gpu_available': self._detect_gpu(),
            'ollama_available': self._check_ollama_availability(),
            'performance_tier': 'medium'  # Will be determined based on specs
        }
        
        # Determine performance tier
        capabilities['performance_tier'] = self._determine_performance_tier(capabilities)
        
        logger.info(f"Detected system capabilities: {capabilities}")
        return capabilities
    
    def _get_memory_info(self) -> float:
        """Get available system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 * 1024 * 1024)
        except ImportError:
            logger.warning("psutil not available, using default memory estimate")
            return 8.0  # Default assumption
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available for LLM processing."""
        try:
            # Try to detect NVIDIA GPU
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except (ImportError, FileNotFoundError):
            return False
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _determine_performance_tier(self, capabilities: Dict[str, Any]) -> str:
        """Determine system performance tier based on capabilities and current load."""
        memory_gb = capabilities.get('memory_gb', 8)
        cpu_cores = capabilities.get('cpu_cores', 4)
        gpu_available = capabilities.get('gpu_available', False)
        
        # Check current system load if psutil is available
        current_load_factor = 1.0
        try:
            import psutil
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Apply load penalty - if system is heavily loaded, downgrade tier
            if memory_usage > 80 or cpu_usage > 80:
                current_load_factor = 0.3  # Heavily loaded - use emergency settings
                logger.warning(f"System under heavy load: RAM {memory_usage}%, CPU {cpu_usage}% - using emergency tier")
            elif memory_usage > 60 or cpu_usage > 60:
                current_load_factor = 0.6  # Moderately loaded - reduce performance
                logger.info(f"System moderately loaded: RAM {memory_usage}%, CPU {cpu_usage}% - reducing tier")
        except ImportError:
            pass
        
        # Determine base tier
        if memory_gb >= 32 and cpu_cores >= 16:
            base_tier = 'high'
        elif memory_gb >= 16 and cpu_cores >= 8:
            base_tier = 'medium-high'
        elif memory_gb >= 8 and cpu_cores >= 4:
            base_tier = 'medium'
        else:
            base_tier = 'low'
        
        # Apply load adjustment
        if current_load_factor <= 0.3:
            return 'emergency'  # New emergency tier for overloaded systems
        elif current_load_factor <= 0.6:
            # Downgrade by one tier
            tier_order = ['low', 'medium', 'medium-high', 'high']
            current_index = tier_order.index(base_tier)
            return tier_order[max(0, current_index - 1)]
        else:
            return base_tier
    
    def get_optimal_llm_config(self) -> Dict[str, Any]:
        """Generate optimal LLM configuration based on system capabilities."""
        tier = self.system_info['performance_tier']
        cpu_cores = self.system_info['cpu_cores']
        memory_gb = self.system_info['memory_gb']
        
        # Base configuration templates by performance tier
        configs = {
            'emergency': {
                'num_ctx': 1024,  # Conservative context for overloaded systems
                'num_thread': 2,  # Based on successful testing with Mistral
                'threads': 2,
                'batch_size': 256,
                'num_keep': 8,
                'parallel': 1,
                'chunk_size': 500,
                'chunk_overlap': 50
            },
            'high': {
                'num_ctx': 32768,
                'num_thread': min(cpu_cores, 20),
                'threads': min(cpu_cores, 20),  # Add consistent field name
                'batch_size': 4096,
                'num_keep': 128,
                'parallel': min(cpu_cores // 2, 10),
                'chunk_size': 2000,
                'chunk_overlap': 300
            },
            'medium-high': {
                'num_ctx': 16384,
                'num_thread': min(cpu_cores, 12),
                'threads': min(cpu_cores, 12),  # Add consistent field name
                'batch_size': 2048,
                'num_keep': 64,
                'parallel': min(cpu_cores // 2, 8),
                'chunk_size': 1500,
                'chunk_overlap': 200
            },
            'medium': {
                'num_ctx': 8192,
                'num_thread': min(cpu_cores, 8),
                'threads': min(cpu_cores, 8),  # Add consistent field name
                'batch_size': 1024,
                'num_keep': 32,
                'parallel': min(cpu_cores // 2, 6),
                'chunk_size': 1000,
                'chunk_overlap': 150
            },
            'low': {
                'num_ctx': 4096,
                'num_thread': max(2, min(cpu_cores, 4)),
                'threads': max(2, min(cpu_cores, 4)),  # Add consistent field name
                'batch_size': 512,
                'num_keep': 16,
                'parallel': 2,
                'chunk_size': 800,
                'chunk_overlap': 100
            }
        }
        
        config = configs.get(tier, configs['medium']).copy()
        
        # Add common settings
        config.update({
            'num_gpu': 0,  # Let Ollama handle GPU detection
            'repeat_penalty': 1.1,
            'temperature': 0.7,
            'top_k': 40,
            'top_p': 0.9
        })
        
        logger.info(f"Generated optimal LLM config for {tier} tier system: {config}")
        return config
    
    def get_optimal_model_recommendation(self) -> str:
        """Recommend the best model based on system capabilities."""
        tier = self.system_info['performance_tier']
        
        models = {
            'emergency': "mistral:7b-instruct-v0.2-q4_0",  # Use working model even under load
            'high': "mistral:7b-instruct-v0.2-q4_0",
            'medium-high': "mistral:7b-instruct-v0.2-q4_0", 
            'medium': "mistral:7b-instruct-v0.2-q4_0",
            'low': "mistral:7b-instruct-v0.2-q4_0"  # Use consistent working model across all tiers
        }
        
        return models.get(tier, "mistral:7b-instruct-v0.2-q4_0")
    
    def auto_configure_system(self, data_folder: str) -> Dict[str, Any]:
        """
        Automatically configure the entire system based on detected capabilities.
        
        Args:
            data_folder: Path to the data folder
            
        Returns:
            Complete system configuration
        """
        config = {
            'system_info': self.system_info,
            'llm_config': self.get_optimal_llm_config(),
            'recommended_model': self.get_optimal_model_recommendation(),
            'data_folder': data_folder,
            'base_url': "http://localhost:11434",
            'processing_settings': self._get_processing_settings()
        }
        
        # Save configuration for reference
        self._save_adaptive_config(config, data_folder)
        
        return config
    
    def _get_processing_settings(self) -> Dict[str, Any]:
        """Get processing-specific settings based on system tier."""
        tier = self.system_info['performance_tier']
        
        settings = {
            'emergency': {
                'max_concurrent_requests': 1,
                'request_timeout': 600,  # 10 minutes for Raspberry Pi and overloaded systems
                'retry_attempts': 1,
                'batch_processing': False
            },
            'high': {
                'max_concurrent_requests': 8,
                'request_timeout': 120,
                'retry_attempts': 3,
                'batch_processing': True
            },
            'medium-high': {
                'max_concurrent_requests': 6,
                'request_timeout': 90,
                'retry_attempts': 3,
                'batch_processing': True
            },
            'medium': {
                'max_concurrent_requests': 4,
                'request_timeout': 60,
                'retry_attempts': 2,
                'batch_processing': False
            },
            'low': {
                'max_concurrent_requests': 2,
                'request_timeout': 45,
                'retry_attempts': 2,
                'batch_processing': False
            }
        }
        
        return settings.get(tier, settings['medium'])
    
    def _save_adaptive_config(self, config: Dict[str, Any], data_folder: str):
        """Save the adaptive configuration for reference and debugging."""
        try:
            config_path = Path(data_folder) / "adaptive_system_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            logger.info(f"Saved adaptive configuration to {config_path}")
        except Exception as e:
            logger.warning(f"Could not save adaptive config: {e}")


def get_adaptive_system_config(data_folder: str = None) -> Dict[str, Any]:
    """
    Convenience function to get adaptive system configuration.
    
    Args:
        data_folder: Path to data folder (optional)
        
    Returns:
        Complete adaptive system configuration
    """
    if data_folder is None:
        data_folder = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    
    adaptive_config = AdaptiveSystemConfig()
    return adaptive_config.auto_configure_system(data_folder)


if __name__ == "__main__":
    # Test the adaptive configuration system
    config = get_adaptive_system_config()
    print("🚀 Adaptive System Configuration:")
    print(json.dumps(config, indent=2, default=str))
