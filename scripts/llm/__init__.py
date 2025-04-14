"""
LLM integration module for interacting with different LLM providers.
"""
import logging
import os

# Set up package-level logger
logger = logging.getLogger(__name__)

# Track initialization to avoid duplicate setups
_initialized = False

def initialize_llm_subsystem():
    """Initialize LLM subsystem if not already done"""
    global _initialized
    if not _initialized:
        logger.info("Initializing LLM subsystem")
        _initialized = True
    else:
        logger.debug("LLM subsystem already initialized")

# Only initialize when explicitly requested, not on import
# This prevents automatic initialization during imports from multiple places
