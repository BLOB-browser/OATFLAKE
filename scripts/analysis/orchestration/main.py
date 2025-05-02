#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main orchestrator module that brings together all specialized components.
This module provides the primary KnowledgeOrchestrator class through multiple inheritance.
"""

import logging
import os
from typing import Dict, Any, Optional

# Import the components
from scripts.analysis.orchestration.base_orchestrator import BaseOrchestrator
from scripts.analysis.orchestration.url_processor import URLProcessor
from scripts.analysis.orchestration.knowledge_processor import KnowledgeProcessor
from scripts.analysis.orchestration.phased_processor import PhasedProcessor

logger = logging.getLogger(__name__)

class KnowledgeOrchestrator(URLProcessor, KnowledgeProcessor, PhasedProcessor):
    """
    Orchestrates the entire knowledge processing workflow by coordinating all steps.
    This is the main entry point for the knowledge processing pipeline.
    
    This class inherits from the specialized components:
    - URLProcessor: Handles URL processing and state checking
    - KnowledgeProcessor: Handles main knowledge processing workflow
    - PhasedProcessor: Handles phased knowledge processing approach
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the knowledge orchestrator.
        
        Args:
            data_folder: Path to the data directory
        """
        super().__init__(data_folder)
        logger.info(f"KnowledgeOrchestrator initialized with data folder: {data_folder}")


# Standalone function for easy import
async def process_knowledge_base(data_folder=None, request_app_state=None, **kwargs):
    """
    Process all knowledge base files as a standalone function.
    
    Args:
        data_folder: Path to data folder (if None, gets from config)
        request_app_state: FastAPI request.app.state
        **kwargs: All other KnowledgeOrchestrator parameters
    
    Returns:
        Dictionary with processing results
    """
    # Get data folder from config if not provided
    if data_folder is None:
        from utils.config import get_data_path
        data_folder = get_data_path()
    
    # Initialize orchestrator
    orchestrator = KnowledgeOrchestrator(data_folder)
    
    # Process knowledge
    return await orchestrator.process_knowledge(request_app_state=request_app_state, **kwargs)
