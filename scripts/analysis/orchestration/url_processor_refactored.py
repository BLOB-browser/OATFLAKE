#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
URLProcessor handles URL processing functionality in OATFLAKE.
This version uses the refactored components for better modularity.
"""

import logging
import os
from typing import Dict, Any, Optional

from scripts.analysis.orchestration.base_orchestrator import BaseOrchestrator
from scripts.analysis.url_batch_processor import URLBatchProcessor
from scripts.analysis.url_level_processor import URLLevelProcessor
from scripts.analysis.url_discovery_manager import URLDiscoveryManager

logger = logging.getLogger(__name__)

class URLProcessor(BaseOrchestrator):
    """
    Handles URL processing functionality including:
    - Checking URL processing state
    - Processing URLs at specific levels
    - URL discovery
    
    This class now delegates to specialized components for better code organization.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the URL Processor.
        
        Args:
            data_folder: Path to the data directory
        """
        super().__init__(data_folder)
        self.url_batch_processor = URLBatchProcessor(data_folder)
        self.level_processor = URLLevelProcessor(data_folder)
        self.discovery_manager = URLDiscoveryManager(data_folder)
    
    def check_url_processing_state(self) -> Dict[str, Any]:
        """
        Check the current state of resource processing to identify potential issues.
        
        Returns:
            Dictionary with diagnostic information
        """
        return self.discovery_manager.check_discovery_needed()
    
    def process_urls_by_levels(self, max_depth: int = 4, batch_size: int = 50, force_fetch: bool = False) -> Dict[str, Any]:
        """
        Process all pending URLs by level, with special handling for higher levels.
        
        Args:
            max_depth: Maximum depth to process (usually 4)
            batch_size: Number of URLs to process in each batch
            force_fetch: Whether to force processing of already processed URLs
            
        Returns:
            Dictionary with processing results
        """
        # Forward cancellation status to batch processor if needed
        self.url_batch_processor.cancel_requested = self.cancel_requested
        
        # Delegate to URL batch processor
        return self.url_batch_processor.process_urls_by_levels(
            max_depth=max_depth,
            batch_size=batch_size,
            force_fetch=force_fetch
        )
    
    def process_urls_at_level(self, level: int, batch_size: int = 50, force_fetch: bool = False) -> Dict[str, Any]:
        """
        Process all pending URLs at a specific level.
        
        Args:
            level: The depth level to process (1=first level, 2=second level, etc.)
            batch_size: Maximum number of URLs to process in one batch
            force_fetch: If True, force reprocessing of already processed URLs
            
        Returns:
            Dictionary with processing results
        """
        # Forward cancellation status to level processor if needed
        self.level_processor.cancel_requested = self.cancel_requested
        
        # Delegate to URL level processor
        return self.level_processor.process_level(
            level=level,
            batch_size=batch_size,
            force_fetch=force_fetch
        )
    
    def request_cancellation(self):
        """Request cancellation of any running URL processes."""
        super().request_cancellation()
        self.url_batch_processor.request_cancellation()
        self.level_processor.request_cancellation()
        self.discovery_manager.cancel_requested = True
