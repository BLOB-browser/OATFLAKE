#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
URLBatchProcessor coordinates processing URLs across multiple levels.
This is extracted from URLProcessor to make the code more modular.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import pandas as pd

from scripts.analysis.url_level_processor import URLLevelProcessor
from scripts.analysis.url_discovery_manager import URLDiscoveryManager

logger = logging.getLogger(__name__)

class URLBatchProcessor:
    """
    Coordinates URL processing across multiple depth levels.
    This component is responsible for handling the sequence of URL processing
    from level 1 through level 4.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the URL Batch Processor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        self.level_processor = URLLevelProcessor(data_folder)
        self.discovery_manager = URLDiscoveryManager(data_folder)
        self.cancel_requested = False
        
    def request_cancellation(self):
        """Request cancellation of processing."""
        self.cancel_requested = True
        self.level_processor.request_cancellation()
        self.discovery_manager.cancel_requested = True
    
    def process_urls_by_levels(self, max_depth: int = 4, batch_size: int = 50, force_fetch: bool = False) -> Dict[str, Any]:
        """
        Process all pending URLs by level.
        
        Args:
            max_depth: Maximum depth to process (usually 4)
            batch_size: Number of URLs to process in each batch
            force_fetch: Whether to force processing of already processed URLs
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing URLs by level up to depth {max_depth}")
        logger.info(f"Force fetch enabled: {force_fetch}")
        
        # Structure to track results
        results = {
            "processed_by_level": {},
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
        
        # Get discovery status to determine which levels have pending URLs
        discovery_status = self.discovery_manager.check_discovery_needed()
        
        # Extract levels with pending URLs
        levels_with_pending = {}
        if discovery_status.get("status") == "success":
            pending_by_level = discovery_status.get("pending_urls_by_level", {})
            for level, count in pending_by_level.items():
                if count > 0:
                    levels_with_pending[level] = count
                    
            # Log the levels with pending URLs
            if levels_with_pending:
                logger.info(f"Found pending URLs at {len(levels_with_pending)} levels: {list(levels_with_pending.keys())}")
                highest_level = max(levels_with_pending.keys())
                logger.info(f"Highest level with pending URLs: {highest_level}")
            else:
                logger.warning(f"No pending URLs found at any level (1-{max_depth})!")
        
        # Process all levels with pending URLs
        if levels_with_pending:
            self._process_pending_levels(levels_with_pending, batch_size, force_fetch, results)
        else:
            # If no levels have pending URLs but we have resources to analyze,
            # we need to try discovery or direct processing
            self._handle_no_pending_urls(max_depth, batch_size, force_fetch, results)

        logger.info(f"URL processing completed: {results['total_processed']} URLs processed total")
        return results
        
    def _process_pending_levels(self, levels_with_pending: Dict[int, int], batch_size: int, force_fetch: bool, results: Dict[str, Any]):
        """
        Process levels that have pending URLs.
        
        Args:
            levels_with_pending: Dictionary mapping levels to URL counts
            batch_size: Number of URLs to process in each batch
            force_fetch: Whether to force processing of already processed URLs
            results: Dictionary to store results (modified in-place)
        """
        # Process each level in ascending order
        for level in sorted(levels_with_pending.keys()):
            logger.info(f"Processing {levels_with_pending[level]} URLs at level {level}")
            
            # Process this level with URL level processor
            try:
                level_result = self.level_processor.process_level(
                    level=level,
                    batch_size=batch_size,
                    force_fetch=force_fetch
                )
                
                # Record results
                processed = level_result.get('processed_urls', 0)
                results["processed_by_level"][level] = processed
                results["total_processed"] += processed
                results["successful"] += level_result.get('success_count', 0)
                results["failed"] += level_result.get('error_count', 0)
                
                logger.info(f"Level {level} processing complete: {processed} URLs processed")
                
                # Check for cancellation
                if self.cancel_requested:
                    logger.info(f"URL processing cancelled during level {level}")
                    break
            except Exception as e:
                logger.error(f"Error processing URLs at level {level}: {e}")
                results["failed"] += 1
                
                # Continue with next level despite errors
                logger.info(f"Continuing with next level despite errors at level {level}")
                continue
                
    def _handle_no_pending_urls(self, max_depth: int, batch_size: int, force_fetch: bool, results: Dict[str, Any]):
        """
        Handle the case when there are no pending URLs at any level.
        
        Args:
            max_depth: Maximum depth to process
            batch_size: Number of URLs to process in each batch
            force_fetch: Whether to force processing of already processed URLs
            results: Dictionary to store results (modified in-place)
        """
        logger.warning("No pending URLs found, attempting recovery strategies")
          # Try to trigger discovery for level 1
        discovery_result = self.discovery_manager.discover_urls_from_resources_sync(level=1, max_depth=max_depth)
        
        if discovery_result.get("status") == "success" and discovery_result.get("discovered_urls", 0) > 0:
            # If discovery worked, process level 1
            level_result = self.level_processor.process_level(
                level=1,
                batch_size=batch_size,
                force_fetch=force_fetch
            )
            
            # Record results
            processed = level_result.get('processed_urls', 0)
            results["processed_by_level"][1] = processed
            results["total_processed"] += processed
            results["successful"] += level_result.get('success_count', 0)
            results["failed"] += level_result.get('error_count', 0)
        else:
            # If discovery didn't work, try a final attempt starting from the lowest levels
            # to ensure proper hierarchical processing
            logger.warning("No URLs found through discovery, making one final attempt with direct processing")
            
            for level in range(1, max_depth + 1):
                logger.warning(f"Final attempt to check level {level} URLs with force_fetch={force_fetch}")
                level_result = self.level_processor.process_level(
                    level=level,
                    batch_size=batch_size,
                    force_fetch=True  # Force fetch for one final attempt
                )
                
                # Record results even if 0
                processed = level_result.get('processed_urls', 0)
                results["processed_by_level"][level] = processed
                results["total_processed"] += processed
                results["successful"] += level_result.get('success_count', 0)
                results["failed"] += level_result.get('error_count', 0)
                
                logger.warning(f"Final attempt at level {level} complete: {processed} URLs processed")
                
                # Check for cancellation
                if self.cancel_requested:
                    logger.info(f"URL processing cancelled during level {level}")
                    break
                    
    def process_specific_level(self, level: int, batch_size: int = 50, force_fetch: bool = False) -> Dict[str, Any]:
        """
        Process URLs at a specific level only.
        
        Args:
            level: The level to process (1-4)
            batch_size: Number of URLs to process in each batch
            force_fetch: Whether to force processing of already processed URLs
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing only level {level} URLs")
        
        # Process this level directly
        level_result = self.level_processor.process_level(
            level=level,
            batch_size=batch_size,
            force_fetch=force_fetch
        )
        
        return level_result
