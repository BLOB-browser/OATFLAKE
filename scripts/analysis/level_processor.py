#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt

logger = logging.getLogger(__name__)

class LevelBasedProcessor:
    """
    Processes all URLs at a specific depth level across all resources.
    This allows processing all level 1 URLs first, followed by level 2, etc.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the LevelBasedProcessor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
        # Import required components
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from scripts.analysis.cleanup_manager import CleanupManager
        from scripts.analysis.vector_generator import VectorGenerator
        from scripts.analysis.url_storage import URLStorageManager
        from utils.config import get_data_path
        
        # Initialize components
        self.single_processor = SingleResourceProcessor(data_folder)
        self.cleanup_manager = CleanupManager(data_folder)
        self.vector_generator = VectorGenerator(data_folder)
        
        # Initialize URL storage manager
        processed_urls_file = os.path.join(get_data_path(), "processed_urls.csv")
        self.url_storage = URLStorageManager(processed_urls_file)
        
        # Track processing state
        self.vector_generation_needed = False
        self._cancel_processing = False
    
    def reset_cancellation(self):
        """Reset the cancellation flag"""
        self._cancel_processing = False
        clear_interrupt()
    
    def check_cancellation(self):
        """Check if processing should be cancelled"""
        return self._cancel_processing or is_interrupt_requested()
    
    def is_level_complete(self, level: int) -> bool:
        """Check if all URLs at a specific level have been processed
        
        Args:
            level: The depth level to check
            
        Returns:
            True if no pending URLs exist at this level, False otherwise
        """
        pending_urls = self.url_storage.get_pending_urls(depth=level)
        return len(pending_urls) == 0
        
    def get_level_status(self) -> Dict[int, Dict[str, Any]]:
        """Get processing status for all levels
        
        Returns:
            Dictionary with statistics for each level
        """
        stats = {}
        
        # Check all levels from 1 to 10 (arbitrary max)
        for level in range(1, 11):
            pending = len(self.url_storage.get_pending_urls(depth=level))
            if pending > 0 or level == 1:  # Always include level 1
                stats[level] = {
                    "pending": pending,
                    "is_complete": pending == 0
                }
        
        return stats
    
    def process_next_available_level(self) -> Dict[str, Any]:
        """Process the next available level, ensuring proper level progression
        
        Returns:
            Processing statistics from process_level
        """
        # Get status for all levels
        level_status = self.get_level_status()
        
        # Find the lowest incomplete level
        next_level = None
        for level in sorted(level_status.keys()):
            if not level_status[level]["is_complete"]:
                next_level = level
                break
        
        if next_level is None:
            return {
                "status": "completed",
                "message": "All levels are complete"
            }
        
        # Process the next level
        return self.process_level(next_level)
    
    def process_level(self, level: int, csv_path: str = None, max_urls: int = None, 
                   skip_vector_generation: bool = False) -> Dict[str, Any]:
        """
        Process all URLs at a specific depth level across all resources.
        
        Args:
            level: The depth level to process (1=first level, 2=second level, etc.)
            csv_path: Path to the resources CSV file (for finding resource metadata)
            max_urls: Maximum number of URLs to process
            skip_vector_generation: Whether to skip vector generation after processing
            
        Returns:
            Dictionary with processing statistics
        """
        # Check if previous levels are complete before processing this level
        for prev_level in range(1, level):
            if not self.is_level_complete(prev_level):
                logger.warning(f"Cannot process level {level} because level {prev_level} is not complete")
                return {
                    "status": "error",
                    "error": f"Previous level {prev_level} is not complete",
                    "urls_processed": 0
                }
        
        # Reset cancellation flag when starting new processing
        self.reset_cancellation()
        
        # Set default CSV path if not provided
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, 'resources.csv')
        
        if not os.path.exists(csv_path):
            logger.error(f"Resources file not found: {csv_path}")
            return {
                "status": "error",
                "error": "Resources file not found",
                "urls_processed": 0
            }
        
        # Load resources data for reference
        try:
            resources_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(resources_df)} resources from {csv_path}")
        except Exception as e:
            logger.error(f"Error loading resources file: {e}")
            return {
                "status": "error",
                "error": f"Error loading resources file: {e}",
                "urls_processed": 0
            }
        
        # Statistics to track
        stats = {
            "level": level,
            "urls_processed": 0,
            "success_count": 0,
            "error_count": 0,
            "definitions_found": 0,
            "projects_found": 0,
            "methods_found": 0,
            "start_time": datetime.now()
        }
        
        try:
            # Get all pending URLs at this level
            pending_urls = self.url_storage.get_pending_urls(depth=level, max_urls=max_urls or 10000)
            if not pending_urls:
                logger.info(f"No pending URLs found at level {level}")
                return {
                    "status": "completed",
                    "message": f"No pending URLs found at level {level}",
                    "urls_processed": 0
                }
            
            logger.info(f"Found {len(pending_urls)} pending URLs at level {level}")
            
            # Create a set of URLs we've already processed to avoid duplicates
            # First from the already processed URLs in the storage system
            processed_urls = self.url_storage.get_processed_urls()
            
            # Create another set for this processing session to avoid duplicates even if they're not yet in storage
            current_session_processed = set()
            
            # Track extracted items
            all_definitions = []
            all_projects = []
            all_methods = []
            
            # Track when to do incremental vector generation
            incremental_build_threshold = 10  # Build vector store after every N URLs
            urls_since_last_vector_build = 0
            
            # Process each URL at this level
            for url_idx, url_info in enumerate(pending_urls):
                # Limit processing if max_urls is specified
                if max_urls is not None and url_idx >= max_urls:
                    logger.info(f"Reached maximum URL limit of {max_urls}")
                    break
                
                # Check for cancellation
                if self.check_cancellation():
                    logger.info("URL processing cancelled by external request")
                    break
                
                url = url_info["url"]
                origin = url_info["origin"]
                depth = url_info["depth"]
                
                # Skip if this URL is already processed (either from storage or current session)
                if url in processed_urls or url in current_session_processed:
                    logger.info(f"Skipping already processed URL: {url}")
                    # Remove from pending URLs since it's already processed
                    self.url_storage.remove_pending_url(url)
                    continue
                
                # Add to current session processed set to avoid duplicates
                current_session_processed.add(url)
                
                logger.info(f"Processing level {depth} URL {url_idx+1}/{len(pending_urls)}: {url}")
                
                # Find the resource this URL belongs to
                resource_row = resources_df[resources_df['url'] == origin]
                if resource_row.empty:
                    logger.warning(f"Could not find origin resource for URL: {url} (origin: {origin})")
                    continue
                
                origin_resource = resource_row.iloc[0].to_dict()
                
                try:
                    # Process just this specific URL
                    result = self.single_processor.process_specific_url(
                        url=url,
                        origin_url=origin,
                        resource=origin_resource,
                        depth=depth
                    )
                    
                    # Update statistics
                    stats["urls_processed"] += 1
                    urls_since_last_vector_build += 1
                    
                    if result.get("success", False):
                        stats["success_count"] += 1
                        stats["definitions_found"] += len(result.get("definitions", []))
                        stats["projects_found"] += len(result.get("projects", []))
                        stats["methods_found"] += len(result.get("methods", []))
                        
                        # Store the extracted data
                        all_definitions.extend(result.get("definitions", []))
                        all_projects.extend(result.get("projects", []))
                        all_methods.extend(result.get("methods", []))
                        
                        # Remove this URL from pending
                        self.url_storage.remove_pending_url(url)
                    else:
                        stats["error_count"] += 1
                    
                    # Check if vector generation is needed
                    if self.single_processor.vector_generation_needed:
                        self.vector_generation_needed = True
                    
                    # Incremental vector store building
                    if (urls_since_last_vector_build >= incremental_build_threshold and 
                        self.vector_generation_needed and not skip_vector_generation):
                        
                        logger.info(f"Starting incremental vector store build after {urls_since_last_vector_build} URLs")
                        
                        try:
                            # Get all content files from the temporary storage path
                            from scripts.storage.content_storage_service import ContentStorageService
                            content_storage = ContentStorageService(self.data_folder)
                            content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
                            
                            logger.info(f"Found {len(content_paths)} content files for incremental vector generation")
                            
                            # Run the incremental vector generation
                            incremental_stats = asyncio.run(self.vector_generator.generate_vector_stores(content_paths))
                            
                            # Store the stats of this incremental build
                            if "incremental_builds" not in stats:
                                stats["incremental_builds"] = []
                                
                            stats["incremental_builds"].append({
                                "build_number": len(stats.get("incremental_builds", [])) + 1,
                                "urls_processed": urls_since_last_vector_build,
                                "content_paths": len(content_paths),
                                "stats": incremental_stats
                            })
                            
                            # Reset the counter for next batch
                            urls_since_last_vector_build = 0
                            logger.info(f"Completed incremental vector store build #{len(stats.get('incremental_builds', []))}")
                            
                        except Exception as ve:
                            logger.error(f"Error during incremental vector generation: {ve}")
                    
                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    stats["error_count"] += 1
                
                # Check for cancellation again after processing each URL
                if self.check_cancellation():
                    logger.info("URL processing cancelled after processing URL")
                    break
            
            # Calculate duration
            stats["duration_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
            stats["status"] = "completed"
            
            # Include extracted data in the stats
            stats["pending_urls_remaining"] = len(self.url_storage.get_pending_urls(depth=level))
            stats["definitions"] = all_definitions
            stats["projects"] = all_projects
            stats["methods"] = all_methods
            
            # Final vector store generation if needed
            if self.vector_generation_needed and not skip_vector_generation:
                logger.info("Starting final vector generation for this level")
                
                try:
                    from scripts.storage.content_storage_service import ContentStorageService
                    content_storage = ContentStorageService(self.data_folder)
                    content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
                    
                    if content_paths:
                        vector_stats = asyncio.run(self.vector_generator.generate_vector_stores(content_paths))
                        stats["vector_stats"] = vector_stats
                        logger.info(f"Vector generation complete: {vector_stats}")
                except Exception as ve:
                    logger.error(f"Error during final vector generation: {ve}")
                    stats["vector_error"] = str(ve)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during level processing: {e}")
            stats["status"] = "error"
            stats["error"] = str(e)
            stats["duration_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
            return stats

# Utility function for command-line usage
def process_level(level, data_folder=None, csv_path=None, max_urls=None):
    """
    Process all URLs at a specific level from the command line.
    """
    from utils.config import get_data_path
    
    if data_folder is None:
        data_folder = get_data_path()
    
    processor = LevelBasedProcessor(data_folder)
    result = processor.process_level(level, csv_path, max_urls)
    
    return result