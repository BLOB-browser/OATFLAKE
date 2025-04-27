#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt

logger = logging.getLogger(__name__)

class LevelBasedURLProcessor:
    """
    Process URLs based on the current_url_level in config.json
    This processor uses a simplified approach where:
    1. It loads current_url_level from config.json (default to 1 if not found)
    2. For each resource, it processes URLs:
       - If level=1: Fetch main URL and its direct links
       - If level=2: Fetch main URL and traverse to depth 2
       - If level=3: Fetch main URL and traverse to depth 3
       - If level=4: Fetch main URL and traverse to depth 4
    3. It processes all pending URLs across levels at once
    4. When all URLs for current level are processed, it increments the level in config
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the simplified URL processor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
        # Import required components
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from scripts.analysis.cleanup_manager import CleanupManager
        from scripts.analysis.url_storage import URLStorageManager
        from utils.config import get_data_path
        
        # Initialize components
        self.single_processor = SingleResourceProcessor(data_folder)
        self.cleanup_manager = CleanupManager(data_folder)
        
        # Initialize URL storage manager
        processed_urls_file = os.path.join(get_data_path(), "processed_urls.csv")
        self.url_storage = URLStorageManager(processed_urls_file)
        
        # Track processing state
        self._cancel_processing = False
        
        # Load config to get current_url_level
        self.config_path = self._get_config_path()
        self.current_url_level = self._get_current_url_level()
        self.max_depth = 4  # Maximum supported depth
    
    def _get_config_path(self):
        """Get the path to the config file"""
        local_config = Path.home() / '.blob' / 'config.json'
        if local_config.exists():
            return local_config
        
        # If it doesn't exist, use a default location
        return Path(os.path.join(os.getcwd(), "config.json"))
    
    def _get_current_url_level(self) -> int:
        """Get current_url_level from config.json, default to 1 if not found"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    return config.get('current_url_level', 1)
        except Exception as e:
            logger.error(f"Error reading current_url_level from config: {e}")
        
        # Default to level 1 if not found or error
        return 1
    
    def _update_current_url_level(self, new_level: int):
        """Update current_url_level in config.json"""
        try:
            if self.config_path.exists():
                # Read existing config
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Update level
                config['current_url_level'] = new_level
                
                # Write back to file
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Update local attribute
                self.current_url_level = new_level
                logger.info(f"Updated current_url_level in config to: {new_level}")
                return True
        except Exception as e:
            logger.error(f"Error updating current_url_level in config: {e}")
        
        return False
    
    def reset_cancellation(self):
        """Reset the cancellation flag"""
        self._cancel_processing = False
        clear_interrupt()
    
    def check_cancellation(self):
        """Check if processing should be cancelled"""
        return self._cancel_processing or is_interrupt_requested()
    
    def get_level_status(self) -> Dict[str, Dict[str, Any]]:
        """Get processing status for all levels
        
        Returns:
            Dictionary with statistics for each level
        """
        stats = {}
        
        # Check all levels from 1 to max_depth
        for level in range(1, self.max_depth + 1):
            pending = len(self.url_storage.get_pending_urls(depth=level))
            processed = len(self.url_storage.get_processed_urls_by_level(level))
            
            stats[str(level)] = {
                "pending": pending,
                "processed": processed,
                "is_complete": pending == 0,
                "total": pending + processed
            }
        
        return stats
    
    def process_resources_at_current_level(self, csv_path: str = None, max_resources: int = None) -> Dict[str, Any]:
        """
        Process original resources to the current depth level set in config.json
        
        Args:
            csv_path: Path to the resources CSV file
            max_resources: Maximum number of resources to process
            
        Returns:
            Dictionary with processing statistics
        """
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
                "resources_processed": 0
            }
        
        # Load resources data
        try:
            resources_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(resources_df)} resources from {csv_path}")
        except Exception as e:
            logger.error(f"Error loading resources file: {e}")
            return {
                "status": "error",
                "error": f"Error loading resources file: {e}",
                "resources_processed": 0
            }
        
        # Statistics to track
        stats = {
            "current_url_level": self.current_url_level,
            "resources_processed": 0,
            "urls_discovered": 0,
            "success_count": 0,
            "error_count": 0,
            "start_time": datetime.now()
        }
        
        # Keep track of resources that were completely processed to current level
        completed_resources = []
        
        # Import content fetcher for crawling
        from scripts.analysis.content_fetcher import ContentFetcher
        content_fetcher = ContentFetcher()
        
        # Process each resource up to current_url_level depth
        for idx, row in enumerate(resources_df.iterrows()):
            # Limit processing if max_resources is specified
            if max_resources is not None and idx >= max_resources:
                logger.info(f"Reached maximum resource limit of {max_resources}")
                break
            
            # Check for cancellation
            if self.check_cancellation():
                logger.info("Resource processing cancelled by external request")
                break
            
            # Extract resource data
            resource_idx, resource = row
            resource_url = resource.get('url')
            resource_id = resource.get('id')
            
            if not resource_url:
                logger.warning(f"Resource at index {resource_idx} has no URL, skipping")
                continue
            
            resource_title = resource.get('title', 'Untitled')
            logger.info(f"Processing resource {idx+1}/{len(resources_df)}: {resource_title} ({resource_url})")
            
            try:
                # Use content_fetcher to discover URLs up to current level
                result = content_fetcher.get_main_page_with_links(
                    url=resource_url,
                    max_depth=self.current_url_level,
                    discover_only_level=None  # Process all levels up to current_url_level
                )
                
                # Track statistics
                if result.get('success', False):
                    stats["success_count"] += 1
                    urls_discovered = result.get('urls_discovered', 0)
                    stats["urls_discovered"] += urls_discovered
                    
                    logger.info(f"Successfully processed {resource_title}, discovered {urls_discovered} URLs")
                    
                    # Check if this resource has been fully processed to current level
                    pending_for_resource = self.url_storage.get_pending_urls_for_origin(resource_url)
                    if not pending_for_resource:
                        completed_resources.append(resource_id)
                else:
                    stats["error_count"] += 1
                    logger.error(f"Error processing {resource_title}: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                stats["error_count"] += 1
                logger.error(f"Exception processing {resource_title}: {e}")
            
            stats["resources_processed"] += 1
        
        # Update resources.csv to mark resources that have been fully processed
        if completed_resources:
            try:
                # Mark resources as analysis_completed if they've been processed to max_depth
                # or if all levels up to current_url_level are complete with no pending URLs
                for resource_id in completed_resources:
                    # Find the row by ID
                    resource_rows = resources_df[resources_df['id'] == resource_id]
                    if not resource_rows.empty:
                        resource_idx = resource_rows.index[0]
                        
                        # Mark as analysis_completed only if we've reached max_depth
                        if self.current_url_level >= self.max_depth:
                            if 'analysis_completed' not in resources_df.columns:
                                resources_df['analysis_completed'] = False
                                
                            resources_df.at[resource_idx, 'analysis_completed'] = True
                            logger.info(f"Marked resource {resource_id} as analysis_completed=True (reached max_depth)")
                
                # Save updated dataframe
                resources_df.to_csv(csv_path, index=False)
                logger.info(f"Updated resources.csv with {len(completed_resources)} completed resources")
                
            except Exception as e:
                logger.error(f"Error updating resources.csv: {e}")
        
        # Check if all resources have been processed at current level
        level_status = self.get_level_status()
        current_level_status = level_status.get(str(self.current_url_level), {})
        all_resources_at_level_complete = current_level_status.get('is_complete', False)
        
        # If all resources are complete at this level and we haven't reached max depth,
        # increment the current_url_level in config
        if all_resources_at_level_complete and self.current_url_level < self.max_depth:
            new_level = self.current_url_level + 1
            self._update_current_url_level(new_level)
            stats["level_incremented"] = True
            stats["new_level"] = new_level
        
        # Add level status to stats
        stats["level_status"] = level_status
        stats["end_time"] = datetime.now()
        stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
        
        return {
            "status": "success",
            "stats": stats
        }
    
    def process_all_pending_urls(self, csv_path: str = None, max_urls: int = None) -> Dict[str, Any]:
        """
        Process all pending URLs across all levels
        
        Args:
            csv_path: Path to the resources CSV file (for finding resource metadata)
            max_urls: Maximum number of URLs to process
            
        Returns:
            Dictionary with processing statistics
        """
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
            "urls_processed": 0,
            "success_count": 0,
            "error_count": 0,
            "by_level": {},
            "start_time": datetime.now()
        }
        
        try:
            # Get all pending URLs across all levels up to current level 
            all_pending_urls = []
            for level in range(1, self.current_url_level + 1):
                level_urls = self.url_storage.get_pending_urls(depth=level)
                all_pending_urls.extend(level_urls)
                stats["by_level"][str(level)] = len(level_urls)
            
            if not all_pending_urls:
                logger.info(f"No pending URLs found across levels 1 to {self.current_url_level}")
                return {
                    "status": "completed",
                    "message": f"No pending URLs found",
                    "urls_processed": 0
                }
            
            # Limit the number of URLs to process if specified
            if max_urls is not None and max_urls < len(all_pending_urls):
                all_pending_urls = all_pending_urls[:max_urls]
            
            logger.info(f"Processing {len(all_pending_urls)} pending URLs across all levels")
            
            # Create a set of URLs we've already processed to avoid duplicates
            processed_urls = set(url['url'] for url in self.url_storage.get_processed_urls())
            current_session_processed = set()
            
            # Process all pending URLs
            for url_idx, url_info in enumerate(all_pending_urls):
                # Check for cancellation
                if self.check_cancellation():
                    logger.info("URL processing cancelled by external request")
                    break
                
                url = url_info["url"]
                origin = url_info["origin"]
                depth = url_info["depth"]
                attempt_count = url_info.get("attempt_count", 0)
                
                # Skip if this URL is already processed
                if url in processed_urls or url in current_session_processed:
                    logger.info(f"Skipping already processed URL: {url}")
                    self.url_storage.remove_pending_url(url)
                    continue
                
                # Force mark as processed after too many attempts
                if attempt_count >= 3:
                    logger.warning(f"URL {url} has been attempted {attempt_count} times, force marking as processed")
                    self.url_storage.save_processed_url(url, depth=depth, origin=origin)
                    self.url_storage.remove_pending_url(url)
                    stats["urls_processed"] += 1
                    stats["forced_skip_count"] = stats.get("forced_skip_count", 0) + 1
                    continue
                
                # Add to current session processed set to avoid duplicates
                current_session_processed.add(url)
                
                logger.info(f"Processing URL {url_idx+1}/{len(all_pending_urls)}: {url} (level {depth})")
                
                # Find the resource this URL belongs to
                resource_row = resources_df[resources_df['url'] == origin]
                resource_data = {}
                
                if not resource_row.empty:
                    resource_data = resource_row.iloc[0].to_dict()
                
                try:
                    # Process the URL
                    result = self.single_processor.process_single_url(
                        url=url,
                        origin=origin,
                        depth=depth,
                        resource_metadata=resource_data,
                        extract_links=depth < self.current_url_level  # Only extract links if below current level
                    )
                    
                    # Mark as processed regardless of result to avoid endless retries
                    self.url_storage.save_processed_url(url, depth=depth, origin=origin)
                    self.url_storage.remove_pending_url(url)
                    
                    if result.get("success", False):
                        stats["success_count"] += 1
                    else:
                        stats["error_count"] += 1
                        logger.warning(f"Error processing URL {url}: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    stats["error_count"] += 1
                    logger.error(f"Exception processing URL {url}: {e}")
                    
                    # Increment attempt count for this URL 
                    self.url_storage.increment_url_attempt(url)
                
                stats["urls_processed"] += 1
            
            # Update end time and duration
            stats["end_time"] = datetime.now()
            stats["duration_seconds"] = (stats["end_time"] - stats["start_time"]).total_seconds()
            
            return {
                "status": "success",
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error in process_all_pending_urls: {e}")
            return {
                "status": "error",
                "error": str(e),
                "urls_processed": stats["urls_processed"]
            }
    
    def handle_stuck_urls(self):
        """
        Handle problematic URLs that might get stuck in the pending queue
        by force-marking them as processed
        
        Returns:
            Number of URLs that were force-marked as processed
        """
        all_pending = self.url_storage.get_pending_urls()
        problematic_urls = []
        
        for url_info in all_pending:
            # Mark URLs with high attempt counts as problematic
            if url_info.get("attempt_count", 0) >= 3:
                problematic_urls.append(url_info)
            
            # Also look for very long URLs or URLs with strange characters that might cause issues
            url = url_info.get("url", "")
            if len(url) > 500:  # Extremely long URLs
                problematic_urls.append(url_info)
        
        # Force mark all problematic URLs as processed
        for url_info in problematic_urls:
            url = url_info["url"]
            origin = url_info["origin"]
            depth = url_info["depth"]
            
            logger.warning(f"Force marking problematic URL as processed: {url}")
            self.url_storage.save_processed_url(url, depth=depth, origin=origin)
            self.url_storage.remove_pending_url(url)
        
        return len(problematic_urls)
