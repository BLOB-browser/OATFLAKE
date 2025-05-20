#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
URLDiscoveryManager handles discovery of URLs from resources.
This is extracted from URLProcessor to make the code more modular.
"""

import logging
import os
import pandas as pd
from typing import Dict, List, Any, Optional

from scripts.analysis.single_resource_processor import SingleResourceProcessor
from scripts.analysis.url_storage import URLStorageManager
from scripts.analysis.content_fetcher import ContentFetcher

logger = logging.getLogger(__name__)

class URLDiscoveryManager:
    """
    Handles discovery of new URLs from resources.
    This component is responsible for finding new URLs when there are no pending URLs
    at a specific level.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the URL Discovery Manager.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        self.resources_csv_path = os.path.join(data_folder, 'resources.csv')
        self.processed_urls_file = os.path.join(data_folder, "processed_urls.csv")
        self.url_storage = URLStorageManager(self.processed_urls_file)
        self.cancel_requested = False
    
    def discover_urls_from_resources(self, level: int = 1, max_depth: int = 4) -> Dict[str, Any]:
        """
        Discover URLs at a specific level by processing all resources.
        This is typically used when there are no pending URLs at a level,
        especially level 1.
        
        Args:
            level: The level to discover URLs for (typically 1)
            max_depth: Maximum depth for URL discovery
            
        Returns:
            Dictionary with discovery results
        """
        logger.info(f"No pending URLs at level {level}, triggering URL discovery on all resources")
        
        # Check if the resources file exists
        if not os.path.exists(self.resources_csv_path):
            logger.warning(f"Resources file not found: {self.resources_csv_path}")
            return {"status": "error", "error": "resources_file_not_found"}
            
        # Use the dedicated discovery function that doesn't use SingleResourceProcessor
        return self.discover_urls_from_resources_direct(level, max_depth)
    
    def discover_urls_from_resources_direct(self, level: int = 1, max_depth: int = 4) -> Dict[str, Any]:
        """
        Discover URLs directly from resources using ContentFetcher, without going through SingleResourceProcessor.
        This provides a cleaner separation between discovery and analysis.
        
        Args:
            level: The level to discover URLs for (typically 1)
            max_depth: Maximum depth for URL discovery
            
        Returns:
            Dictionary with discovery results
        """
        logger.info(f"Starting direct URL discovery at level {level} with max_depth={max_depth}")
        
        # Load resources with URLs
        resources_df = pd.read_csv(self.resources_csv_path)
        resources_with_url = resources_df[resources_df['url'].notna()]
        
        # Initialize ContentFetcher directly
        content_fetcher = ContentFetcher(timeout=120)  # Use longer timeout for discovery
        
        # Set discovery_only_mode to True
        content_fetcher.discovery_only_mode = True
        logger.info("Set discovery_only_mode=True in ContentFetcher for dedicated discovery")
        
        # Process every resource to trigger URL discovery
        processed_resources = 0
        total_discovered_urls = 0
        
        for idx, row in resources_with_url.iterrows():
            # Check for cancellation
            if self.cancel_requested:
                logger.info("URL discovery cancelled by external request")
                break
                
            resource = row.to_dict()
            resource_url = resource.get('url', '')
            resource_id = resource.get('title', '')  # Use title as resource ID
            
            if not resource_url:
                continue
                
            logger.info(f"Triggering direct URL discovery for resource: {resource_id} - {resource_url}")
            
            # Use ContentFetcher's discovery_phase method directly
            result = content_fetcher.discovery_phase(
                urls=[resource_url],
                max_depth=max_depth,
                force_reprocess=True,  # Always force reprocessing for discovery
                resource_ids={resource_url: resource_id}  # Map URLs to resource IDs
            )
            
            # Log discovery results
            discovered_urls = result.get("total_discovered", 0)
            total_discovered_urls += discovered_urls
            logger.info(f"Discovered {discovered_urls} URLs from resource {resource_id}")
            
            processed_resources += 1
        
        # After discovery, check for pending URLs
        pending_urls = self.url_storage.get_pending_urls(depth=level)
        
        if not pending_urls:
            logger.info(f"Still no pending URLs found at level {level} after discovery")
            return {
                "status": "skipped", 
                "reason": "no_pending_urls_after_discovery", 
                "resources_processed": processed_resources,
                "total_discovered": total_discovered_urls
            }
        else:
            logger.info(f"Found {len(pending_urls)} pending URLs at level {level} after discovery")
            return {
                "status": "success", 
                "discovered_urls": len(pending_urls),
                "resources_processed": processed_resources,
                "total_discovered": total_discovered_urls
            }
    
    def check_and_perform_discovery(self, level: int = 1, max_depth: int = 4) -> Dict[str, Any]:
        """
        Check if URL discovery is needed and perform it automatically if necessary.
        
        Args:
            level: The level to check and potentially discover URLs for
            max_depth: Maximum depth for URL discovery
            
        Returns:
            Dictionary with status and action taken
        """
        # Check if there are any pending URLs at the specified level
        pending_urls = self.url_storage.get_pending_urls(depth=level)
        
        if not pending_urls:
            # No pending URLs, check if there are unanalyzed resources
            if not os.path.exists(self.resources_csv_path):
                return {"status": "error", "error": "resources_file_not_found"}
                
            resources_df = pd.read_csv(self.resources_csv_path)
            resources_with_url = resources_df['url'].notna().sum()
            analyzed_resources = resources_df['analysis_completed'].fillna(False).sum()
            unanalyzed_resources = resources_with_url - analyzed_resources
            
            if unanalyzed_resources > 0:
                # We have unanalyzed resources and no pending URLs, so perform discovery
                logger.info("No pending URLs but unanalyzed resources exist. Automatically performing URL discovery.")
                # Use the direct discovery method instead of going through SingleResourceProcessor
                return self.discover_urls_from_resources_direct(level=level, max_depth=max_depth)
            else:
                return {
                    "status": "skipped",
                    "reason": "no_unanalyzed_resources",
                    "message": "No pending URLs and no unanalyzed resources. Nothing to discover."
                }
        else:
            return {
                "status": "skipped",
                "reason": "pending_urls_exist",
                "message": f"Found {len(pending_urls)} pending URLs at level {level}. No need for discovery.",
                "pending_count": len(pending_urls)
            }
    
    def check_discovery_needed(self) -> Dict[str, Any]:
        """
        Check if URL discovery is needed based on resource and URL status.
        
        Returns:
            Dictionary with discovery status information
        """
        try:
            if not os.path.exists(self.resources_csv_path):
                return {
                    "status": "error",
                    "error": f"Resources file not found: {self.resources_csv_path}"
                }
                
            # Load resources
            resources_df = pd.read_csv(self.resources_csv_path)
            
            # Get pending URLs for all levels
            pending_urls_by_level = {}
            total_pending_urls = 0
            for level in range(1, 5):
                pending_urls = self.url_storage.get_pending_urls(depth=level)
                pending_urls_by_level[level] = len(pending_urls)
                total_pending_urls += len(pending_urls)
            
            # Get overall resource statistics
            total_resources = len(resources_df)
            resources_with_url = resources_df['url'].notna().sum()
            analyzed_resources = resources_df['analysis_completed'].fillna(False).sum()
            unanalyzed_resources = resources_with_url - analyzed_resources
            
            # Check if we need to trigger URL discovery
            # We need to discover URLs when there are no pending URLs but still unanalyzed resources
            discovery_needed = (total_pending_urls == 0) and (unanalyzed_resources > 0)
            
            result = {
                "total_resources": total_resources,
                "resources_with_url": int(resources_with_url),
                "analyzed_resources": int(analyzed_resources),
                "unanalyzed_resources": int(unanalyzed_resources),
                "total_pending_urls": total_pending_urls,
                "pending_urls_by_level": pending_urls_by_level,
                "discovery_needed": discovery_needed,
                "status": "success"
            }
            
            # If discovery is needed, add a note suggesting to use the new check_and_perform_discovery method
            if discovery_needed:
                result["recommendation"] = "No pending URLs found but there are unanalyzed resources. Use check_and_perform_discovery() to automatically handle discovery."
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking discovery status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
