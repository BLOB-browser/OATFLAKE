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
        
        # Load resources to trigger discovery
        if not os.path.exists(self.resources_csv_path):
            logger.warning(f"Resources file not found: {self.resources_csv_path}")
            return {"status": "error", "error": "resources_file_not_found"}
            
        resources_df = pd.read_csv(self.resources_csv_path)
        
        # Only process resources that have a URL
        resources_with_url = resources_df[resources_df['url'].notna()]
        
        # Initialize the SingleResourceProcessor with forced fetch
        single_processor = SingleResourceProcessor(self.data_folder)
        single_processor.force_fetch = True
        
        # Process every resource to trigger URL discovery
        processed_resources = 0
        for idx, row in resources_with_url.iterrows():
            # Check for cancellation
            if self.cancel_requested:
                logger.info("URL discovery cancelled by external request")
                break
                
            resource = row.to_dict()
            resource_url = resource.get('url', '')
            
            if not resource_url:
                continue
                
            logger.info(f"Triggering URL discovery for resource: {resource.get('title', 'Unnamed')} - {resource_url}")
            
            # Process the resource to discover URLs
            result = single_processor.process_resource(
                resource=resource,
                resource_id=f"{idx+1}/{len(resources_with_url)}",
                idx=idx,
                csv_path=self.resources_csv_path,
                max_depth=max_depth,  # Use maximum depth for discovery
                process_by_level=True,  # Ensure we use level-based processing for proper URL discovery
                discovery_only=True  # Only discover URLs, don't process content
            )
            
            processed_resources += 1
        
        # After discovery, check for pending URLs
        pending_urls = self.url_storage.get_pending_urls(depth=level)
        
        if not pending_urls:
            logger.info(f"Still no pending URLs found at level {level} after discovery")
            return {
                "status": "skipped", 
                "reason": "no_pending_urls_after_discovery", 
                "resources_processed": processed_resources
            }
        else:
            logger.info(f"Found {len(pending_urls)} pending URLs at level {level} after discovery")
            return {
                "status": "success", 
                "discovered_urls": len(pending_urls),
                "resources_processed": processed_resources
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
            
            # If discovery is needed and there are resources with URLs, suggest triggering discovery
            if discovery_needed and resources_with_url > 0:
                result["recommendation"] = "No pending URLs found but there are unanalyzed resources. Consider triggering URL discovery by running with force_url_fetch=True"
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking discovery status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
