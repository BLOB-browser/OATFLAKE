#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import csv
import pandas as pd
from typing import Dict, Any, List, Optional

from scripts.analysis.orchestration.base_orchestrator import BaseOrchestrator
from scripts.analysis.url_storage import URLStorageManager
from scripts.analysis.single_resource_processor import SingleResourceProcessor
from scripts.analysis.url_discovery_manager import URLDiscoveryManager

logger = logging.getLogger(__name__)

class URLProcessor(BaseOrchestrator):
    """
    Handles URL processing functionality including:
    - Checking URL processing state
    - Processing URLs at specific levels
    - URL discovery
    """
    
    def check_url_processing_state(self) -> Dict[str, Any]:
        """
        Check the current state of resource processing to identify potential issues.
        
        Returns:
            Dictionary with diagnostic information
        """
        try:
            # Get resources CSV path
            resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            
            if not os.path.exists(resources_csv_path):
                return {
                    "status": "error",
                    "error": f"Resources file not found: {resources_csv_path}"
                }
                
            # Load resources
            resources_df = pd.read_csv(resources_csv_path)
            
            # Get URL statistics
            url_storage = URLStorageManager(processed_urls_file)
            
            # Get pending URLs for all levels
            pending_urls_by_level = {}
            total_pending_urls = 0
            for level in range(1, 5):
                pending_urls = url_storage.get_pending_urls(depth=level)
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
                result["recommendation"] = "No pending URLs found but there are unanalyzed resources. Discovery will happen automatically in the next processing cycle."
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking resource processing state: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_urls_by_levels(self, max_depth: int = 4, batch_size: int = 50) -> Dict[str, Any]:
        """
        Process all pending URLs by level, automatically performing discovery when needed.
        
        Args:
            max_depth: Maximum depth to process (usually 4)
            batch_size: Number of URLs to process in each batch
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing URLs by level up to depth {max_depth}")
        
        # Initialize URL storage
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Initialize URL discovery manager
        url_discovery_manager = URLDiscoveryManager(self.data_folder)
        url_discovery_manager.cancel_requested = self.cancel_requested
        
        # Structure to track results
        results = {
            "processed_by_level": {},
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "discovery_performed": False
        }
        
        # Process each level
        for level in range(1, max_depth + 1):
            # First check and perform discovery if needed
            discovery_result = url_discovery_manager.check_and_perform_discovery(level=level, max_depth=max_depth)
            
            if discovery_result.get("status") == "success":
                results["discovery_performed"] = True
                results["discovered_urls"] = discovery_result.get("discovered_urls", 0)
                logger.info(f"Discovery performed for level {level}: found {discovery_result.get('discovered_urls', 0)} URLs")
            
            # Get all pending URLs for this level
            pending_urls = url_storage.get_pending_urls(depth=level)
            
            if not pending_urls:
                logger.info(f"No pending URLs at level {level} to process")
                continue
                
            logger.info(f"Processing {len(pending_urls)} URLs at level {level}")
            
            # Process the URLs for this level
            level_result = self.process_urls_at_level(
                level=level,
                batch_size=batch_size
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
                
        logger.info(f"URL processing completed: {results['total_processed']} URLs processed total")
        return results
        
    def process_urls_at_level(self, level: int, batch_size: int = 50) -> Dict[str, Any]:
        """
        Process all pending URLs at a specific level.
        
        Args:
            level: The depth level to process (1=first level, 2=second level, etc.)
            batch_size: Maximum number of URLs to process in one batch
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing all pending URLs at level {level}")
        
        # Initialize components
        single_processor = SingleResourceProcessor(self.data_folder)
        resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Get all pending URLs at this level
        pending_urls = url_storage.get_pending_urls(depth=level)
        logger.info(f"URLProcessor: Retrieved {len(pending_urls) if pending_urls else 0} pending URLs at level {level}")
        
        if not pending_urls:
            logger.info(f"No pending URLs found at level {level}")
            return {"status": "skipped", "reason": "no_pending_urls"}
        
        # Initialize counters
        total_urls = len(pending_urls)
        processed_urls_count = 0
        success_count = 0
        error_count = 0
        
        try:
            # Load resources
            if not os.path.exists(resources_csv_path):
                logger.error(f"Resources file not found: {resources_csv_path}")
                return {"status": "error", "error": "resources_file_not_found"}
                
            resources_df = pd.read_csv(resources_csv_path)
            resources_with_url = resources_df[resources_df['url'].notna()]
            
            # Group URLs by origin (or resource_id if available)
            # First check if we have resource IDs in the pending URLs
            has_resource_ids = False
            for pending_url_data in pending_urls:
                if pending_url_data.get('resource_id'):
                    has_resource_ids = True
                    break
                    
            if has_resource_ids:
                # Group URLs by resource ID
                urls_by_resource = {}
                for pending_url_data in pending_urls:
                    url = pending_url_data.get('url')
                    resource_id = pending_url_data.get('resource_id', '')
                    origin = pending_url_data.get('origin', '')
                    
                    # If no resource ID, use origin as fallback
                    group_key = resource_id or origin
                    
                    if group_key not in urls_by_resource:
                        urls_by_resource[group_key] = []
                    
                    urls_by_resource[group_key].append(pending_url_data)
                
                logger.info(f"Processing {total_urls} URLs at level {level} grouped by resource ID")
                
                # Find resources for each resource ID
                resource_mapping = {}
                for resource_id in urls_by_resource.keys():
                    if not resource_id:
                        continue
                        
                    # Try to find resource by title match
                    resource_found = False
                    for _, row in resources_with_url.iterrows():
                        title = row.get('title', '').strip()
                        if title == resource_id:
                            resource_mapping[resource_id] = row.to_dict()
                            resource_found = True
                            logger.info(f"Found resource for ID: {resource_id}")
                            break
                    
                    if not resource_found:
                        logger.info(f"Could not find exact resource for ID: {resource_id}, using first resource as fallback")
                        if not resources_with_url.empty:
                            resource_mapping[resource_id] = resources_with_url.iloc[0].to_dict()
                
                # Process URLs by resource ID
                for resource_id, urls_for_resource in urls_by_resource.items():
                    # Check for cancellation
                    if self.cancel_requested:
                        logger.info(f"URL processing at level {level} cancelled")
                        break
                    
                    # Get resource for this group
                    resource_for_processing = None
                    if resource_id and resource_id in resource_mapping:
                        resource_for_processing = resource_mapping[resource_id]
                        logger.info(f"Using resource {resource_id} for processing {len(urls_for_resource)} URLs")
                    else:
                        # Fallback to first resource if we can't find a match
                        if not resources_with_url.empty:
                            resource_for_processing = resources_with_url.iloc[0].to_dict()
                            logger.info(f"Using fallback resource for URLs without resource ID")
                        else:
                            logger.warning(f"No resources available for processing URLs")
                            continue
                    
                    # Process each URL using the specific_url_processor
                    for idx, pending_url_data in enumerate(urls_for_resource):
                        # Check for cancellation
                        if self.cancel_requested:
                            logger.info(f"URL processing at level {level} cancelled")
                            break
                        
                        url_to_process = pending_url_data.get('url')
                        origin = pending_url_data.get('origin', '')
                        
                        # Process the URL
                        logger.info(f"Processing URL: {url_to_process}")
                        result = single_processor.process_specific_url(
                            url=url_to_process,
                            origin_url=origin,
                            resource=resource_for_processing,
                            depth=level
                        )
                        
                        processed_urls_count += 1
                        if result.get('success', False):
                            success_count += 1
                        else:
                            error_count += 1
                            
                        # Log progress
                        if processed_urls_count % 10 == 0:
                            logger.info(f"Processed {processed_urls_count}/{total_urls} URLs at level {level}")
                        
                        # If we've reached our batch size, break
                        if batch_size and processed_urls_count >= batch_size:
                            logger.info(f"Reached batch size of {batch_size} URLs, stopping processing")
                            break
                    
                    # If we've reached our batch size, break out of the resource loop too
                    if batch_size and processed_urls_count >= batch_size:
                        break
            else:
                # Original approach - group by origin URL
                origin_urls = set()
                for pending_url_data in pending_urls:
                    origin = pending_url_data.get('origin', '')
                    if origin:
                        origin_urls.add(origin)
                        
                logger.info(f"Found {len(origin_urls)} origin URLs for level {level}")
                
                # Map each origin URL to a resource
                resource_mapping = {}
                
                # For each origin, find the matching resource
                for origin_url in origin_urls:
                    # First try exact match
                    resource_found = False
                    for _, row in resources_with_url.iterrows():
                        if row['url'] == origin_url:
                            resource_mapping[origin_url] = row.to_dict()
                            resource_found = True
                            break
                    
                    # If no exact match, try domain-based matching
                    if not resource_found:
                        import re
                        origin_domain_match = re.search(r'https?://([^/]+)', origin_url)
                        if origin_domain_match:
                            origin_domain = origin_domain_match.group(1)
                            
                            for _, row in resources_with_url.iterrows():
                                resource_url = row['url']
                                resource_domain_match = re.search(r'https?://([^/]+)', resource_url)
                                if resource_domain_match and resource_domain_match.group(1) == origin_domain:
                                    resource_mapping[origin_url] = row.to_dict()
                                    resource_found = True
                                    break
                    
                    # If still not found, use first resource as fallback
                    if not resource_found and not resources_with_url.empty:
                        resource_mapping[origin_url] = resources_with_url.iloc[0].to_dict()
                
                # Group URLs by origin
                urls_by_origin = {}
                for pending_url_data in pending_urls:
                    url = pending_url_data.get('url')
                    origin = pending_url_data.get('origin', '')
                    
                    if origin not in urls_by_origin:
                        urls_by_origin[origin] = []
                    
                    urls_by_origin[origin].append(url)
                
                # Process URLs by origin
                for origin, urls in urls_by_origin.items():
                    # Check for cancellation
                    if self.cancel_requested:
                        logger.info(f"URL processing at level {level} cancelled")
                        break
                    
                    # Get resource for this origin
                    resource = resource_mapping.get(origin)
                    
                    # If no resource found, use the first one
                    if not resource and not resources_with_url.empty:
                        resource = resources_with_url.iloc[0].to_dict()
                    
                    if not resource:
                        logger.warning(f"No resource found for origin {origin}, skipping URLs")
                        continue
                    
                    logger.info(f"Processing {len(urls)} URLs from origin {origin}")
                    
                    # Process each URL
                    for url in urls:
                        # Check for cancellation
                        if self.cancel_requested:
                            logger.info(f"URL processing at level {level} cancelled")
                            break
                            
                        logger.info(f"Processing URL: {url}")
                        result = single_processor.process_specific_url(
                            url=url,
                            origin_url=origin,
                            resource=resource,
                            depth=level
                        )
                        
                        processed_urls_count += 1
                        if result.get('success', False):
                            success_count += 1
                        else:
                            error_count += 1
                            
                        # Log progress
                        if processed_urls_count % 10 == 0:
                            logger.info(f"Processed {processed_urls_count}/{total_urls} URLs at level {level}")
                        
                        # If we've reached our batch size, break
                        if batch_size and processed_urls_count >= batch_size:
                            logger.info(f"Reached batch size of {batch_size} URLs, stopping processing")
                            break
                    
                    # If we've reached our batch size, break out of the origin loop too
                    if batch_size and processed_urls_count >= batch_size:
                        break
            
            return {
                "status": "success",
                "processed_urls": processed_urls_count,
                "success_count": success_count,
                "error_count": error_count
            }
            
        except Exception as e:
            logger.error(f"Error processing URLs at level {level}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processed_urls": processed_urls_count,
                "success_count": success_count,
                "error_count": error_count
            }
