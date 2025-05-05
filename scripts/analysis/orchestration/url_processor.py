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
                result["recommendation"] = "No pending URLs found but there are unanalyzed resources. Consider triggering URL discovery by running with force_url_fetch=True"
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking resource processing state: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
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
        logger.info(f"Processing URLs by level up to depth {max_depth}")
        logger.info(f"Force fetch enabled: {force_fetch}")
        
        # Initialize URL storage
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Set force_fetch in the URL storage manager
        logger.info(f"Setting force_fetch={force_fetch} for URL processing")
        
        # Structure to track results
        results = {
            "processed_by_level": {},
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
        
        # Check which levels have pending URLs
        levels_with_pending = {}
        highest_level_with_pending = 0
        
        for level in range(1, max_depth + 1):
            pending_urls = url_storage.get_pending_urls(depth=level)
            if pending_urls:
                levels_with_pending[level] = len(pending_urls)
                highest_level_with_pending = max(highest_level_with_pending, level)
                logger.info(f"Found {len(pending_urls)} pending URLs at level {level}")
            else:
                logger.info(f"No pending URLs at level {level}")
        
        # If we have no pending URLs at any level, log a warning
        if not levels_with_pending:
            logger.warning(f"No pending URLs found at any level (1-{max_depth})!")
            logger.warning("Checking raw pending_urls.csv file for level 3-4 URLs...")
        
        # Process all levels with pending URLs
        if levels_with_pending:
            logger.info(f"Found pending URLs at {len(levels_with_pending)} levels: {list(levels_with_pending.keys())}")
            logger.info(f"Highest level with pending URLs: {highest_level_with_pending}")
            
            # Process each level that has pending URLs
            for level in sorted(levels_with_pending.keys()):
                logger.info(f"Processing {levels_with_pending[level]} URLs at level {level}")
                
                # Process this level with URL storage manager
                level_result = self.process_urls_at_level(
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
        else:
            # If no levels have pending URLs, check raw pending_urls.csv for all URLs
            # This is critical when URLs have been marked as processed but never analyzed
            import csv
            
            # Check pending_urls.csv directly for URLs at ALL levels (not just 3-4)
            pending_csv_path = os.path.join(os.path.dirname(processed_urls_file), "pending_urls.csv")
            raw_urls_by_level = {}
            
            if os.path.exists(pending_csv_path):
                try:
                    with open(pending_csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        next(reader, None)  # Skip header
                        
                        # Check each row for level 3-4 URLs
                        for row in reader:
                            if len(row) >= 2 and row[1].isdigit():
                                url = row[0]
                                depth = int(row[1])
                                
                                # Track all levels, not just 3 and 4
                                if depth not in raw_urls_by_level:
                                    raw_urls_by_level[depth] = []
                                raw_urls_by_level[depth].append({
                                    'url': url,
                                    'depth': depth,
                                    'origin': row[2] if len(row) > 2 else ""
                                })
                    
                    # Log what we found
                    for level, urls in raw_urls_by_level.items():
                        logger.warning(f"Found {len(urls)} raw URLs at level {level} in pending_urls.csv")
                    
                    # If we found any level 3-4 URLs, force process them
                    if raw_urls_by_level:
                        logger.warning("Found URLs in raw CSV, attempting to process with force_fetch=True")
                        
                        # Process all levels that have URLs, not just 3 and 4
                        # This ensures we process all URLs that might have been missed
                        for level in sorted(raw_urls_by_level.keys()):
                            if raw_urls_by_level[level]:
                                urls_to_process = raw_urls_by_level[level]
                                logger.warning(f"Force processing {len(urls_to_process)} URLs at level {level}")
                                
                                # Create a custom level processing method that directly uses these URLs
                                processed_count = 0
                                success_count = 0
                                error_count = 0
                                
                                # Initialize SingleResourceProcessor
                                single_processor = SingleResourceProcessor(self.data_folder)
                                single_processor.force_fetch = True  # Force fetch ALL URLs
                                
                                # Load resources for origin mapping
                                resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
                                resources_df = pd.read_csv(resources_csv_path)
                                resources_with_url = resources_df[resources_df['url'].notna()]
                                
                                # Process each URL directly
                                for url_data in urls_to_process:
                                    url_to_process = url_data['url']
                                    origin_url = url_data['origin']
                                    
                                    # Check if URL is already processed before calling the processor
                                    # If force_fetch is false, skip already processed URLs
                                    if url_storage.url_is_processed(url_to_process) and not force_fetch:
                                        logger.info(f"Skipping already processed URL (in raw URLs): {url_to_process}")
                                        # Make sure it's removed from pending list
                                        url_storage.remove_pending_url(url_to_process)
                                        # Update counters since we're skipping
                                        processed_count += 1
                                        success_count += 1
                                        continue
                                    
                                    # Find resource for this origin - with better matching
                                    resource_for_origin = None
                                    
                                    # First try exact match
                                    for _, row in resources_with_url.iterrows():
                                        if row['url'] == origin_url:
                                            resource_for_origin = row.to_dict()
                                            logger.info(f"Found exact match for origin URL: {origin_url}")
                                            break
                                    
                                    # If exact match fails, try base domain matching
                                    if not resource_for_origin:
                                        # Extract main domain from the origin URL (e.g., https://fabacademy.org/about/index.html -> fabacademy.org)
                                        import re
                                        origin_domain_match = re.search(r'https?://([^/]+)', origin_url)
                                        if origin_domain_match:
                                            origin_domain = origin_domain_match.group(1)
                                            logger.info(f"Trying domain match for: {origin_domain}")
                                            
                                            # Look for resources with the same domain
                                            for _, row in resources_with_url.iterrows():
                                                resource_url = row['url']
                                                resource_domain_match = re.search(r'https?://([^/]+)', resource_url)
                                                if resource_domain_match and resource_domain_match.group(1) == origin_domain:
                                                    resource_for_origin = row.to_dict()
                                                    logger.info(f"Found domain match for origin: {origin_url} -> resource: {resource_url}")
                                                    break
                                    
                                    if not resource_for_origin:
                                        logger.warning(f"Could not find resource for origin URL: {origin_url}")
                                        logger.warning(f"Looking for any matching main domain resource")
                                        
                                        # Try to find any resource with similar domain 
                                        import re
                                        main_domain = None
                                        origin_domain_match = re.search(r'https?://([^/]+)', origin_url)
                                        if origin_domain_match:
                                            main_domain = origin_domain_match.group(1)
                                        
                                        # Look for any resource that belongs to this domain
                                        if main_domain:
                                            # Extract just the base domain (e.g., fabacademy.org from academy.fabacademy.org)
                                            base_domain = re.sub(r'^[^.]+\.', '', main_domain) if '.' in main_domain else main_domain
                                            logger.warning(f"Looking for resources matching base domain: {base_domain}")
                                            
                                            # Find any resource with this base domain
                                            for _, row in resources_with_url.iterrows():
                                                resource_url = row['url']
                                                if base_domain in resource_url:
                                                    resource_for_origin = row.to_dict()
                                                    logger.warning(f"Found base domain match: {base_domain} in {resource_url}")
                                                    break
                                        
                                        # If still not found, use the main fabacademy resource as fallback
                                        if not resource_for_origin and not resources_with_url.empty:
                                            # First look for fabacademy.org as a final fallback
                                            for _, row in resources_with_url.iterrows():
                                                if 'fabacademy.org' in row['url']:
                                                    resource_for_origin = row.to_dict()
                                                    logger.warning(f"Using fabacademy.org resource as fallback for {origin_url}")
                                                    break
                                            
                                            # If still not found, use the first resource as last resort
                                            if not resource_for_origin:
                                                resource_for_origin = resources_with_url.iloc[0].to_dict()
                                                logger.warning(f"Using first resource as last-resort fallback for {origin_url}")
                                        
                                        # If we still don't have a resource, we have to skip
                                        if not resource_for_origin:
                                            logger.error(f"No resources found for URL {url_to_process}")
                                            continue
                                    
                                    # Process the URL - respect force_fetch setting
                                    if force_fetch:
                                        logger.warning(f"Force processing URL: {url_to_process}")
                                        # Set force_fetch to match our parameter
                                        single_processor.force_fetch = force_fetch
                                    else:
                                        logger.info(f"Processing URL (not forcing): {url_to_process}")
                                        single_processor.force_fetch = False
                                        
                                    result = single_processor.process_specific_url(
                                        url=url_to_process,
                                        origin_url=origin_url,
                                        resource=resource_for_origin,
                                        depth=level
                                    )
                                    
                                    processed_count += 1
                                    if result.get('success', False):
                                        success_count += 1
                                    else:
                                        error_count += 1
                                
                                # Record results
                                results["processed_by_level"][level] = processed_count
                                results["total_processed"] += processed_count
                                results["successful"] += success_count
                                results["failed"] += error_count
                                
                                logger.warning(f"Force processed {processed_count} URLs at level {level}: {success_count} success, {error_count} errors")
                except Exception as e:
                    logger.error(f"Error checking or processing raw pending URLs: {e}")
            
            # If no raw URLs found either, try direct level-based processing
            if not raw_urls_by_level:
                logger.warning("No URLs found in raw CSV either, making one final attempt with process_urls_at_level")
                
                # Focus on levels 3 and 4 which might have been missed
                for level in [3, 4]:
                    logger.warning(f"Final attempt to check level {level} URLs with force_fetch={force_fetch}")
                    level_result = self.process_urls_at_level(
                        level=level,
                        batch_size=batch_size,
                        force_fetch=force_fetch  # Use the same force_fetch setting that was passed to this method
                    )
                    
                    # Record results even if 0
                    processed = level_result.get('processed_urls', 0)
                    results["processed_by_level"][level] = processed
                    results["total_processed"] += processed
                    results["successful"] += level_result.get('success_count', 0)
                    results["failed"] += level_result.get('error_count', 0)
                    
                    logger.warning(f"Final attempt at level {level} complete: {processed} URLs processed")

        logger.info(f"URL processing completed: {results['total_processed']} URLs processed total")
        return results
        
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
        logger.info(f"Processing all pending URLs at level {level}")
        
        # Initialize components
        single_processor = SingleResourceProcessor(self.data_folder)
        resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Set force_fetch if enabled
        if force_fetch:
            logger.info(f"Setting force_fetch=True in SingleResourceProcessor for level {level}")
            single_processor.force_fetch = True
        else:
            logger.info(f"Setting force_fetch=False in SingleResourceProcessor for level {level}")
            single_processor.force_fetch = False
        
        # Get all pending URLs at this level
        pending_urls = url_storage.get_pending_urls(depth=level)
        logger.info(f"URLProcessor: Retrieved {len(pending_urls) if pending_urls else 0} pending URLs at level {level}")
        
        # If force_fetch is enabled, we'll also check for URLs directly from pending_urls.csv
        # This handles cases where URLs are in processed_urls.csv but weren't actually analyzed
        if force_fetch and not pending_urls:
            logger.info(f"No pending URLs found at level {level}, but force_fetch=True, checking raw pending_urls.csv")
            
            # Check pending_urls.csv directly for level-specific URLs
            pending_csv_path = os.path.join(os.path.dirname(processed_urls_file), "pending_urls.csv")
            raw_urls = []
            
            if os.path.exists(pending_csv_path):
                try:
                    with open(pending_csv_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        next(reader, None)  # Skip header
                        
                        # Check each row for the specific level URLs
                        for row in reader:
                            if len(row) >= 2 and row[1].isdigit():
                                url = row[0]
                                depth = int(row[1])
                                
                                # Only consider URLs at the specified level
                                if depth == level:
                                    raw_urls.append({
                                        'url': url,
                                        'depth': depth,
                                        'origin': row[2] if len(row) > 2 else ""
                                    })
                except Exception as e:
                    logger.error(f"Error reading raw pending_urls.csv: {e}")
                    
                if raw_urls:
                    logger.info(f"Found {len(raw_urls)} raw URLs at level {level} in pending_urls.csv to force process")
                    pending_urls = raw_urls
        
        if not pending_urls:
            logger.info(f"No pending URLs found at level {level}")
            
            # When there are no pending URLs at this level, we should trigger URL discovery
            # by processing all resources with force_fetch enabled to discover new URLs
            if level == 1:  # Only do this for level 1, as deeper levels depend on level 1 discoveries
                logger.info(f"No pending URLs at level {level}, triggering URL discovery on all resources")
                # Load resources to trigger discovery
                if os.path.exists(resources_csv_path):
                    resources_df = pd.read_csv(resources_csv_path)
                    
                    # Only process resources that have a URL
                    resources_with_url = resources_df[resources_df['url'].notna()]
                    
                    # Process every resource to trigger URL discovery
                    # We'll do this by forcing URL fetching on each resource
                    processed_resources = 0
                    for idx, row in resources_with_url.iterrows():
                        resource = row.to_dict()
                        resource_url = resource.get('url', '')
                        
                        if not resource_url:
                            continue
                            
                        logger.info(f"Triggering URL discovery for resource: {resource.get('title', 'Unnamed')} - {resource_url}")
                        
                        # Store the original force_fetch value
                        original_force_fetch = single_processor.force_fetch
                        # Enable force_fetch for this operation
                        single_processor.force_fetch = True
                        
                        # Process the resource to discover URLs
                        result = single_processor.process_resource(
                            resource=resource,
                            resource_id=f"{idx+1}/{len(resources_with_url)}",
                            idx=idx,
                            csv_path=resources_csv_path,
                            max_depth=4,  # Use maximum depth for discovery
                            process_by_level=True  # Ensure we use level-based processing for proper URL discovery
                        )
                        
                        # Restore original force_fetch setting
                        single_processor.force_fetch = original_force_fetch
                        processed_resources += 1
                    
                    # After discovery, check again for pending URLs
                    pending_urls = url_storage.get_pending_urls(depth=level)
                    
                    if not pending_urls:
                        logger.info(f"Still no pending URLs found at level {level} after discovery")
                        return {"status": "skipped", "reason": "no_pending_urls_after_discovery", "resources_processed": processed_resources}
                    else:
                        logger.info(f"Found {len(pending_urls)} pending URLs at level {level} after discovery")
                else:
                    logger.warning(f"Resources file not found: {resources_csv_path}")
                    return {"status": "error", "error": "resources_file_not_found"}
            else:
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
            
            # First check if we have resource IDs in the pending URLs
            # If so, we'll group URLs by resource ID instead of origin
            has_resource_ids = False
            for pending_url_data in pending_urls:
                if pending_url_data.get('resource_id'):
                    has_resource_ids = True
                    break
                    
            if has_resource_ids:
                # Group URLs by resource ID (simpler approach)
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
                        
                        # Check if URL is already processed before calling the processor
                        if url_storage.url_is_processed(url_to_process) and not force_fetch:
                            logger.info(f"Skipping already processed URL: {url_to_process}")
                            # Make sure it's removed from pending list
                            url_storage.remove_pending_url(url_to_process)
                            # Update counters since we're skipping
                            processed_urls_count += 1
                            success_count += 1
                            continue
                        
                        # Process this URL
                        logger.info(f"Processing URL for resource {resource_id}: {url_to_process}")
                        url_result = single_processor.process_specific_url(
                            url=url_to_process,
                            origin_url=origin,
                            resource=resource_for_processing,
                            depth=level
                        )
                        
                        processed_urls_count += 1
                        
                        if url_result.get('success', False):
                            success_count += 1
                        else:
                            error_count += 1
                            logger.error(f"Error processing URL at level {level}: {url_to_process}")
                        
                        # Show progress every 10 URLs
                        if processed_urls_count % 10 == 0 or processed_urls_count == total_urls:
                            logger.info(f"Progress: {processed_urls_count}/{total_urls} URLs ({int(processed_urls_count/total_urls*100)}%)")
            
            else:
                # Original approach - group by origin
                logger.info(f"No resource IDs found in pending URLs, using origin-based grouping")
                urls_by_origin = {}
                for pending_url_data in pending_urls:
                    url = pending_url_data.get('url')
                    origin = pending_url_data.get('origin', '')
                    
                    if origin not in urls_by_origin:
                        urls_by_origin[origin] = []
                    
                    urls_by_origin[origin].append(pending_url_data)
                
                logger.info(f"Processing {total_urls} URLs at level {level} from {len(urls_by_origin)} origin URLs")
                
                # Process URLs in batches, grouping by origin
                for origin_url, urls_for_origin in urls_by_origin.items():
                    # Check for cancellation
                    if self.cancel_requested:
                        logger.info(f"URL processing at level {level} cancelled")
                        break
                    
                    # Find the resource this origin URL belongs to - with better matching
                    resource_for_origin = None
                    
                    # First try exact match
                    for _, row in resources_with_url.iterrows():
                        if row['url'] == origin_url:
                            resource_for_origin = row.to_dict()
                            logger.info(f"Found exact match for origin URL: {origin_url}")
                            break
                    
                    # If exact match fails, try base domain matching
                    if not resource_for_origin:
                        # Extract main domain from the origin URL (e.g., https://fabacademy.org/about/index.html -> fabacademy.org)
                        import re
                        origin_domain_match = re.search(r'https?://([^/]+)', origin_url)
                        if origin_domain_match:
                            origin_domain = origin_domain_match.group(1)
                            logger.info(f"Trying domain match for: {origin_domain}")
                            
                            # Look for resources with the same domain
                            for _, row in resources_with_url.iterrows():
                                resource_url = row['url']
                                resource_domain_match = re.search(r'https?://([^/]+)', resource_url)
                                if resource_domain_match and resource_domain_match.group(1) == origin_domain:
                                    resource_for_origin = row.to_dict()
                                    logger.info(f"Found domain match for origin: {origin_url} -> resource: {resource_url}")
                                    break
                
                if resource_for_origin:
                    logger.info(f"Processing {len(urls_for_origin)} URLs at level {level} from origin: {origin_url}")
                    origin_resource_title = resource_for_origin.get('title', 'Unnamed')
                    
                    # Process each URL using the specific_url_processor
                    for idx, pending_url_data in enumerate(urls_for_origin):
                        # Check for cancellation
                        if self.cancel_requested:
                            logger.info(f"URL processing at level {level} cancelled")
                            break
                        
                        url_to_process = pending_url_data.get('url')
                        
                        # Check if URL is already processed before calling the processor
                        if url_storage.url_is_processed(url_to_process) and not force_fetch:
                            logger.info(f"Skipping already processed URL (checking before processing): {url_to_process}")
                            # Make sure it's removed from pending list
                            url_storage.remove_pending_url(url_to_process)
                            # Update counters since we're skipping
                            processed_urls_count += 1
                            success_count += 1
                            continue
                        
                        # Process this URL
                        logger.info(f"Processing URL with force_fetch={force_fetch}: {url_to_process}")
                        url_result = single_processor.process_specific_url(
                            url=url_to_process,
                            origin_url=origin_url,
                            resource=resource_for_origin,
                            depth=level
                        )
                        
                        processed_urls_count += 1
                        
                        if url_result.get('success', False):
                            success_count += 1
                        else:
                            error_count += 1
                            logger.error(f"Error processing URL at level {level}: {url_to_process}")
                        
                        # Show progress every 10 URLs
                        if processed_urls_count % 10 == 0 or processed_urls_count == total_urls:
                            logger.info(f"Progress: {processed_urls_count}/{total_urls} URLs ({int(processed_urls_count/total_urls*100)}%)")
                else:
                    logger.warning(f"Could not find resource for origin URL: {origin_url}")
                    logger.warning(f"Looking for any matching main domain resource")
                    
                    # Try to find any resource with similar domain 
                    import re
                    main_domain = None
                    origin_domain_match = re.search(r'https?://([^/]+)', origin_url)
                    if origin_domain_match:
                        main_domain = origin_domain_match.group(1)
                    
                    # Look for any resource that belongs to this domain
                    if main_domain:
                        # Extract just the base domain (e.g., fabacademy.org from academy.fabacademy.org)
                        base_domain = re.sub(r'^[^.]+\.', '', main_domain) if '.' in main_domain else main_domain
                        logger.warning(f"Looking for resources matching base domain: {base_domain}")
                        
                        # Find any resource with this base domain
                        for _, row in resources_with_url.iterrows():
                            resource_url = row['url']
                            if base_domain in resource_url:
                                resource_for_origin = row.to_dict()
                                logger.warning(f"Found base domain match: {base_domain} in {resource_url}")
                                break
                    
                    # If still not found, use the main fabacademy resource as fallback
                    if not resource_for_origin and not resources_with_url.empty:
                        # First look for fabacademy.org as a final fallback
                        for _, row in resources_with_url.iterrows():
                            if 'fabacademy.org' in row['url']:
                                resource_for_origin = row.to_dict()
                                logger.warning(f"Using fabacademy.org resource as fallback for {origin_url}")
                                break
                        
                        # If still not found, use the first resource
                        if not resource_for_origin:
                            resource_for_origin = resources_with_url.iloc[0].to_dict()
                            logger.warning(f"Using first resource as last-resort fallback for {origin_url}")
                    
                    # Process URLs with this fallback resource
                    logger.info(f"Processing {len(urls_for_origin)} URLs using fallback resource")
                    
                    for idx, pending_url_data in enumerate(urls_for_origin):
                        # Check for cancellation
                        if self.cancel_requested:
                            logger.info(f"URL processing at level {level} cancelled")
                            break
                        
                        url_to_process = pending_url_data.get('url')
                        
                        # Check if URL is already processed before calling the processor
                        if url_storage.url_is_processed(url_to_process) and not force_fetch:
                            logger.info(f"Skipping already processed URL (checking before processing): {url_to_process}")
                            # Make sure it's removed from pending list
                            url_storage.remove_pending_url(url_to_process)
                            # Update counters since we're skipping
                            processed_urls_count += 1
                            success_count += 1
                            continue
                        
                        # Process this URL with the fallback resource
                        logger.info(f"Processing URL with fallback resource and force_fetch={force_fetch}: {url_to_process}")
                        url_result = single_processor.process_specific_url(
                            url=url_to_process,
                            origin_url=origin_url,
                            resource=resource_for_origin,  # Use resource_for_origin instead of undefined fallback_resource
                            depth=level
                        )
                        
                        processed_urls_count += 1
                        
                        if url_result.get('success', False):
                            success_count += 1
                        else:
                            error_count += 1
                            logger.error(f"Error processing URL at level {level}: {url_to_process}")
                            
                        # Show progress every 10 URLs
                        if processed_urls_count % 10 == 0 or processed_urls_count == total_urls:
                            logger.info(f"Progress: {processed_urls_count}/{total_urls} URLs ({int(processed_urls_count/total_urls*100)}%)")
                    
            logger.info(f"Level {level} processing complete: {processed_urls_count} URLs processed, {success_count} successful, {error_count} errors")
            
            return {
                "status": "success",
                "level": level,
                "total_urls": total_urls,
                "processed_urls": processed_urls_count,
                "success_count": success_count,
                "error_count": error_count
            }
            
        except Exception as e:
            logger.error(f"Error processing URLs at level {level}: {e}", exc_info=True)
            return {
                "status": "error",
                "level": level,
                "error": str(e),
                "processed_urls": processed_urls_count,
                "success_count": success_count,
                "error_count": error_count
            }
