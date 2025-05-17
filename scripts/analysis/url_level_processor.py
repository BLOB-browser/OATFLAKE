#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
URLLevelProcessor handles processing URLs at specific levels.
This is extracted from URLProcessor to make the code more modular.
"""

import logging
import os
import csv
import re
import pandas as pd
from typing import Dict, List, Any, Optional
import itertools

from scripts.analysis.single_resource_processor import SingleResourceProcessor
from scripts.analysis.url_storage import URLStorageManager
from scripts.analysis.url_discovery_manager import URLDiscoveryManager
from scripts.analysis.resource_resolver import ResourceResolver

logger = logging.getLogger(__name__)

class URLLevelProcessor:
    """
    Processes URLs at a specific depth level.
    This component is responsible for handling URL processing logic
    for a single level (1, 2, 3, or 4).
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the URL Level Processor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        self.resources_csv_path = os.path.join(data_folder, 'resources.csv')
        self.processed_urls_file = os.path.join(data_folder, "processed_urls.csv")
        self.url_storage = URLStorageManager(self.processed_urls_file)
        self.resource_resolver = ResourceResolver()
        self.discovery_manager = URLDiscoveryManager(data_folder)
        self.cancel_requested = False
        
    def request_cancellation(self):
        """Request cancellation of processing."""
        self.cancel_requested = True
        self.discovery_manager.cancel_requested = True
    
    def process_level(self, level: int, batch_size: int = 50, force_fetch: bool = False) -> Dict[str, Any]:
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
        single_processor.force_fetch = force_fetch
        
        # Get all pending URLs at this level
        pending_urls = self.url_storage.get_pending_urls(depth=level)
        logger.info(f"URLLevelProcessor: Retrieved {len(pending_urls) if pending_urls else 0} pending URLs at level {level}")
        
        # If force_fetch is enabled, we'll also check for URLs directly from pending_urls.csv
        # This handles cases where URLs are in processed_urls.csv but weren't actually analyzed
        if force_fetch and not pending_urls:
            pending_urls = self._get_raw_pending_urls(level)
            
        # If still no pending URLs, try discovery for level 1
        if not pending_urls:
            if level == 1:
                discovery_result = self.discovery_manager.discover_urls_from_resources(level=level)
                if discovery_result.get("status") == "success":
                    pending_urls = self.url_storage.get_pending_urls(depth=level)
                else:
                    return discovery_result
            else:
                return {"status": "skipped", "reason": "no_pending_urls"}
                
        # At this point, if we still have no pending URLs, there's nothing to do
        if not pending_urls:
            return {"status": "skipped", "reason": "no_pending_urls_after_discovery"}
            
        # Initialize counters
        total_urls = len(pending_urls)
        processed_urls_count = 0
        success_count = 0
        error_count = 0
        
        try:
            # Load resources
            if not os.path.exists(self.resources_csv_path):
                logger.error(f"Resources file not found: {self.resources_csv_path}")
                return {"status": "error", "error": "resources_file_not_found"}
                
            resources_df = pd.read_csv(self.resources_csv_path)
            
            # Check if we should group by resource ID or origin
            urls_to_process = self._prepare_urls_for_processing(pending_urls, resources_df)
            
            # Process grouped URLs - always process ALL URLs in each group
            for group_key, urls_group in urls_to_process.items():
                # Check for cancellation
                if self.cancel_requested:
                    logger.info(f"URL processing at level {level} cancelled")
                    break
                
                # Get resource for this group
                resource = self._get_resource_for_group(group_key, urls_group[0].get('origin', ''), resources_df)
                if not resource:
                    logger.warning(f"No resource available for group {group_key}, skipping {len(urls_group)} URLs")
                    
                    # Important: Don't mark these as processed since we're skipping due to no resource
                    error_count += len(urls_group)
                    continue
                
                # Process each URL in this group - process ALL URLs not just one per group
                for url_data in urls_group:
                    # Process URL, handling errors, and update counters
                    result = self._process_single_url(url_data, resource, level, single_processor)
                    processed_urls_count += 1
                    
                    if result == "success":
                        success_count += 1
                    elif result == "error":
                        error_count += 1
                    
                    # Show progress
                    if processed_urls_count % 10 == 0 or processed_urls_count == total_urls:
                        logger.info(f"Progress: {processed_urls_count}/{total_urls} URLs ({int(processed_urls_count/total_urls*100)}%)")
            
            # Check if there are any remaining pending URLs at this level
            remaining_urls = self.url_storage.get_pending_urls(depth=level)
            remaining_count = len(remaining_urls) if remaining_urls else 0
            
            if remaining_count > 0:
                logger.warning(f"Level {level} processing incomplete: {remaining_count} URLs still pending")
                logger.warning(f"This suggests URLs were added while processing or some URLs were not properly removed from the pending queue")
                
                # Try to process these remaining URLs if the count is reasonable
                if remaining_count <= batch_size:
                    logger.warning(f"Attempting to process {remaining_count} remaining URLs in this call")
                    
                    # Process remaining URLs as a new batch
                    more_urls_to_process = self._prepare_urls_for_processing(remaining_urls, resources_df)
                    for group_key, urls_group in more_urls_to_process.items():
                        resource = self._get_resource_for_group(group_key, urls_group[0].get('origin', ''), resources_df)
                        if not resource:
                            continue
                            
                        for url_data in urls_group:
                            result = self._process_single_url(url_data, resource, level, single_processor)
                            processed_urls_count += 1
                            
                            if result == "success":
                                success_count += 1
                            elif result == "error":
                                error_count += 1
            
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
    
    def _get_raw_pending_urls(self, level: int) -> List[Dict[str, Any]]:
        """
        Get pending URLs directly from the CSV file.
        
        Args:
            level: URL depth level to filter by
            
        Returns:
            List of URL data dictionaries
        """
        logger.info(f"No pending URLs found at level {level}, but force_fetch=True, checking raw pending_urls.csv")
        
        raw_urls = []
        pending_csv_path = os.path.join(os.path.dirname(self.processed_urls_file), "pending_urls.csv")
        
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
            
        return raw_urls
        
    def _prepare_urls_for_processing(self, pending_urls: List[Dict[str, Any]], resources_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group URLs by either resource ID or origin for more efficient processing.
        
        Args:
            pending_urls: List of pending URL data dictionaries
            resources_df: DataFrame containing resources
            
        Returns:
            Dictionary grouping URLs by resource ID or origin
        """
        # First check if we have resource IDs in the pending URLs
        has_resource_ids = any(url_data.get('resource_id') for url_data in pending_urls)
        
        urls_by_group = {}
        
        if has_resource_ids:
            # Group URLs by resource ID (simpler approach)
            for pending_url_data in pending_urls:
                url = pending_url_data.get('url')
                resource_id = pending_url_data.get('resource_id', '')
                origin = pending_url_data.get('origin', '')
                
                # If no resource ID, use origin as fallback
                group_key = resource_id or origin
                
                if group_key not in urls_by_group:
                    urls_by_group[group_key] = []
                
                urls_by_group[group_key].append(pending_url_data)
                
            logger.info(f"Processing URLs grouped by resource ID")
        else:
            # Group by origin
            for pending_url_data in pending_urls:
                url = pending_url_data.get('url')
                origin = pending_url_data.get('origin', '')
                
                if origin not in urls_by_group:
                    urls_by_group[origin] = []
                
                urls_by_group[origin].append(pending_url_data)
                
            logger.info(f"Processing URLs grouped by origin URL")
            
        return urls_by_group
        
    def _get_resource_for_group(self, group_key: str, origin_url: str, resources_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Find the appropriate resource for a group of URLs.
        
        Args:
            group_key: The group key (resource ID or origin URL)
            origin_url: Fallback origin URL if group key is not a resource ID
            resources_df: DataFrame containing resources
            
        Returns:
            Resource dictionary or None
        """
        resources_with_url = resources_df[resources_df['url'].notna()]
        
        # First try to find resource by title match (for resource IDs)
        for _, row in resources_with_url.iterrows():
            title = row.get('title', '').strip()
            if title == group_key:
                resource = row.to_dict()
                logger.info(f"Found resource for ID: {group_key}")
                return resource
                
        # If not found by ID, try URL matching
        return self.resource_resolver.resolve_resource_for_url(group_key, origin_url, resources_df)
        
    def _process_single_url(self, url_data: Dict[str, Any], resource: Dict[str, Any], level: int, processor: SingleResourceProcessor) -> str:
        """
        Process a single URL, handling errors appropriately.
        
        Args:
            url_data: URL data dictionary
            resource: Resource dictionary
            level: URL depth level
            processor: SingleResourceProcessor instance
            
        Returns:
            Status string: "success", "error", or "skipped"
        """
        url_to_process = url_data.get('url')
        origin_url = url_data.get('origin', '')
        
        # Check if URL is already processed before calling the processor
        if self.url_storage.url_is_processed(url_to_process) and not processor.force_fetch:
            logger.info(f"Skipping already processed URL: {url_to_process}")
            # Make sure it's removed from pending list
            self.url_storage.remove_pending_url(url_to_process)
            return "skipped"
        
        # Process this URL
        logger.info(f"Processing URL: {url_to_process}")
        try:
            url_result = processor.process_specific_url(
                url=url_to_process,
                origin_url=origin_url,
                resource=resource,
                depth=level
            )
            
            if url_result.get('success', False):
                return "success"
            else:
                logger.error(f"Error processing URL at level {level}: {url_to_process}")
                # CRITICAL: Always remove failed URLs from pending queue
                try:
                    self.url_storage.remove_pending_url(url_to_process)
                    logger.info(f"Removed failed URL from pending queue: {url_to_process}")
                    
                    # Mark as processed to avoid future attempts
                    self.url_storage.save_processed_url(url_to_process, depth=level, origin=origin_url, error=True)
                    logger.info(f"Marked failed URL as processed with error flag: {url_to_process}")
                except Exception as rm_error:
                    logger.error(f"Error handling failed URL removal: {rm_error}")
                
                return "error"
        except Exception as url_error:
            logger.error(f"Exception processing URL at level {level}: {url_to_process} - Error: {url_error}")
            # Make sure we remove this URL from the pending list to avoid getting stuck
            try:
                self.url_storage.remove_pending_url(url_to_process)
                logger.info(f"Removed problematic URL from pending queue: {url_to_process}")
                
                # Also mark as processed with error flag to ensure we don't try again
                self.url_storage.save_processed_url(url_to_process, depth=level, origin=origin_url, error=True)
                logger.info(f"Marked exception URL as processed with error flag: {url_to_process}")
            except Exception as rm_error:
                logger.error(f"Error removing URL from pending queue: {rm_error}")
            return "error"
