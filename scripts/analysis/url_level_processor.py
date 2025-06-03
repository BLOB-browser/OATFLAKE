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

from scripts.analysis.single_resource_processor_universal import SingleResourceProcessorUniversal
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
        self.single_processor = SingleResourceProcessorUniversal(data_folder)
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
        
        # Import necessary components
        from scripts.analysis.single_resource_processor_universal import SingleResourceProcessorUniversal
        self.single_processor = SingleResourceProcessorUniversal(self.data_folder)
        
        # Get pending URLs at this level
        pending_urls = self.url_storage.get_pending_urls(depth=level)
        if not pending_urls:
            logger.info(f"No pending URLs at level {level}")
            
            if force_fetch:
                # Try getting raw URLs from file as a backup
                pending_urls = self._get_raw_pending_urls(level)
                logger.info(f"Found {len(pending_urls)} raw pending URLs at level {level} for force processing")
            
            if not pending_urls:
                return {
                    "level": level,
                    "processed_urls": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "resources_created": 0
                }

        # Track processed URLs and results
        processed = set()
        processed_count = 0
        success_count = 0 
        error_count = 0
        resources_created = 0

        # Process URLs in batches
        for start_idx in range(0, len(pending_urls), batch_size):
            batch = pending_urls[start_idx:start_idx + batch_size]
            logger.info(f"Processing batch {start_idx//batch_size + 1} with {len(batch)} URLs")
            
            for url_data in batch:
                url = url_data["url"]
                origin = url_data["origin"]
                depth = url_data["depth"]
                
                # Skip if already processed in this run
                if url in processed:
                    continue
                    
                # Skip URLs above our current level
                if depth > level:
                    continue

                # Process the URL
                try:
                    result = self.single_processor.process_single_url(
                        url=url,
                        origin=origin,
                        depth=depth
                    )
                    
                    if result.get("success", False):
                        success_count += 1
                        resources_created += result.get("resources_created", 0)
                        # Mark as processed on success
                        self.url_storage.save_processed_url(url, depth=depth, origin=origin)
                        self.url_storage.remove_pending_url(url)
                    else:
                        error_count += 1
                        # Mark errors after too many attempts
                        if url_data.get("attempt_count", 0) >= 3:
                            logger.warning(f"URL {url} had {url_data['attempt_count']} failed attempts, marking as error")
                            self.url_storage.save_processed_url(url, depth=depth, origin=origin, error=True)
                            self.url_storage.remove_pending_url(url)
                        else:
                            # Increment attempt count and keep in pending
                            self.url_storage.increment_url_attempt(url)
                            logger.info(f"URL {url} failed, attempt #{url_data.get('attempt_count', 0) + 1}")
                    
                    processed.add(url)
                    processed_count += 1

                except Exception as e:
                    logger.error(f"Error processing URL {url}: {e}")
                    error_count += 1
                    # Increment attempt count
                    self.url_storage.increment_url_attempt(url)

        logger.info(f"Level {level} processing complete: {processed_count} URLs processed ({success_count} success, {error_count} errors)")

        return {
            "level": level,
            "processed_urls": processed_count,
            "success_count": success_count,
            "error_count": error_count,
            "resources_created": resources_created
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
        # Check for origin_url field first (new schema), then url field (legacy schema)
        if 'origin_url' in resources_df.columns:
            resources_with_url = resources_df[resources_df['origin_url'].notna()]
        else:
            resources_with_url = resources_df[resources_df['url'].notna()]
          # First try to find resource by title match (for resource IDs)
        for _, row in resources_with_url.iterrows():
            title = row.get('title', '').strip()
            if title == group_key:
                resource = row.to_dict()
                # Make sure the resource has origin_url field (new schema format)
                if 'origin_url' not in resource and 'url' in resource:
                    resource['origin_url'] = resource['url']
                logger.info(f"Found resource for ID: {group_key}")
                return resource
                
        # If not found by ID, try URL matching
        return self.resource_resolver.resolve_resource_for_url(group_key, origin_url, resources_df)

    def _process_single_url(self, url_data: Dict[str, Any], resource: Dict[str, Any], level: int, processor: SingleResourceProcessorUniversal) -> str:
        """
        Process a single URL, handling errors appropriately.
        
        Args:
            url_data: URL data dictionary
            resource: Resource dictionary
            level: URL depth level
            processor: SingleResourceProcessorUniversal instance
            
        Returns:
            Status string: "success", "error", or "skipped"
        """
        url_to_process = url_data.get('url')
        # Use 'origin' field as origin_url (consistent with universal schema)
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
            # Ensure resource has origin_url field (universal schema)
            if resource and 'origin_url' not in resource and 'url' in resource:
                resource['origin_url'] = resource['url']
            
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
