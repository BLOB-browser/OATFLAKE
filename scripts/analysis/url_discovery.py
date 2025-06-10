#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def discover_urls_for_level(
    data_folder: str, 
    target_level: int, 
    url_storage=None, 
    max_discovery_per_level: int = 50
) -> Dict[str, Any]:
    """
    Discovers URLs for a specific target level by processing original resources.
    
    Args:
        data_folder: Path to the data directory
        target_level: The level to discover URLs for (2, 3, or 4)
        url_storage: Optional URLStorageManager instance
        max_discovery_per_level: Maximum number of sources to use for discovery
        
    Returns:
        Dictionary with discovery statistics
    """
    # Import necessary components
    from scripts.analysis.web_fetcher import WebFetcher
    from scripts.analysis.url_processor import (
        extract_navigation_links, extract_project_links, extract_detail_links, 
        process_links, get_base_domain_and_url
    )
    
    # Initialize URL storage if not provided
    if url_storage is None:
        from scripts.analysis.url_storage import URLStorageManager
        processed_urls_file = os.path.join(data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
    
    # Enable discovery mode for improved performance during URL discovery
    logger.info(f"Enabling discovery mode for level {target_level} URL discovery")
    url_storage.set_discovery_mode(True)
    
    # Initialize web fetcher
    web_fetcher = WebFetcher()
    
    # Get resources CSV path
    resources_csv_path = os.path.join(data_folder, 'resources.csv')
    
    # Statistics to track
    stats = {
        "target_level": target_level,
        "sources_processed": 0,
        "urls_discovered": 0,
        "successful_sources": 0,
        "failed_sources": 0
    }
    
    if not os.path.exists(resources_csv_path):
        logger.error(f"Resources file not found: {resources_csv_path}")
        return stats
    
    try:
        # Load resources from CSV to get original URLs
        resources_df = pd.read_csv(resources_csv_path)
        if resources_df.empty:
            logger.warning(f"No resources found in {resources_csv_path}")
            return stats
        
        # Get only resources with URLs        # Check for either origin_url or legacy url field
        if 'origin_url' in resources_df.columns:
            resources_with_url = resources_df[resources_df['origin_url'].notna()]
            url_field = 'origin_url'
        else:
            # Use origin_url instead of url for compatibility with universal schema
            logger.error("No origin_url field found in resources.csv. Please use universal schema.")
            return stats
        
        # Limit the number of resources to process
        max_sources = min(len(resources_with_url), max_discovery_per_level)
        logger.info(f"Using up to {max_sources} sources to discover level {target_level} URLs")        # Process each original resource URL
        for _, row in resources_with_url.iterrows():
            # Limit the number of sources to process
            if stats["sources_processed"] >= max_discovery_per_level:
                break
                
            origin_url = row.get(url_field)
            resource_id = str(row.get('title', ''))  # Get resource ID from title
            if not origin_url or not isinstance(origin_url, str) or not origin_url.startswith('http'):
                continue
                
            # Ensure origin_url is stored properly for universal schema compatibility
            if 'origin_url' not in row and url_field == 'url':
                row['origin_url'] = origin_url
            
            stats["sources_processed"] += 1
            
            # Create headers for the request
            headers = web_fetcher.create_headers()
            
            try:
                logger.info(f"Fetching source for level {target_level} discovery: {origin_url}")
                
                # Get base domain for resolving relative URLs
                base_domain, base_path, base_url = get_base_domain_and_url(origin_url)
                
                # Fetch the HTML content even if already processed
                success, html_content = web_fetcher.fetch_page(origin_url, headers)
                
                if success and html_content:
                    # Parse into BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract links for the appropriate level
                    if target_level == 2:
                        links = extract_navigation_links(soup)
                        logger.info(f"Extracting navigation links for level 2 from {origin_url}")
                    elif target_level == 3:
                        links = extract_project_links(soup)
                        logger.info(f"Extracting project links for level 3 from {origin_url}")
                    else:
                        links = extract_detail_links(soup)
                        logger.info(f"Extracting detail links for level 4+ from {origin_url}")
                    
                    # Process links to filter and resolve URLs
                    visited_urls = set([origin_url])  # Start with the current URL as visited
                    found_urls = process_links(links, base_domain, origin_url, visited_urls)                    # Save URLs to pending using enhanced storage for the target level
                    added_count = 0
                    for found_url in found_urls:                        # Only save URLs with valid resource IDs
                        if resource_id:
                            url_storage.save_pending_url(found_url, depth=target_level, origin=origin_url, resource_id=resource_id)
                            added_count += 1
                        else:
                            logger.warning(f"Skipping URL without resource ID: {found_url}")
                    
                    stats["urls_discovered"] += added_count
                    stats["successful_sources"] += 1
                    logger.info(f"Added {added_count} new URLs for level {target_level} from {origin_url}")
                else:
                    logger.error(f"Failed to fetch content from {origin_url} for level {target_level}")
                    stats["failed_sources"] += 1
            except Exception as e:
                logger.error(f"Error discovering URLs for level {target_level} from {origin_url}: {e}")
                stats["failed_sources"] += 1
    
    finally:
        # Disable discovery mode and cleanup processed URLs before analysis
        logger.info(f"Disabling discovery mode and cleaning up processed URLs for level {target_level}")
        url_storage.set_discovery_mode(False)
        url_storage.cleanup_processed_urls_from_pending()
    
    # Return statistics
    return stats

# Standalone function for easy import
def discover_next_level_urls(data_folder, current_level, max_depth=4, url_storage=None):
    """
    Discover URLs for the next level based on the current level.
    
    Args:
        data_folder: Path to the data directory
        current_level: The current level (1, 2, or 3)
        max_depth: Maximum depth to discover (default 4)
        url_storage: Optional URLStorageManager instance
    
    Returns:
        Dictionary with discovery statistics
    """
    next_level = current_level + 1
    if next_level > max_depth:
        logger.info(f"Already at maximum depth {max_depth}")
        return {"status": "skipped", "reason": "max_depth_reached"}
    
    # Perform discovery for next level with enhanced storage
    results = discover_urls_for_level(data_folder, next_level, url_storage)
    
    # Return discovery results
    return {
        "status": "completed",
        "target_level": next_level,
        "urls_discovered": results.get("urls_discovered", 0),
        "sources_processed": results.get("sources_processed", 0)
    }
