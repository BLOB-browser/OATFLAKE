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
    max_discovery_per_level: int = 20
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
        
        # Get only resources with URLs
        resources_with_url = resources_df[resources_df['url'].notna()]
        
        # Limit the number of resources to process
        max_sources = min(len(resources_with_url), max_discovery_per_level)
        logger.info(f"Using up to {max_sources} sources to discover level {target_level} URLs")
        
        # Process each original resource URL
        for _, row in resources_with_url.iterrows():
            stats["sources_processed"] += 1
            
            # Limit the number of sources to process
            if stats["sources_processed"] > max_discovery_per_level:
                break
                
            url = row.get('url')
            if not url or not isinstance(url, str) or not url.startswith('http'):
                continue
            
            # Create headers for the request
            headers = web_fetcher.create_headers()
            
            try:
                logger.info(f"Fetching source for level {target_level} discovery: {url}")
                
                # Get base domain for resolving relative URLs
                base_domain, base_path, base_url = get_base_domain_and_url(url)
                
                # Fetch the HTML content even if already processed
                success, html_content = web_fetcher.fetch_page(url, headers)
                
                if success and html_content:
                    # Parse into BeautifulSoup
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract links for the appropriate level
                    if target_level == 2:
                        links = extract_navigation_links(soup)
                        logger.info(f"Extracting navigation links for level 2 from {url}")
                    elif target_level == 3:
                        links = extract_project_links(soup)
                        logger.info(f"Extracting project links for level 3 from {url}")
                    else:
                        links = extract_detail_links(soup)
                        logger.info(f"Extracting detail links for level 4+ from {url}")
                    
                    # Process links to filter and resolve URLs
                    visited_urls = set([url])  # Start with the current URL as visited
                    found_urls = process_links(links, base_domain, url, visited_urls)
                    
                    # Save URLs to pending for the target level
                    added_count = 0
                    for found_url in found_urls:
                        # MODIFIED: Remove check for already processed URLs
                        # Only check if it's already in pending URLs to avoid duplicates
                        pending_urls = url_storage.get_pending_urls(depth=target_level)
                        if not any(p.get('url') == found_url for p in pending_urls):
                            url_storage.save_pending_url(found_url, depth=target_level, origin=url)
                            added_count += 1
                    
                    stats["urls_discovered"] += added_count
                    stats["successful_sources"] += 1
                    logger.info(f"Added {added_count} new URLs for level {target_level} from {url}")
                else:
                    logger.error(f"Failed to fetch content from {url} for level {target_level}")
                    stats["failed_sources"] += 1
            except Exception as e:
                logger.error(f"Error discovering URLs for level {target_level} from {url}: {e}")
                stats["failed_sources"] += 1
    
    except Exception as e:
        logger.error(f"Error reading resources.csv for level {target_level} URLs: {e}")
    
    # Return statistics
    return stats

# Standalone function for easy import
def discover_next_level_urls(data_folder, current_level, max_depth=4):
    """
    Discover URLs for the next level based on the current level.
    
    Args:
        data_folder: Path to the data directory
        current_level: The current level (1, 2, or 3)
        max_depth: Maximum depth to discover (default 4)
    
    Returns:
        Dictionary with discovery statistics
    """
    next_level = current_level + 1
    if next_level > max_depth:
        logger.info(f"Already at maximum depth {max_depth}")
        return {"status": "skipped", "reason": "max_depth_reached"}
    
    # Perform discovery for next level
    results = discover_urls_for_level(data_folder, next_level)
    
    # Return discovery results
    return {
        "status": "completed",
        "target_level": next_level,
        "urls_discovered": results.get("urls_discovered", 0),
        "sources_processed": results.get("sources_processed", 0)
    }
