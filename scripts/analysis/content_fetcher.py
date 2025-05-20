#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import csv
import requests
import platform
from typing import Tuple, Dict, Any, List, Set
from bs4 import BeautifulSoup
from pathlib import Path
from utils.config import get_data_path

# Import our refactored modules
from scripts.analysis.url_processor import (
    clean_url, get_base_domain_and_url, resolve_absolute_url,
    extract_navigation_links, extract_project_links, extract_detail_links,
    process_links
)
from scripts.analysis.html_extractor import extract_text, extract_page_texts
from scripts.analysis.url_storage import URLStorageManager
from scripts.analysis.web_fetcher import WebFetcher

logger = logging.getLogger(__name__)

class ContentFetcher:
    """Responsible for fetching and extracting content from websites"""
    
    def __init__(self, timeout: int = 90):
        self.timeout = timeout  # Increased from 30 to 90 seconds for Raspberry Pi
        # Add size limits matching ResourceLLM
        self.size_limits = {
            'description': 4000,    # For description generation
            'tags': 2500,          # For tag generation
            'definitions': 4000,    # For definition extraction
            'projects': 4000,      # For project identification
            'default': 2000        # Default limit
        }
        # Add SSL verification flag - useful for Raspberry Pi which might have issues
        self.verify_ssl = False  # Disable SSL verification for compatibility
        
        # Track URL count for incremental vector store generation
        self.urls_processed_since_last_build = 0
        self.url_build_threshold = 10  # Build vector store after every 10 URLs
        
        # Add platform-specific User-Agents with updated Chrome version
        self.user_agents = {
            'windows': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'mac': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'raspberry': 'Mozilla/5.0 (Linux; Android 13; Raspberry Pi 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
        }
        
        # Detect platform and set appropriate User-Agent
        self.system = platform.system().lower()
        
        # Set User-Agent based on platform
        if 'darwin' in self.system:
            self.user_agent = self.user_agents['mac']
        elif 'linux' in self.system:
            self.user_agent = self.user_agents['raspberry']
        else:
            self.user_agent = self.user_agents['windows']
            
        logger.info(f"ContentFetcher initialized for {self.system} platform")
        logger.info(f"ContentFetcher using User-Agent for: {self.user_agent[:30]}...")
        logger.info(f"ContentFetcher initialized with size limits: {self.size_limits}")
        
        # Initialize web fetcher
        self.web_fetcher = WebFetcher(
            user_agent=self.user_agent,
            timeout=self.timeout,
            verify_ssl=self.verify_ssl
        )
        
        # Get data path from config and create directory if needed
        data_path = get_data_path()
        os.makedirs(data_path, exist_ok=True)
        
        # Initialize URL storage manager
        processed_urls_file = os.path.join(data_path, "processed_urls.csv")
        self.url_storage = URLStorageManager(processed_urls_file)
        logger.info(f"Using processed URLs file at: {processed_urls_file}")

        # Add flag to control whether to analyze immediately or defer analysis
        self.defer_analysis = False
        self.discovery_only_mode = False  # When True, focus on URL discovery only
        self.allow_processed_url_discovery = False  # When True, allow discovery from processed URLs when no pending URLs
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate if a URL is potentially valid before adding it to the pending queue.
        Filters out obviously malformed URLs and checks for problematic patterns.
        
        Args:
            url: The URL to validate
            
        Returns:
            Boolean indicating if the URL appears valid
        """
        # Skip empty URLs
        if not url:
            return False
            
        # Skip URLs with file:/// protocol
        if "file:///" in url:
            logger.warning(f"Skipping URL with file:/// protocol: {url}")
            return False
            
        # Skip URLs that mix protocols (http://...file:///)
        if url.count("://") > 1:
            logger.warning(f"Skipping malformed URL with multiple protocols: {url}")
            return False
            
        # Make sure URL starts with http:// or https://
        if not url.startswith(("http://", "https://")):
            logger.warning(f"Skipping URL with invalid protocol: {url}")
            return False
            
        # Check URL length - too long URLs are often malformed
        if len(url) > 500:
            logger.warning(f"Skipping URL that is too long ({len(url)} characters): {url[:100]}...")
            return False
            
        # Check for common problematic patterns
        problematic_patterns = [
            "localhost", 
            "127.0.0.1",
            ".git",
            "undefined",
            "javascript:",
            "mailto:",
            "tel:",
            "data:",
            "about:"
        ]
        
        for pattern in problematic_patterns:
            if pattern in url:
                logger.warning(f"Skipping URL with problematic pattern '{pattern}': {url}")
                return False
                
        return True
    
    def get_main_page_with_links(self, url: str, max_depth: int = 4, go_deeper: bool = None, breadth_first: bool = True, 
                               discover_only_level: int = 4, resource_id: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieves the main page content and identifies links to additional pages.
        Can go multiple layers deep to find more content-rich pages.
        
        Args:
            url: The URL to fetch
            max_depth: Maximum depth level to crawl (1=just main page links, 2=two levels, 3=three levels)
            go_deeper: (Deprecated) If True, will look for links two levels deep (use max_depth instead)
            breadth_first: If True, will discover all URLs in breadth-first order (completing all level 1 URLs before level 2)
            discover_only_level: Maximum level to discover URLs (default 4). URLs beyond this level will not be discovered.
            resource_id: Optional resource ID to associate with discovered URLs
            
        Returns:
            Tuple of (success_flag, dict with main_html and additional_urls)
        """
        # Handle backward compatibility with go_deeper parameter
        if go_deeper is not None:
            logger.warning("Parameter 'go_deeper' is deprecated, use 'max_depth' instead")
            if go_deeper is False:
                max_depth = 1  # Only get main page links
            else:
                max_depth = 2  # Go two levels deep (original behavior)
        
        try:
            # Clean URL, handling trailing slashes and common issues
            url = clean_url(url)
                
            # Special handling for discovery_only mode
            if self.discovery_only_mode:
                # In discovery-only mode, ALWAYS proceed with URL discovery regardless of whether the URL is processed
                # We want to find new URLs at deeper levels, not reanalyze content
                if self.url_storage.url_is_processed(url):
                    logger.info(f"URL {url} already processed, but continuing with discovery to find deeper URLs")
                # Continue with the discovery process
            else:
                # Normal analysis mode - check if URL is already processed
                if self.url_storage.url_is_processed(url):
                    logger.info(f"Main URL {url} already processed, skipping fetch")
                    return False, {"error": "URL already processed"}
            
            # Store the base domain for building absolute URLs
            base_domain, base_path, base_url = get_base_domain_and_url(url)
            
            # Create headers using WebFetcher
            headers = self.web_fetcher.create_headers()
            
            # Get the main page using WebFetcher
            success, main_html = self.web_fetcher.fetch_page(url, headers)
            if not success:
                return False, {"error": "Failed to fetch main page"}
            
            # Parse the page to find important links
            soup = BeautifulSoup(main_html, 'html.parser')
            
            # Track visited URLs to avoid duplicates
            visited_urls = {url}
            
            # Initialize a dictionary to store URLs by level
            urls_by_level = {level: [] for level in range(1, max_depth + 1)}
            
            # IMPORTANT: Determine discovery depth based on mode
            if self.discovery_only_mode:
                # In discovery-only mode, ALWAYS use the full max_depth (4) to ensure complete discovery
                # regardless of any other parameters
                discovery_depth = 4  # Always go to 4 levels deep in discovery
                logger.info(f"Discovery-only mode: Forcing discovery to full depth of 4 levels")
            else:
                # In analysis mode, respect the original limits
                discovery_depth = min(max_depth, discover_only_level)
                logger.info(f"Analysis mode: URL discovery set to level {discovery_depth} (max_depth={max_depth}, discover_only_level={discover_only_level})")
            
            # Load processed URLs to avoid re-fetching content we've already analyzed
            processed_urls = self.url_storage.get_processed_urls()
            
            # Always use the simplified breadth-first crawling
            logger.info(f"Using simplified breadth-first discovery directly from the resource")
            if resource_id:
                logger.info(f"Resource ID provided for URL association: {resource_id}")
            
            # Use our new simplified breadth-first discovery that doesn't use pending URLs
            self._discover_urls_breadth_first(
                root_soup=soup,
                root_url=url,
                base_domain=base_domain,
                headers=headers,
                max_depth=discovery_depth,  # Use our calculated discovery depth
                visited_urls=visited_urls,
                urls_by_level=urls_by_level,
                resource_id=resource_id  # Pass the resource ID for tracking
            )
            
            # Create an ordered list of all URLs, sorted by level
            # This ensures we process level 1 URLs before level 2, etc.
            all_additional_urls = []
            for level in range(1, max_depth + 1):
                if level in urls_by_level:
                    level_urls = urls_by_level[level]
                    logger.info(f"Found {len(level_urls)} unprocessed URLs at level {level}")
                    all_additional_urls.extend(level_urls)
                    
            # Return the main page HTML and the list of all additional URLs
            return True, {
                "main_html": main_html,
                "additional_urls": all_additional_urls,
                "base_url": base_url,
                "base_domain": base_domain,
                "headers": headers,
                "visited_urls": visited_urls,
                # Include level information compatible with the old format
                "level1_urls": urls_by_level.get(1, []),
                "level2_urls": urls_by_level.get(2, []),
                "level3_urls": urls_by_level.get(3, []),
                # Add the full dictionary for more levels if needed
                "urls_by_level": urls_by_level,
                # Add flag to indicate this was processed in breadth-first mode
                "breadth_first": breadth_first
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching main page {url}: {e}")
            return False, {"error": str(e)}
            
    def _crawl_recursive(self, soup, base_url, base_domain, headers, current_level, max_depth, visited_urls, urls_by_level, force_reprocess=False):
        """
        Recursively crawls pages up to a specified depth, collecting unprocessed URLs.
        This uses a depth-first approach (not recommended for level-by-level processing).
        
        Args:
            soup: BeautifulSoup object of the current page
            base_url: URL of the current page
            base_domain: Base domain for resolving relative URLs
            headers: HTTP headers for requests
            current_level: Current depth level (1-based)
            max_depth: Maximum depth to crawl
            visited_urls: Set of already visited URLs
            urls_by_level: Dictionary to store URLs by their depth level
            force_reprocess: If True, include URLs even if they've already been processed
        """
        if current_level > max_depth:
            return
            
        # Extract links appropriate for the current level
        if current_level == 1:
            # For the first level, get navigation links
            links = extract_navigation_links(soup)
            logger.info(f"Extracting navigation links at level {current_level}")
        elif current_level == 2:
            # For the second level, get project links
            links = extract_project_links(soup)
            logger.info(f"Extracting project links at level {current_level}")
        else:
            # For deeper levels, get detail links
            links = extract_detail_links(soup)
            logger.info(f"Extracting detail links at level {current_level}")
            
        # Process the found links
        found_urls = process_links(links, base_domain, base_url, visited_urls)
        
        # Filter out already processed URLs (unless force_reprocess is True)
        filtered_urls = []
        for url in found_urls:
            if self.url_storage.url_is_processed(url) and not force_reprocess:
                logger.info(f"URL {url} already processed, filtering out from level {current_level}")
            elif self.url_storage.url_is_processed(url) and force_reprocess:
                logger.info(f"URL {url} already processed but force_reprocess is True, including at level {current_level}")
                filtered_urls.append(url)
            else:
                filtered_urls.append(url)
                
        # Store the filtered URLs at this level
        urls_by_level[current_level].extend(filtered_urls)
        logger.info(f"Found {len(filtered_urls)} unprocessed URLs at level {current_level}")
        
        # If we haven't reached max depth, fetch each URL and continue crawling
        if current_level < max_depth and filtered_urls:
            # Limit the number of URLs to check at deeper levels to avoid excessive crawling
            max_urls_to_check = min(len(filtered_urls), 20)
            urls_to_check = filtered_urls[:max_urls_to_check]
            
            logger.info(f"Going deeper to level {current_level + 1}: examining {len(urls_to_check)} pages from level {current_level}")
            
            for next_url in urls_to_check:
                try:
                    # Mark as visited to avoid re-processing
                    visited_urls.add(next_url)
                    
                    # Fetch the page
                    success, html_content = self.web_fetcher.fetch_page(
                        next_url,
                        headers,
                        custom_timeout=60  # Increased timeout for deeper pages
                    )
                    
                    if not success:
                        continue
                        
                    # Parse the HTML
                    next_soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Recursively crawl this page
                    self._crawl_recursive(
                        soup=next_soup,
                        base_url=next_url,
                        base_domain=base_domain,
                        headers=headers,
                        current_level=current_level + 1,
                        max_depth=max_depth,
                        visited_urls=visited_urls,
                        urls_by_level=urls_by_level,
                        force_reprocess=force_reprocess  # Pass the force_reprocess flag
                    )
                    
                except Exception as e:
                    logger.warning(f"Error processing URL {next_url} at level {current_level}: {e}")
                    
    def _discover_urls_breadth_first(self, root_soup, root_url, base_domain, headers, max_depth=4, visited_urls=None, urls_by_level=None, force_reprocess=False, resource_id=None):
        """
        Discovers URLs in a breadth-first manner, directly from the initial resource only.
        No recursive discovery from pending or processed URLs.
        
        Args:
            root_soup: BeautifulSoup object of the root/main page
            root_url: URL of the root/main page
            base_domain: Base domain for resolving relative URLs
            headers: HTTP headers for requests
            max_depth: Maximum depth to crawl
            visited_urls: Set of already visited URLs
            urls_by_level: Dictionary to store URLs by their depth level
            force_reprocess: If True, include URLs even if they've already been processed
            resource_id: Optional ID of the resource for tracking association
        """
        # Initialize collections if they're None
        if visited_urls is None:
            visited_urls = set()
        if urls_by_level is None:
            urls_by_level = {}
        
        # Make sure all level keys exist in the urls_by_level dictionary
        for level in range(1, max_depth + 1):
            if level not in urls_by_level:
                urls_by_level[level] = []
        
        # For level 1, extract links directly from the root page
        logger.info(f"Processing URL discovery at level 1 (direct from resource)")
        
        # Get navigation links from the root page
        links = extract_navigation_links(root_soup)
        logger.info(f"Extracting navigation links from root URL: {root_url}")
        found_urls = process_links(links, base_domain, root_url, visited_urls)
        filtered_urls = []
        
        # Store level 1 URLs and mark them as originated from the root resource
        for url in found_urls:
            # Check if we should include this URL
            should_include = True  # Always include URLs for discovery regardless of processed status
            
            # Skip URLs we've already visited in this session to avoid duplicates
            if url in visited_urls:
                should_include = False
                
            # Validate the URL to filter out problematic ones
            if should_include and not self._is_valid_url(url):
                should_include = False
                
            # Log inclusion of already processed URLs
            if should_include and self.url_storage.url_is_processed(url):
                logger.info(f"Including already processed URL in discovery: {url}")
            
            if should_include:
                filtered_urls.append(url)
                visited_urls.add(url)
                
                # Save to pending URLs with resource_id if provided
                if resource_id:
                    self.url_storage.save_pending_url(url, depth=1, origin=root_url, resource_id=resource_id)
                    self.url_storage.save_resource_url(resource_id, url, depth=1)
                    logger.info(f"URL discovered at level 1: {url} (associated with resource {resource_id})")
                else:
                    self.url_storage.save_pending_url(url, depth=1, origin=root_url)
                    logger.info(f"URL discovered at level 1: {url}")
                    
        # Store the filtered URLs at level 1
        urls_by_level[1].extend(filtered_urls)
        logger.info(f"Discovered {len(filtered_urls)} new URLs at level 1")
        
        # Fetch each level 1 URL to discover level 2 URLs (if needed)
        if max_depth >= 2 and filtered_urls:
            # Limit the number of URLs to check for level 2
            max_urls_to_check = min(len(filtered_urls), 20)
            urls_to_check = filtered_urls[:max_urls_to_check]
            
            logger.info(f"Fetching {len(urls_to_check)} level 1 URLs to discover level 2 links")
            level_2_urls = []
            
            # For each level 1 URL, fetch the page and extract level 2 links
            for level_1_url in urls_to_check:
                try:
                    logger.info(f"Fetching level 1 URL to discover level 2 links: {level_1_url}")
                    success, html_content = self.web_fetcher.fetch_page(level_1_url, headers)
                    
                    if success and html_content:
                        level_1_soup = BeautifulSoup(html_content, 'html.parser')
                        
                        # Extract project links (level 2)
                        level_2_links = extract_project_links(level_1_soup)
                        level_2_found_urls = process_links(level_2_links, base_domain, level_1_url, visited_urls)
                        
                        for url in level_2_found_urls:
                            # Check if we should include this URL
                            should_include = True  # Always include URLs for discovery regardless of processed status
                            
                            # Skip URLs we've already visited in this session to avoid duplicates
                            if url in visited_urls:
                                should_include = False
                            
                            # Validate the URL to filter out problematic ones
                            if should_include and not self._is_valid_url(url):
                                should_include = False
                                
                            # Log inclusion of already processed URLs
                            if should_include and self.url_storage.url_is_processed(url):
                                logger.info(f"Including already processed URL at level 2: {url}")
                            
                            if should_include:
                                level_2_urls.append(url)
                                visited_urls.add(url)
                                
                                # Save to pending URLs with resource_id if provided
                                if resource_id:
                                    self.url_storage.save_pending_url(url, depth=2, origin=level_1_url, resource_id=resource_id)
                                    self.url_storage.save_resource_url(resource_id, url, depth=2)
                                    logger.info(f"URL discovered at level 2: {url} (associated with resource {resource_id})")
                                else:
                                    self.url_storage.save_pending_url(url, depth=2, origin=level_1_url)
                                    logger.info(f"URL discovered at level 2: {url}")
                                
                except Exception as e:
                    logger.warning(f"Error processing level 1 URL {level_1_url}: {e}")
                    
            # Store level 2 URLs
            urls_by_level[2].extend(level_2_urls)
            logger.info(f"Discovered {len(level_2_urls)} new URLs at level 2")
            
            # If needed, process level 2 URLs to find level 3 URLs
            if max_depth >= 3 and level_2_urls:
                # Limit the number of level 2 URLs to check
                max_urls_to_check = min(len(level_2_urls), 15)
                urls_to_check = level_2_urls[:max_urls_to_check]
                
                logger.info(f"Fetching {len(urls_to_check)} level 2 URLs to discover level 3 links")
                level_3_urls = []
                
                # Process each level 2 URL to find level 3 links
                for level_2_url in urls_to_check:
                    try:
                        logger.info(f"Fetching level 2 URL to discover level 3 links: {level_2_url}")
                        success, html_content = self.web_fetcher.fetch_page(level_2_url, headers)
                        
                        if success and html_content:
                            level_2_soup = BeautifulSoup(html_content, 'html.parser')
                            
                            # Extract detail links (level 3)
                            level_3_links = extract_detail_links(level_2_soup)
                            level_3_found_urls = process_links(level_3_links, base_domain, level_2_url, visited_urls)
                            
                            for url in level_3_found_urls:
                                # Check if we should include this URL
                                should_include = True  # Always include URLs for discovery regardless of processed status
                                
                                # Skip URLs we've already visited in this session to avoid duplicates
                                if url in visited_urls:
                                    should_include = False
                                
                                # Validate the URL to filter out problematic ones
                                if should_include and not self._is_valid_url(url):
                                    should_include = False
                                
                                # Log inclusion of already processed URLs
                                if should_include and self.url_storage.url_is_processed(url):
                                    logger.info(f"Including already processed URL at level 3: {url}")
                                
                                if should_include:
                                    level_3_urls.append(url)
                                    visited_urls.add(url)
                                    
                                    # Save to pending URLs with resource_id if provided
                                    if resource_id:
                                        self.url_storage.save_pending_url(url, depth=3, origin=level_2_url, resource_id=resource_id)
                                        self.url_storage.save_resource_url(resource_id, url, depth=3)
                                        logger.info(f"URL discovered at level 3: {url} (associated with resource {resource_id})")
                                    else:
                                        self.url_storage.save_pending_url(url, depth=3, origin=level_2_url)
                                        logger.info(f"URL discovered at level 3: {url}")
                                    
                    except Exception as e:
                        logger.warning(f"Error processing level 2 URL {level_2_url}: {e}")
                        
                # Store level 3 URLs - FIXED to check if level 3 exists in the dictionary
                urls_by_level[3].extend(level_3_urls)
                logger.info(f"Discovered {len(level_3_urls)} new URLs at level 3")
                
                # If needed, process level 3 URLs to find level 4 URLs
                if max_depth >= 4 and level_3_urls:
                    # Limit the number of level 3 URLs to check
                    max_urls_to_check = min(len(level_3_urls), 10)
                    urls_to_check = level_3_urls[:max_urls_to_check]
                    
                    logger.info(f"Fetching {len(urls_to_check)} level 3 URLs to discover level 4 links")
                    level_4_urls = []
                    
                    # Process each level 3 URL to find level 4 links
                    for level_3_url in urls_to_check:
                        try:
                            logger.info(f"Fetching level 3 URL to discover level 4 links: {level_3_url}")
                            success, html_content = self.web_fetcher.fetch_page(level_3_url, headers)
                            
                            if success and html_content:
                                level_3_soup = BeautifulSoup(html_content, 'html.parser')
                                
                                # Extract detail links (level 4)
                                level_4_links = extract_detail_links(level_3_soup)
                                level_4_found_urls = process_links(level_4_links, base_domain, level_3_url, visited_urls)
                                
                                for url in level_4_found_urls:
                                    # Check if we should include this URL
                                    should_include = True  # Always include URLs for discovery regardless of processed status
                                    
                                    # Skip URLs we've already visited in this session to avoid duplicates
                                    if url in visited_urls:
                                        should_include = False
                                    
                                    # Validate the URL to filter out problematic ones
                                    if should_include and not self._is_valid_url(url):
                                        should_include = False
                                    
                                    # Log inclusion of already processed URLs
                                    if should_include and self.url_storage.url_is_processed(url):
                                        logger.info(f"Including already processed URL at level 4: {url}")
                                    
                                    if should_include:
                                        level_4_urls.append(url)
                                        visited_urls.add(url)
                                        
                                        # Save to pending URLs with resource_id if provided
                                        if resource_id:
                                            self.url_storage.save_pending_url(url, depth=4, origin=level_3_url, resource_id=resource_id)
                                            self.url_storage.save_resource_url(resource_id, url, depth=4)
                                            logger.info(f"URL discovered at level 4: {url} (associated with resource {resource_id})")
                                        else:
                                            self.url_storage.save_pending_url(url, depth=4, origin=level_3_url)
                                            logger.info(f"URL discovered at level 4: {url}")
                                        
                        except Exception as e:
                            logger.warning(f"Error processing level 3 URL {level_3_url}: {e}")
                            
                    # Store level 4 URLs - FIXED to check if level 4 exists in the dictionary
                    urls_by_level[4].extend(level_4_urls)
                    logger.info(f"Discovered {len(level_4_urls)} new URLs at level 4")
        
        # Log total URLs discovered at each level
        for level in range(1, max_depth + 1):
            if level in urls_by_level:
                logger.info(f"Total URLs discovered at level {level}: {len(urls_by_level[level])}")

    def fetch_additional_page(self, url: str, headers: Dict = None, timeout: int = 60) -> Tuple[bool, str]:
        """
        Fetch a single additional page using WebFetcher.
        
        Args:
            url: The URL to fetch
            headers: Request headers
            timeout: Request timeout
            
        Returns:
            Tuple of (success_flag, html_content)
        """
        # This method is kept for backward compatibility and for direct page fetching without crawling
        return self.web_fetcher.fetch_page(url, headers, timeout)
    
    def fetch_content(self, url: str, max_depth: int = 4, process_by_level: bool = True, 
                     batch_size: int = 50, current_level: int = None,
                     discovery_only: bool = False, resource_id: str = None) -> Tuple[bool, Dict[str, str]]:
        """
        Safely retrieves website content with error handling.
        Fetches main page and all discovered related pages up to the specified depth.
        
        Args:
            url: The URL to fetch
            max_depth: Maximum depth level to crawl (can be set to any value, e.g. 10 for deep crawling)
            process_by_level: If True, processes URLs strictly level by level (all level 1 URLs before level 2)
            batch_size: Maximum number of URLs to process at each level to limit memory usage
            current_level: If specified, only discover URLs at this specific level (for level-based processing)
            discovery_only: If True, focus on URL discovery without analyzing content
            
        Returns:
            Tuple of (success_flag, dict_of_pages)
            dict_of_pages format: {'main': main_html, 'url1': page1_html, ...}
        """
        # Store the discovery_only mode for this operation
        original_discovery_mode = self.discovery_only_mode
        self.discovery_only_mode = discovery_only
        
        try:
            # Always allow discovery even for processed URLs
            if self.discovery_only_mode:
                # In discovery mode, always process URLs regardless of processed status
                if self.url_storage.url_is_processed(url):
                    logger.info(f"URL {url} is already processed, but we're in discovery mode so continuing anyway")
            else:
                # In analysis mode, check if URL is already processed
                if self.url_storage.url_is_processed(url):
                    logger.info(f"URL {url} is already processed, skipping content analysis")
                    # Since we're skipping, return empty result with success=True
                    return True, {"main": "", "additional_urls": [], "error": "URL already processed"}

            # Special handling for discovery_only mode
            if self.discovery_only_mode:
                logger.info(f"Running in discovery-only mode for URL: {url}")
                
                # Get the main page and discover all URLs up to max_depth
                success, main_data = self.get_main_page_with_links(
                    url, 
                    max_depth=max_depth,
                    breadth_first=True,
                    discover_only_level=max_depth,
                    resource_id=resource_id  # Pass the resource ID for URL association
                )
                
                if not success or "error" in main_data:
                    logger.warning(f"Failed URL discovery for {url}: {main_data.get('error', 'Unknown error')}")
                    # Mark URL as processed with error flag even on failure to avoid repeated attempts
                    self.url_storage.save_processed_url(url, depth=0, origin="", discovery_only=True, error=True)
                    # Remove from pending URLs to avoid getting stuck
                    self.url_storage.remove_pending_url(url)
                    logger.info(f"Marked failed URL {url} as processed with error flag and removed from pending queue")
                    return False, {"error": main_data.get("error", "Unknown error fetching main page")}

                # In discovery-only mode, don't process any URLs, just discover and save to pending queue
                urls_by_level = main_data.get("urls_by_level", {})
                total_discovered = 0
                
                # Save all discovered URLs to pending queue
                for level in range(1, max_depth + 1):
                    if level in urls_by_level and urls_by_level[level]:
                        level_urls = urls_by_level[level]
                        saved_count = 0
                        
                        for discovered_url in level_urls:
                            # Always save URLs for discovery regardless of processed status
                            should_save = True
                            
                            # Log if URL has already been processed
                            if self.url_storage.url_is_processed(discovered_url):
                                logger.info(f"Re-adding already processed URL to pending for complete discovery: {discovered_url}")
                            
                            # Save all URLs to ensure complete discovery
                            self.url_storage.save_pending_url(discovered_url, depth=level, origin=url)
                            saved_count += 1
                            total_discovered += 1
                            logger.info(f"Discovered URL at level {level}: {discovered_url}")
                                
                        logger.info(f"Discovery-only mode: Saved {saved_count} URLs at level {level} to pending queue")
                
                # CRITICAL: In discovery-only mode, SKIP ALL CONTENT EXTRACTION AND ANALYSIS
                # Return early with a minimal empty result - don't extract or analyze
                logger.info(f"Discovery-only mode: Completed for {url}, total {total_discovered} URLs discovered across all levels")
                logger.info(f"Discovery-only mode: SKIPPING all content extraction and analysis")
                
                # Mark URL as having completed discovery phase
                self.url_storage.save_discovery_status(url, True)
                
                # Return with minimal empty content to avoid analysis
                return True, {'main': ''}
            
            # For non-discovery mode (actual content analysis)
            # First discover all URLs from this URL
            logger.info(f"Discovering URLs at all levels up to max_depth={max_depth}")
            
            # Get the main page and discover all URLs up to max_depth
            success, main_data = self.get_main_page_with_links(
                url, 
                max_depth=max_depth,
                breadth_first=True,
                discover_only_level=max_depth,
                resource_id=resource_id  # Pass the resource ID for URL association
            )

            if not success or "error" in main_data:
                return False, {"error": main_data.get("error", "Unknown error fetching main page")}

            # Initialize the pages dictionary with the main page
            pages_dict = {'main': main_data["main_html"]}

            # Get headers and metadata from the main page request
            headers = main_data["headers"]
            visited_urls = main_data["visited_urls"]
            urls_by_level = main_data.get("urls_by_level", {})
            
            # Save all discovered URLs to the pending queue first without analyzing them
            # This change ensures all URLs are discovered and saved before any analysis begins
            logger.info("Saving all discovered URLs to pending queue without analyzing them")
            total_saved_urls = 0
            
            for level in range(1, max_depth + 1):
                if level in urls_by_level and urls_by_level[level]:
                    level_urls = urls_by_level[level]
                    saved_count = 0
                    
                    # Save all URLs at this level to the pending queue
                    for discovered_url in level_urls:
                        # Always save URLs regardless of processed status to ensure complete discovery
                        self.url_storage.save_pending_url(discovered_url, depth=level, origin=url)
                        saved_count += 1
                        total_saved_urls += 1
                    
                    logger.info(f"Saved {saved_count} URLs at level {level} to pending queue for later processing")
            
            logger.info(f"Total of {total_saved_urls} URLs saved to pending queue across all levels")
            
            # If specified, process a batch of level 1 URLs immediately
            if process_by_level and current_level == 1 and 1 in urls_by_level:
                level_urls = urls_by_level[1]
                if level_urls:
                    total_urls = len(level_urls)
                    # Process only a batch to avoid memory issues
                    urls_to_process = level_urls[:batch_size]
                    
                    logger.info(f"Processing {len(urls_to_process)} URLs at level 1 immediately")
                    
                    for idx, additional_url in enumerate(urls_to_process):
                        if self.url_storage.url_is_processed(additional_url):
                            logger.info(f"Skipping already processed subpage URL at level 1: {additional_url}")
                            continue
                            
                        logger.info(f"Fetching level 1 page {idx+1}/{len(urls_to_process)}: {additional_url}")
                        success, html_content = self.fetch_additional_page(additional_url, headers)
                        
                        if success and html_content:
                            # Use page name as key (simplified URL)
                            section_name = additional_url.rsplit('/', 1)[-1] or f'level1_page{idx+1}'
                            # Store each page separately
                            pages_dict[section_name] = html_content
                            visited_urls.add(additional_url)
                            
                            # Don't mark URL as processed yet - we'll do this after successful analysis
                            # We're only tracking it for vector generation counts
                            self.urls_processed_since_last_build += 1
                    
                    logger.info(f"Completed fetching HTML content for level 1 URLs from {url} - ready for analysis")
            
            # Restore original discovery mode
            self.discovery_only_mode = original_discovery_mode
            
            # Return all pages as a dictionary
            # For level 1 processing, this includes the main page and level 1 pages
            # For discovery-only or other levels, this is just the main page
            return True, pages_dict

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            # Restore original discovery mode
            self.discovery_only_mode = original_discovery_mode
            return False, {"error": str(e)}
    
    def mark_url_as_processed(self, url, depth=1, origin=""):
        """
        Mark a URL as processed after its content has been successfully analyzed.
        This method can be called by external components like MainProcessor.
        
        Args:
            url: The URL to mark as processed
            depth: The depth level of the URL (0=main, 1=first level, etc.)
            origin: The parent URL that led to this URL
        """
        logger.info(f"Marking URL as processed after successful content analysis: {url}")
        self.url_storage.save_processed_url(url, depth=depth, origin=origin)

    def should_build_vector_store(self):
        """
        Check if we should trigger a vector store build based on URL count.
        Returns:
            bool: True if enough URLs have been processed to trigger a build
        """
        if self.urls_processed_since_last_build >= self.url_build_threshold:
            logger.info(f"URL threshold reached ({self.urls_processed_since_last_build}/{self.url_build_threshold}), should build vector stores")
            return True
        return False

    def reset_url_counter(self):
        """
        Reset the URL counter after a vector store build.
        """
        logger.info(f"Resetting URL counter from {self.urls_processed_since_last_build} to 0")
        self.urls_processed_since_last_build = 0
        
    def check_discovery_needed(self, max_depth: int = 4) -> Dict[str, Any]:
        """
        Check if URL discovery is needed by counting pending URLs at each level.
        
        Args:
            max_depth: Maximum depth to check
            
        Returns:
            Dictionary with discovery status information
        """
        logger.info(f"Checking if URL discovery is needed (max_depth={max_depth})")
        
        # Count pending URLs by level
        pending_by_level = {}
        total_pending = 0
        
        for level in range(1, max_depth + 1):
            pending_urls = self.url_storage.get_pending_urls(depth=level)
            count = len(pending_urls) if pending_urls else 0
            pending_by_level[level] = count
            total_pending += count
        
        # If we have some pending URLs, no discovery is needed
        discovery_needed = total_pending == 0
        
        # Log results
        if discovery_needed:
            logger.info(f"Discovery needed: No pending URLs found at any level")
        else:
            level_info = ", ".join([f"level {level}: {count}" for level, count in pending_by_level.items() if count > 0])
            logger.info(f"Discovery not needed: Found {total_pending} pending URLs ({level_info})")
        
        # Always enable discovery from processed URLs to prevent getting stuck
        self.allow_processed_url_discovery = True
        logger.info(f"Always allowing discovery from processed URLs to ensure complete crawling")
        
        return {
            "discovery_needed": discovery_needed,
            "total_pending": total_pending,
            "pending_by_level": pending_by_level
        }

    def discovery_phase(self, urls: List[str], max_depth: int = 4, force_reprocess: bool = False, resource_ids: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Run STRICT discovery-only phase on a list of URLs.
        This will ONLY discover all URLs from the provided list WITHOUT analyzing content.
        All discovered URLs will be saved to pending_urls.csv for later analysis.
        
        Args:
            urls: List of URLs to process for discovery
            max_depth: Maximum depth for discovery
            force_reprocess: Whether to force reprocessing of already processed URLs
            resource_ids: Optional dictionary mapping URLs to their resource IDs
            
        Returns:
            Dictionary with discovery statistics
        """
        logger.info(f"==================================================")
        logger.info(f"STRICT DISCOVERY-ONLY PHASE - NO ANALYSIS")
        logger.info(f"==================================================")
        logger.info(f"Processing {len(urls)} URLs with max_depth={max_depth}")
        logger.info(f"This phase ONLY discovers and saves URLs to pending_urls.csv")
        logger.info(f"NO CONTENT ANALYSIS will be performed during this phase")
        
        # Ensure discovery_only_mode is True for this entire phase
        original_discovery_mode = self.discovery_only_mode
        self.discovery_only_mode = True
        
        # Check if we have pending URLs to potentially enable discovery from processed URLs
        discovery_status = self.check_discovery_needed(max_depth)
        total_pending = discovery_status["total_pending"]
        
        # If there are already enough pending URLs, skip discovery
        # This is a critical addition to prevent unnecessary URL rediscovery
        if total_pending > 100 and not force_reprocess:  # Consider 100 URLs as plenty
            logger.info(f"SKIPPING DISCOVERY: Found {total_pending} pending URLs already. No need for additional discovery.")
            logger.info(f"To force rediscovery, run with force_reprocess=True")
            
            # Return early with stats about existing URLs
            results = {
                "total_urls": len(urls),
                "successful": 0,
                "failed": 0,
                "skipped": len(urls),
                "discovered_by_level": discovery_status["pending_by_level"],
                "total_discovered": total_pending,
                "discovery_skipped": True,
                "reason": f"Already have {total_pending} pending URLs"
            }
            
            # Restore original discovery mode
            self.discovery_only_mode = original_discovery_mode
            return results
        
        # If no pending URLs and we're at level 0 in config, force_reprocess
        if discovery_status["total_pending"] == 0:
            original_force_reprocess = force_reprocess
            force_reprocess = True
            logger.info(f"No pending URLs found - forcing discovery from processed URLs (force_reprocess=True)")
        
        results = {
            "total_urls": len(urls),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "discovered_by_level": {},
            "discovered_by_resource": {},
            "total_discovered": 0
        }
        
        # Store initial pending URL counts before discovery
        initial_pending = {}
        for level in range(1, max_depth + 1):
            pending_urls = self.url_storage.get_pending_urls(depth=level)
            initial_pending[level] = len(pending_urls) if pending_urls else 0
        
        try:
            # Process each URL for discovery only
            for idx, url in enumerate(urls):
                # Get resource ID for this URL if available
                resource_id = None
                if resource_ids and url in resource_ids:
                    resource_id = resource_ids[url]
                    logger.info(f"[{idx+1}/{len(urls)}] Running URL discovery for resource {resource_id}: {url}")
                else:
                    logger.info(f"[{idx+1}/{len(urls)}] Running URL discovery for: {url}")
                
                # Check if we've already done discovery for this URL and skip if so
                if self.url_storage.get_discovery_status(url) and not force_reprocess:
                    logger.info(f"URL {url} already completed discovery phase, skipping")
                    results["skipped"] += 1
                    continue
                
                # Track all URLs discovered from this resource
                resource_discovered = 0
                
                # Run a specialized discovery-only fetch that only discovers URLs - NO ANALYSIS
                # This is critical to ensure separation between discovery and analysis phases
                success, content = self.fetch_content(
                    url=url,
                    max_depth=max_depth,
                    discovery_only=True,     # STRICT discovery only mode
                    force_reprocess=force_reprocess,
                    resource_id=resource_id  # Pass the resource ID to associate URLs with resource
                )
                
                if success:
                    results["successful"] += 1
                    # Mark this URL as having gone through discovery (but not analysis)
                    self.url_storage.set_discovery_status(url, True)
                    
                    # Associate this URL with the resource if we have a resource ID
                    if resource_id:
                        self.url_storage.save_resource_url(resource_id, url, depth=0)
                        
                        # Now check all pending URLs created by this discovery process
                        # and associate them with this resource
                        for level in range(1, max_depth + 1):
                            pending_urls = self.url_storage.get_pending_urls(max_urls=10000, depth=level)
                            for pending_url_data in pending_urls:
                                pending_url = pending_url_data.get('url')
                                origin = pending_url_data.get('origin', '')
                                
                                # Check if this pending URL came from our current URL or any URL
                                # that we've already associated with this resource
                                if origin == url or self.url_storage.url_belongs_to_resource(origin, resource_id):
                                    # This URL belongs to our resource - update it with resource ID
                                    self.url_storage.save_pending_url(
                                        url=pending_url,
                                        depth=level,
                                        origin=origin,
                                        attempt_count=pending_url_data.get('attempt_count', 0),
                                        resource_id=resource_id
                                    )
                                    
                                    # Also save direct resource association
                                    self.url_storage.save_resource_url(resource_id, pending_url, depth=level)
                                    resource_discovered += 1
                        
                        # Update resource-specific discovery stats
                        if resource_id not in results["discovered_by_resource"]:
                            results["discovered_by_resource"][resource_id] = 0
                        results["discovered_by_resource"][resource_id] += resource_discovered
                        
                        logger.info(f"Associated {resource_discovered} discovered URLs with resource {resource_id}")
                else:
                    results["failed"] += 1
                    logger.warning(f"Failed to process URL for discovery: {url}")
            
            # Collect statistics on discovered URLs by level after discovery
            final_pending = {}
            for level in range(1, max_depth + 1):
                pending_urls = self.url_storage.get_pending_urls(depth=level)
                count = len(pending_urls) if pending_urls else 0
                final_pending[level] = count
                results["discovered_by_level"][level] = count
                results["total_discovered"] += count
            
            # Calculate newly discovered URLs by comparing before and after
            new_discovered = {}
            for level in range(1, max_depth + 1):
                new_discovered[level] = final_pending[level] - initial_pending[level]
            
            logger.info(f"==================================================")
            logger.info(f"DISCOVERY PHASE COMPLETED")
            logger.info(f"==================================================")
            logger.info(f"Results: {results['successful']} successful, {results['failed']} failed, {results['skipped']} skipped")
            logger.info(f"Total discovered URLs across all levels: {results['total_discovered']}")
            
            # Log details by level
            for level in range(1, max_depth + 1):
                count = results["discovered_by_level"].get(level, 0)
                new_count = new_discovered.get(level, 0)
                logger.info(f"  Level {level}: {count} total URLs ({new_count} newly discovered)")
            
            # Log details by resource
            if results["discovered_by_resource"]:
                logger.info("URLs discovered by resource:")
                for resource_id, count in results["discovered_by_resource"].items():
                    logger.info(f"  Resource {resource_id}: {count} URLs")
            
            logger.info("Discovery phase complete - all URLs are now saved to pending_urls.csv with resource IDs")
            logger.info("These URLs will be analyzed in a separate phase")
            
        finally:
            # Always restore the original discovery mode
            logger.info(f"Restoring discovery mode to original value: {original_discovery_mode}")
            self.discovery_only_mode = original_discovery_mode
        
        return results
    
    def analysis_phase(self, batch_size: int = 50, max_level: int = 4) -> Dict[str, Any]:
        """
        Process all pending URLs for analysis.
        This should be called after the discovery phase has completed.
        
        Args:
            batch_size: Number of URLs to process in each batch
            max_level: Maximum level to process
            
        Returns:
            Dictionary with analysis statistics
        """
        logger.info(f"Starting URL analysis phase with batch_size={batch_size}")
        
        results = {
            "processed_by_level": {},
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
        
        # Process each level sequentially
        for level in range(1, max_level + 1):
            pending_urls = self.url_storage.get_pending_urls(depth=level)
            
            if not pending_urls:
                logger.info(f"No pending URLs at level {level}")
                results["processed_by_level"][level] = 0
                continue
                
            logger.info(f"Processing {len(pending_urls)} pending URLs at level {level}")
            
            # Process URLs in batches
            processed_count = 0
            success_count = 0
            failed_count = 0
            
            # Process in batches to avoid memory issues
            for batch_start in range(0, len(pending_urls), batch_size):
                batch_end = min(batch_start + batch_size, len(pending_urls))
                batch = pending_urls[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(pending_urls)-1)//batch_size + 1} "
                           f"({batch_end - batch_start} URLs)")
                
                for pending_url_data in batch:
                    url_to_process = pending_url_data.get('url')
                    origin_url = pending_url_data.get('origin', '')
                    
                    # Check if already processed
                    if self.url_storage.url_is_processed(url_to_process):
                        logger.info(f"URL already processed: {url_to_process}, skipping analysis")
                        continue
                        
                    # Process this URL (analysis should be handled by caller)
                    # Here we just mark that it's ready for processing
                    logger.info(f"URL ready for analysis: {url_to_process}")
                    processed_count += 1
                    
                    # This would be where actual content analysis happens,
                    # but we're just preparing URLs for the calling code to analyze
            
            results["processed_by_level"][level] = processed_count
            results["total_processed"] += processed_count
            results["successful"] += success_count
            results["failed"] += failed_count
            
            logger.info(f"Level {level} processing: {processed_count} URLs processed")
        
        logger.info(f"URL analysis phase completed: {results['total_processed']} URLs processed")
        return results