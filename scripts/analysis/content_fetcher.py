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

    def get_main_page_with_links(self, url: str, max_depth: int = 4, go_deeper: bool = None, breadth_first: bool = True, discover_only_level: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieves the main page content and identifies links to additional pages.
        Can go multiple layers deep to find more content-rich pages.
        
        Args:
            url: The URL to fetch
            max_depth: Maximum depth level to crawl (1=just main page links, 2=two levels, 3=three levels)
            go_deeper: (Deprecated) If True, will look for links two levels deep (use max_depth instead)
            breadth_first: If True, will discover all URLs in breadth-first order (completing all level 1 URLs before level 2)
            discover_only_level: Maximum level to discover URLs (default 1). URLs beyond this level will not be discovered.
            
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
                
            # Check if the main URL has already been processed - do this check first
            # to avoid unnecessary network requests
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
            
            # Limit discovery to only specified level (default: level 1)
            effective_max_depth = min(max_depth, discover_only_level)
            
            # Log the max depth being used
            logger.info(f"URL discovery limited to level {effective_max_depth} (max_depth={max_depth}, discover_only_level={discover_only_level})")
            
            # Load processed URLs to avoid re-fetching content we've already analyzed
            processed_urls = self.url_storage.get_processed_urls()
            
            if breadth_first:
                # Use breadth-first crawling to discover all URLs level by level
                self._discover_urls_breadth_first(
                    root_soup=soup,
                    root_url=url,
                    base_domain=base_domain,
                    headers=headers,
                    max_depth=effective_max_depth,  # Use the limited depth for discovery
                    visited_urls=visited_urls,
                    urls_by_level=urls_by_level
                )
            else:
                # Use the original recursive (depth-first) crawling
                self._crawl_recursive(
                    soup=soup, 
                    base_url=url,
                    base_domain=base_domain, 
                    headers=headers,
                    current_level=1, 
                    max_depth=effective_max_depth,  # Use the limited depth for discovery
                    visited_urls=visited_urls,
                    urls_by_level=urls_by_level
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
            
    def _crawl_recursive(self, soup, base_url, base_domain, headers, current_level, max_depth, visited_urls, urls_by_level):
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
        
        # Filter out already processed URLs
        filtered_urls = []
        for url in found_urls:
            if self.url_storage.url_is_processed(url):
                logger.info(f"URL {url} already processed, filtering out from level {current_level}")
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
                        urls_by_level=urls_by_level
                    )
                    
                except Exception as e:
                    logger.warning(f"Error processing URL {next_url} at level {current_level}: {e}")
                    
    def _discover_urls_breadth_first(self, root_soup, root_url, base_domain, headers, max_depth=4, visited_urls=None, urls_by_level=None):
        """
        Discovers URLs in a breadth-first manner, completing all URLs at one level before
        moving to the next. This ensures all level 1 URLs are found before level 2, etc.
        
        Args:
            root_soup: BeautifulSoup object of the root/main page
            root_url: URL of the root/main page
            base_domain: Base domain for resolving relative URLs
            headers: HTTP headers for requests
            max_depth: Maximum depth to crawl
            visited_urls: Set of already visited URLs
            urls_by_level: Dictionary to store URLs by their depth level
        """
        # Start with the root URL at level 0
        pages_to_process = {
            1: [(root_soup, root_url)]  # Initial page at level 1
        }
        
        # Process each level completely before moving to the next
        for current_level in range(1, max_depth + 1):
            logger.info(f"Processing URL discovery at level {current_level}")
            
            # Skip if we don't have pages at this level
            if current_level not in pages_to_process or not pages_to_process[current_level]:
                logger.info(f"No pages to process at level {current_level}")
                break
                
            # Initialize container for next level
            if current_level + 1 <= max_depth:
                pages_to_process[current_level + 1] = []
                
            # Process all pages at current level
            for soup, page_url in pages_to_process[current_level]:
                # Extract links appropriate for the current level
                if current_level == 1:
                    # For the first level, get navigation links
                    links = extract_navigation_links(soup)
                    logger.info(f"Extracting navigation links from {page_url}")
                elif current_level == 2:
                    # For the second level, get project links
                    links = extract_project_links(soup)
                    logger.info(f"Extracting project links from {page_url}")
                else:
                    # For deeper levels, get detail links
                    links = extract_detail_links(soup)
                    logger.info(f"Extracting detail links from {page_url}")
                
                # Process the found links
                found_urls = process_links(links, base_domain, page_url, visited_urls)
                
                # Filter out already processed URLs
                filtered_urls = []
                for url in found_urls:
                    if self.url_storage.url_is_processed(url):
                        logger.info(f"URL {url} already processed, filtering out")
                    else:
                        filtered_urls.append(url)
                        # Save to pending URLs file to reduce memory usage
                        self.url_storage.save_pending_url(url, depth=current_level, origin=page_url)
                
                # Store the filtered URLs at this level
                urls_by_level[current_level].extend(filtered_urls)
                
                # If we're not at max depth, fetch a sample of next level's pages
                # Full list of URLs is already saved to pending_urls.csv for later processing
                if current_level < max_depth and filtered_urls and current_level + 1 in pages_to_process:
                    # Limit the number of URLs to check at deeper levels to avoid excessive crawling
                    max_urls_to_check = min(len(filtered_urls), 20)
                    urls_to_check = filtered_urls[:max_urls_to_check]
                    
                    logger.info(f"Queueing {len(urls_to_check)} sample URLs for level {current_level + 1} discovery (total {len(filtered_urls)} saved to pending)")
                    
                    # Process each URL to discover next level
                    for next_url in urls_to_check:
                        try:
                            # Mark as visited to avoid re-processing
                            visited_urls.add(next_url)
                            
                            # We don't need to fetch the full content during discovery
                            # Just add the URL to the pending queue for later processing
                            # This will be processed in the main content extraction phase
                            logger.info(f"Adding URL to discovery queue (level {current_level+1}): {next_url}")
                            
                            # When we're not fetching content during discovery phase, we still need to 
                            # provide some structure for pages_to_process to maintain level tracking
                            
                            # Create a minimal BeautifulSoup object that won't cause errors if accessed
                            # This avoids issues with None values in the _discover_urls_breadth_first method
                            next_soup = BeautifulSoup("<html><body></body></html>", 'html.parser')
                            
                            # Skip the actual content fetching and parsing during discovery
                            # This significantly reduces network traffic and processing time
                            
                            # Add to the next level's pages to process
                            pages_to_process[current_level + 1].append((next_soup, next_url))
                            
                        except Exception as e:
                            logger.warning(f"Error processing URL {next_url} at level {current_level}: {e}")
            
            # Log the number of pages discovered for the next level
            if current_level + 1 in pages_to_process:
                logger.info(f"Discovered {len(pages_to_process[current_level + 1])} pages to process at level {current_level + 1}")
                
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
                     batch_size: int = 50, current_level: int = None) -> Tuple[bool, Dict[str, str]]:
        """
        Safely retrieves website content with error handling.
        Fetches main page and all discovered related pages up to the specified depth.
        
        Args:
            url: The URL to fetch
            max_depth: Maximum depth level to crawl (can be set to any value, e.g. 10 for deep crawling)
            process_by_level: If True, processes URLs strictly level by level (all level 1 URLs before level 2)
            batch_size: Maximum number of URLs to process at each level to limit memory usage
            current_level: If specified, only discover URLs at this specific level (for level-based processing)
            
        Returns:
            Tuple of (success_flag, dict_of_pages)
            dict_of_pages format: {'main': main_html, 'url1': page1_html, ...}
        """
        try:
            # Check if URL is already processed
            if self.url_storage.url_is_processed(url):
                logger.info(f"Skipping already processed URL: {url}")
                return True, {"main": "", "error": "URL already processed"}

            # Get the main page and extract links based on the current processing level
            # When current_level is specified, we're in level-based processing mode
            
            # Determine what level to discover URLs for
            if current_level is not None:
                # Only discover URLs at the specific level requested
                discover_only_level = current_level
                # For level discovery, we need current_level depth
                current_max_depth = current_level
                logger.info(f"Level-based processing: Discovering ONLY level {discover_only_level} URLs")
            else:
                # Default to level 1 for initial processing
                discover_only_level = 1
                current_max_depth = 1
                logger.info(f"Initial processing: Discovering ONLY level {discover_only_level} URLs")
            
            # Get the main page and discover only the exact level we want
            success, main_data = self.get_main_page_with_links(
                url, 
                max_depth=current_max_depth,  # Limited to exactly the current level
                breadth_first=True,  # Always use breadth-first to ensure level discipline
                discover_only_level=discover_only_level  # Only discover this exact level
            )

            if not success or "error" in main_data:
                return False, {"error": main_data.get("error", "Unknown error fetching main page")}

            # Initialize the pages dictionary with the main page
            pages_dict = {'main': main_data["main_html"]}

            # Get headers and metadata from the main page request
            headers = main_data["headers"]
            visited_urls = main_data["visited_urls"]
            urls_by_level = main_data.get("urls_by_level", {})
            
            # Don't mark the URL as processed yet - we'll do this after successful extraction
            # The URL will only be marked as processed after the entire analysis is complete
            # This prevents URLs from being skipped if extraction fails
            
            if process_by_level:
                # Modified to only process level 1 URLs and save higher levels to pending queue
                # Process only the first level URLs immediately
                current_level = 1
                
                if current_level in urls_by_level and urls_by_level[current_level]:
                    level_urls = urls_by_level[current_level]
                    total_urls = len(level_urls)
                    
                    # Process only a batch to avoid memory issues, rest are stored in pending_urls.csv
                    urls_to_process = level_urls[:batch_size]
                    remaining = max(0, total_urls - batch_size)
                    
                    logger.info(f"Processing {len(urls_to_process)} URLs at level {current_level} (remaining {remaining} URLs saved to pending_urls.csv)")
                    
                    # Save remaining level 1 URLs to pending
                    if remaining > 0:
                        for remaining_url in level_urls[batch_size:]:
                            if not self.url_storage.url_is_processed(remaining_url):
                                self.url_storage.save_pending_url(remaining_url, depth=current_level, origin=url)
                    
                    # Process URLs in this batch
                    for idx, additional_url in enumerate(urls_to_process):
                        if self.url_storage.url_is_processed(additional_url):
                            logger.info(f"Skipping already processed subpage URL at level {current_level}: {additional_url}")
                            continue

                        logger.info(f"Fetching level {current_level} page {idx+1}/{len(urls_to_process)}: {additional_url}")
                        success, html_content = self.fetch_additional_page(additional_url, headers)

                        if success and html_content:
                            # Use page name as key (simplified URL)
                            section_name = additional_url.rsplit('/', 1)[-1] or f'level{current_level}_page{idx+1}'
                            # Store each page separately
                            pages_dict[section_name] = html_content
                            visited_urls.add(additional_url)
                            
                            # Don't mark URL as processed yet - we'll do this after successful analysis
                            # We're only tracking it for vector generation counts
                            self.urls_processed_since_last_build += 1
                    
                    logger.info(f"Completed fetching HTML content for level 1 URLs from {url} - ready for analysis")
                
                # Save all higher level URLs to pending for later cross-resource processing
                for future_level in range(2, max_depth + 1):
                    if future_level in urls_by_level and urls_by_level[future_level]:
                        future_urls = urls_by_level[future_level]
                        logger.info(f"Saving {len(future_urls)} URLs at level {future_level} to pending queue for later processing")
                        
                        # Save all these URLs to pending
                        for future_url in future_urls:
                            if not self.url_storage.url_is_processed(future_url):
                                self.url_storage.save_pending_url(future_url, depth=future_level, origin=url)
                
                # Check if we should build vector store after processing level 1
                if self.should_build_vector_store():
                    logger.info(f"URL threshold reached after processing level 1, time to build vector stores")
                    # This will be handled by the caller
            else:
                # Original approach: process URLs in the order they were discovered
                additional_urls = main_data["additional_urls"]
                logger.info(f"Processing all {len(additional_urls)} additional pages from {url} (mixed levels)")
                
                # Fetch additional pages one by one
                for idx, additional_url in enumerate(additional_urls):
                    if self.url_storage.url_is_processed(additional_url):
                        logger.info(f"Skipping already processed subpage URL: {additional_url}")
                        continue

                    logger.info(f"Fetching additional page {idx+1}/{len(additional_urls)}: {additional_url}")
                    success, html_content = self.fetch_additional_page(additional_url, headers)

                    if success and html_content:
                        # Use page name as key (simplified URL)
                        section_name = additional_url.rsplit('/', 1)[-1] or f'page{idx+1}'
                        # Store each page separately
                        pages_dict[section_name] = html_content
                        visited_urls.add(additional_url)

                        # Determine the depth level using urls_by_level
                        depth = 1  # Default to depth 1 
                        
                        # Loop through all levels to find where this URL belongs
                        for level, urls in urls_by_level.items():
                            if additional_url in urls:
                                depth = int(level)
                                break
                        
                        # Don't mark the subpage URL as processed yet
                        # URLs will be marked as processed after successful analysis by the resource processor
                        # Just track it for vector generation counts
                        self.urls_processed_since_last_build += 1

            # Return all pages as a dictionary
            return True, pages_dict

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
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