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

    def get_main_page_with_links(self, url: str, max_depth: int = 3, go_deeper: bool = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieves the main page content and identifies links to additional pages.
        Can go multiple layers deep to find more content-rich pages.
        
        Args:
            url: The URL to fetch
            max_depth: Maximum depth level to crawl (1=just main page links, 2=two levels, 3=three levels)
            go_deeper: (Deprecated) If True, will look for links two levels deep (use max_depth instead)
            
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
            
            # Load processed URLs to avoid re-fetching content we've already analyzed
            processed_urls = self.url_storage.get_processed_urls()
            
            # Start recursive crawling from the main page (level 0)
            self._crawl_recursive(
                soup=soup, 
                base_url=url,
                base_domain=base_domain, 
                headers=headers,
                current_level=1, 
                max_depth=max_depth,
                visited_urls=visited_urls,
                urls_by_level=urls_by_level
            )
            
            # Create a flattened list of all URLs
            all_additional_urls = []
            for level in range(1, max_depth + 1):
                if level in urls_by_level:
                    logger.info(f"Found {len(urls_by_level[level])} unprocessed URLs at level {level}")
                    all_additional_urls.extend(urls_by_level[level])
                    
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
                "urls_by_level": urls_by_level
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching main page {url}: {e}")
            return False, {"error": str(e)}
            
    def _crawl_recursive(self, soup, base_url, base_domain, headers, current_level, max_depth, visited_urls, urls_by_level):
        """
        Recursively crawls pages up to a specified depth, collecting unprocessed URLs.
        
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
    
    def fetch_content(self, url: str, max_depth: int = 3) -> Tuple[bool, Dict[str, str]]:
        """
        Safely retrieves website content with error handling.
        Fetches main page and all discovered related pages up to the specified depth.
        
        Args:
            url: The URL to fetch
            max_depth: Maximum depth level to crawl (can be set to any value, e.g. 10 for deep crawling)
            
        Returns:
            Tuple of (success_flag, dict_of_pages)
            dict_of_pages format: {'main': main_html, 'url1': page1_html, ...}
        """
        try:
            # Check if URL is already processed
            if self.url_storage.url_is_processed(url):
                logger.info(f"Skipping already processed URL: {url}")
                return True, {"main": "", "error": "URL already processed"}

            # Get the main page and extract links to other pages with specified depth
            success, main_data = self.get_main_page_with_links(url, max_depth=max_depth)

            if not success or "error" in main_data:
                return False, {"error": main_data.get("error", "Unknown error fetching main page")}

            # Initialize the pages dictionary with the main page
            pages_dict = {'main': main_data["main_html"]}

            # Get the list of additional URLs
            additional_urls = main_data["additional_urls"]
            headers = main_data["headers"]
            visited_urls = main_data["visited_urls"]
            urls_by_level = main_data.get("urls_by_level", {})

            # Log the number of pages we'll be fetching
            logger.info(f"Processing all {len(additional_urls)} additional pages from {url}")
            
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
                    
                    # Save the processed subpage URL with depth and origin information
                    self.url_storage.save_processed_url(additional_url, depth=depth, origin=url)
                    self.urls_processed_since_last_build += 1

            # Save the main URL as processed (depth 0, no origin)
            self.url_storage.save_processed_url(url, depth=0, origin="")
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