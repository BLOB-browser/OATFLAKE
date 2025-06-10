#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import csv
import json
import requests
import platform
from urllib.parse import urljoin
from typing import Tuple, Dict, Any, List, Set
from bs4 import BeautifulSoup
from datetime import datetime
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
    """Responsible for fetching and extracting content from websites - CONTENT ONLY"""
    
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
        logger.info(f"Using processed URLs file at: {processed_urls_file}")        # Initialize enhanced URL storage manager for unique URL identification
        try:
            # Try to import from project root first
            import sys
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from enhanced_url_storage import EnhancedURLStorageManager
            self.enhanced_url_storage = EnhancedURLStorageManager(processed_urls_file)
            logger.info("Enhanced URL storage manager initialized successfully")
        except ImportError as e:
            logger.warning(f"Enhanced URL storage not available: {e}")
            self.enhanced_url_storage = None

        # Add flag to control whether to analyze immediately or defer analysis
        self.defer_analysis = False
        self.discovery_only_mode = False  # When True, focus on URL discovery only
        self.allow_processed_url_discovery = False  # When True, allow discovery from processed URLs when no pending URLs
    

    def _get_main_resource_url(self, resource_id: str) -> str:
        """Get the main resource URL for a given resource ID.
        
        Args:
            resource_id: The resource ID to look up
            
        Returns:
            The main resource URL or empty string if not found
        """
        if not resource_id or resource_id == "0":
            return ""
            
        # Try enhanced storage first if available
        if hasattr(self, 'enhanced_url_storage') and self.enhanced_url_storage:
            try:
                main_url = self.enhanced_url_storage.get_main_resource_url(resource_id)
                if main_url:
                    return main_url
            except Exception as e:
                logger.warning(f"Error getting main resource URL from enhanced storage: {e}")
          # Fallback to traditional storage
        try:
            # In traditional storage, get_resource_urls returns a Set[str] of URLs
            resource_urls = self.url_storage.get_resource_urls(resource_id)
            
            # Since get_resource_urls returns a set of URL strings, not dictionaries,
            # we just return the first URL if available
            if resource_urls:
                # Convert set to list to access first element
                urls_list = list(resource_urls)
                return urls_list[0] if urls_list else ""
        except Exception as e:
            logger.warning(f"Error getting main resource URL from traditional storage: {e}")
        
        return ""

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
    def fetch_content_simple(self, url: str, headers: Dict = None) -> Tuple[bool, str]:
        """
        Simple content fetching method - just gets the HTML content of a single URL.
        This is the core functionality that ContentFetcher should provide.
        
        Args:
            url: The URL to fetch content from
            headers: Optional headers to use for the request
            
        Returns:
            Tuple of (success_flag, html_content)
        """
        try:
            if not headers:
                headers = self.web_fetcher.create_headers()
            
            success, html_content = self.web_fetcher.fetch_page(url, headers)
            
            if success:
                logger.debug(f"Successfully fetched content from {url}")
                return True, html_content
            else:
                logger.warning(f"Failed to fetch content from {url}: {html_content}")
                return False, html_content
                    
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return False, str(e)
                    
    
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
    def fetch_content(self, url: str, discover_links: bool = False) -> Tuple[bool, Dict[str, str]]:
        """
        Simplified content fetching method - CONTENT ONLY.
        No URL discovery, no crawling, just fetches the content of the specified URL.
        
        Args:
            url: The URL to fetch content from
            discover_links: Deprecated parameter, kept for compatibility but ignored
            
        Returns:
            Tuple of (success_flag, content_dict)
            content_dict format: {'main': html_content}
        """
        try:
            # Check if URL is already processed
            if self.url_storage.url_is_processed(url):
                logger.info(f"URL {url} is already processed, skipping content fetch")
                return True, {"main": "", "error": "URL already processed"}

            # Create headers using WebFetcher
            headers = self.web_fetcher.create_headers()
            
            # Fetch the single URL content without any discovery
            success, html_content = self.web_fetcher.fetch_page(url, headers)
            
            if not success:
                logger.warning(f"Failed to fetch content for {url}")
                return False, {"error": "Failed to fetch page content"}
            
            logger.info(f"Successfully fetched content from {url}")
            return True, {'main': html_content}
            
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
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