#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
from typing import Tuple, Dict, Any
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class WebFetcher:
    """Handles HTTP requests for fetching web content"""
    
    def __init__(self, user_agent: str, timeout: int = 90, verify_ssl: bool = False):
        self.user_agent = user_agent
        self.timeout = timeout
        self.verify_ssl = verify_ssl
    
    def create_headers(self) -> Dict[str, str]:
        """Create HTTP headers for requests
        
        Returns:
            Dictionary of HTTP headers
        """
        return {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    
    def fetch_page(self, url: str, custom_headers: Dict = None, custom_timeout: int = None) -> Tuple[bool, str]:
        """Fetch a web page
        
        Args:
            url: The URL to fetch
            custom_headers: Optional custom headers to use
            custom_timeout: Optional custom timeout to use
            
        Returns:
            Tuple of (success_flag, html_content)
        """
        try:
            headers = custom_headers or self.create_headers()
            timeout = custom_timeout or self.timeout
            
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=self.verify_ssl,
                allow_redirects=True
            )
              # Handle different status codes explicitly
            if response.status_code == 200:
                logger.info(f"Successfully fetched content from {url}")
                return True, response.text
            elif response.status_code == 404:
                # Explicit handling for 404 errors
                logger.warning(f"Error fetching page {url}: 404 Client Error: Not Found for url: {url}")
                return False, f"HTTP 404 Not Found: {url}"
            elif response.status_code in (403, 401):
                # Access forbidden or unauthorized
                logger.warning(f"Access denied for {url}: HTTP {response.status_code}")
                return False, f"HTTP {response.status_code} Access Denied: {url}"
            elif response.status_code in (500, 502, 503, 504):
                # Server errors that should trigger fallback discovery
                logger.warning(f"Server error for {url}: HTTP {response.status_code} - this should trigger fallback discovery")
                return False, f"HTTP {response.status_code} Server Error: {url}"
            else:
                # Other HTTP errors
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                response.raise_for_status()  # Will trigger exception handler
                return False, ""
                
        except requests.RequestException as e:
            logger.warning(f"Error fetching page {url}: {e}")
            return False, f"Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return False, ""

    def is_server_error(self, error_message: str) -> bool:
        """Check if an error message indicates a server error that should trigger fallbacks
        
        Args:
            error_message: The error message returned from fetch_page
            
        Returns:
            True if this is a server error that should trigger fallback discovery
        """
        server_error_indicators = [
            "HTTP 500 Server Error",
            "HTTP 502 Server Error", 
            "HTTP 503 Server Error",
            "HTTP 504 Server Error"
        ]
        
        return any(indicator in error_message for indicator in server_error_indicators)
