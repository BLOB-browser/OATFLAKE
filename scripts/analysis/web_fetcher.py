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
            
            response.raise_for_status()
            
            if response.status_code == 200:
                logger.info(f"Successfully fetched content from {url}")
                return True, response.text
            else:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return False, ""
                
        except requests.RequestException as e:
            logger.warning(f"Error fetching page {url}: {e}")
            return False, ""
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return False, ""
