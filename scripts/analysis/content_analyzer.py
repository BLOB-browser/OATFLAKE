#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import Tuple, Dict, Any, List, Set
import requests
import os
import platform
from scripts.analysis.web_fetcher import WebFetcher
from utils.config import get_data_path  # Fixed import

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """Handles content fetching and analysis operations"""
    
    def __init__(self, web_fetcher=None, url_storage_manager=None, timeout: int = 90):
        self.timeout = timeout
        
        # Add size limits matching ResourceLLM
        self.size_limits = {
            'description': 4000,    # For description generation
            'tags': 2500,          # For tag generation
            'definitions': 4000,    # For definition extraction
            'projects': 4000,      # For project identification
            'default': 2000        # Default limit
        }
        
        # Track URL count for incremental vector store generation
        self.urls_processed_since_last_build = 0
        self.url_build_threshold = 10  # Build vector store after every 10 URLs
        
        # Initialize web fetcher if not provided
        if web_fetcher is None:
            # Add platform-specific User-Agents with updated Chrome version
            user_agents = {
                'windows': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'mac': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'raspberry': 'Mozilla/5.0 (Linux; Android 13; Raspberry Pi 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
            }
            
            # Detect platform and set appropriate User-Agent
            system = platform.system().lower()
            
            # Set User-Agent based on platform
            if 'darwin' in system:
                user_agent = user_agents['mac']
            elif 'linux' in system:
                user_agent = user_agents['raspberry']
            else:
                user_agent = user_agents['windows']
                
            self.web_fetcher = WebFetcher(
                user_agent=user_agent,
                timeout=self.timeout,
                verify_ssl=False  # Disable SSL verification for compatibility
            )
        else:
            self.web_fetcher = web_fetcher
            
        self.url_storage = url_storage_manager
        
        logger.info(f"ContentAnalyzer initialized with size limits: {self.size_limits}")
    
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

    def fetch_content_for_analysis(self, url: str, max_depth: int = 4, process_by_level: bool = True, 
                                  batch_size: int = 50, current_level: int = None,
                                  resource_id: str = None) -> Tuple[bool, Dict[str, str]]:
        """
        Fetch content for analysis without URL discovery.
        This method focuses purely on content retrieval and analysis.
        
        Args:
            url: The URL to fetch content from
            max_depth: Maximum depth level to consider (for level-based processing)
            process_by_level: If True, processes URLs strictly level by level
            batch_size: Maximum number of URLs to process at each level
            current_level: If specified, only process URLs at this specific level
            resource_id: Optional resource ID to associate with the content
            
        Returns:
            Tuple of (success_flag, dict_of_pages)
            dict_of_pages format: {'main': main_html, 'url1': page1_html, ...}
        """
        try:
            # Check if URL is already processed
            if self.url_storage and self.url_storage.url_is_processed(url):
                logger.info(f"URL {url} is already processed, skipping content analysis")
                return True, {"main": "", "additional_urls": [], "error": "URL already processed"}

            # Create headers using WebFetcher
            headers = self.web_fetcher.create_headers()
            
            # Fetch the single URL content without link discovery
            logger.info(f"Fetching content for analysis: {url}")
            success, html_content = self.web_fetcher.fetch_page(url, headers)
            
            if not success:
                logger.warning(f"Failed to fetch content for {url}")
                return False, {"error": "Failed to fetch page content"}
            
            # Return just the content without any link discovery
            return True, {'main': html_content}
            
        except Exception as e:
            logger.error(f"Error fetching content for analysis {url}: {e}")
            return False, {"error": str(e)}

    def analyze_page_content(self, content_dict: Dict[str, str], url: str = None) -> Dict[str, Any]:
        """
        Analyze page content and extract useful information.
        
        Args:
            content_dict: Dictionary containing page content (e.g., {'main': html_content})
            url: Optional URL for context
            
        Returns:
            Dictionary with analysis results
        """
        analysis_results = {
            "url": url,
            "pages_analyzed": len(content_dict),
            "total_content_length": 0,
            "main_content_length": 0,
            "additional_pages": [],
            "analysis_metadata": {}
        }
        
        try:
            # Analyze main content
            if 'main' in content_dict:
                main_content = content_dict['main']
                analysis_results["main_content_length"] = len(main_content)
                analysis_results["total_content_length"] += len(main_content)
                
                # Basic content analysis
                if main_content:
                    # Count basic elements (this could be expanded with more sophisticated analysis)
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(main_content, 'html.parser')
                    
                    analysis_results["analysis_metadata"] = {
                        "title": soup.title.string if soup.title else "No title",
                        "paragraph_count": len(soup.find_all('p')),
                        "link_count": len(soup.find_all('a')),
                        "image_count": len(soup.find_all('img')),
                        "heading_count": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
                    }
            
            # Analyze additional pages
            for page_key, page_content in content_dict.items():
                if page_key != 'main':
                    analysis_results["additional_pages"].append({
                        "page_key": page_key,
                        "content_length": len(page_content)
                    })
                    analysis_results["total_content_length"] += len(page_content)
            
            logger.info(f"Content analysis completed for {url}: {analysis_results['pages_analyzed']} pages, {analysis_results['total_content_length']} total characters")
            
        except Exception as e:
            logger.error(f"Error analyzing page content for {url}: {e}")
            analysis_results["error"] = str(e)
        
        return analysis_results

    def analysis_phase(self, batch_size: int = 50, max_level: int = 4) -> Dict[str, Any]:
        """
        Process all pending URLs for content analysis.
        This should be called after the discovery phase has completed.
        
        Args:
            batch_size: Number of URLs to process in each batch
            max_level: Maximum level to process
            
        Returns:
            Dictionary with analysis statistics
        """
        logger.info(f"Starting content analysis phase with batch_size={batch_size}")
        
        results = {
            "processed_by_level": {},
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
        
        if not self.url_storage:
            logger.error("URL storage manager not available for analysis phase")
            return results
        
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
                    origin_url = pending_url_data.get('origin_url', '')
                    resource_id = pending_url_data.get('resource_id', '')
                    
                    # Check if already processed
                    if self.url_storage.url_is_processed(url_to_process):
                        logger.info(f"URL already processed: {url_to_process}, skipping analysis")
                        continue
                        
                    # Process this URL for content analysis
                    logger.info(f"Analyzing content for URL: {url_to_process}")
                    
                    try:
                        # Fetch content for analysis
                        success, content_dict = self.fetch_content_for_analysis(
                            url=url_to_process,
                            resource_id=resource_id
                        )
                        
                        if success and content_dict:
                            # Analyze the content
                            analysis_results = self.analyze_page_content(content_dict, url_to_process)
                            
                            # Track processing success
                            success_count += 1
                            processed_count += 1
                            
                            # Update URL processing count for vector store generation
                            self.urls_processed_since_last_build += 1
                            
                            logger.info(f"Successfully analyzed content for: {url_to_process}")
                            
                        else:
                            failed_count += 1
                            logger.warning(f"Failed to fetch/analyze content for: {url_to_process}")
                    
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error analyzing content for {url_to_process}: {e}")
            
            results["processed_by_level"][level] = processed_count
            results["total_processed"] += processed_count
            results["successful"] += success_count
            results["failed"] += failed_count
            
            logger.info(f"Level {level} analysis: {processed_count} URLs processed")
        
        logger.info(f"Content analysis phase completed: {results['total_processed']} URLs processed")
        return results

    def mark_url_as_processed(self, url, depth=1, origin=""):
        """
        Mark a URL as processed after its content has been successfully analyzed.
        This method can be called by external components like MainProcessor.
        
        Args:
            url: The URL to mark as processed
            depth: The depth level of the URL (0=main, 1=first level, etc.)
            origin: The parent URL that led to this URL
        """
        if self.url_storage:
            logger.info(f"Marking URL as processed after successful content analysis: {url}")
            self.url_storage.save_processed_url(url, depth=depth, origin=origin)
        else:
            logger.warning("URL storage manager not available to mark URL as processed")

    def should_build_vector_store(self):
        """
        Check if we should build the vector store based on processed URL count.
        
        Returns:
            True if vector store should be built, False otherwise
        """
        return self.urls_processed_since_last_build >= self.url_build_threshold

    def reset_url_counter(self):
        """
        Reset the URL counter after vector store has been built.
        """
        self.urls_processed_since_last_build = 0
        logger.info("Reset URL counter for vector store generation")

    def extract_page_data(self, html_content: str, url: str = None) -> Dict[str, Any]:
        """
        Extract structured data from HTML content.
        
        Args:
            html_content: Raw HTML content
            url: Optional URL for context
            
        Returns:
            Dictionary with extracted data
        """
        extracted_data = {
            "url": url,
            "title": "",
            "text_content": "",
            "metadata": {},
            "links": [],
            "images": []
        }
        
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            if soup.title:
                extracted_data["title"] = soup.title.string.strip()
            
            # Extract text content (remove scripts and styles)
            for script in soup(["script", "style"]):
                script.decompose()
            
            extracted_data["text_content"] = soup.get_text()
            
            # Extract metadata
            extracted_data["metadata"] = {
                "description": "",
                "keywords": "",
                "author": ""
            }
            
            # Look for meta tags
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc:
                extracted_data["metadata"]["description"] = meta_desc.get("content", "")
            
            meta_keywords = soup.find("meta", attrs={"name": "keywords"})
            if meta_keywords:
                extracted_data["metadata"]["keywords"] = meta_keywords.get("content", "")
            
            meta_author = soup.find("meta", attrs={"name": "author"})
            if meta_author:
                extracted_data["metadata"]["author"] = meta_author.get("content", "")
            
            # Extract links
            for link in soup.find_all('a', href=True):
                extracted_data["links"].append({
                    "text": link.get_text().strip(),
                    "href": link['href']
                })
            
            # Extract images
            for img in soup.find_all('img', src=True):
                extracted_data["images"].append({
                    "alt": img.get('alt', ''),
                    "src": img['src']
                })
            
            logger.debug(f"Extracted data from {url}: {len(extracted_data['text_content'])} characters")
            
        except Exception as e:
            logger.error(f"Error extracting page data from {url}: {e}")
            extracted_data["error"] = str(e)
        
        return extracted_data

    def set_enhanced_url_storage(self, enhanced_storage):
        """
        Set enhanced URL storage manager for better URL tracking.
        
        Args:
            enhanced_storage: Enhanced URL storage manager instance
        """
        self.enhanced_url_storage = enhanced_storage
        logger.info("Enhanced URL storage manager set for ContentAnalyzer")