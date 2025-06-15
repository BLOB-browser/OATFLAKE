#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from typing import Tuple, Dict, Any, List, Set
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests

from scripts.analysis.url_processor import (
    clean_url, get_base_domain_and_url
)

logger = logging.getLogger(__name__)

class URLDiscoveryEngine:
    """Dedicated engine for discovering URLs from web pages"""
    
    def __init__(self, url_storage_manager, web_fetcher):
        self.url_storage = url_storage_manager
        self.web_fetcher = web_fetcher
        self.enhanced_url_storage = None
        self.discovery_only_mode = False  # When True, focus on URL discovery only
        self.allow_processed_url_discovery = False  # When True, allow discovery from processed URLs when no pending URLs
        self.use_enhanced_storage_primary = True  # When True, use enhanced storage as primary system

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
                  
            # In discovery mode, always proceed regardless of processed status
            # This ensures we discover all possible URLs even from processed pages
            if self.discovery_only_mode:
                logger.info(f"Discovery-only mode: proceeding with URL {url} without checking processed status")
            else:
                # Only in content analysis mode do we check if URL is processed
                if self.url_storage.url_is_processed(url):
                    logger.info(f"Main URL {url} already processed, skipping fetch")
                    return False, {"error": "URL already processed"}
            
            # Store the base domain for building absolute URLs
            base_domain, base_path, base_url = get_base_domain_and_url(url)
            
            # Create headers using WebFetcher
            headers = self.web_fetcher.create_headers()
              
            # Get the main page using WebFetcher
            success, main_html = self.web_fetcher.fetch_page(url, headers)
            
            # Check for server errors that should trigger fallback discovery
            if not success and self.web_fetcher.is_server_error(main_html):
                logger.warning(f"Main page failed with server error: {main_html}")
                logger.info(f"Attempting fallback discovery for {url}")
                
                # Attempt fallback discovery when main page fails with server errors
                fallback_result = self._attempt_fallback_discovery(url, base_domain, headers, max_depth, resource_id)
                
                if fallback_result and fallback_result.get("success"):
                    logger.info(f"Fallback discovery successful for {url}")
                    
                    # Process URLs discovered through fallback to ensure they're properly saved 
                    # in the pending_urls system for later analysis
                    if resource_id:
                        discovered_urls = fallback_result.get("additional_urls", [])
                        if discovered_urls:
                            logger.info(f"Saving {len(discovered_urls)} URLs discovered through fallback")
                            for discovered_url in discovered_urls:
                                # Use centralized method for both traditional and enhanced storage
                                self._save_discovered_url(
                                    url=discovered_url,
                                    depth=1,  # Fallback discoveries are always level 1
                                    origin_url=url,
                                    resource_id=resource_id
                                )
                    
                    return True, fallback_result
                else:
                    logger.warning(f"Both main page and fallback discovery failed for {url}")
                    return False, {"error": f"Main page server error and fallback failed: {main_html}"}
            
            if not success:
                return False, {"error": "Failed to fetch main page"}
            
            # Parse the page to find important links
            soup = BeautifulSoup(main_html, 'html.parser')
            
            # Track visited URLs to avoid duplicates
            visited_urls = {url}
            
            # Initialize a dictionary to store URLs by level
            urls_by_level = {level: [] for level in range(1, max_depth + 1)}
            
            # FIXED: Use the actual max_depth instead of hardcoded values
            if self.discovery_only_mode:
                # In discovery-only mode, use the full max_depth to ensure complete discovery
                discovery_depth = max_depth
                logger.info(f"Discovery-only mode: Using full max_depth of {discovery_depth} levels")
            else:
                # In analysis mode, respect the original limits
                discovery_depth = min(max_depth, discover_only_level)
                logger.info(f"Analysis mode: URL discovery set to level {discovery_depth} (max_depth={max_depth}, discover_only_level={discover_only_level})")
            
            # Always use the simplified breadth-first crawling
            logger.info(f"Using simplified breadth-first discovery directly from the resource")
            if resource_id:
                logger.info(f"Resource ID provided for URL association: {resource_id}")
            
            # Use our new simplified breadth-first discovery that doesn't use pending URLs
            self._discover_urls_breadth_first(
                root_soup=soup,
                url=url,
                base_domain=base_domain,
                headers=headers,
                max_depth=discovery_depth,  # Use our calculated discovery depth
                visited_urls=visited_urls,
                urls_by_level=urls_by_level,
                force_reprocess=False,
                resource_id=resource_id
            )
            
            # Flatten all discovered URLs for backward compatibility
            all_additional_urls = []
            for level in range(1, discovery_depth + 1):
                if level in urls_by_level:
                    all_additional_urls.extend(urls_by_level[level])
            
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
                "breadth_first": True
            }
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return False, {"error": str(e)}
            
            # Parse the page to find important links
            soup = BeautifulSoup(main_html, 'html.parser')
            
            # Track visited URLs to avoid duplicates
            visited_urls = {url}
            
            # Initialize a dictionary to store URLs by level
            urls_by_level = {level: [] for level in range(1, max_depth + 1)}
            
            # FIXED: Use the actual max_depth instead of hardcoded values
            if self.discovery_only_mode:
                # In discovery-only mode, use the full max_depth to ensure complete discovery
                discovery_depth = max_depth
                logger.info(f"Discovery-only mode: Using full max_depth of {discovery_depth} levels")
            else:
                # In analysis mode, respect the original limits
                discovery_depth = min(max_depth, discover_only_level)
                logger.info(f"Analysis mode: URL discovery set to level {discovery_depth} (max_depth={max_depth}, discover_only_level={discover_only_level})")
            
            # Always use the simplified breadth-first crawling
            logger.info(f"Using simplified breadth-first discovery directly from the resource")
            if resource_id:
                logger.info(f"Resource ID provided for URL association: {resource_id}")
            
            # Use our new simplified breadth-first discovery that doesn't use pending URLs
            self._discover_urls_breadth_first(
                root_soup=soup,
                url=url,
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
        
    def _discover_urls_breadth_first(self, root_soup, url, base_domain, headers, max_depth=4, visited_urls=None, urls_by_level=None, force_reprocess=False, resource_id=None):
        """
        Discovers URLs in a breadth-first manner, keeping track of depth levels.
        
        Args:
            root_soup: BeautifulSoup object of the current page
            url: URL of the current page
            base_domain: Base domain for resolving URLs
            headers: HTTP headers for requests
            max_depth: Maximum depth to discover URLs at
            visited_urls: Set of visited URLs
            urls_by_level: Dictionary to store URLs by level
            force_reprocess: Whether to process already processed URLs
            resource_id: Resource ID to associate with discovered URLs
        """
        if visited_urls is None:
            visited_urls = {url}  # Start with root URL
        if urls_by_level is None:
            urls_by_level = {level: [] for level in range(1, max_depth + 1)}
            
        logger.info(f"Starting breadth-first URL discovery from {url} (max_depth={max_depth})")
        
        # Process level 1 first
        found_urls = self.extract_all_page_links(root_soup, base_domain, url)
        level1_urls = []
        
        # Track processed URLs to avoid duplicates
        processed_this_run = set()
          # Track URL IDs for relationship mapping
        url_id_mapping = {}  # url -> url_id mapping
        main_url_id = None
        
        # Get the main URL ID if this is the root URL
        if self.enhanced_url_storage:
            try:
                # Check if the main URL already has an ID
                existing_urls = self.enhanced_url_storage.get_enhanced_pending_urls(resource_id=resource_id)
                for url_data in existing_urls:
                    if url_data.get('origin_url') == url and int(url_data.get('depth', 0)) == 0:
                        main_url_id = url_data.get('url_id')
                        url_id_mapping[url] = main_url_id
                        break
            except Exception as e:
                logger.warning(f"Error checking for existing main URL ID: {e}")

        # Process level 1 URLs
        for found_url in found_urls:
            if found_url not in visited_urls and found_url not in processed_this_run and self._is_valid_url(found_url):
                level1_urls.append(found_url)
                visited_urls.add(found_url)
                processed_this_run.add(found_url)
                
                # Save level 1 URLs with proper depth - ONLY if we have a valid resource ID
                if resource_id and resource_id != "0":
                    logger.info(f"💾 LEVEL1: Attempting to save URL with resource_id='{resource_id}': {found_url}")
                    url_id = self._save_discovered_url(
                        url=found_url,
                        depth=1,
                        origin_url=url,
                        resource_id=resource_id,
                        main_resource_url=url,
                        origin_url_id=main_url_id
                    )
                    if url_id:
                        url_id_mapping[found_url] = url_id
                        logger.info(f"✅ LEVEL1: Successfully saved URL with ID: {url_id[:8]}... for {found_url}")
                    else:
                        logger.error(f"❌ LEVEL1: Failed to save URL (no ID returned): {found_url}")
                else:
                    logger.warning(f"⚠️ LEVEL1: No valid resource_id (current='{resource_id}'), trying inheritance for: {found_url}")
                    # Try to find proper resource ID from the current URL chain
                    inherited_resource_id = self.url_storage.get_resource_id_for_url(url)
                    if inherited_resource_id and inherited_resource_id != "0":
                        logger.info(f"💾 LEVEL1: Using inherited resource_id='{inherited_resource_id}' for: {found_url}")
                        url_id = self._save_discovered_url(
                            url=found_url,
                            depth=1,
                            origin_url=url,
                            resource_id=inherited_resource_id,
                            main_resource_url=url,
                            origin_url_id=main_url_id
                        )
                        if url_id:
                            url_id_mapping[found_url] = url_id
                            logger.info(f"✅ LEVEL1: Successfully saved URL with inherited resource: {found_url}")
                        else:
                            logger.error(f"❌ LEVEL1: Failed to save URL with inherited resource: {found_url}")
                    else:
                        # Don't save URLs without a valid resource ID to prevent defaulting to resource ID "0"
                        logger.warning(f"⚠️ LEVEL1: Skipping URL - no valid resource ID found (inherited='{inherited_resource_id}'): {found_url}")

        # Store level 1 URLs
        urls_by_level[1].extend(level1_urls)
        logger.info(f"Found {len(level1_urls)} URLs at level 1")

        # Now process remaining levels in breadth-first order
        for level in range(2, max_depth + 1):
            previous_level_urls = urls_by_level[level - 1]
            current_level_urls = []
            
            logger.info(f"Processing {len(previous_level_urls)} URLs from level {level-1}")
            
            # Process each URL from the previous level
            for parent_url in previous_level_urls:
                try:
                    if parent_url in processed_this_run:
                        continue
                        
                    # Try to get content from parent URL
                    success, parent_html = self.web_fetcher.fetch_page(parent_url, headers)
                    
                    if not success or not parent_html:
                        logger.warning(f"Failed to fetch content from {parent_url} for level {level}")
                        continue
                    
                    # Parse the parent page
                    parent_soup = BeautifulSoup(parent_html, 'html.parser')
                    
                    # Extract URLs from this page
                    found_urls = self.extract_all_page_links(parent_soup, base_domain, parent_url)
                    
                    # Determine the effective resource ID for this branch
                    effective_resource_id = resource_id
                    if not effective_resource_id:
                        effective_resource_id = self.url_storage.get_resource_id_for_url(parent_url)
                      # Process and save discovered URLs
                    for found_url in found_urls:
                        if found_url not in visited_urls and found_url not in processed_this_run and self._is_valid_url(found_url):
                            current_level_urls.append(found_url)
                            visited_urls.add(found_url)
                            processed_this_run.add(found_url)
                            
                            # Save with proper depth level - ONLY if we have a valid resource ID
                            if effective_resource_id and effective_resource_id != "0":
                                # Get the parent URL ID for relationship tracking
                                parent_url_id = url_id_mapping.get(parent_url, "")
                                
                                # Determine main resource URL
                                main_resource_url = url if level == 2 else self._get_main_resource_url(effective_resource_id)
                                if not main_resource_url:
                                    main_resource_url = url  # Fallback
                                
                                url_id = self._save_discovered_url(
                                    url=found_url,
                                    depth=level,
                                    origin_url=parent_url,
                                    resource_id=effective_resource_id,
                                    main_resource_url=main_resource_url,
                                    origin_url_id=parent_url_id
                                )
                                if url_id:
                                    url_id_mapping[found_url] = url_id
                                logger.debug(f"Saved level {level} URL with resource: {found_url}")
                            else:
                                # Don't save URLs without a valid resource ID to prevent defaulting to resource ID "0"
                                logger.debug(f"Skipping level {level} URL - no valid resource ID found: {found_url}")

                except Exception as e:
                    logger.warning(f"Error processing URL {parent_url} at level {level-1}: {e}")

            # Store URLs for this level
            urls_by_level[level].extend(current_level_urls)
            logger.info(f"Found {len(current_level_urls)} URLs at level {level}")
              # If we found no URLs at this level, we can stop early
            if not current_level_urls:
                logger.info(f"No URLs found at level {level}, stopping discovery")
                break

        # Print enhanced discovery summary if available
        if self.enhanced_url_storage and resource_id:
            try:
                logger.info(f"Enhanced URL discovery summary for resource {resource_id}:")
                self.enhanced_url_storage.print_discovery_summary(resource_id)
            except Exception as e:
                logger.warning(f"Error printing enhanced discovery summary: {e}")

        return urls_by_level

    def _discover_urls_breadth_first_simplified(self, url: str, max_depth: int = 4, resource_id: str = None) -> tuple[bool, dict]:
        """
        Simplified breadth-first URL discovery with minimal redundant checks.
        
        Args:
            url: Starting URL
            max_depth: Maximum depth to discover
            resource_id: Resource ID for discovered URLs
            
        Returns:
            Tuple of (success, data)
        """
        logger.info(f"=== Simplified breadth-first discovery for: {url} ===")
        
        try:
            # Get base domain for URL resolution
            base_domain, _, base_url = get_base_domain_and_url(url)
            
            # Track visited URLs and initialize queues
            visited_urls = {url}
            current_level_urls = [url]
            
            # Process each level
            for level in range(1, max_depth + 1):
                if not current_level_urls:
                    logger.info(f"No URLs to process at level {level}, stopping")
                    break
                    
                next_level_urls = []
                logger.info(f"Processing {len(current_level_urls)} URLs at level {level}")
                
                for parent_url in current_level_urls:
                    try:
                        # Fetch page content
                        success, parent_html = self.web_fetcher.fetch_page(parent_url)
                        if not success or not parent_html:
                            logger.debug(f"Failed to fetch: {parent_url}")
                            continue
                        
                        # Parse and extract links
                        soup = BeautifulSoup(parent_html, 'html.parser')
                        found_urls = self.extract_all_page_links(soup, base_domain, parent_url)
                        
                        # Process found URLs
                        for found_url in found_urls:
                            if (found_url not in visited_urls and 
                                self._is_valid_url(found_url) and
                                found_url.startswith('http')):
                                
                                # Add to next level and mark as visited
                                next_level_urls.append(found_url)
                                visited_urls.add(found_url)
                                
                                # Save URL directly to pending queue
                                self._save_discovered_url(
                                    url=found_url,
                                    depth=level + 1,  # Save at next level
                                    origin_url=parent_url,
                                    resource_id=resource_id
                                )
                                
                    except Exception as e:
                        logger.debug(f"Error processing {parent_url}: {e}")
                        continue
                
                # Move to next level
                current_level_urls = next_level_urls
                logger.info(f"Found {len(current_level_urls)} URLs for level {level + 1}")
            
            logger.info(f"Simplified discovery complete for {url}")
            return True, {"total_urls": len(visited_urls)}
            
        except Exception as e:
            logger.error(f"Error in simplified breadth-first discovery: {e}")
            return False, {"error": str(e)}

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
        Simplified direct discovery flow - streamlined URL discovery without redundant checks.
        
        Args:
            urls: List of URLs to process for discovery
            max_depth: Maximum discovery depth
            force_reprocess: If True, forces reprocessing of already processed URLs
            resource_ids: Dictionary mapping URLs to resource IDs
            
        Returns:
            Dictionary with discovery statistics
        """
        logger.info(f"=== SIMPLIFIED DIRECT DISCOVERY FLOW ===")
        logger.info(f"Processing {len(urls)} URLs with max_depth={max_depth}")
        
        # Store original discovery mode to restore later
        original_discovery_mode = self.discovery_only_mode
        self.discovery_only_mode = True
        
        # Enable discovery mode in URL storage
        if hasattr(self.url_storage, 'set_discovery_mode'):
            self.url_storage.set_discovery_mode(True)
        
        # Initialize simplified results tracking
        results = {
            "total_urls": len(urls),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "discovered_by_level": {},
            "total_discovered": 0
        }
        
        # Get initial counts for comparison
        initial_pending = {}
        for level in range(1, max_depth + 1):
            pending_urls = self.url_storage.get_pending_urls(depth=level)
            initial_pending[level] = len(pending_urls) if pending_urls else 0
        
        try:
            # Direct discovery loop - simplified processing
            for idx, url in enumerate(urls):
                resource_id = resource_ids.get(url) if resource_ids else None
                logger.info(f"[{idx+1}/{len(urls)}] Discovering URLs from: {url}")
                
                # Skip discovery check if force_reprocess is False (simplified logic)
                if not force_reprocess and self.url_storage.get_discovery_status(url):
                    logger.info(f"URL already discovered, skipping: {url}")
                    results["skipped"] += 1
                    continue
                
                # Direct breadth-first discovery - simplified call
                success, _ = self._discover_urls_breadth_first_simplified(
                    url, 
                    max_depth=max_depth,
                    resource_id=resource_id
                )
                
                if success:
                    results["successful"] += 1
                    # Mark discovery as complete for this URL
                    self.url_storage.mark_discovery_complete(url)
                else:
                    results["failed"] += 1
                    logger.warning(f"Failed to discover URLs from: {url}")
            
            # Calculate final counts and discovered URLs
            for level in range(1, max_depth + 1):
                pending_urls = self.url_storage.get_pending_urls(depth=level)
                final_count = len(pending_urls) if pending_urls else 0
                initial_count = initial_pending[level]
                
                results["discovered_by_level"][level] = final_count
                newly_discovered = final_count - initial_count
                results["total_discovered"] += newly_discovered
                
                logger.info(f"Level {level}: {initial_count} → {final_count} URLs ({newly_discovered:+d} new)")
            
            logger.info(f"=== DISCOVERY COMPLETED ===")
            logger.info(f"Processed: {results['successful']}/{len(urls)} URLs successfully")
            logger.info(f"Total new URLs discovered: {results['total_discovered']}")
            
        finally:
            # Restore original modes
            self.discovery_only_mode = original_discovery_mode
            if hasattr(self.url_storage, 'set_discovery_mode'):
                self.url_storage.set_discovery_mode(False)
        
        return results
    
    def _attempt_fallback_discovery(self, original_url: str, base_domain: str, headers: Dict[str, str], 
                                   max_depth: int = 4, resource_id: str = None) -> Dict[str, Any]:
        """
        Attempt fallback discovery when the main page fails with server errors.
        Uses multiple strategies to continue URL exploration despite main page failures.
        
        Args:
            original_url: The original URL that failed
            base_domain: Base domain for URL resolution
            headers: HTTP headers for requests
            max_depth: Maximum depth for discovery
            resource_id: Optional resource ID for URL association
            
        Returns:
            Dictionary with fallback discovery results or None if all strategies fail
        """
        logger.info(f"=== FALLBACK DISCOVERY STARTED for {original_url} ===")
        
        fallback_result = {
            "success": False,
            "main_html": "",  # Empty since main page failed
            "additional_urls": [],
            "base_url": original_url,
            "base_domain": base_domain,
            "headers": headers,
            "visited_urls": {original_url},
            "urls_by_level": {level: [] for level in range(1, max_depth + 1)},
            "breadth_first": True,
            "fallback_used": True,
            "fallback_strategies": []
        }
        
        try:
            # Strategy 1: Find alternative URLs from same domain in resources.csv
            logger.info("Strategy 1: Finding alternative URLs from same domain in resources.csv")
            alternative_urls = self._find_alternative_domain_urls(base_domain)
            if alternative_urls:
                logger.info(f"Found {len(alternative_urls)} alternative URLs from same domain")
                fallback_result["fallback_strategies"].append("alternative_domain_urls")
                
                for alt_url in alternative_urls[:3]:  # Try up to 3 alternative URLs
                    if self._attempt_discovery_from_url(alt_url, headers, max_depth, fallback_result, resource_id):
                        logger.info(f"Successful discovery from alternative URL: {alt_url}")
                        break
            
            # Strategy 2: Use processed URLs from same domain as starting points
            if not fallback_result["success"]:
                logger.info("Strategy 2: Using processed URLs from same domain as starting points")
                processed_domain_urls = self._find_processed_domain_urls(base_domain)
                if processed_domain_urls:
                    logger.info(f"Found {len(processed_domain_urls)} processed URLs from same domain")
                    fallback_result["fallback_strategies"].append("processed_domain_urls")
                    
                    for proc_url in processed_domain_urls[:2]:  # Try up to 2 processed URLs
                        if self._attempt_discovery_from_url(proc_url, headers, max_depth, fallback_result, resource_id):
                            logger.info(f"Successful discovery from processed URL: {proc_url}")
                            break
            
            # Strategy 3: Try common URL patterns
            if not fallback_result["success"]:
                logger.info("Strategy 3: Trying common URL patterns")
                common_patterns = self._generate_common_url_patterns(original_url)
                if common_patterns:
                    logger.info(f"Trying {len(common_patterns)} common URL patterns")
                    fallback_result["fallback_strategies"].append("common_url_patterns")
                    
                    for pattern_url in common_patterns:
                        if self._attempt_discovery_from_url(pattern_url, headers, max_depth, fallback_result, resource_id):
                            logger.info(f"Successful discovery from common pattern: {pattern_url}")
                            break
            
            # Strategy 4: Use existing pending URLs from same domain
            logger.info("Strategy 4: Using existing pending URLs from same domain")
            pending_domain_urls = self._find_pending_domain_urls(base_domain)
            if pending_domain_urls:
                logger.info(f"Found {len(pending_domain_urls)} pending URLs from same domain")
                fallback_result["fallback_strategies"].append("pending_domain_urls")
                
                # Add these to our discovery results since they're already known good URLs
                for pending_url_data in pending_domain_urls[:10]:  # Take up to 10 pending URLs
                    try:
                        pending_url = pending_url_data.get('origin_url')
                        depth = pending_url_data.get('depth', 1)
                        if pending_url and depth <= max_depth:
                            fallback_result["additional_urls"].append(pending_url)
                            if depth not in fallback_result["urls_by_level"]:
                                fallback_result["urls_by_level"][depth] = []
                            fallback_result["urls_by_level"][depth].append(pending_url)
                    except Exception as e:
                        logger.warning(f"Error processing pending URL: {e}")
                        continue
            
            # Calculate total URLs discovered through fallback
            fallback_result["total_urls"] = len(fallback_result["additional_urls"])
            
            # Mark as successful if we found any URLs through fallback strategies
            if fallback_result["total_urls"] > 0:
                fallback_result["success"] = True
                logger.info(f"=== FALLBACK DISCOVERY SUCCESSFUL: {fallback_result['total_urls']} URLs found ===")
                logger.info(f"Fallback strategies used: {', '.join(fallback_result['fallback_strategies'])}")
            else:
                logger.warning(f"=== FALLBACK DISCOVERY FAILED: No URLs found through any strategy ===")
        
        except Exception as e:
            logger.error(f"Error during fallback discovery: {e}")
            # Add error information to the result
            fallback_result["error"] = str(e)
            fallback_result["fallback_strategies"].append("error_occurred")

        # Even if an exception occurred, still return any useful results we might have found
        return fallback_result if fallback_result.get("success", False) else None

    def _find_alternative_domain_urls(self, domain: str) -> List[str]:
        """Find alternative URLs from the same domain in resources.csv"""
        try:
            from utils.config import get_data_path
            # Fixed: Use correct path construction for resources file
            data_path = get_data_path()
            resources_file = os.path.join(data_path, "resources.csv")
            
            if not os.path.exists(resources_file):
                logger.warning(f"Resources file not found: {resources_file}")
                return []
            
            alternative_urls = []
            
            # Read resources.csv and find URLs from the same domain
            import csv
            with open(resources_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    url = row.get('origin_url', '').strip()
                    if url and domain in url:
                        alternative_urls.append(url)
            
            logger.info(f"Found {len(alternative_urls)} alternative URLs for domain {domain}")
            return alternative_urls
            
        except Exception as e:
            logger.warning(f"Error finding alternative domain URLs: {e}")
            return []

    def _find_processed_domain_urls(self, domain: str) -> List[str]:
        """Find processed URLs from the same domain"""
        try:
            processed_urls = self.url_storage.get_processed_urls()
            domain_urls = []
            
            for url in processed_urls:
                if domain in url:
                    domain_urls.append(url)
            
            logger.info(f"Found {len(domain_urls)} processed URLs for domain {domain}")
            return domain_urls
            
        except Exception as e:
            logger.warning(f"Error finding processed domain URLs: {e}")
            return []

    def _find_pending_domain_urls(self, domain: str) -> List[Dict[str, Any]]:
        """Find pending URLs from the same domain"""
        try:
            all_pending = []
            
            # Get pending URLs from all levels
            for depth in range(1, 5):  # Check levels 1-4
                pending_urls = self.url_storage.get_pending_urls(depth=depth)
                if pending_urls:
                    all_pending.extend(pending_urls)
            
            # Filter by domain
            domain_urls = []
            for url_data in all_pending:
                url = url_data.get('origin_url', '')
                if domain in url:
                    domain_urls.append(url_data)
            
            logger.info(f"Found {len(domain_urls)} pending URLs for domain {domain}")
            return domain_urls
            
        except Exception as e:
            logger.warning(f"Error finding pending domain URLs: {e}")
            return []

    def _generate_common_url_patterns(self, original_url: str) -> List[str]:
        """Generate common URL patterns to try as fallbacks"""
        try:
            from urllib.parse import urlparse
            
            parsed = urlparse(original_url)
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            
            patterns = [
                f"{base_url}/index.html",
                f"{base_url}/home",
                f"{base_url}/projects",
                f"{base_url}/blog",
                f"{base_url}/about",
                f"{base_url}/news",
                f"{base_url}/events",
                f"{base_url}/resources",
                f"{base_url}/documentation",
                f"{base_url}/docs"
            ]
            
            # Only return patterns that are different from the original URL
            return [p for p in patterns if p != original_url]
        except Exception as e:
            logger.warning(f"Error generating common URL patterns: {e}")
            return []

    def _attempt_discovery_from_url(self, url: str, headers: Dict[str, str], max_depth: int, 
                                   fallback_result: Dict[str, Any], resource_id: str = None) -> bool:
        """
        Attempt to discover URLs from a specific fallback URL
        
        Args:
            url: The URL to attempt discovery from
            headers: HTTP headers for requests
            max_depth: Maximum discovery depth
            fallback_result: Dictionary to store results in
            resource_id: Optional resource ID for URL association
            
        Returns:
            True if discovery was successful, False otherwise
        """
        try:
            logger.info(f"Attempting discovery from fallback URL: {url}")
            
            # Try to fetch this URL
            success, html_content = self.web_fetcher.fetch_page(url, headers)
            
            if not success:
                logger.info(f"Failed to fetch fallback URL: {url}")
                return False
            
            # Parse the content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract links using existing methods
            from scripts.analysis.url_processor import (
                extract_navigation_links, extract_project_links, extract_detail_links,
                process_links, get_base_domain_and_url
            )
            
            base_domain, base_path, base_url = get_base_domain_and_url(url)
            
            # Extract different types of links
            nav_links = extract_navigation_links(soup)
            project_links = extract_project_links(soup)
            detail_links = extract_detail_links(soup)
            
            all_links = nav_links + project_links + detail_links
            found_urls = process_links(all_links, base_domain, url, fallback_result["visited_urls"])
            
            if found_urls:
                logger.info(f"Discovered {len(found_urls)} URLs from fallback URL: {url}")
                
                # Add discovered URLs to results
                for discovered_url in found_urls:
                    if discovered_url not in fallback_result["additional_urls"]:
                        fallback_result["additional_urls"].append(discovered_url)
                        fallback_result["visited_urls"].add(discovered_url)
                        
                        # Add to level 1 by default for fallback discoveries
                        if 1 not in fallback_result["urls_by_level"]:
                            fallback_result["urls_by_level"][1] = []
                        fallback_result["urls_by_level"][1].append(discovered_url)
                          # Save to pending URLs for later processing using centralized method
                        if resource_id:
                            # Use centralized storage method for both traditional and enhanced storage
                            self._save_discovered_url(
                                url=discovered_url,
                                depth=1,
                                origin_url=url,
                                resource_id=resource_id
                            )
                        else:
                            # Don't save URLs without valid resource ID to prevent defaulting to resource ID "0"
                            logger.debug(f"Skipping fallback URL - no valid resource ID: {discovered_url}")
                
                return True
            else:
                logger.info(f"No URLs discovered from fallback URL: {url}")
                return False
                
        except Exception as e:
            logger.warning(f"Error attempting discovery from fallback URL {url}: {e}")
            return False

    def set_enhanced_url_storage(self, enhanced_storage):
        """Set the enhanced URL storage manager for use with URL discovery
        
        Args:
            enhanced_storage: The enhanced URL storage manager
        """
        logger.info("Setting enhanced URL storage manager for URL discovery engine")
        self.enhanced_url_storage = enhanced_storage

    def _save_discovered_url(self, url: str, depth: int, origin_url: str, resource_id: str, 
                            main_resource_url: str = None, origin_url_id: str = None) -> str:
        """
        Save a discovered URL using the enhanced storage system as primary, with traditional storage as fallback.
        
        Args:
            url: The URL to save
            depth: The depth level of the URL
            origin_url: The URL that led to this URL's discovery
            resource_id: The associated resource ID
            main_resource_url: The main resource URL (for level 1+ URLs)
            origin_url_id: The ID of the URL that led to this URL's discovery
            
        Returns:
            URL ID if enhanced storage is used, empty string otherwise
        """
        logger.info(f"🔍 _save_discovered_url called with: url={url}, depth={depth}, origin_url={origin_url}, resource_id={resource_id}")
        
        # Variable to track URL ID for relationship mapping
        url_id = ""
        
        # PRIMARY: Use enhanced URL storage as the main storage system
        if hasattr(self, 'enhanced_url_storage') and self.enhanced_url_storage:
            logger.info(f"📦 Enhanced storage available, attempting save...")
            try:
                # Use main_resource_url from parameter or derive from origin
                if not main_resource_url:
                    main_resource_url = origin_url if depth == 1 else self._get_main_resource_url(resource_id)
                    if not main_resource_url:
                        main_resource_url = origin_url  # Fallback to origin URL
                
                # Use the enhanced storage save method with full parameter set
                url_id = self.enhanced_url_storage.save_enhanced_pending_url(
                    url=url,
                    depth=depth,
                    origin_url=origin_url,
                    resource_id=resource_id,
                    main_resource_url=main_resource_url,
                    origin_url_id=origin_url_id
                )
                logger.debug(f"✅ PRIMARY: Saved URL with enhanced storage - ID: {url_id[:8]}...")
                
                # Also save to the URL storage manager's enhanced system for compatibility
                try:
                    self.url_storage.save_pending_url(
                        url=url,
                        depth=depth,
                        origin_url=origin_url,
                        resource_id=resource_id,
                        main_resource_url=main_resource_url,
                        origin_url_id=origin_url_id
                    )
                    logger.debug(f"✅ COMPATIBILITY: Also saved to URL storage enhanced system")
                except Exception as e:
                    logger.warning(f"⚠️ Could not save to URL storage enhanced system: {e}")
                
                return url_id
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to save URL with enhanced storage: {e}")
                # Try a simplified version on failure
                try:
                    url_id = self.enhanced_url_storage.save_enhanced_pending_url(
                        url=url,
                        depth=depth,
                        origin_url=origin_url,
                        resource_id=resource_id
                    )
                    logger.debug("✅ FALLBACK: Saved URL with simplified enhanced storage parameters")
                    return url_id
                except Exception as e2:
                    logger.error(f"❌ Critical failure in enhanced URL storage, falling back to traditional: {e2}")
        
        # FALLBACK: Use traditional storage only if enhanced storage fails or is unavailable
        logger.warning(f"⚠️ FALLBACK: Using traditional storage for URL: {url}")
        try:
            self.url_storage.save_pending_url(
                url=url, 
                depth=depth, 
                origin=origin_url, 
                resource_id=resource_id
            )
            
            # Also save direct resource association - use safe wrapper
            from scripts.analysis.url_storage_helper import save_url_resource_relationship
            save_url_resource_relationship(self.url_storage, url, resource_id)
            logger.debug(f"✅ FALLBACK: Saved level {depth} URL with traditional storage: {url}")
        except Exception as e:
            logger.error(f"❌ CRITICAL: Both enhanced and traditional storage failed for URL {url}: {e}")
        
        return url_id

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
        Check if a URL is valid for processing.
        
        Args:
            url: The URL to validate
            
        Returns:
            True if URL is valid, False otherwise
        """
        if not url or not isinstance(url, str):
            return False
            
        # Clean the URL first
        url = url.strip()
        
        # Basic format check
        if not url.startswith(('http://', 'https://')):
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

    def extract_all_page_links(self, soup, base_domain, parent_url):
        """
        Extract all relevant links from a page using the comprehensive link extraction methods.
        
        Args:
            soup: BeautifulSoup object of the parsed HTML
            base_domain: Base domain for resolving relative URLs
            parent_url: URL of the parent page
            
        Returns:
            List of discovered URLs
        """
        try:
            # Import link extraction functions
            from scripts.analysis.url_processor import (
                extract_navigation_links, extract_project_links, extract_detail_links,
                process_links
            )
            
            # Extract different types of links
            nav_links = extract_navigation_links(soup)
            project_links = extract_project_links(soup)
            detail_links = extract_detail_links(soup)
            
            # Combine all links
            all_links = nav_links + project_links + detail_links
            
            # Process the links to get clean URLs
            visited_set = set()  # Empty set since we're extracting all links
            found_urls = process_links(all_links, base_domain, parent_url, visited_set)
            
            logger.debug(f"Extracted {len(found_urls)} links from {parent_url}")
            return found_urls
            
        except Exception as e:
            logger.warning(f"Error extracting links from {parent_url}: {e}")
            return []
    
    def _safe_get_discovery_status(self, url: str) -> bool:
        """Safely check discovery status without raising errors when method doesn't exist"""
        try:
            if hasattr(self.url_storage, 'get_discovery_status'):
                return self.url_storage.get_discovery_status(url)
            else:
                # Method doesn't exist, assume URL needs discovery
                return False
        except Exception as e:
            logger.warning(f"Error checking discovery status: {e}")
            return False  # Default to needing discovery on error
    
    def _safe_set_discovery_status(self, url: str, status: bool) -> None:
        """Safely set discovery status without raising errors when method doesn't exist"""
        try:
            if hasattr(self.url_storage, 'set_discovery_status'):
                self.url_storage.set_discovery_status(url, status)
            # Silently do nothing if method doesn't exist
        except Exception as e:
            logger.warning(f"Error setting discovery status: {e}")
            # Continue without failing