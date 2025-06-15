#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import csv
from datetime import datetime
from typing import Set, Dict, Any, List

logger = logging.getLogger(__name__)

class URLStorageManager:
    """Handles storing and retrieving processed URLs"""
    
    def __init__(self, processed_urls_file: str):
        self.processed_urls_file = processed_urls_file
        self._processed_urls_cache = set()
        self.url_metadata = {}
        
        # For pending URLs that haven't been processed yet
        self.pending_urls_file = os.path.join(
            os.path.dirname(self.processed_urls_file), 
            "pending_urls.csv"
        )
        
        # For tracking URLs that have completed discovery (but not necessarily analysis)
        self.discovery_status_file = os.path.join(
            os.path.dirname(self.processed_urls_file),
            "discovery_status.csv"
        )
        
        # For tracking resource to URL relationships
        self.resource_urls_file = os.path.join(
            os.path.dirname(self.processed_urls_file),
            "resource_urls.csv"
        )
        
        # The main resources file (this is what we're adding/fixing)
        self.resources_file = os.path.join(
            os.path.dirname(self.processed_urls_file),
            "resources.csv"
        )
        
        # Cache for URLs that have completed discovery 
        self._discovery_completed_cache = set()
        
        # Cache for pending URLs - mapping of URL to metadata for quick checks
        self._pending_urls_cache = {}
        
        # Cache for resource URLs - maps resource ID to set of discovered URLs
        self._resource_urls_cache = {}
        
        # Flag to indicate if we should allow already processed URLs to be rediscovered
        self.allow_processed_url_rediscovery = False
        
        # Ensure the file directory exists
        os.makedirs(os.path.dirname(self.processed_urls_file), exist_ok=True)
          # Check if the processed_urls file exists and create it with header if it doesn't
        if not os.path.exists(self.processed_urls_file):
            try:
                with open(self.processed_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "depth", "origin", "timestamp", "error", "resource_id"])
                logger.info(f"Created new processed_urls.csv file with enhanced structure")
            except Exception as e:
                logger.error(f"Failed to create processed_urls.csv: {e}")
                # Fallback to current directory
                self.processed_urls_file = "processed_urls.csv"
                with open(self.processed_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "depth", "origin", "timestamp", "error", "resource_id"])
        
        # Initialize the pending URLs file if it doesn't exist
        if not os.path.exists(self.pending_urls_file):
            try:
                with open(self.pending_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Add attempt_count field to track how many times we've tried to process this URL
                    # Add resource_id field to track which resource this URL belongs to
                    writer.writerow(["url", "depth", "origin", "discovery_timestamp", "attempt_count", "resource_id"])
                logger.info(f"Created new pending_urls.csv file for URL queue with resource tracking")
            except Exception as e:
                logger.error(f"Failed to create pending_urls.csv: {e}")
                # Fallback to current directory
                self.pending_urls_file = "pending_urls.csv"
                with open(self.pending_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "depth", "origin", "discovery_timestamp", "attempt_count", "resource_id"])
        
        # Initialize the resource URLs file if it doesn't exist
        if not os.path.exists(self.resource_urls_file):
            try:
                with open(self.resource_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["resource_id", "url", "depth", "timestamp"])
                logger.info(f"Created new resource_urls.csv file for tracking URLs by resource")
            except Exception as e:
                logger.error(f"Failed to create resource_urls.csv: {e}")
                # Fallback to current directory
                self.resource_urls_file = "resource_urls.csv"
                with open(self.resource_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["resource_id", "url", "depth", "timestamp"])
        
        # Important: Load processed URLs during initialization
        logger.info("Loading processed URLs during initialization")
        self.load_processed_urls()
        
        # Load resource URL mappings
        self.load_resource_urls()
        
        # Load pending URLs cache for quick lookups
        self.load_pending_urls_cache()
    
    def load_processed_urls(self) -> Set[str]:
        """Load processed URLs from the CSV file.
        
        Returns:
            Set of processed URLs for quick lookup
        """
        processed_urls = set()
        self.url_metadata = {}  # Dictionary to store additional metadata about URLs
        
        if os.path.exists(self.processed_urls_file):
            try:
                logger.info(f"Loading processed URLs from: {self.processed_urls_file}")
                with open(self.processed_urls_file, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    header = next(reader, None)  # Skip header
                    
                    # Check if we're dealing with old format (just URL) or new format (with depth, origin, etc.)
                    is_new_format = header and len(header) >= 3 and "depth" in header
                    
                    for row in reader:
                        if row and len(row) > 0:  # Make sure row has data
                            url = row[0]
                            processed_urls.add(url)
                            
                            # If we have the enhanced format, store the metadata
                            if is_new_format and len(row) >= 3:
                                depth = int(row[1]) if row[1].isdigit() else 0
                                origin = row[2] if len(row) >= 3 else ""
                                timestamp = row[3] if len(row) >= 4 else ""
                                # Check for error flag (might be in column 5 for newer records)
                                error = False
                                if len(row) >= 5 and row[4] == "error":
                                    error = True
                                self.url_metadata[url] = {
                                    "depth": depth,
                                    "origin": origin,
                                    "timestamp": timestamp,
                                    "error": error
                                }
                logger.info(f"Loaded {len(processed_urls)} processed URLs")
            except Exception as e:
                logger.error(f"Error loading processed URLs: {e}")
                
        # Store the processed URLs in an instance attribute so we can add to it when saving
        self._processed_urls_cache = processed_urls
        logger.info(f"URL cache initialized with {len(self._processed_urls_cache)} URLs")
        return processed_urls
    
    def load_pending_urls_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load pending URLs into memory cache for quick access.
        
        Returns:
            Dictionary mapping URLs to their metadata
        """
        pending_urls = {}
        
        if os.path.exists(self.pending_urls_file):
            try:
                logger.info(f"Loading pending URLs into cache from: {self.pending_urls_file}")
                with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    header = next(reader, None)  # Skip header
                    
                    for row in reader:
                        if row and len(row) > 0:  # Make sure row has data
                            url = row[0]
                            depth = int(row[1]) if len(row) > 1 and row[1].isdigit() else 0
                            origin = row[2] if len(row) > 2 else ""
                            timestamp = row[3] if len(row) > 3 else ""
                            attempt_count = int(row[4]) if len(row) > 4 and row[4].isdigit() else 0
                            resource_id = row[5] if len(row) > 5 else ""                            # Skip if already processed, no point keeping in pending cache
                            if url in self._processed_urls_cache:
                                continue
                                
                            pending_urls[url] = {
                                "depth": depth,
                                "origin": origin,
                                "timestamp": timestamp,
                                "attempt_count": attempt_count,
                                "resource_id": resource_id
                            }
                logger.info(f"Loaded {len(pending_urls)} pending URLs into cache")
            except Exception as e:
                logger.error(f"Error loading pending URLs cache: {e}")
        self._pending_urls_cache = pending_urls
        return pending_urls
    
    def save_processed_url(self, url: str, depth: int = 0, origin: str = "", error: bool = False, resource_id: str = "") -> bool:
        """Save a URL as being processed (completed).
        
        Args:
            url: URL that has been processed
            depth: The crawl depth level of this URL (0=main, 1=first level, etc.)
            origin: The URL that led to this URL (empty for main URLs)
            error: If True, this URL experienced an error during processing
            resource_id: ID of the resource this URL belongs to
            
        Returns:
            Success flag
        """
        try:
            # Make sure the file and directory exist
            if not os.path.exists(self.processed_urls_file):
                os.makedirs(os.path.dirname(self.processed_urls_file), exist_ok=True)
              # Only append if not already processed
            if url in self._processed_urls_cache:
                return True
                
            error_status = "error" if error else ""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Get resource_id from pending URLs cache if not provided
            current_resource_id = resource_id
            if not current_resource_id and url in self._pending_urls_cache:
                current_resource_id = self._pending_urls_cache[url].get("resource_id", "")
            
            logger.info(f"Saving processed URL to {self.processed_urls_file}: {url} (depth={depth}, origin={origin or 'main'}, error={error}, resource_id={current_resource_id})")
                
            # Keep existing depth if URL was pending with lower depth
            if url in self._pending_urls_cache:
                cached_depth = self._pending_urls_cache[url].get("depth", depth)
                depth = min(depth, cached_depth)  # Use the lower depth
            
            with open(self.processed_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([url, depth, origin, timestamp, error_status, current_resource_id])
                file.flush()
                os.fsync(file.fileno())  # Force write to disk
            
            # Store in memory cache as well
            if not hasattr(self, 'url_metadata'):
                self.url_metadata = {}
                
            self.url_metadata[url] = {
                "depth": depth, 
                "origin": origin, 
                "timestamp": timestamp, 
                "error": error            }
            
            # Add to processed URLs cache
            self._processed_urls_cache.add(url)
            logger.info(f"Added URL to processed URLs cache (now has {len(self._processed_urls_cache)} URLs)")
        
            # Remove from pending URLs cache if it exists there
            if url in self._pending_urls_cache:
                del self._pending_urls_cache[url]
                logger.info(f"Removed URL from pending cache: {url}")
            
            # Also remove from pending URLs file
            self.remove_pending_url(url)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processed URL {url}: {e}")
            try:
                # Try a fallback to the local directory if the data path fails
                if self.processed_urls_file != "processed_urls.csv":
                    fallback_file = "processed_urls.csv"
                    with open(fallback_file, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([url, depth, origin, timestamp])
                        file.flush()
                    logger.info(f"Saved URL to fallback processed_urls.csv in project directory")
                    return True
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {fallback_error}")
            return False
            
    def url_is_processed(self, url: str) -> bool:
        """Check if a URL has already been processed
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL has been processed
        """
        # If cache is empty, try loading URLs first
        if not self._processed_urls_cache:
            logger.info("URL cache empty, reloading from file")
            self.load_processed_urls()
            
        return url in self._processed_urls_cache
    
    def url_is_pending(self, url: str) -> bool:
        """Check if a URL is in the pending queue
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is in the pending queue
        """
        # If not in memory cache, check if we need to reload the cache
        if not self._pending_urls_cache and os.path.exists(self.pending_urls_file):
            self.load_pending_urls_cache()
            
        # First check memory cache for efficiency
        if url in self._pending_urls_cache:
            return True
            
        # For safety, also check the file directly
        if os.path.exists(self.pending_urls_file):
            try:
                with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if row and row[0] == url:
                            # Add to cache for future lookups
                            depth = int(row[1]) if len(row) > 1 and row[1].isdigit() else 0
                            origin = row[2] if len(row) > 2 else ""
                            timestamp = row[3] if len(row) > 3 else ""
                            attempt_count = int(row[4]) if len(row) > 4 and row[4].isdigit() else 0
                            resource_id = row[5] if len(row) > 5 else ""
                            
                            self._pending_urls_cache[url] = {
                                "depth": depth,
                                "origin": origin,
                                "timestamp": timestamp,
                                "attempt_count": attempt_count,
                                "resource_id": resource_id
                            }
                            return True
            except Exception as e:
                logger.error(f"Error checking if URL is pending: {e}")
                
        return False
    
    def get_processed_urls(self) -> Set[str]:
        """Get all processed URLs
        
        Returns:
            Set of all processed URLs
        """
        return self._processed_urls_cache
    
    def get_urls_by_depth(self, depth: int) -> List[str]:
        """Get all processed URLs at a specific depth level
        
        Args:
            depth: The depth level to filter by (0=main, 1=first level, etc.)
            
        Returns:
            List of URLs at the specified depth
        """
        urls = []
        for url, metadata in self.url_metadata.items():
            if metadata.get("depth") == depth:
                urls.append(url)
        return urls
        
    def get_url_metadata(self, url: str) -> Dict[str, Any]:
        """Get metadata for a specific URL
        
        Args:
            url: The URL to get metadata for
            
        Returns:
            Dictionary with metadata (depth, origin, timestamp)
        """
        return self.url_metadata.get(url, {"depth": 0, "origin": "", "timestamp": ""})
        
    def get_depth_statistics(self) -> Dict[int, int]:
        """Get statistics about URL counts at each depth level
        
        Returns:
            Dictionary mapping depth levels to URL counts
        """
        stats = {}
        for url, metadata in self.url_metadata.items():
            depth = metadata.get("depth", 0)
            if depth not in stats:
                stats[depth] = 0
            stats[depth] += 1
        return stats
        
    def load_resource_urls(self) -> Dict[str, Set[str]]:
        """Load resource URL mappings from the CSV file.
        
        Returns:
            Dictionary mapping resource IDs to sets of URLs
        """
        resource_urls = {}
        
        # FIRST CHECK THE MAIN RESOURCES FILE
        # This is the fix - look for resources.csv first
        if os.path.exists(self.resources_file):
            try:
                logger.info(f"Loading resources from main resources file: {self.resources_file}")
                with open(self.resources_file, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    header = next(reader, None)  # Skip header
                      # Find the URL column index - check for origin_url first, then url
                    url_index = 0
                    title_index = 0
                    if header:
                        try:
                            # Use only origin_url field (universal schema)
                            if "origin_url" in header:
                                url_index = header.index("origin_url")
                                logger.info("Found origin_url column in resources.csv")
                            else:
                                logger.error("origin_url column not found in resources.csv - universal schema requires origin_url field")
                                return resource_urls
                            
                            title_index = header.index("title") if "title" in header else 0
                        except ValueError:
                            logger.error("Could not find required origin_url column in resources.csv")
                            return resource_urls
                            
                    resource_count = 0
                    for idx, row in enumerate(reader):
                        if row and len(row) > url_index and row[url_index]:
                            resource_id = str(idx)  # Use index as resource ID
                            url = row[url_index]
                            
                            # Create a resource entry if URL is valid
                            if url and "://" in url:
                                if resource_id not in resource_urls:
                                    resource_urls[resource_id] = set()
                                    
                                resource_urls[resource_id].add(url)
                                resource_count += 1
                    
                    logger.info(f"Loaded {resource_count} resource URLs from {len(resource_urls)} resources in main resources file")
                    
            except Exception as e:
                logger.error(f"Error loading resources from main file: {e}")
        
        # THEN ALSO CHECK THE RESOURCE_URLS FILE FOR BACKWARD COMPATIBILITY
        # This ensures we don't lose any existing mappings
        if os.path.exists(self.resource_urls_file):
            try:
                logger.info(f"Loading resource URL mappings from: {self.resource_urls_file}")
                with open(self.resource_urls_file, mode='r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    next(reader, None)  # Skip header
                    
                    for row in reader:
                        if row and len(row) >= 2:  # Make sure row has data
                            resource_id = row[0]
                            url = row[1]
                            
                            if resource_id not in resource_urls:
                                resource_urls[resource_id] = set()
                                
                            resource_urls[resource_id].add(url)
                            
                logger.info(f"Additionally loaded URL mappings for {len(resource_urls)} resources from resource_urls.csv")
            except Exception as e:
                logger.error(f"Error loading resource URL mappings: {e}")
                
        # Store the resource URLs in an instance attribute
        self._resource_urls_cache = resource_urls
        return resource_urls
        
    def save_resource_url(self, resource_id, origin_url, depth=0) -> bool:
        """Save a URL associated with a resource ID.
        
        Args:
            resource_id: ID of the resource this URL belongs to
            url: The URL to associate with the resource
            depth: The crawl depth level of this URL
            
        Returns:
            Success flag
        """
        try:
            # Check if this resource-URL association already exists
            if resource_id in self._resource_urls_cache and origin_url in self._resource_urls_cache[resource_id]:
                logger.debug(f"URL {origin_url} already associated with resource {resource_id}")
                return True
                
            # Add to cache
            if resource_id not in self._resource_urls_cache:
                self._resource_urls_cache[resource_id] = set()
                
            self._resource_urls_cache[resource_id].add(origin_url)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.resource_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([resource_id, origin_url, depth, timestamp])
                file.flush()
                os.fsync(file.fileno())  # Force write to disk
                
            logger.info(f"Associated URL {origin_url} with resource {resource_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save resource URL mapping: {e}")
            return False
            
    def get_resource_urls(self, resource_id: str) -> Set[str]:
        """Get all URLs associated with a resource ID.
        
        Args:
            resource_id: ID of the resource
            
        Returns:
            Set of associated URLs
        """
        return self._resource_urls_cache.get(resource_id, set())
        
    def url_belongs_to_resource(self, url: str, resource_id: str) -> bool:
        """Check if a URL belongs to a specific resource.
        
        Args:
            url: URL to check
            resource_id: Resource ID to check against
            
        Returns:
            True if the URL belongs to the resource
        """
        return resource_id in self._resource_urls_cache and url in self._resource_urls_cache[resource_id]
    
    def get_resource_id_for_url(self, url: str) -> str:
        """Get the resource ID associated with a URL.
        
        Args:
            url: URL to find the resource ID for
            
        Returns:
            Resource ID if found, empty string otherwise
        """
        # Make sure resource URLs are loaded
        if not self._resource_urls_cache:
            self.load_resource_urls()
        
        # Check each resource ID to see if it contains this URL
        for resource_id, urls in self._resource_urls_cache.items():
            if url in urls:
                return resource_id
        
        return ""
        
    def save_pending_url(self, url: str, depth: int = 0, origin: str = "", attempt_count: int = 0, resource_id: str = "") -> bool:
        """Save a pending URL to the CSV file with depth, origin, attempt count and resource ID information.
        
        Args:
            url: The URL to be processed later
            depth: The crawl depth level of this URL (0=main, 1=first level, etc.)
            origin: The URL that led to this URL (empty for main URLs)
            attempt_count: Number of times this URL has been attempted to process
            resource_id: ID of the resource this URL belongs to (empty if unknown)
            
        Returns:
            Success flag
        """
        try:
            # First check if already processed - only skip if both processed AND discovery completed
            if self.url_is_processed(url):
                if not self.allow_processed_url_rediscovery and self.get_discovery_status(url):
                    logger.debug(f"URL {url} is already processed and discovery completed, not saving to pending")
                    return True
                elif not self.allow_processed_url_rediscovery:
                    logger.debug(f"URL {url} is already processed, but discovery may be incomplete. Not saving due to rediscovery being disabled.")
                    return True
                else:
                    logger.info(f"URL {url} is already processed, but rediscovery is allowed. Adding to pending.")
            
            # Initialize tracking variables
            current_attempt_count = attempt_count
            current_resource_id = resource_id
            current_depth = depth  # Always use the provided depth
            already_pending = False
            needs_update = False
            
            # Check if already in pending cache
            if url in self._pending_urls_cache:
                already_pending = True
                cached_data = self._pending_urls_cache[url]
                cached_depth = cached_data.get("depth", 0)
                cached_attempt_count = cached_data.get("attempt_count", 0)
                cached_resource_id = cached_data.get("resource_id", "")
                
                # Only update if new depth is lower (closer to root) or other attributes need updating
                if (depth < cached_depth or
                    attempt_count > cached_attempt_count or
                    (resource_id and resource_id != cached_resource_id)):
                    needs_update = True
                    current_depth = min(depth, cached_depth)  # Keep the lower depth
                    current_attempt_count = max(attempt_count, cached_attempt_count)
                    if resource_id:
                        current_resource_id = resource_id
                    logger.info(f"Updating URL {url}: depth {cached_depth}->{current_depth}, attempts {current_attempt_count}, resource {current_resource_id}")
            
            # Check pending URLs file if not in cache
            if not already_pending and os.path.exists(self.pending_urls_file):
                try:
                    with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                        reader = csv.reader(file)
                        next(reader, None)  # Skip header
                        for row in reader:
                            if row and row[0] == url:
                                already_pending = True
                                file_depth = int(row[1]) if row[1].isdigit() else 0
                                current_attempt = int(row[4]) if len(row) >= 5 and row[4].isdigit() else 0
                                file_resource_id = row[5] if len(row) >= 6 else ""
                                
                                # Only update if new depth is lower or other attributes need updating
                                if (depth < file_depth or
                                    attempt_count > current_attempt or
                                    (resource_id and resource_id != file_resource_id)):
                                    needs_update = True
                                    current_depth = min(depth, file_depth)  # Keep the lower depth
                                    current_attempt_count = max(attempt_count, current_attempt)
                                    if resource_id:
                                        current_resource_id = resource_id
                                break
                except Exception as e:
                    logger.error(f"Error checking existing pending URL: {e}")
            
            # If already pending and no updates needed, we're done  
            if already_pending and not needs_update:
                return True
                
            # Remove old entry if updating
            if already_pending and needs_update:
                self.remove_pending_url(url)
            
            # Save new/updated entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.pending_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([url, current_depth, origin, timestamp, current_attempt_count, current_resource_id])
                file.flush()
                os.fsync(file.fileno())
            
            # Update cache
            self._pending_urls_cache[url] = {
                "depth": current_depth,
                "origin": origin, 
                "timestamp": timestamp,
                "attempt_count": current_attempt_count,
                "resource_id": current_resource_id
            }
            
            # Save resource association if needed
            if current_resource_id:
                self.save_resource_url(current_resource_id, url, current_depth)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pending URL {url}: {e}")
            return False
            
    def get_pending_urls(self, max_urls: int = 0, depth: int = None, resource_id: str = None) -> List[Dict[str, Any]]:
        """Get pending URLs to process from the CSV file
        
        Args:
            max_urls: Maximum number of URLs to return (0 = no limit, return all URLs)
            depth: Optional depth level to filter by
            resource_id: Optional resource ID to filter by
            
        Returns:
            List of dictionaries with URL information
        """
        pending_urls = []
        
        if not os.path.exists(self.pending_urls_file):
            logger.warning(f"Pending URLs file doesn't exist: {self.pending_urls_file}")
            return pending_urls
        
        # First check if we need to reload the cache
        if not self._pending_urls_cache:
            self.load_pending_urls_cache()
            
        # If we have a populated cache, use it for better performance
        if self._pending_urls_cache:
            for url, data in self._pending_urls_cache.items():
                # Skip if this URL has already been processed
                if self.url_is_processed(url):
                    continue
                
                # Filter by depth if specified
                url_depth = data.get("depth", 0)
                if depth is not None and url_depth != depth:
                    continue
                    
                # Filter by resource ID if specified
                url_resource_id = data.get("resource_id", "")
                if resource_id is not None and url_resource_id != resource_id:
                    continue
                
                pending_urls.append({
                    "url": url,
                    "depth": url_depth,
                    "origin": data.get("origin", ""),
                    "attempt_count": data.get("attempt_count", 0),
                    "resource_id": url_resource_id
                })
                
                # Stop if we've reached the maximum (only if max_urls > 0)
                if max_urls > 0 and len(pending_urls) >= max_urls:
                    logger.info(f"Reached max_urls limit of {max_urls}, more URLs may be pending")
                    break
                    
            if resource_id:
                logger.info(f"Retrieved {len(pending_urls)} pending URLs for resource {resource_id} from cache")
            else:
                logger.info(f"Retrieved {len(pending_urls)} pending URLs from cache")
                
            return pending_urls

        # If cache is empty or has issues, fall back to reading directly from file
        try:
            # Read the CSV file
            with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Skip header
                
                # Process each row
                for row in reader:
                    if len(row) >= 3:
                        url = row[0]
                        row_depth = int(row[1]) if row[1].isdigit() else 0
                        origin = row[2]
                        
                        # Get attempt count if available (in newer format)
                        attempt_count = 0
                        if len(row) >= 5 and row[4] and row[4].isdigit():
                            attempt_count = int(row[4])
                            
                        # Get resource ID if available
                        row_resource_id = ""
                        if len(row) >= 6 and row[5]:
                            row_resource_id = row[5]
                        
                        # Skip if this URL has already been processed
                        if self.url_is_processed(url):
                            continue
                        
                        # Filter by depth if specified
                        if depth is not None and row_depth != depth:
                            continue
                            
                        # Filter by resource ID if specified
                        if resource_id is not None and row_resource_id != resource_id:
                            continue
                        
                        pending_urls.append({
                            "url": url,
                            "depth": row_depth,
                            "origin": origin,
                            "attempt_count": attempt_count,
                            "resource_id": row_resource_id
                        })
                        
                        # Stop if we've reached the maximum (only if max_urls > 0)
                        if max_urls > 0 and len(pending_urls) >= max_urls:
                            logger.info(f"Reached max_urls limit of {max_urls}, more URLs may be pending")
                            break
                
            if resource_id:
                logger.info(f"Retrieved {len(pending_urls)} pending URLs for resource {resource_id}")
            else:
                logger.info(f"Retrieved {len(pending_urls)} pending URLs from file")
                
            return pending_urls
            
        except Exception as e:
            logger.error(f"Error reading pending URLs: {e}")
            return []
            
    def remove_resource_url(self, url: str) -> bool:
        """Remove a URL from the resource_urls.csv file after it's been processed
        
        Args:
            url: The URL to remove from resource_urls.csv
            
        Returns:
            Success flag
        """
        if not os.path.exists(self.resource_urls_file):
            logger.warning(f"Resource URLs file doesn't exist: {self.resource_urls_file}")
            return False
        
        try:
            # Read all URLs from the file
            resource_urls = []
            url_found = False
            resource_id = None
            
            with open(self.resource_urls_file, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Get header
                
                # Keep all rows except those containing the URL to remove
                for row in reader:
                    if len(row) >= 2 and row[1] == url:
                        url_found = True
                        resource_id = row[0]
                        logger.info(f"Found URL to remove from resource_urls: {url} (resource_id: {resource_id})")
                        continue  # Skip this row (don't add to resource_urls)
                    resource_urls.append(row)
            
            # If the URL wasn't found, log it but continue
            if not url_found:
                logger.info(f"URL not found in resource_urls file: {url}")
                # Return True anyway since the URL is not in the list
                return True
            
            # Write the remaining URLs back to the file
            with open(self.resource_urls_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Make sure to write the header
                if header:
                    writer.writerow(header)
                else:
                    # If header wasn't found, write a default one
                    writer.writerow(["resource_id", "url", "depth", "timestamp"])
                
                # Write all rows except the ones we removed
                writer.writerows(resource_urls)
                
                # Force write to disk
                file.flush()
                os.fsync(file.fileno())
            
            # Also update the memory cache if it exists
            if resource_id and resource_id in self._resource_urls_cache:
                self._resource_urls_cache[resource_id].discard(url)
                logger.info(f"Removed URL from resource_urls cache: {url} (resource_id: {resource_id})")
            
            logger.info(f"Successfully removed URL from resource_urls file: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing URL from resource_urls file {url}: {e}")
            return False
    
    def remove_pending_url(self, url: str) -> bool:
        """Remove a URL from the pending queue
        
        Args:
            url: URL to remove from pending queue
            
        Returns:
            Success flag
        """
        if not os.path.exists(self.pending_urls_file):
            return True  # Nothing to remove
            
        try:
            # First remove from cache
            if url in self._pending_urls_cache:
                del self._pending_urls_cache[url]
                logger.debug(f"Removed URL from pending cache: {url}")
            
            # Then update the file
            pending_urls = []
            header = None
            
            # Read all URLs except the one we want to remove
            with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Save header
                for row in reader:
                    if row and row[0] != url:  # Keep all except target URL
                        pending_urls.append(row)
                        
            # Write back all URLs except removed one
            with open(self.pending_urls_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if header:
                    writer.writerow(header)
                else:
                    # If header wasn't found, write a default one
                    writer.writerow(["url", "depth", "origin", "discovery_timestamp", "attempt_count", "resource_id"])
                
                # Write all rows except the one we removed
                writer.writerows(pending_urls)
                
                # Force write to disk
                file.flush()
                os.fsync(file.fileno())
              # Verify the file was updated
            pending_count_after = 0
            try:
                with open(self.pending_urls_file, mode='r', encoding='utf-8') as check_file:
                    reader = csv.reader(check_file)
                    next(reader, None)  # Skip header
                    for _ in reader:
                        pending_count_after += 1
                        
                logger.info(f"After removal, pending URLs file has {pending_count_after} URLs")
            except Exception as check_err:
                logger.error(f"Error verifying pending URLs after removal: {check_err}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing pending URL {url}: {e}")
            return False
    
    def save_discovery_status(self, url, completed=True):
        """
        Mark a URL as having completed the discovery phase.
        
        Args:
            url: The URL to mark
            completed: Whether discovery is completed (True) or not (False)
        """
        # Use the dedicated discovery status tracking method
        self.set_discovery_status(url, completed)
        logger.info(f"Marked URL as discovery completed: {url}")

    def set_discovery_status(self, url: str, status: bool = True) -> bool:
        """
        Set the discovery status for a URL.
        
        Args:
            url: The URL to mark
            status: True if discovery completed, False to reset
            
        Returns:
            Success flag
        """
        try:
            # Initialize discovery status file if it doesn't exist
            if not os.path.exists(self.discovery_status_file):
                with open(self.discovery_status_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "timestamp"])
            
            # If marking as not discovered, remove from file if present
            if not status:
                # Remove URL from discovery status file if it exists
                if not os.path.exists(self.discovery_status_file):
                    return True  # Nothing to do
                
                try:
                    # Read all URLs except the one to remove
                    entries = []
                    with open(self.discovery_status_file, mode='r', encoding='utf-8') as file:
                        reader = csv.reader(file)
                        header = next(reader, None)  # Skip header
                        for row in reader:
                            if len(row) >= 1 and row[0] != url:  # Keep all except target URL
                                entries.append(row)
                    
                    # Write back the remaining entries
                    with open(self.discovery_status_file, mode='w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow(["url", "timestamp"])  # Write header
                        writer.writerows(entries)
                    
                    # Also remove from cache
                    self._discovery_completed_cache.discard(url)
                    return True
                except Exception as e:
                    logger.error(f"Error removing URL from discovery status file: {e}")
                    return False            
            # Check if URL is already in discovery_status
            if url in self._discovery_completed_cache:
                return True  # Already marked as discovered
            
            # Add URL to discovery status file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.discovery_status_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([url, timestamp])
            
            # Add to cache
            self._discovery_completed_cache.add(url)
            logger.info(f"Marked URL as discovery completed: {url}")
            return True
        except Exception as e:
            logger.error(f"Error setting discovery status for URL {url}: {e}")
            return False
    
    def get_discovery_status(self, url: str) -> bool:
        """
        Check if a URL has completed the discovery phase.
        
        Args:
            url: The URL to check
            
        Returns:
            True if discovery phase completed, False otherwise
        """
        # Check cache first
        if url in self._discovery_completed_cache:
            return True
        
        # Check if URL exists in discovery_status.csv 
        if os.path.exists(self.discovery_status_file):
            try:
                with open(self.discovery_status_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Skip header
                    for row in reader:
                        if len(row) >= 1 and row[0] == url:
                            # Add to cache for future lookups
                            self._discovery_completed_cache.add(url)
                            return True
            except Exception as e:
                logger.error(f"Error checking discovery status: {e}")
        
        # Also check old method (compatibility) - using processed_urls with discovery_only flag
        if os.path.exists(self.processed_urls_file):
            try:
                with open(self.processed_urls_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get('origin_url') == url and row.get('discovery_only') == 'True':
                            # Add to cache for future lookups
                            self._discovery_completed_cache.add(url)
                            return True
            except Exception as e:
                logger.error(f"Error checking discovery status in processed_urls: {e}")
                
        return False

    def set_rediscovery_mode(self, allow_rediscovery: bool):
        """
        Set whether processed URLs can be added to the pending list.
        This is used when no pending URLs are available and we need to break
        out of a "stuck" discovery state.
        
        Args:
            allow_rediscovery: If True, processed URLs can be rediscovered
        """
        self.allow_processed_url_rediscovery = allow_rediscovery
        logger.info(f"URL rediscovery mode set to: {allow_rediscovery}")
    
    def get_rediscovery_mode(self) -> bool:
        """
        Check if processed URLs can be rediscovered.
        
        Returns:
            True if rediscovery is allowed, False otherwise
        """
        return self.allow_processed_url_rediscovery
    
    def increment_url_attempt(self, url: str) -> bool:
        """
        Increment the attempt count for a pending URL.
        
        Args:
            url: The URL to update
            
        Returns:
            Success flag
        """
        # Check if URL is in memory cache first
        if url in self._pending_urls_cache:
            current_attempt = self._pending_urls_cache[url].get("attempt_count", 0)
            self._pending_urls_cache[url]["attempt_count"] = current_attempt + 1
            logger.info(f"Incremented attempt count in cache for {url} to {current_attempt + 1}")
            
            # Now update the file
            self.save_pending_url(
                url=url,
                depth=self._pending_urls_cache[url].get("depth", 0),
                origin=self._pending_urls_cache[url].get("origin", ""),
                attempt_count=current_attempt + 1,
                resource_id=self._pending_urls_cache[url].get("resource_id", "")
            )
            return True
            
        # Fall back to file-based update if not in cache
        if not os.path.exists(self.pending_urls_file):
            logger.warning(f"Pending URLs file doesn't exist: {self.pending_urls_file}")
            return False
        
        try:
            # Read all pending URLs
            pending_urls = []
            updated = False
            
            # Read the current state
            with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Get header
                
                # Process each row
                for row in reader:
                    if len(row) >= 1:
                        if row[0] == url:
                            # Get current attempt count
                            current_attempt = 0
                            if len(row) >= 5 and row[4] and row[4].isdigit():
                                current_attempt = int(row[4])
                            
                            # Update attempt count
                            row[4] = str(current_attempt + 1)
                            updated = True
                            
                            # Also update cache
                            if url not in self._pending_urls_cache:
                                self._pending_urls_cache[url] = {
                                    "depth": int(row[1]) if row[1].isdigit() else 0,
                                    "origin": row[2] if len(row) > 2 else "",
                                    "timestamp": row[3] if len(row) > 3 else "",
                                    "attempt_count": current_attempt + 1,
                                    "resource_id": row[5] if len(row) > 5 else ""
                                }
                            else:
                                self._pending_urls_cache[url]["attempt_count"] = current_attempt + 1
                                
                            logger.info(f"Incremented attempt count for {url} to {current_attempt + 1}")
                    
                    pending_urls.append(row)
            
            # Only write back if updated
            if updated:
                with open(self.pending_urls_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "depth", "origin", "discovery_timestamp", "attempt_count", "resource_id"])  # Write header
                    writer.writerows(pending_urls)
                
                return True
            else:
                logger.warning(f"URL {url} not found in pending URLs, could not increment attempt count")
                return False
                
        except Exception as e:
            logger.error(f"Error incrementing attempt count for URL {url}: {e}")
            return False

    def mark_level_as_processed(self, level: int) -> Dict[str, Any]:
        """Mark all pending URLs at a specific level as processed when advancing to the next level.
        
        This ensures that when we move to a deeper level, we don't keep trying to process
        URLs from previous levels that might be stuck in the pending queue.
        
        Args:
            level: The level to mark as completely processed
            
        Returns:
            Stats about how many URLs were marked as processed
        """
        # First get all pending URLs at this level
        pending_urls = self.get_pending_urls(depth=level)
        
        if not pending_urls:
            logger.info(f"No pending URLs found at level {level} to mark as processed")
            return {
                "level": level,
                "urls_processed": 0,
                "success": True
            }
        
        logger.info(f"Marking {len(pending_urls)} pending URLs at level {level} as processed before advancing")
        
        # Count of URLs successfully processed
        processed_count = 0
          # Process each URL
        for url_data in pending_urls:
            url = url_data.get("url")
            depth = url_data.get("depth", level)
            origin = url_data.get("origin", "")
            attempt_count = url_data.get("attempt_count", 0)
            
            # Only mark as processed if we've tried it at least once (attempt_count >= 1)
            # This prevents URLs that haven't been properly attempted yet from being marked as processed
            if attempt_count >= 1:
                # Mark the URL as processed but with error=True since we're force-completing it
                if self.save_processed_url(url, depth=depth, origin=origin, error=True):
                    # Remove from pending queue
                    self.remove_pending_url(url)
                    processed_count += 1
            else:
                logger.info(f"Skipping marking URL as processed at level {level} because it hasn't been attempted yet: {url}")
        
        logger.info(f"Successfully marked {processed_count}/{len(pending_urls)} URLs at level {level} as processed")
        
        return {
            "level": level,
            "urls_processed": processed_count,
            "total_urls": len(pending_urls),
            "success": processed_count > 0
        }

    def get_pending_urls_by_level(self) -> Dict[int, int]:
        """Get counts of pending URLs grouped by level.
        
        Returns:
            Dictionary with level numbers as keys and pending URL counts as values
        """
        level_counts = {}
        
        # Ensure we have the correct path to the pending_urls file
        pending_urls_file = os.path.join(
            os.path.dirname(self.processed_urls_file), 
            "pending_urls.csv"
        )
        
        logger.info(f"Looking for pending URLs in: {pending_urls_file}")
        
        # Fall back to file-based counting - always read directly from file for accuracy
        if not os.path.exists(pending_urls_file):
            logger.warning(f"Pending URLs file doesn't exist: {pending_urls_file}")
            return level_counts
        
        try:
            # Read the CSV file
            with open(pending_urls_file, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Skip header
                
                # Process each row
                for row in reader:
                    if len(row) >= 2 and row[1].isdigit():
                        depth = int(row[1])
                        url = row[0]
                        
                        # Skip if this URL has already been processed
                        if self.url_is_processed(url):
                            continue
                        
                        # Increment count for this level
                        if depth not in level_counts:
                            level_counts[depth] = 0
                        level_counts[depth] += 1
                        
            logger.info(f"Found pending URLs by level (from file): {level_counts}")
            return level_counts
        except Exception as e:
            logger.error(f"Error reading pending URLs file for level counting: {e}")
            return level_counts
