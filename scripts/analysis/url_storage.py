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
        
        # Cache for URLs that have completed discovery 
        self._discovery_completed_cache = set()
        
        # Cache for resource URLs - maps resource ID to set of discovered URLs
        self._resource_urls_cache = {}
        
        # Ensure the file directory exists
        os.makedirs(os.path.dirname(self.processed_urls_file), exist_ok=True)
        
        # Check if the processed_urls file exists and create it with header if it doesn't
        if not os.path.exists(self.processed_urls_file):
            try:
                with open(self.processed_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "depth", "origin", "timestamp"])
                logger.info(f"Created new processed_urls.csv file with enhanced structure")
            except Exception as e:
                logger.error(f"Failed to create processed_urls.csv: {e}")
                # Fallback to current directory
                self.processed_urls_file = "processed_urls.csv"
                with open(self.processed_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "depth", "origin", "timestamp"])
        
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
                                self.url_metadata[url] = {
                                    "depth": depth,
                                    "origin": origin,
                                    "timestamp": timestamp
                                }
                logger.info(f"Loaded {len(processed_urls)} processed URLs")
            except Exception as e:
                logger.error(f"Error loading processed URLs: {e}")
                
        # Store the processed URLs in an instance attribute so we can add to it when saving
        self._processed_urls_cache = processed_urls
        logger.info(f"URL cache initialized with {len(self._processed_urls_cache)} URLs")
        return processed_urls
    
    def save_processed_url(self, url: str, depth: int = 0, origin: str = "", discovery_only: bool = False) -> bool:
        """Save a processed URL to the CSV file with depth and origin information.
        
        Args:
            url: The URL that was processed
            depth: The crawl depth level of this URL (0=main, 1=first level, etc.)
            origin: The URL that led to this URL (empty for main URLs)
            discovery_only: Whether this URL is only for discovery (defaults to False)
            
        Returns:
            Success flag
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extra diagnostics about file path and directory
            file_dir = os.path.dirname(self.processed_urls_file)
            logger.info(f"Directory for processed URLs: {file_dir}")
            logger.info(f"Directory exists: {os.path.exists(file_dir)}")
            logger.info(f"Directory writable: {os.access(file_dir, os.W_OK)}")
            logger.info(f"File path: {self.processed_urls_file}")
            logger.info(f"File exists: {os.path.exists(self.processed_urls_file)}")
            
            # Make sure directory exists
            os.makedirs(os.path.dirname(self.processed_urls_file), exist_ok=True)
            
            logger.info(f"Saving processed URL to {self.processed_urls_file}: {url} (depth={depth}, origin={origin or 'main'})")
            
            # Open in append mode with explicit flush
            with open(self.processed_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([url, depth, origin, timestamp])
                file.flush()
                os.fsync(file.fileno())  # Force write to disk
            
            # Verify the file exists and has content
            if os.path.exists(self.processed_urls_file):
                file_size = os.path.getsize(self.processed_urls_file)
                logger.info(f"After write: file exists with size {file_size} bytes")
            else:
                logger.warning(f"After write: file does not exist!")
            
            # Store in memory cache as well
            if not hasattr(self, 'url_metadata'):
                self.url_metadata = {}
            self.url_metadata[url] = {"depth": depth, "origin": origin, "timestamp": timestamp}
            
            # IMPORTANT: Also add to the in-memory processed_urls set
            self._processed_urls_cache.add(url)
            logger.info(f"Added URL to in-memory processed URLs cache (now has {len(self._processed_urls_cache)} URLs)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processed URL {url}: {e}")
            # Try a fallback to the local directory if the data path fails
            try:
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
                            
                logger.info(f"Loaded URL mappings for {len(resource_urls)} resources")
            except Exception as e:
                logger.error(f"Error loading resource URL mappings: {e}")
                
        # Store the resource URLs in an instance attribute
        self._resource_urls_cache = resource_urls
        return resource_urls
        
    def save_resource_url(self, resource_id: str, url: str, depth: int = 0) -> bool:
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
            if resource_id in self._resource_urls_cache and url in self._resource_urls_cache[resource_id]:
                logger.debug(f"URL {url} already associated with resource {resource_id}")
                return True
                
            # Add to cache
            if resource_id not in self._resource_urls_cache:
                self._resource_urls_cache[resource_id] = set()
                
            self._resource_urls_cache[resource_id].add(url)
            
            # Save to file
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.resource_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([resource_id, url, depth, timestamp])
                file.flush()
                os.fsync(file.fileno())  # Force write to disk
                
            logger.info(f"Associated URL {url} with resource {resource_id}")
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
            # First check if the URL is already in the processed list
            if self.url_is_processed(url):
                logger.debug(f"URL {url} is already processed, not saving to pending")
                return True
                
            # Next, check if the URL is already in the pending list
            # by reading the pending URLs file
            current_attempt_count = attempt_count
            already_pending = False
            current_resource_id = resource_id
            
            if os.path.exists(self.pending_urls_file):
                try:
                    with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                        reader = csv.reader(file)
                        next(reader, None)  # Skip header
                        for row in reader:
                            if row and len(row) > 0 and row[0] == url:
                                # URL is already in pending, might need to update attempt count
                                already_pending = True
                                # Get current attempt count if available (in newer format)
                                if len(row) >= 5 and row[4] and row[4].isdigit():
                                    current_attempt_count = int(row[4])
                                
                                # Get existing resource ID if available
                                if len(row) >= 6 and row[5]:
                                    current_resource_id = row[5]
                                    
                                logger.debug(f"URL {url} is already in pending list with {current_attempt_count} attempts, resource={current_resource_id}")
                                
                                # If we're explicitly trying to increment the attempt count or update resource ID, do it
                                if attempt_count > current_attempt_count or (resource_id and resource_id != current_resource_id):
                                    # We need to update this URL, so we'll do that by removing and re-adding
                                    self.remove_pending_url(url)
                                    already_pending = False
                                    current_attempt_count = attempt_count
                                    if resource_id:  # Only update resource_id if a new one is provided
                                        current_resource_id = resource_id
                                    logger.info(f"Updating URL {url} attempt count to {attempt_count}, resource={current_resource_id}")
                                else:
                                    # No need to update, keep as is
                                    return True
                except Exception as e:
                    logger.error(f"Error checking pending URLs file: {e}")
            
            # If the URL is already pending and we don't need to update it, we're done
            if already_pending:
                return True
            
            # URL is not processed and either not pending or needs updating, so save it
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Saving pending URL to {self.pending_urls_file}: {url} (depth={depth}, origin={origin or 'main'}, attempts={current_attempt_count}, resource={current_resource_id or 'unknown'})")
            
            # Open in append mode with explicit flush
            with open(self.pending_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([url, depth, origin, timestamp, current_attempt_count, current_resource_id])
                file.flush()
                os.fsync(file.fileno())  # Force write to disk
            
            # If we have a resource ID, save the association
            if current_resource_id:
                self.save_resource_url(current_resource_id, url, depth)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pending URL {url}: {e}")
            return False
            
    def get_pending_urls(self, max_urls: int = 100, depth: int = None, resource_id: str = None) -> List[Dict[str, Any]]:
        """Get pending URLs to process from the CSV file
        
        Args:
            max_urls: Maximum number of URLs to return
            depth: Optional depth level to filter by
            resource_id: Optional resource ID to filter by
            
        Returns:
            List of dictionaries with URL information
        """
        pending_urls = []
        
        if not os.path.exists(self.pending_urls_file):
            logger.warning(f"Pending URLs file doesn't exist: {self.pending_urls_file}")
            return pending_urls
        
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
                        
                        # Stop if we've reached the maximum
                        if len(pending_urls) >= max_urls:
                            break
                
            if resource_id:
                logger.info(f"Retrieved {len(pending_urls)} pending URLs for resource {resource_id}")
            else:
                logger.info(f"Retrieved {len(pending_urls)} pending URLs from file")
                
            return pending_urls
            
        except Exception as e:
            logger.error(f"Error reading pending URLs: {e}")
            return []
            
    def remove_pending_url(self, url: str) -> bool:
        """Remove a URL from the pending URLs file after it's been processed
        
        Args:
            url: The URL to remove
            
        Returns:
            Success flag
        """
        if not os.path.exists(self.pending_urls_file):
            logger.warning(f"Pending URLs file doesn't exist: {self.pending_urls_file}")
            return False
        
        try:
            # Add detailed debug logging
            logger.info(f"DEBUG URL STORAGE: Removing URL {url} from pending URLs file: {self.pending_urls_file}")
            
            # Read all URLs from the file
            pending_urls = []
            url_found = False
            
            with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                header = next(reader, None)  # Get header
                
                # Keep all rows except the one to remove
                for row in reader:
                    if len(row) >= 1 and row[0] == url:
                        url_found = True
                        logger.info(f"DEBUG URL STORAGE: Found URL {url} to remove at row: {row}")
                        continue
                    pending_urls.append(row)
            
            # If the URL wasn't found, there's nothing to do
            if not url_found:
                logger.warning(f"DEBUG URL STORAGE: URL {url} not found in pending URLs file")
                return False
            
            # Write the remaining URLs back to the file
            with open(self.pending_urls_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["url", "depth", "origin", "discovery_timestamp"])  # Write header
                writer.writerows(pending_urls)
            
            # Verify the file was updated
            pending_count_after = 0
            try:
                with open(self.pending_urls_file, mode='r', encoding='utf-8') as check_file:
                    reader = csv.reader(check_file)
                    next(reader, None)  # Skip header
                    for _ in reader:
                        pending_count_after += 1
                        
                logger.info(f"DEBUG URL STORAGE: After removal, pending URLs file has {pending_count_after} URLs")
            except Exception as check_err:
                logger.error(f"DEBUG URL STORAGE: Error verifying pending URLs count: {check_err}")
            
            logger.info(f"DEBUG URL STORAGE: Successfully removed URL {url} from pending URLs file")
            return True
            
        except Exception as e:
            logger.error(f"DEBUG URL STORAGE: Error removing pending URL {url}: {e}")
            return False
        
    def save_discovery_status(self, url, completed=True):
        """
        Mark a URL as having completed the discovery phase.
        
        Args:
            url: The URL to mark
            completed: Whether discovery is completed (True) or not (False)
        """
        # We'll use the processed_urls file but with a special flag
        self.save_processed_url(url, depth=0, origin="", discovery_only=True)
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
                        if row.get('url') == url and row.get('discovery_only') == 'True':
                            # Add to cache for future lookups
                            self._discovery_completed_cache.add(url)
                            return True
            except Exception as e:
                logger.error(f"Error checking discovery status in processed_urls: {e}")
                
        return False
