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
                    writer.writerow(["url", "depth", "origin", "discovery_timestamp"])
                logger.info(f"Created new pending_urls.csv file for URL queue")
            except Exception as e:
                logger.error(f"Failed to create pending_urls.csv: {e}")
                # Fallback to current directory
                self.pending_urls_file = "pending_urls.csv"
                with open(self.pending_urls_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "depth", "origin", "discovery_timestamp"])
        
        # Important: Load processed URLs during initialization
        logger.info("Loading processed URLs during initialization")
        self.load_processed_urls()
    
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
    
    def save_processed_url(self, url: str, depth: int = 0, origin: str = "") -> bool:
        """Save a processed URL to the CSV file with depth and origin information.
        
        Args:
            url: The URL that was processed
            depth: The crawl depth level of this URL (0=main, 1=first level, etc.)
            origin: The URL that led to this URL (empty for main URLs)
            
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
        
    def save_pending_url(self, url: str, depth: int = 0, origin: str = "") -> bool:
        """Save a pending URL to the CSV file with depth and origin information.
        
        Args:
            url: The URL to be processed later
            depth: The crawl depth level of this URL (0=main, 1=first level, etc.)
            origin: The URL that led to this URL (empty for main URLs)
            
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
            if os.path.exists(self.pending_urls_file):
                try:
                    with open(self.pending_urls_file, mode='r', encoding='utf-8') as file:
                        reader = csv.reader(file)
                        next(reader, None)  # Skip header
                        for row in reader:
                            if row and len(row) > 0 and row[0] == url:
                                logger.debug(f"URL {url} is already in pending list, not saving again")
                                return True
                except Exception as e:
                    logger.error(f"Error checking pending URLs file: {e}")
            
            # URL is not processed and not pending, so save it
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Saving pending URL to {self.pending_urls_file}: {url} (depth={depth}, origin={origin or 'main'})")
            
            # Open in append mode with explicit flush
            with open(self.pending_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([url, depth, origin, timestamp])
                file.flush()
                os.fsync(file.fileno())  # Force write to disk
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save pending URL {url}: {e}")
            return False
            
    def get_pending_urls(self, max_urls: int = 100, depth: int = None) -> List[Dict[str, Any]]:
        """Get pending URLs to process from the CSV file
        
        Args:
            max_urls: Maximum number of URLs to return
            depth: Optional depth level to filter by
            
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
                        
                        # Skip if this URL has already been processed
                        if self.url_is_processed(url):
                            continue
                        
                        # Filter by depth if specified
                        if depth is not None and row_depth != depth:
                            continue
                        
                        pending_urls.append({
                            "url": url,
                            "depth": row_depth,
                            "origin": origin
                        })
                        
                        # Stop if we've reached the maximum
                        if len(pending_urls) >= max_urls:
                            break
                
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
                        continue
                    pending_urls.append(row)
            
            # If the URL wasn't found, there's nothing to do
            if not url_found:
                logger.warning(f"URL {url} not found in pending URLs file")
                return False
            
            # Write the remaining URLs back to the file
            with open(self.pending_urls_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["url", "depth", "origin", "discovery_timestamp"])  # Write header
                writer.writerows(pending_urls)
            
            logger.info(f"Removed URL {url} from pending URLs file")
            return True
            
        except Exception as e:
            logger.error(f"Error removing pending URL {url}: {e}")
            return False
