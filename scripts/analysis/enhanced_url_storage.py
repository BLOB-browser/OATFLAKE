#!/usr/bin/env python3
"""
Enhanced URL Storage Manager with unique URL identification system
"""

import csv
import os
import hashlib
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class URLIdentifierGenerator:
    """Generates unique identifiers for URLs"""
    
    @staticmethod
    def generate_url_id(url: str) -> str:
        """Generate a unique URL ID based on URL content + timestamp"""
        # Use first 8 characters of MD5 hash + timestamp suffix for readability
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = str(int(time.time()))[-6:]  # last 6 digits of timestamp
        return f"url_{url_hash}_{timestamp}"
    
    @staticmethod
    def generate_sequential_id(counter: int) -> str:
        """Generate a sequential ID"""
        return f"url_{counter:06d}"  # url_000001, url_000002, etc.

class EnhancedURLData:
    """Enhanced URL data structure with unique identification"""
    
    def __init__(self, 
                 url: str,
                 depth: int,
                 origin_url: str,
                 resource_id: str,
                 main_resource_url: str,
                 url_id: str = None,
                 origin_url_id: str = None):
        self.url_id = url_id or URLIdentifierGenerator.generate_url_id(url)
        self.url = url
        self.depth = depth
        self.origin_url = origin_url
        self.origin_url_id = origin_url_id  # ID of the URL that led to this URL
        self.resource_id = resource_id
        self.main_resource_url = main_resource_url
        self.discovery_timestamp = datetime.now().isoformat()
        self.attempt_count = 0
        self.status = "pending"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV storage"""
        return {
            "url_id": self.url_id,
            "url": self.url,
            "depth": self.depth,
            "origin_url": self.origin_url,
            "origin_url_id": self.origin_url_id or "",
            "resource_id": self.resource_id,
            "main_resource_url": self.main_resource_url,
            "discovery_timestamp": self.discovery_timestamp,
            "attempt_count": self.attempt_count,
            "status": self.status
        }

class EnhancedURLStorageManager:
    """Enhanced URL storage manager with unique URL identification"""
    
    def __init__(self, processed_urls_file: str):
        self.processed_urls_file = processed_urls_file
        self._processed_urls_cache = set()
        self.url_metadata = {}
        
        # Enhanced pending URLs file with unique IDs
        self.enhanced_pending_urls_file = os.path.join(
            os.path.dirname(self.processed_urls_file), 
            "enhanced_pending_urls.csv"
        )
        
        # URL lookup tables
        self._url_id_to_url = {}  # url_id -> url
        self._url_to_url_id = {}  # url -> url_id
        self._url_cache = {}      # url_id -> EnhancedURLData
        
        # Initialize enhanced pending URLs file
        self._initialize_enhanced_pending_file()
        
    def _initialize_enhanced_pending_file(self):
        """Initialize the enhanced pending URLs file with proper headers"""
        if not os.path.exists(self.enhanced_pending_urls_file):
            try:
                with open(self.enhanced_pending_urls_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "url_id", "url", "depth", "origin_url", "origin_url_id",
                        "resource_id", "main_resource_url", "discovery_timestamp", 
                        "attempt_count", "status"
                    ])
                logger.info(f"Created enhanced pending URLs file: {self.enhanced_pending_urls_file}")
            except Exception as e:
                logger.error(f"Failed to create enhanced pending URLs file: {e}")
    
    def save_enhanced_pending_url(self, 
                                url: str,
                                depth: int,
                                origin_url: str,
                                resource_id: str,
                                main_resource_url: str,
                                origin_url_id: str = None) -> str:
        """
        Save a URL with enhanced tracking information.
        
        Args:
            url: The URL to save
            depth: Depth level 
            origin_url: URL that led to this URL
            resource_id: ID of the main resource
            main_resource_url: Original resource URL
            origin_url_id: ID of the origin URL (if known)
            
        Returns:
            The unique URL ID assigned to this URL
        """
        try:
            # Check if URL already exists
            if url in self._url_to_url_id:
                existing_url_id = self._url_to_url_id[url]
                logger.debug(f"URL {url} already exists with ID {existing_url_id}")
                return existing_url_id
            
            # If origin_url_id not provided, try to find it
            if not origin_url_id and origin_url:
                origin_url_id = self._url_to_url_id.get(origin_url, "")
            
            # Create enhanced URL data
            url_data = EnhancedURLData(
                url=url,
                depth=depth,
                origin_url=origin_url,
                resource_id=resource_id,
                main_resource_url=main_resource_url,
                origin_url_id=origin_url_id
            )
            
            # Update lookup tables
            self._url_id_to_url[url_data.url_id] = url
            self._url_to_url_id[url] = url_data.url_id
            self._url_cache[url_data.url_id] = url_data
            
            # Write to CSV file
            with open(self.enhanced_pending_urls_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                row_data = url_data.to_dict()
                writer.writerow([
                    row_data["url_id"],
                    row_data["url"],
                    row_data["depth"],
                    row_data["origin_url"],
                    row_data["origin_url_id"],
                    row_data["resource_id"],
                    row_data["main_resource_url"],
                    row_data["discovery_timestamp"],
                    row_data["attempt_count"],
                    row_data["status"]
                ])
            
            logger.debug(f"Saved enhanced URL: {url} with ID {url_data.url_id}")
            return url_data.url_id
            
        except Exception as e:
            logger.error(f"Error saving enhanced pending URL {url}: {e}")
            return ""
    
    def get_enhanced_pending_urls(self, resource_id: str = None, depth: int = None, max_urls: int = 0) -> List[Dict[str, Any]]:
        """Get enhanced pending URLs with filtering options"""
        urls = []
        
        if not os.path.exists(self.enhanced_pending_urls_file):
            return urls
        
        try:
            with open(self.enhanced_pending_urls_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Apply filters
                    if resource_id and row.get('resource_id') != resource_id:
                        continue
                    if depth is not None and int(row.get('depth', 0)) != depth:
                        continue
                    if row.get('status') != 'pending':
                        continue
                    
                    urls.append(dict(row))
                    
                    # Limit results if specified
                    if max_urls > 0 and len(urls) >= max_urls:
                        break
        
        except Exception as e:
            logger.error(f"Error reading enhanced pending URLs: {e}")
        
        return urls
    
    def get_urls_by_origin(self, origin_url_id: str) -> List[Dict[str, Any]]:
        """Get all URLs that were discovered from a specific origin URL"""
        urls = []
        
        try:
            with open(self.enhanced_pending_urls_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    if row.get('origin_url_id') == origin_url_id:
                        urls.append(dict(row))
        
        except Exception as e:
            logger.error(f"Error getting URLs by origin {origin_url_id}: {e}")
        
        return urls
    
    def get_url_discovery_tree(self, resource_id: str) -> Dict[str, Any]:
        """Build a discovery tree for a resource showing URL relationships"""
        tree = {
            "resource_id": resource_id,
            "main_url": "",
            "children": {}
        }
        
        try:
            # Get all URLs for this resource
            resource_urls = self.get_enhanced_pending_urls(resource_id=resource_id)
            
            # Find main URL (depth 0)
            main_urls = [url for url in resource_urls if int(url.get('depth', 0)) == 0]
            if main_urls:
                tree["main_url"] = main_urls[0].get('url', '')
                tree["main_url_id"] = main_urls[0].get('url_id', '')
            
            # Build tree structure
            def add_children(parent_url_id, parent_node):
                children = [url for url in resource_urls if url.get('origin_url_id') == parent_url_id]
                for child in children:
                    child_id = child.get('url_id')
                    parent_node["children"][child_id] = {
                        "url": child.get('url'),
                        "url_id": child_id,
                        "depth": int(child.get('depth', 0)),
                        "children": {}
                    }
                    # Recursively add children
                    add_children(child_id, parent_node["children"][child_id])
            
            if tree.get("main_url_id"):
                add_children(tree["main_url_id"], tree)
        
        except Exception as e:
            logger.error(f"Error building discovery tree for resource {resource_id}: {e}")
        
        return tree

    def print_discovery_summary(self, resource_id: str = None):
        """Print a summary of the discovery structure"""
        if resource_id:
            tree = self.get_url_discovery_tree(resource_id)
            print(f"\n📊 Discovery Tree for Resource {resource_id}:")
            print(f"Main URL: {tree.get('main_url', 'Unknown')}")
            
            def print_tree_level(children, level=1):
                for url_id, data in children.items():
                    indent = "  " * level
                    url = data.get('url', '')[:60]
                    depth = data.get('depth', 0)
                    print(f"{indent}├─ [{depth}] {url}... (ID: {url_id[:8]})")
                    if data.get('children'):
                        print_tree_level(data['children'], level + 1)
            
            if tree.get('children'):
                print_tree_level(tree['children'])
        else:
            # Print summary for all resources
            urls = self.get_enhanced_pending_urls()
            resources = {}
            for url in urls:
                rid = url.get('resource_id', 'unknown')
                if rid not in resources:
                    resources[rid] = {'total': 0, 'by_depth': {}}
                resources[rid]['total'] += 1
                depth = int(url.get('depth', 0))
                resources[rid]['by_depth'][depth] = resources[rid]['by_depth'].get(depth, 0) + 1
            
            print(f"\n📊 Enhanced URL Discovery Summary:")
            for rid, data in resources.items():
                print(f"Resource {rid}: {data['total']} URLs")
                for depth, count in sorted(data['by_depth'].items()):
                    print(f"  Level {depth}: {count} URLs")
    
    def get_main_resource_url(self, resource_id: str) -> str:
        """
        Get the main resource URL for a given resource ID.
        
        Args:
            resource_id: The resource ID to look up
            
        Returns:
            The main resource URL, or empty string if not found
        """
        try:
            # Look through our URL cache for URLs with this resource_id and depth 0 or 1
            for url_id, url_data in self._url_cache.items():
                if url_data.resource_id == resource_id and url_data.depth <= 1:
                    return url_data.main_resource_url or url_data.url
            
            # If not found in cache, try reading from file
            if os.path.exists(self.enhanced_pending_urls_file):
                with open(self.enhanced_pending_urls_file, 'r', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if (row.get('resource_id') == resource_id and 
                            int(row.get('depth', 0)) <= 1):
                            return row.get('main_resource_url', '') or row.get('url', '')
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting main resource URL for resource {resource_id}: {e}")
            return ""

if __name__ == "__main__":
    # Test the enhanced URL storage
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_file = os.path.join(temp_dir, "processed_urls.csv")
        
        # Create enhanced storage manager
        storage = EnhancedURLStorageManager(processed_file)
        
        # Simulate URL discovery
        print("Testing enhanced URL storage...")
        
        # Main resource URL
        main_url_id = storage.save_enhanced_pending_url(
            url="https://fablabbcn.org/",
            depth=0,
            origin_url="",
            resource_id="1",
            main_resource_url="https://fablabbcn.org/"
        )
        print(f"Main URL ID: {main_url_id}")
        
        # First level URLs
        projects_url_id = storage.save_enhanced_pending_url(
            url="https://fablabbcn.org/projects",
            depth=1,
            origin_url="https://fablabbcn.org/",
            resource_id="1",
            main_resource_url="https://fablabbcn.org/",
            origin_url_id=main_url_id
        )
        
        about_url_id = storage.save_enhanced_pending_url(
            url="https://fablabbcn.org/about",
            depth=1,
            origin_url="https://fablabbcn.org/",
            resource_id="1", 
            main_resource_url="https://fablabbcn.org/",
            origin_url_id=main_url_id
        )
        
        # Second level URL
        project_detail_url_id = storage.save_enhanced_pending_url(
            url="https://fablabbcn.org/projects/item1",
            depth=2,
            origin_url="https://fablabbcn.org/projects",
            resource_id="1",
            main_resource_url="https://fablabbcn.org/",
            origin_url_id=projects_url_id
        )
        
        # Print discovery summary
        storage.print_discovery_summary("1")
