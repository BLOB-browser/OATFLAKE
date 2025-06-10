#!/usr/bin/env python3
"""
Enhanced URL ID System Design

Current Problem:
- All URLs from the same resource get the same resource_id (e.g., '1')
- Cannot distinguish individual URLs within a resource
- No unique identifier per discovered URL

Proposed Solution:
1. URL_ID: Unique identifier for each discovered URL (uuid or sequential)
2. RESOURCE_ID: Identifier of the parent resource (from resources.csv)
3. ORIGIN_URL_ID: Identifier of the immediate parent URL that led to this URL
4. MAIN_RESOURCE_URL: The original resource URL this discovery chain started from

New CSV Structure:
pending_urls.csv:
- url_id (unique per URL)
- url (actual URL)
- depth (level)
- origin_url (immediate parent URL)
- origin_url_id (unique ID of parent URL) 
- resource_id (main resource this belongs to)
- main_resource_url (original resource URL)
- discovery_timestamp
- attempt_count
- status (pending/processing/completed/failed)

Benefits:
1. Each URL has unique identity
2. Can trace discovery path: main_resource → origin_url → current_url  
3. Can group URLs by resource for batch processing
4. Can find all child URLs of any given URL
5. Can rebuild the discovery tree structure
"""

import uuid
import hashlib
from typing import Dict, List, Optional

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
    def generate_uuid_id() -> str:
        """Generate a UUID-based unique ID"""
        return str(uuid.uuid4())[:8]  # Short UUID
    
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
            "origin_url_id": self.origin_url_id,
            "resource_id": self.resource_id,
            "main_resource_url": self.main_resource_url,
            "discovery_timestamp": self.discovery_timestamp,
            "attempt_count": self.attempt_count,
            "status": self.status
        }

# Example usage:
"""
Resource: "Fab Lab Barcelona" (resource_id: "1", url: "https://fablabbcn.org/")

Discovery chain:
1. Main resource URL: https://fablabbcn.org/ 
   - url_id: "url_abc12345_123456"
   - resource_id: "1" 
   - depth: 0
   - origin_url: ""
   - origin_url_id: ""

2. First level URLs discovered from main:
   - https://fablabbcn.org/projects
     - url_id: "url_def67890_123457"
     - resource_id: "1"
     - depth: 1
     - origin_url: "https://fablabbcn.org/"
     - origin_url_id: "url_abc12345_123456"
   
   - https://fablabbcn.org/about
     - url_id: "url_ghi11111_123458"
     - resource_id: "1"
     - depth: 1
     - origin_url: "https://fablabbcn.org/"
     - origin_url_id: "url_abc12345_123456"

3. Second level URLs discovered from projects page:
   - https://fablabbcn.org/projects/item1
     - url_id: "url_jkl22222_123459"
     - resource_id: "1"
     - depth: 2
     - origin_url: "https://fablabbcn.org/projects"
     - origin_url_id: "url_def67890_123457"

Now we can:
- Find all URLs belonging to resource "1"
- Find all child URLs of "https://fablabbcn.org/projects"
- Trace the discovery path for any URL
- Process URLs individually while maintaining relationships
"""

if __name__ == "__main__":
    import time
    from datetime import datetime
    
    # Example URL data creation
    main_url = EnhancedURLData(
        url="https://fablabbcn.org/",
        depth=0,
        origin_url="",
        resource_id="1",
        main_resource_url="https://fablabbcn.org/"
    )
    
    child_url = EnhancedURLData(
        url="https://fablabbcn.org/projects",
        depth=1,
        origin_url="https://fablabbcn.org/",
        resource_id="1",
        main_resource_url="https://fablabbcn.org/",
        origin_url_id=main_url.url_id
    )
    
    print("Main URL data:")
    print(main_url.to_dict())
    print("\nChild URL data:")
    print(child_url.to_dict())
