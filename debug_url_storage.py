#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Debug script to investigate why URL storage is not finding pending URLs
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.analysis.url_storage import URLStorageManager
from utils.config import get_data_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Debug URL storage issues"""
    try:
        # Get data path
        data_path = get_data_path()
        print(f"Data path: {data_path}")
        
        # Initialize URL storage
        processed_urls_file = os.path.join(data_path, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        print(f"Discovery mode: {url_storage.discovery_mode}")
        print(f"Allow processed URL rediscovery: {url_storage.allow_processed_url_rediscovery}")
        print(f"Pending URLs file: {url_storage.pending_urls_file}")
        print(f"Processed URLs file: {url_storage.processed_urls_file}")
        
        # Check if files exist
        print(f"Pending URLs file exists: {os.path.exists(url_storage.pending_urls_file)}")
        print(f"Processed URLs file exists: {os.path.exists(url_storage.processed_urls_file)}")
        
        # Count processed URLs
        processed_urls = url_storage.get_processed_urls()
        print(f"Processed URLs count: {len(processed_urls)}")
        
        # Check cache
        print(f"Pending URLs cache size: {len(url_storage._pending_urls_cache)}")
        
        # Try to get pending URLs at different levels
        for level in [1, 2, 3, 4]:
            pending_at_level = url_storage.get_pending_urls(depth=level)
            print(f"Pending URLs at level {level}: {len(pending_at_level)}")
        
        # Get all pending URLs
        all_pending = url_storage.get_pending_urls()
        print(f"Total pending URLs: {len(all_pending)}")
        
        # Set discovery mode and try again
        print("\n--- Setting discovery mode to True ---")
        url_storage.set_discovery_mode(True)
        
        # Reload cache and try again
        url_storage.load_pending_urls_cache()
        print(f"Pending URLs cache size after discovery mode: {len(url_storage._pending_urls_cache)}")
        
        # Try to get pending URLs at different levels again
        for level in [1, 2, 3, 4]:
            pending_at_level = url_storage.get_pending_urls(depth=level)
            print(f"Pending URLs at level {level} (discovery mode): {len(pending_at_level)}")
        
        # Get all pending URLs in discovery mode
        all_pending_discovery = url_storage.get_pending_urls()
        print(f"Total pending URLs (discovery mode): {len(all_pending_discovery)}")
        
        # Check a few sample URLs
        if all_pending_discovery:
            print(f"\nSample pending URLs:")
            for i, url_data in enumerate(all_pending_discovery[:5]):
                print(f"  {i+1}. {url_data['url']} (depth: {url_data['depth']}, resource: {url_data['resource_id']})")
        
    except Exception as e:
        logger.error(f"Error during debug: {e}", exc_info=True)

if __name__ == "__main__":
    main()
