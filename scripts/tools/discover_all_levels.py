#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Discover all URLs at all levels (1-max_depth) for all resources.
This script ensures complete crawling of all resources to the maximum depth, but does NOT analyze/process them.
"""
import sys
import os
import logging
import asyncio
import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path

# Add the parent directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def discover_all_levels(max_depth: int = None, force_discovery: bool = True) -> Dict[str, Any]:
    """
    For each resource, discover all levels (1 to max_depth) sequentially before moving to the next resource.
    No analysis/processing is performed.
    """
    try:
        from scripts.analysis.url_discovery_manager import URLDiscoveryManager
        from utils.config import get_data_path
        
        # Get data path from config
        data_path = get_data_path()
        logger.info(f"Using data folder: {data_path}")
        
        # Read max_depth from config.json if not provided
        if max_depth is None:
            config_path = Path(__file__).parents[2] / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                max_depth = config.get('crawl_config', {}).get('max_depth', 4)
                logger.info(f"Read max_depth={max_depth} from config.json")
            else:
                max_depth = 4
                logger.warning("config.json not found, using default max_depth=4")
        
        # Initialize URL discovery manager
        discovery_manager = URLDiscoveryManager(data_path)
        # Enable discovery-only mode in the URL discovery engine
        discovery_manager.url_discovery.discovery_only_mode = True
        
        # Load resources
        resources_csv_path = Path(data_path) / 'resources.csv'
        if not resources_csv_path.exists():
            logger.error(f"Resources file not found: {resources_csv_path}")
            return {"status": "error", "message": f"Resources file not found: {resources_csv_path}"}
        resources_df = pd.read_csv(resources_csv_path)
        resources_with_url = resources_df[resources_df['url'].notna()]
        
        results = {
            "status": "success",
            "resources": {},
            "total_urls_discovered": 0
        }
        
        # For each resource, discover all levels before moving to the next
        for idx, row in resources_with_url.iterrows():
            resource = row.to_dict()
            resource_url = resource.get('url', '')
            resource_id = resource.get('id', '') or str(idx)
            if not resource_url:
                continue
                
            logger.info(f"===== DISCOVERING ALL LEVELS FOR RESOURCE: {resource_id} - {resource_url} =====")
            resource_result = {"levels": {}, "total_discovered": 0}
            
            for level in range(1, max_depth + 1):
                logger.info(f"Discovering level {level} for resource: {resource_id}")
                
                # Call discovery_phase to discover URLs at this level using URL discovery engine
                discovery_result = discovery_manager.url_discovery.discovery_phase(
                    urls=[resource_url],
                    max_depth=level,
                    force_reprocess=force_discovery,
                    resource_ids={resource_url: resource_id}
                )
                
                # Extract the number of discovered URLs at this level
                discovered = discovery_result.get("total_discovered", 0)
                discovered_by_level = discovery_result.get("discovered_by_level", {})
                
                # Store the results for this level
                resource_result["levels"][f"level_{level}"] = {
                    "discovered": discovered,
                    "by_level": discovered_by_level
                }
                
                # Accumulate total discovered URLs for this resource
                resource_result["total_discovered"] += discovered
                
                # Add to overall total
                results["total_urls_discovered"] += discovered
                
                logger.info(f"Resource {resource_id} level {level} discovery: {discovered} URLs discovered")
            
            # Store results for this resource
            results["resources"][resource_id] = resource_result
        
        logger.info(f"===== ALL RESOURCES DISCOVERED =====")
        logger.info(f"Total URLs discovered: {results['total_urls_discovered']}")
        
        # Check if any URLs were discovered and saved to pending
        from scripts.analysis.url_storage import URLStorageManager
        processed_urls_file = os.path.join(data_path, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Get counts of pending URLs by level for verification
        pending_counts = {}
        for level in range(1, max_depth + 1):
            pending_urls = url_storage.get_pending_urls(depth=level)
            pending_counts[level] = len(pending_urls)
        
        logger.info(f"Verification - Pending URLs in the system by level: {pending_counts}")
        results["pending_url_counts"] = pending_counts
        
        return results
    except Exception as e:
        logger.error(f"Error discovering all levels: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# If run as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover all URLs at all depth levels across all resources (discovery only, no analysis)')
    parser.add_argument('--max-depth', type=int, default=None, help='Maximum depth level to discover (default: read from config.json)')
    parser.add_argument('--no-force', action='store_true', help='Disable force discovery (by default, discovery is forced)')
    args = parser.parse_args()
    
    asyncio.run(discover_all_levels(max_depth=args.max_depth, force_discovery=not args.no_force))