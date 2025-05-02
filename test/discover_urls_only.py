#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
from pathlib import Path
import pandas as pd
import asyncio

# Add the parent directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    """
    Discover URLs from all resources without analyzing them.
    """
    try:
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from utils.config import get_data_path
        
        # Get data path
        data_folder = get_data_path()
        logger.info(f"Using data folder: {data_folder}")
        
        # Initialize SingleResourceProcessor
        single_processor = SingleResourceProcessor(data_folder)
        
        # Get resources CSV path
        resources_csv_path = os.path.join(data_folder, 'resources.csv')
        if not os.path.exists(resources_csv_path):
            logger.error(f"Resources file not found: {resources_csv_path}")
            return
            
        # Load resources
        resources_df = pd.read_csv(resources_csv_path)
        resources_with_url = resources_df[resources_df['url'].notna()]
        
        # Initialize results
        total_discovered = 0
        
        # Process each resource for URL discovery only
        for idx, row in resources_with_url.iterrows():
            resource = row.to_dict()
            resource_url = resource.get('url', '')
            resource_id = f"{idx+1}/{len(resources_with_url)}"
            resource_title = resource.get('title', 'Unnamed')
            
            logger.info(f"[{resource_id}] Discovering URLs from: {resource_title} - {resource_url}")
            
            # Add the discover_all_urls_from_resource method to SingleResourceProcessor
            # This will be a new method that discovers URLs without analyzing content
            # Call the method here once it's implemented
            urls_discovered = await single_processor.discover_all_urls(
                resource=resource,
                resource_id=resource_id,
                max_depth=4  # Default max depth
            )
            
            logger.info(f"[{resource_id}] Discovered {urls_discovered} URLs from {resource_title}")
            total_discovered += urls_discovered
            
        logger.info(f"URL discovery complete. Total: {total_discovered} URLs discovered")
        
    except Exception as e:
        logger.error(f"Error in URL discovery: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
