#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process all URLs at a specific depth level across all resources.
This allows prioritizing breadth over depth, ensuring all level 1 URLs
are processed before level 2, and so on.
"""

import sys
import os
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def check_config_level():
    """Check the current_process_level in config.json."""
    try:
        config_path = os.path.join(os.getcwd(), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                level = config.get('current_process_level', 1)
                logger.info(f"Current process level from config: {level}")
                return level
        else:
            logger.warning("Config file not found")
            return 1
    except Exception as e:
        logger.error(f"Error reading current_process_level from config: {e}")
        return 1

def update_config_level(level: int):
    """Update current_process_level in config.json."""
    try:
        config_path = os.path.join(os.getcwd(), "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config['current_process_level'] = level
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Updated current_process_level to {level} in config.json")
            return True
        else:
            logger.warning("Config file not found, couldn't update level")
            return False
    except Exception as e:
        logger.error(f"Error updating current_process_level in config: {e}")
        return False

def main():
    """
    Main entry point for the level-based processor script.
    Processes all URLs at a specified depth level across all resources.
    """
    parser = argparse.ArgumentParser(description='Process all URLs at a specific depth level across all resources')
    parser.add_argument('--level', type=int, help='Depth level to process (1=first level, 2=second level, etc.)')
    parser.add_argument('--next', action='store_true', help='Process the next available level automatically')
    parser.add_argument('--status', action='store_true', help='Show status of all levels')
    parser.add_argument('--data-folder', type=str, help='Path to the data folder (default: from config)')
    parser.add_argument('--csv-path', type=str, help='Path to the resources CSV file (default: resources.csv in data folder)')
    parser.add_argument('--max-urls', type=int, help='Maximum number of URLs to process')
    parser.add_argument('--skip-vectors', action='store_true', help='Skip vector generation after processing')
    parser.add_argument('--reset', action='store_true', help='Reset to level 0 and enable rediscovery mode')
    
    args = parser.parse_args()
    
    # Get data folder from config if not specified
    if not args.data_folder:
        try:
            from utils.config import get_data_path
            args.data_folder = get_data_path()
        except Exception as e:
            logger.error(f"Error getting data path from config: {e}")
            sys.exit(1)
    
    # Set CSV path if not specified
    if not args.csv_path:
        args.csv_path = os.path.join(args.data_folder, 'resources.csv')
    
    # Start time for tracking
    start_time = datetime.now()
    
    try:
        # Import the level processor
        from scripts.analysis.level_processor import LevelBasedProcessor
        
        # Initialize the processor
        processor = LevelBasedProcessor(args.data_folder)
        
        # Handle reset flag to enable rediscovery
        if args.reset:
            logger.info("Resetting to level 0 and enabling URL rediscovery mode")
            # Update config to level 0
            update_config_level(0)
            
            # Import URL storage and ContentFetcher to enable rediscovery mode
            try:
                from scripts.analysis.url_storage import URLStorageManager
                from scripts.analysis.content_fetcher import ContentFetcher
                from utils.config import get_data_path
                
                # Initialize components
                processed_urls_file = os.path.join(get_data_path(), "processed_urls.csv")
                url_storage = URLStorageManager(processed_urls_file)
                content_fetcher = ContentFetcher()
                
                # Enable rediscovery mode
                url_storage.set_rediscovery_mode(True)
                content_fetcher.allow_processed_url_discovery = True
                
                logger.info("URL rediscovery mode enabled. Run with level 1 to start discovery process.")
                return
            except Exception as e:
                logger.error(f"Error setting up rediscovery mode: {e}")
                sys.exit(1)
        
        # Determine which action to take
        if args.status:
            # Show status of all levels
            level_status = processor.get_level_status()
            logger.info("Level Processing Status:")
            for level, status in level_status.items():
                logger.info(f"Level {level}: {status['pending']} pending URLs, Complete: {status['is_complete']}")
            return
        elif args.next:
            # Process the next available level
            logger.info("Processing the next available level")
            result = processor.process_next_available_level()
        elif args.level is not None:
            # Process specific level
            # Log settings
            logger.info(f"Processing level {args.level} URLs")
            logger.info(f"Data folder: {args.data_folder}")
            logger.info(f"Resources CSV: {args.csv_path}")
            if args.max_urls:
                logger.info(f"Max URLs: {args.max_urls}")
            if args.skip_vectors:
                logger.info("Vector generation will be skipped")
            
            # Process the level
            result = processor.process_level(
                level=args.level,
                csv_path=args.csv_path,
                max_urls=args.max_urls,
                skip_vector_generation=args.skip_vectors
            )
        else:
            # Check current level from config
            current_level = check_config_level()
            
            if current_level == 0:
                # We're at level 0, check if we have pending URLs
                # Get data from URL storage
                try:
                    from scripts.analysis.url_storage import URLStorageManager
                    from scripts.analysis.content_fetcher import ContentFetcher
                    from utils.config import get_data_path
                    
                    # Initialize components
                    processed_urls_file = os.path.join(get_data_path(), "processed_urls.csv")
                    url_storage = URLStorageManager(processed_urls_file)
                    content_fetcher = ContentFetcher()
                    
                    # Check if we need discovery
                    discovery_status = content_fetcher.check_discovery_needed(max_depth=4)
                    
                    if discovery_status["discovery_needed"]:
                        logger.info("No pending URLs found and system is at level 0 - enabling URL rediscovery")
                        url_storage.set_rediscovery_mode(True)
                        content_fetcher.allow_processed_url_discovery = True
                        logger.info("URL rediscovery mode enabled.")
                        
                        # Suggest next step
                        logger.info("Please run with level 1 to start discovery process.")
                        return
                    else:
                        logger.info(f"Found {discovery_status['total_pending']} pending URLs. Processing level 1")
                        # Process level 1
                        result = processor.process_level(
                            level=1,
                            csv_path=args.csv_path,
                            max_urls=args.max_urls,
                            skip_vector_generation=args.skip_vectors
                        )
                except Exception as e:
                    logger.error(f"Error checking pending URLs: {e}")
                    sys.exit(1)
            else:
                # Process the current level from config
                logger.info(f"Processing current level {current_level} from config")
                result = processor.process_level(
                    level=current_level,
                    csv_path=args.csv_path,
                    max_urls=args.max_urls,
                    skip_vector_generation=args.skip_vectors
                )
        
        # Log result summary
        if result.get("status") == "completed":
            logger.info(f"Level {args.level} processing completed successfully")
            logger.info(f"URLs processed: {result.get('urls_processed', 0)}")
            logger.info(f"Success count: {result.get('success_count', 0)}")
            logger.info(f"Error count: {result.get('error_count', 0)}")
            logger.info(f"Definitions found: {result.get('definitions_found', 0)}")
            logger.info(f"Projects found: {result.get('projects_found', 0)}")
            logger.info(f"Methods found: {result.get('methods_found', 0)}")
            logger.info(f"Pending URLs remaining at level {args.level}: {result.get('pending_urls_remaining', 0)}")
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Total processing time: {total_time:.2f} seconds")
        else:
            logger.error(f"Level processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Error in level processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()