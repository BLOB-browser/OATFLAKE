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

def main():
    """
    Main entry point for the level-based processor script.
    Processes all URLs at a specified depth level across all resources.
    """
    parser = argparse.ArgumentParser(description='Process all URLs at a specific depth level across all resources')
    parser.add_argument('--level', type=int, required=True, help='Depth level to process (1=first level, 2=second level, etc.)')
    parser.add_argument('--data-folder', type=str, help='Path to the data folder (default: from config)')
    parser.add_argument('--csv-path', type=str, help='Path to the resources CSV file (default: resources.csv in data folder)')
    parser.add_argument('--max-urls', type=int, help='Maximum number of URLs to process')
    parser.add_argument('--skip-vectors', action='store_true', help='Skip vector generation after processing')
    
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