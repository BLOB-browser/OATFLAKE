#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge Processing Script

This script processes web resources to extract knowledge and generate vector stores.

Usage:
    python process_knowledge.py [--max-resources N] [--force-reanalysis] [--csv-path PATH]
"""

import argparse
import logging
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_data_path_from_config():
    """Get data path from config file."""
    config_paths = [
        Path("config.json"),
        Path.home() / '.blob' / 'config.json',
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'data_path' in config:
                        return Path(config['data_path'])
            except Exception as e:
                logger.warning(f"Error reading config file {config_path}: {e}")
    
    # Default fallback
    return Path.cwd() / "data"

def main():
    """Main entry point for knowledge processing."""
    parser = argparse.ArgumentParser(description="Process web resources to extract knowledge")
    parser.add_argument("--csv-path", help="Path to resources CSV file (default: resources.csv in data folder)")
    parser.add_argument("--max-resources", type=int, help="Maximum resources to process")
    parser.add_argument("--force-reanalysis", action="store_true", help="Force reanalysis of already processed resources")
    
    args = parser.parse_args()
    
    # Get data path from config
    data_folder = get_data_path_from_config()
    logger.info(f"Using data folder: {data_folder}")
    data_folder.mkdir(exist_ok=True, parents=True)
    
    # Default CSV path
    csv_path = args.csv_path or (data_folder / "resources.csv")
    
    # Create main processor
    from scripts.analysis.main_processor import MainProcessor
    processor = MainProcessor(str(data_folder))
    
    # Start processing
    start_time = time.time()
    logger.info(f"Starting knowledge processing at {datetime.now().isoformat()}")
    logger.info(f"Processing resources from: {csv_path}")
    
    # Run the processor
    result = processor.process_resources(
        csv_path=str(csv_path),
        max_resources=args.max_resources,
        force_reanalysis=args.force_reanalysis
    )
    
    # Log results
    duration = time.time() - start_time
    logger.info(f"Knowledge processing completed in {duration:.2f} seconds")
    
    # Display summary
    logger.info("Processing Summary:")
    logger.info(f"- Resources processed: {result.get('resources_processed', 0)}")
    logger.info(f"- Successful: {result.get('success_count', 0)}")
    logger.info(f"- Errors: {result.get('error_count', 0)}")
    logger.info(f"- Definitions found: {result.get('definitions_found', 0)}")
    logger.info(f"- Projects identified: {result.get('projects_found', 0)}")
    logger.info(f"- Methods extracted: {result.get('methods_found', 0)}")
    
    # Vector generation results
    if "vector_stats" in result:
        v_stats = result["vector_stats"]
        logger.info("Vector Generation:")
        logger.info(f"- Status: {v_stats.get('status', 'unknown')}")
        logger.info(f"- Documents processed: {v_stats.get('documents_processed', 0)}")
        logger.info(f"- Stores created: {v_stats.get('stores_created', [])}")
    
    return 0

if __name__ == "__main__":
    exit(main())
