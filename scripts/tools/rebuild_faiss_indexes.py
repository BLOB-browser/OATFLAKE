#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rebuild FAISS Indexes

This script rebuilds all FAISS indexes from stored document data without 
needing to reprocess or re-embed documents. It's designed to be run after 
knowledge processing is complete or interrupted, to ensure vector store
consistency.

Usage:
    python rebuild_faiss_indexes.py [--data-path PATH]
"""

import asyncio
import logging
import argparse
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import from scripts.groups
try:
    # Try the new modules first
    from scripts.data.faiss_builder import FAISSBuilder
    use_new_modules = True
except ImportError:
    # Fall back to the original implementation
    from scripts.data.data_processor import DataProcessor
    use_new_modules = False

async def rebuild_indexes(data_path):
    """Rebuild all FAISS indexes from document data."""
    try:
        start_time = time.time()
        
        if use_new_modules:
            logger.info("Using new modular architecture for rebuilding")
            faiss_builder = FAISSBuilder(data_path)
            result = await faiss_builder.rebuild_all_indexes()
        else:
            logger.info("Using legacy DataProcessor for rebuilding")
            data_processor = DataProcessor(data_path)
            result = await data_processor.rebuild_all_vector_stores()
        
        if result and result.get("status") == "success":
            logger.info("✅ Successfully rebuilt all FAISS indexes")
            logger.info(f"Rebuilt {len(result.get('stores_rebuilt', []))} stores")
            logger.info(f"Total documents indexed: {result.get('total_documents', 0)}")
            
            # Show individual store stats
            for store_name, doc_count in result.get('document_counts', {}).items():
                logger.info(f"  - {store_name}: {doc_count} documents")
                
            # Log completion time
            duration = time.time() - start_time
            logger.info(f"⏱️ Rebuild completed in {duration:.2f} seconds")
            return True
        else:
            error_msg = result.get("message", "Unknown error") if result else "No result returned"
            logger.error(f"❌ Failed to rebuild indexes: {error_msg}")
            return False
    except Exception as e:
        logger.error(f"❌ Error rebuilding indexes: {e}")
        return False

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
    return Path.cwd()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rebuild FAISS indexes for vector stores")
    parser.add_argument("--data-path", type=str, help="Path to data directory (defaults to config.json setting)")
    
    args = parser.parse_args()
    
    # Get data path from arguments or config
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_data_path_from_config()
    
    logger.info(f"Using data path: {data_path}")
    
    # Run the rebuild process
    if asyncio.run(rebuild_indexes(data_path)):
        logger.info("🎉 FAISS index rebuild successful")
        return 0
    else:
        logger.error("💥 FAISS index rebuild failed")
        return 1

if __name__ == "__main__":
    exit(main())