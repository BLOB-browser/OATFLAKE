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
        
        # First rebuild all existing stores
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
            
            # Check if we need to generate topic stores
            # Count how many topic stores were rebuilt
            topic_stores = [store for store in result.get('stores_rebuilt', []) if store.startswith("topic_")]
            
            if len(topic_stores) < 3:
                logger.info(f"Few topic stores created ({len(topic_stores)}), attempting to generate topic stores")
                
                # Import vector store manager for topic store generation
                from scripts.storage.vector_store_manager import VectorStoreManager
                
                # Initialize with the data path
                vector_store_manager = VectorStoreManager(base_path=data_path)
                
                # Check if content_store exists
                stores = vector_store_manager.list_stores()
                if "content_store" in [store.get("name") for store in stores]:
                    logger.info("Content store exists, getting representative docs for topics")
                    
                    # Get representative chunks to create topic stores
                    rep_docs = await vector_store_manager.get_representative_chunks(
                        store_name="content_store", 
                        num_chunks=100
                    )
                    
                    if rep_docs:
                        logger.info(f"Got {len(rep_docs)} representative documents for topic generation")
                        
                        # Try to create topic stores from these docs
                        topic_results = await vector_store_manager.create_topic_stores(rep_docs)
                        
                        if topic_results:
                            logger.info(f"Created {len(topic_results)} additional topic stores")
                            # Log the created topic stores
                            for topic, success in topic_results.items():
                                if success:
                                    logger.info(f"  - Created topic store for: {topic}")
                    else:
                        logger.warning("No representative documents found for topic generation")
                else:
                    logger.warning("Content store not found, cannot generate topic stores")
            else:
                logger.info(f"Found {len(topic_stores)} topic stores, no need to generate more")
                
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