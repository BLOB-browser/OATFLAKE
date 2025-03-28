#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run Incremental Knowledge Processing

This script processes knowledge incrementally with batching and FAISS index rebuilding.
It's designed to run the complete knowledge pipeline but in smaller chunks to avoid
overwhelming the system, with regular index rebuilding to maintain consistency.

Usage:
    python run_incremental_processing.py [--batch-size 10] [--data-path PATH]
"""

import asyncio
import logging
import argparse
import json
import time
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Document types to process in order
DOCUMENT_TYPES = [
    "methods",       # Process first since methods are critical reference content
    "definitions",   # Process definitions early as they're used in other contexts
    "projects",      # Projects are reference material
    "materials",     # Materials include PDFs
    "resources"      # Resources are typically web content
]

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

def run_batch_processing(doc_type, data_path, batch_size, no_rebuild=False):
    """Run batch processing script for a document type."""
    try:
        script_path = Path(__file__).parent / "process_batch.py"
        
        # Build command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            str(script_path),
            "--type", doc_type,
            "--batch-size", str(batch_size),
            "--data-path", str(data_path)
        ]
        
        # Add no-rebuild flag if specified
        if no_rebuild:
            cmd.append("--no-rebuild")
            
        logger.info(f"Running batch processing: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Handle result
        if result.returncode == 0:
            logger.info(f"Batch processing successful: {doc_type}")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Batch processing failed: {doc_type}")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running batch processing: {e}")
        return False

def run_index_rebuild(data_path):
    """Run FAISS index rebuild script."""
    try:
        script_path = Path(__file__).parent / "rebuild_faiss_indexes.py"
        
        # Build command
        cmd = [
            sys.executable,  # Use the same Python interpreter
            str(script_path),
            "--data-path", str(data_path)
        ]
            
        logger.info(f"Running FAISS index rebuild: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Handle result
        if result.returncode == 0:
            logger.info("FAISS index rebuild successful")
            logger.info(result.stdout)
            return True
        else:
            logger.error("FAISS index rebuild failed")
            logger.error(f"Error: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running FAISS index rebuild: {e}")
        return False

def count_unprocessed_documents(doc_type, data_path):
    """Count unprocessed documents of a specific type."""
    try:
        import pandas as pd
        
        file_path = Path(data_path) / f"{doc_type}.csv"
        if not file_path.exists():
            return 0
            
        df = pd.read_csv(file_path)
        
        # If processed column doesn't exist, all are unprocessed
        if "processed" not in df.columns:
            return len(df)
            
        # Count unprocessed documents
        return len(df[~df["processed"]])
    except Exception as e:
        logger.error(f"Error counting unprocessed {doc_type}: {e}")
        return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run incremental knowledge processing")
    parser.add_argument("--batch-size", type=int, default=10, 
                      help="Number of documents to process in each batch")
    parser.add_argument("--data-path", type=str, 
                      help="Path to data directory (defaults to config.json setting)")
    parser.add_argument("--rebuilds", type=int, default=1,
                      help="Number of times to rebuild FAISS index after each document type")
    
    args = parser.parse_args()
    
    # Get data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_data_path_from_config()
    
    # Display configuration
    logger.info("=== Starting Incremental Knowledge Processing ===")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"FAISS rebuilds per type: {args.rebuilds}")
    
    # Count unprocessed documents
    total_count = 0
    for doc_type in DOCUMENT_TYPES:
        count = count_unprocessed_documents(doc_type, data_path)
        logger.info(f"Unprocessed {doc_type}: {count}")
        total_count += count
    
    logger.info(f"Total unprocessed documents: {total_count}")
    
    if total_count == 0:
        logger.info("No unprocessed documents found. Running final index rebuild...")
        run_index_rebuild(data_path)
        logger.info("Processing complete.")
        return 0
    
    # Process each document type
    start_time = time.time()
    
    for doc_type in DOCUMENT_TYPES:
        type_start_time = time.time()
        logger.info(f"=== Processing {doc_type} ===")
        
        # Keep processing batches until no more unprocessed documents
        batch_count = 0
        while count_unprocessed_documents(doc_type, data_path) > 0:
            batch_count += 1
            logger.info(f"Processing batch {batch_count} of {doc_type}")
            
            # Process this batch
            if not run_batch_processing(doc_type, data_path, args.batch_size, no_rebuild=True):
                logger.error(f"Failed to process batch {batch_count} of {doc_type}")
                break
                
            # Rebuild index periodically
            if batch_count % args.rebuilds == 0:
                logger.info(f"Rebuilding FAISS indexes after batch {batch_count}")
                run_index_rebuild(data_path)
        
        # Rebuild index after completing this document type
        logger.info(f"Completed processing {doc_type}, rebuilding FAISS indexes")
        run_index_rebuild(data_path)
        
        type_duration = time.time() - type_start_time
        logger.info(f"Processed {doc_type} in {type_duration:.2f} seconds")
    
    # Final rebuild to ensure consistency
    logger.info("=== Final FAISS Index Rebuild ===")
    run_index_rebuild(data_path)
    
    # Display completion info
    total_duration = time.time() - start_time
    logger.info(f"=== Processing Complete ===")
    logger.info(f"Total processing time: {total_duration:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    exit(main())