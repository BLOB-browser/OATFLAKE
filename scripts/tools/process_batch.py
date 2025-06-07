#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Document Batch

This script demonstrates processing a specific number of documents from one source
and then rebuilding the FAISS index. It's designed to be used for incremental
processing with a fixed batch size to prevent system overload.

Usage:
    python process_batch.py --type resources --batch-size 10 [--data-path PATH]
"""

import asyncio
import logging
import argparse
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import required components
from langchain.schema import Document

async def process_batch(doc_type, data_path, batch_size=10, rebuild_index=True):
    """Process a batch of documents and optionally rebuild the index."""
    try:
        # Get file path
        base_path = Path(data_path)
        file_path = base_path / f"{doc_type}.csv"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Try to import the modules
        try:
            from scripts.data.embedding_service import EmbeddingService
            from scripts.data.faiss_builder import FAISSBuilder
            
            embedding_service = EmbeddingService()
            faiss_builder = FAISSBuilder(base_path)
            have_modules = True
        except ImportError:
            logger.error("Required modules not found. Make sure scripts/groups has the necessary modules.")
            return False
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Found {len(df)} total {doc_type} entries")
        
        # Find unprocessed items
        if "processed" not in df.columns:
            df["processed"] = False
        
        unprocessed_df = df[~df["processed"]]
        logger.info(f"Found {len(unprocessed_df)} unprocessed entries")
        
        if unprocessed_df.empty:
            logger.info(f"No unprocessed {doc_type} to handle")
            return True
        
        # Limit to batch size
        batch_df = unprocessed_df.head(batch_size)
        logger.info(f"Processing batch of {len(batch_df)} entries")
        
        # Process the batch
        documents = []
        
        for idx, row in batch_df.iterrows():
            try:
                # Convert row to dict
                row_data = {}
                for key, value in row.items():
                    if pd.isna(value):
                        row_data[key] = ''
                    elif isinstance(value, (int, float)):
                        row_data[key] = str(value)
                    else:
                        row_data[key] = value
                
                # Format content based on document type
                content = ""
                if doc_type == "definitions":
                    content = f"Term: {row_data.get('term', '')}\nDefinition: {row_data.get('definition', '')}"
                elif doc_type == "methods":
                    content = f"Method: {row_data.get('title', '')}\nDescription: {row_data.get('description', '')}\nSteps: {row_data.get('steps', '')}"
                elif doc_type == "projects":
                    content = f"Project: {row_data.get('title', '')}\nDescription: {row_data.get('description', '')}\nGoals: {row_data.get('goals', '')}\nAchievement: {row_data.get('achievement', '')}"
                elif doc_type == "resources":
                    content = f"Resource: {row_data.get('title', '')}\nURL: {row_data.get('url', '')}\nDescription: {row_data.get('description', '')}\nTags: {row_data.get('tags', '')}"
                elif doc_type == "materials":
                    content = f"Material: {row_data.get('title', '')}\nDescription: {row_data.get('description', '')}"
                
                # Create Document object
                from datetime import datetime
                doc = Document(
                    page_content=content,
                    metadata={
                        "source_type": doc_type,
                        "csv_file": file_path.name,
                        "processed_at": datetime.now().isoformat(),
                        **row_data
                    }
                )
                documents.append(doc)
                
                # Mark as processed in the DataFrame
                df.at[idx, "processed"] = True
                
            except Exception as row_error:
                logger.error(f"Error processing row: {row_error}")
        
        # Save the updated DataFrame
        df.to_csv(file_path, index=False)
        logger.info(f"Marked {len(documents)} {doc_type} as processed")
        
        if not documents:
            logger.warning("No documents processed")
            return False
        
        # Get store type
        store_name = "reference_store"
        if doc_type in ["resources", "materials"]:
            store_name = "content_store"
          # Process documents with optimized chunking for performance
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,    # Optimized chunk size for better performance
            chunk_overlap=200   # Optimized overlap for better context preservation
        )
        chunks = text_splitter.split_documents(documents)
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = await embedding_service.generate_embeddings(texts)
        
        # Add to store
        logger.info(f"Adding to {store_name}...")
        result = await faiss_builder.create_index(
            texts=texts,
            embeddings_list=embeddings,
            metadatas=metadatas,
            store_name=store_name
        )
        
        # Rebuild all indexes if requested after batch
        if rebuild_index:
            logger.info("Rebuilding all FAISS indexes...")
            rebuild_result = await faiss_builder.rebuild_all_indexes()
            if rebuild_result["status"] == "success":
                logger.info(f"Successfully rebuilt all indexes with {rebuild_result.get('total_documents', 0)} documents")
            else:
                logger.warning(f"Index rebuild warning: {rebuild_result.get('message', 'Unknown issue')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
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
    parser = argparse.ArgumentParser(description="Process document batch")
    parser.add_argument("--type", type=str, required=True, 
                      choices=["definitions", "methods", "projects", "resources", "materials"],
                      help="Document type to process")
    parser.add_argument("--batch-size", type=int, default=10, 
                      help="Number of documents to process in this batch")
    parser.add_argument("--data-path", type=str, 
                      help="Path to data directory (defaults to config.json setting)")
    parser.add_argument("--no-rebuild", action="store_true", 
                      help="Skip rebuilding indexes after processing")
    
    args = parser.parse_args()
    
    # Get data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_data_path_from_config()
    
    logger.info(f"Processing {args.batch_size} {args.type} from {data_path}")
    
    # Run batch processing
    rebuild_index = not args.no_rebuild
    if asyncio.run(process_batch(args.type, data_path, args.batch_size, rebuild_index)):
        logger.info(f"🎉 Batch processing of {args.type} successful")
        return 0
    else:
        logger.error(f"💥 Batch processing of {args.type} failed")
        return 1

if __name__ == "__main__":
    exit(main())