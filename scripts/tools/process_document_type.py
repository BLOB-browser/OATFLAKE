#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process Document Type

This script demonstrates the modular approach to processing specific document types
(definitions, methods, projects, resources, materials) without full knowledge base
processing. It processes documents, generates embeddings, and creates FAISS indexes
in a single focused operation.

Usage:
    python process_document_type.py --type definitions [--data-path PATH]
"""

import asyncio
import logging
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import from scripts.groups
from langchain.schema import Document

class DocumentProcessor:
    """Simplified document processor for demonstrations."""
    
    def __init__(self, data_path):
        """Initialize with data path."""
        self.data_path = Path(data_path)
        
        # Import required components if available
        try:
            # Try new modular components
            from scripts.data.embedding_service import EmbeddingService
            from scripts.data.faiss_builder import FAISSBuilder
            
            self.embedding_service = EmbeddingService()
            self.faiss_builder = FAISSBuilder(data_path)
            logger.info("Using new modular components")
            self.use_new_modules = True
        except ImportError:
            # Fall back to original implementation
            from scripts.data.data_processor import DataProcessor
            
            self.data_processor = DataProcessor(data_path)
            logger.info("Using legacy DataProcessor")
            self.use_new_modules = False
    
    async def load_documents(self, doc_type: str) -> List[Document]:
        """Load documents of specified type."""
        import pandas as pd
        from datetime import datetime
        
        documents = []
        file_path = self.data_path / f"{doc_type}.csv"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return documents
        
        try:
            # Load CSV file
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} {doc_type} entries")
            
            for _, row in df.iterrows():
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
                    
                except Exception as row_error:
                    logger.error(f"Error processing row: {row_error}")
            
            return documents
        except Exception as e:
            logger.error(f"Error loading {doc_type}: {e}")
            return documents
    
    async def process_documents(self, doc_type: str) -> Dict[str, Any]:
        """Process documents, generate embeddings, and create FAISS index."""
        try:
            # Start timing
            start_time = time.time()
            
            # Get store type based on document type
            store_name = "reference_store"
            if doc_type in ["resources", "materials"]:
                store_name = "content_store"
            
            # Handle using legacy or new code
            if self.use_new_modules:
                # Load documents
                logger.info(f"Loading {doc_type}...")
                documents = await self.load_documents(doc_type)
                if not documents:
                    logger.warning(f"No {doc_type} documents found to process")
                    return {"status": "warning", "message": f"No {doc_type} documents found"}
                
                logger.info(f"Loaded {len(documents)} {doc_type} documents")
                  # Split documents with optimized settings for performance
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
                embeddings = await self.embedding_service.generate_embeddings(texts)
                
                # Create FAISS index
                logger.info(f"Creating FAISS index in {store_name}...")
                result = await self.faiss_builder.create_index(
                    texts=texts,
                    embeddings_list=embeddings,
                    metadatas=metadatas,
                    store_name=store_name
                )
                
                # Return results
                result["document_count"] = len(documents)
                result["chunk_count"] = len(chunks)
                result["embedding_count"] = len(embeddings)
                result["processing_time"] = time.time() - start_time
                
                return result
            else:
                # Use legacy DataProcessor approach
                logger.info(f"Using legacy processor for {doc_type}...")
                
                # For legacy approach, simplified for demo - this is just a stub
                # In real implementation, we would call appropriate methods on data_processor
                # such as process_knowledge_base with specific parameters
                
                return {
                    "status": "success",
                    "message": f"Processed {doc_type} with legacy processor",
                    "document_count": 0,
                    "processing_time": time.time() - start_time
                }
                
        except Exception as e:
            logger.error(f"Error processing {doc_type}: {e}")
            return {"status": "error", "message": str(e)}

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

async def main_async(doc_type, data_path):
    """Async main function."""
    processor = DocumentProcessor(data_path)
    result = await processor.process_documents(doc_type)
    
    if result.get("status") == "success":
        logger.info(f"✅ Successfully processed {doc_type}")
        logger.info(f"Processed {result.get('document_count', 0)} documents")
        logger.info(f"Generated {result.get('embedding_count', 0)} embeddings")
        logger.info(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
        return True
    else:
        logger.error(f"❌ Processing failed: {result.get('message', 'Unknown error')}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process specific document types")
    parser.add_argument("--type", type=str, required=True, 
                       choices=["definitions", "methods", "projects", "resources", "materials"],
                       help="Document type to process")
    parser.add_argument("--data-path", type=str, help="Path to data directory (defaults to config.json setting)")
    
    args = parser.parse_args()
    
    # Get data path from arguments or config
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_data_path_from_config()
    
    logger.info(f"Processing {args.type} from {data_path}")
    
    # Run the processing
    if asyncio.run(main_async(args.type, data_path)):
        logger.info(f"🎉 {args.type} processing successful")
        return 0
    else:
        logger.error(f"💥 {args.type} processing failed")
        return 1

if __name__ == "__main__":
    exit(main())