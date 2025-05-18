#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rebuild FAISS Indexes

This script rebuilds all FAISS indexes from stored document data without 
needing to reprocess or re-embed documents. It's designed to be run after 
knowledge processing is complete or interrupted, to ensure vector store
consistency.

Usage:
    python rebuild_faiss_indexes.py [--data-path PATH] [--rebuild-all] [--rebuild-reference]

Options:
    --data-path PATH       Path to data directory (defaults to config.json setting)
    --rebuild-all          Force complete rebuild of all vector stores
    --rebuild-reference    Force complete rebuild of reference store with all document types
"""

import asyncio
import logging
import argparse
import json
import time
import os
import sys
import pandas as pd
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

async def rebuild_indexes(data_path, rebuild_all=False, rebuild_reference=False):
    """Rebuild all FAISS indexes from document data.
    
    Args:
        data_path (Path): Path to data directory
        rebuild_all (bool): Force rebuild of all vector stores
        rebuild_reference (bool): Force rebuild of reference store with all document types
    """
    try:
        start_time = time.time()
        
        # If we're doing a complete reference store rebuild, handle that first
        if rebuild_reference:
            logger.info("🔄 Starting complete reference store rebuild with all document types...")
            from scripts.storage.vector_store_manager import VectorStoreManager
            
            # Initialize vector store manager
            vector_store_manager = VectorStoreManager(base_path=data_path)
            
            # Get reference store path
            vector_stores_path = data_path / "vector_stores" / "default"
            reference_store_path = vector_stores_path / "reference_store"
            
            # Ensure the directory exists
            reference_store_path.mkdir(parents=True, exist_ok=True)
            
            # Create a clean reference store by removing existing documents.json
            if (reference_store_path / "documents.json").exists():
                logger.info("Removing existing reference store documents")
                (reference_store_path / "documents.json").unlink()
                
            # Create empty documents.json
            with open(reference_store_path / "documents.json", "w") as f:
                json.dump([], f)
                
            # Now add all document types
            logger.info("Creating reference store with all document types from scratch")
            total_docs = await add_document_types(data_path, force_all=True)
            logger.info(f"Added {total_docs} total documents to reference store")
            
            # Rebuild the FAISS index for the reference store
            reference_store = await vector_store_manager.get_vector_store("reference_store")
            if reference_store:
                logger.info("Rebuilding reference_store FAISS index")
                await vector_store_manager.rebuild_store("reference_store")
                logger.info("Reference store FAISS index rebuilt successfully")
        
        # First rebuild all existing stores (or if --rebuild-reference isn't specified)
        if use_new_modules:
            logger.info("Using new modular architecture for rebuilding")
            faiss_builder = FAISSBuilder(data_path)
            # Check if FAISSBuilder has a rebuild_all_indexes method with a rebuild_all parameter
            import inspect
            faiss_rebuild_params = inspect.signature(faiss_builder.rebuild_all_indexes).parameters
            if 'rebuild_all' in faiss_rebuild_params:
                result = await faiss_builder.rebuild_all_indexes(rebuild_all=rebuild_all)
            else:
                # Fallback if the parameter isn't supported
                result = await faiss_builder.rebuild_all_indexes()
        else:
            logger.info("Using legacy DataProcessor for rebuilding")
            data_processor = DataProcessor(data_path)
            # Check if DataProcessor has a rebuild_all_vector_stores method with a rebuild_all parameter
            import inspect
            data_processor_params = inspect.signature(data_processor.rebuild_all_vector_stores).parameters
            if 'rebuild_all' in data_processor_params:
                result = await data_processor.rebuild_all_vector_stores(rebuild_all=rebuild_all)
            else:
                # Fallback if the parameter isn't supported
                result = await data_processor.rebuild_all_vector_stores()
        
        if result and result.get("status") == "success":
            logger.info("✅ Successfully rebuilt all FAISS indexes")
            logger.info(f"Rebuilt {len(result.get('stores_rebuilt', []))} stores")
            logger.info(f"Total documents indexed: {result.get('total_documents', 0)}")
            
            # Show individual store stats
            for store_name, doc_count in result.get('document_counts', {}).items():
                logger.info(f"  - {store_name}: {doc_count} documents")
                
            # If we're not forcing a complete rebuild of the reference store,
            # check for missing document types and add them
            if not rebuild_reference:
                logger.info("Checking for missing document types...")
                added_missing = await add_document_types(data_path, check_existing=True)
                if added_missing > 0:
                    logger.info(f"Successfully added {added_missing} missing documents to reference_store")
                else:
                    logger.info("No missing document types needed to be added")
            
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
        import traceback
        logger.error(traceback.format_exc())
        return False

async def add_document_types(data_path, force_all=False, check_existing=False):
    """Add document types (definitions, projects, methods) to reference_store.
    
    Args:
        data_path (Path): Path to data directory
        force_all (bool): Force addition of all document types, even if they exist
        check_existing (bool): Check if document types already exist before adding
        
    Returns:
        int: Total number of documents added
    """
    try:
        import pandas as pd
        from scripts.storage.vector_store_manager import VectorStoreManager
        from langchain.schema import Document
        
        logger.info("Processing document types for reference_store...")
        vector_store_manager = VectorStoreManager(base_path=data_path)
        
        # Check if reference_store exists
        stores = vector_store_manager.list_stores()
        if "reference_store" not in [store.get("name") for store in stores]:
            logger.warning("Reference store not found, creating it")
            
            # Create an empty reference store
            vector_stores_path = data_path / "vector_stores" / "default" / "reference_store"
            vector_stores_path.mkdir(parents=True, exist_ok=True)
            
            # Create empty documents.json
            with open(vector_stores_path / "documents.json", "w") as f:
                json.dump([], f)
            
        # Get correct reference_store path
        vector_stores_path = data_path / "vector_stores" / "default"
        reference_store_path = vector_stores_path / "reference_store"
        
        # Check if directory exists
        if not reference_store_path.exists():
            logger.warning(f"Reference store directory not found at {reference_store_path}")
            
            # As a fallback, try listing stores and finding reference_store
            stores = vector_store_manager.list_stores()
            reference_store_info = None
            
            for store in stores:
                if store.get("name") == "reference_store":
                    reference_store_info = store
                    break
            
            if not reference_store_info:
                logger.warning("Reference store not found in store list")
                return 0
            
            # Get reference store path from the store info
            reference_store_path = Path(reference_store_info.get("path", ""))
            
        logger.info(f"Using reference store path: {reference_store_path}")

        # If checking for existing types, read the documents.json file
        existing_types = set()
        if check_existing and (reference_store_path / "documents.json").exists():
            with open(reference_store_path / "documents.json", "r") as f:
                existing_docs = json.load(f)
                
            # Check for existing document types and fix any issues
            source_types = set()
            for doc in existing_docs:
                if "metadata" in doc:
                    if "source_type" in doc["metadata"]:
                        source_types.add(doc["metadata"]["source_type"])
                    if "type" in doc["metadata"]:
                        existing_types.add(doc["metadata"]["type"])
                    
                    # If document has source_type but wrong or missing type, fix it
                    if "source_type" in doc["metadata"] and (
                        "type" not in doc["metadata"] or 
                        doc["metadata"].get("type") == "unknown" or
                        doc["metadata"].get("type") != doc["metadata"].get("source_type")
                    ):
                        # Set type to match source_type
                        doc["metadata"]["type"] = doc["metadata"]["source_type"]
            
            logger.info(f"Existing document types in reference_store: {existing_types}")

        # Process all document types
        added_docs = 0
        
        # Process definitions if needed
        if "definitions" not in existing_types or not check_existing or force_all:
            # Try multiple possible paths for definitions.csv
            definitions_paths = [
                data_path / "definitions.csv",                    # Root directory
                data_path / "data" / "definitions.csv",           # Data subdirectory
                data_path / "vector_stores" / "definitions.csv",  # Vector stores directory
            ]
            
            definitions_path = None
            for path in definitions_paths:
                if path.exists():
                    logger.info(f"Found definitions at {path}")
                    definitions_path = path
                    break
                    
            if definitions_path and definitions_path.exists():
                try:
                    # Read definitions CSV
                    df = pd.read_csv(definitions_path)
                    logger.info(f"Found {len(df)} definitions to add")
                    
                    # Check for column names - look for either title/description or term/definition
                    has_title = 'title' in df.columns
                    has_term = 'term' in df.columns
                    has_description = 'description' in df.columns
                    has_definition = 'definition' in df.columns
                    
                    logger.info(f"Definition columns: title={has_title}, term={has_term}, description={has_description}, definition={has_definition}")
                    
                    # Convert to documents
                    documents = []
                    for _, row in df.iterrows():
                        # First check for title/description fields, fall back to term/definition if not found
                        title = row.get('title', row.get('term', ''))
                        description = row.get('description', row.get('definition', ''))
                        content = f"TITLE: {title}\nDESCRIPTION: {description}"
                        
                        # Create metadata from row data
                        metadata = {
                            "source_type": "definitions",
                            "type": "definitions",  # Explicitly set type to match source_type
                            "processed_at": datetime.now().isoformat(),
                        }
                        
                        # Add scalar values from row to metadata
                        for col in df.columns:
                            if col in ["title", "description", "term", "definition"]:
                                continue
                                
                            value = row.get(col)
                            if pd.notna(value) and isinstance(value, (str, int, float, bool)):
                                metadata[col] = value
                        
                        # Create document
                        documents.append(Document(page_content=content, metadata=metadata))
                    
                    if documents:
                        logger.info(f"Adding {len(documents)} definitions to reference_store")
                        
                        # Create metadata for tracking
                        metadata = {
                            "data_type": "definitions",
                            "item_count": len(documents),
                            "source": "rebuild_faiss_indexes",
                            "added_at": datetime.now().isoformat()
                        }
                        
                        # Add to reference store
                        result = await vector_store_manager.add_documents_to_store(
                            "reference_store", 
                            documents, 
                            metadata=metadata,
                            update_stats=True
                        )
                        
                        if result:
                            logger.info(f"✅ Successfully added {len(documents)} definitions")
                            added_docs += len(documents)
                        else:
                            logger.error("❌ Failed to add definitions")
                except Exception as e:
                    logger.error(f"Error adding definitions: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.info("No definitions.csv file found")
        else:
            logger.info("Definitions already in reference store, skipping")
        
        # Process projects if needed
        if "projects" not in existing_types or not check_existing or force_all:
            # Try multiple possible paths for projects.csv
            projects_paths = [
                data_path / "projects.csv",                    # Root directory
                data_path / "data" / "projects.csv",           # Data subdirectory
                data_path / "vector_stores" / "projects.csv",  # Vector stores directory
            ]
            
            projects_path = None
            for path in projects_paths:
                if path.exists():
                    logger.info(f"Found projects at {path}")
                    projects_path = path
                    break
                    
            if projects_path and projects_path.exists():
                try:
                    # Read projects CSV
                    df = pd.read_csv(projects_path)
                    logger.info(f"Found {len(df)} projects to add")
                    
                    # Convert to documents
                    documents = []
                    for _, row in df.iterrows():
                        content = f"PROJECT: {row.get('title', '')}\nDESCRIPTION: {row.get('description', '')}\nGOALS: {row.get('goals', '')}"
                        
                        # Create metadata from row data
                        metadata = {
                            "source_type": "projects",
                            "type": "projects",  # Explicitly set type to match source_type
                            "processed_at": datetime.now().isoformat(),
                        }
                        
                        # Add scalar values from row to metadata
                        for col in df.columns:
                            if col in ["title", "description", "goals"]:
                                continue
                                
                            value = row.get(col)
                            if pd.notna(value) and isinstance(value, (str, int, float, bool)):
                                metadata[col] = value
                        
                        # Create document
                        documents.append(Document(page_content=content, metadata=metadata))
                    
                    if documents:
                        logger.info(f"Adding {len(documents)} projects to reference_store")
                        
                        # Create metadata for tracking
                        metadata = {
                            "data_type": "projects",
                            "item_count": len(documents),
                            "source": "rebuild_faiss_indexes",
                            "added_at": datetime.now().isoformat()
                        }
                        
                        # Add to reference store
                        result = await vector_store_manager.add_documents_to_store(
                            "reference_store", 
                            documents, 
                            metadata=metadata,
                            update_stats=True
                        )
                        
                        if result:
                            logger.info(f"✅ Successfully added {len(documents)} projects")
                            added_docs += len(documents)
                        else:
                            logger.error("❌ Failed to add projects")
                except Exception as e:
                    logger.error(f"Error adding projects: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.info("No projects.csv file found")
        else:
            logger.info("Projects already in reference store, skipping")
        
        # Process methods if needed
        if "methods" not in existing_types or not check_existing or force_all:
            # Try multiple possible paths for methods.csv
            methods_paths = [
                data_path / "methods.csv",                    # Root directory
                data_path / "data" / "methods.csv",           # Data subdirectory
                data_path / "vector_stores" / "methods.csv",  # Vector stores directory
            ]
            
            methods_path = None
            for path in methods_paths:
                if path.exists():
                    logger.info(f"Found methods at {path}")
                    methods_path = path
                    break
                    
            if methods_path and methods_path.exists():
                try:
                    # Read methods CSV
                    df = pd.read_csv(methods_path)
                    logger.info(f"Found {len(df)} methods to add")
                    
                    # Convert to documents
                    documents = []
                    for _, row in df.iterrows():
                        # Create content based on available fields
                        content_parts = []
                        if 'name' in df.columns and pd.notna(row.get('name')):
                            content_parts.append(f"METHOD: {row.get('name')}")
                        if 'description' in df.columns and pd.notna(row.get('description')):
                            content_parts.append(f"DESCRIPTION: {row.get('description')}")
                        if 'steps' in df.columns and pd.notna(row.get('steps')):
                            content_parts.append(f"STEPS: {row.get('steps')}")
                            
                        content = "\n".join(content_parts)
                        
                        # Create metadata
                        metadata = {
                            "source_type": "methods",
                            "type": "methods",  # Explicitly set type to match source_type
                            "processed_at": datetime.now().isoformat(),
                        }
                        
                        # Add scalar values from row to metadata
                        for col in df.columns:
                            if col in ["name", "description", "steps"]:
                                continue
                                
                            value = row.get(col)
                            if pd.notna(value) and isinstance(value, (str, int, float, bool)):
                                metadata[col] = value
                            
                        # Create document
                        documents.append(Document(page_content=content, metadata=metadata))
                    
                    if documents:
                        logger.info(f"Adding {len(documents)} methods to reference_store")
                        
                        # Create metadata for tracking
                        metadata = {
                            "data_type": "methods",
                            "item_count": len(documents),
                            "source": "rebuild_faiss_indexes",
                            "added_at": datetime.now().isoformat()
                        }
                        
                        # Add to reference store
                        result = await vector_store_manager.add_documents_to_store(
                            "reference_store", 
                            documents, 
                            metadata=metadata,
                            update_stats=True
                        )
                        
                        if result:
                            logger.info(f"✅ Successfully added {len(documents)} methods")
                            added_docs += len(documents)
                        else:
                            logger.error("❌ Failed to add methods")
                except Exception as e:
                    logger.error(f"Error adding methods: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.info("No methods.csv file found")
        else:
            logger.info("Methods already in reference store, skipping")
        
        return added_docs
        
    except Exception as e:
        logger.error(f"Error adding document types: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

def get_data_path_from_config():
    """Get data path from config file."""
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Prioritize config.json in the project root directory
    config_paths = [
        project_root / "config.json",  # Project root config
        Path.cwd() / "config.json",    # Current working directory
        Path.home() / '.blob' / 'config.json',  # User home directory
    ]
    
    for config_path in config_paths:
        logger.info(f"Checking for config at: {config_path}")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'data_path' in config:
                        data_path = Path(config['data_path'])
                        logger.info(f"Found data_path in config: {data_path}")
                        return data_path
            except Exception as e:
                logger.warning(f"Error reading config file {config_path}: {e}")
    
    # Default fallback - use the project root
    logger.warning("No data_path found in config files, using project root as fallback")
    return project_root

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rebuild FAISS indexes for vector stores")
    parser.add_argument("--data-path", type=str, help="Path to data directory (defaults to config.json setting)")
    parser.add_argument("--rebuild-all", action="store_true", help="Force complete rebuild of all vector stores")
    parser.add_argument("--rebuild-reference", action="store_true", help="Force complete rebuild of reference store with all document types")
    
    args = parser.parse_args()
    
    # Get data path from arguments or config
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_data_path_from_config()
    
    logger.info(f"Using data path: {data_path}")
    
    # Run the rebuild process
    if asyncio.run(rebuild_indexes(data_path, rebuild_all=args.rebuild_all, rebuild_reference=args.rebuild_reference)):
        if args.rebuild_reference:
            logger.info("🎉 Complete rebuild successful")
        else:
            logger.info("🎉 FAISS index rebuild successful")
        return 0
    else:
        logger.error("💥 FAISS index rebuild failed")
        return 1

if __name__ == "__main__":
    exit(main())
