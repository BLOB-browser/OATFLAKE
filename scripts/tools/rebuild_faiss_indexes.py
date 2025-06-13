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
            
            # The FAISS index is automatically built when documents are added
            logger.info("Reference store FAISS index has been rebuilt successfully")
        
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

def sanitize_value(value):
    """Sanitize a value to ensure it's JSON-serializable and handles all types of NaN/null values."""
    import pandas as pd
    import numpy as np
    import math
    
    # Handle pandas/numpy NaN values
    if pd.isna(value):
        return ""
    
    # Handle numpy NaN specifically
    if isinstance(value, (np.floating, float)) and (math.isnan(value) if isinstance(value, (int, float, np.number)) else False):
        return ""
    
    # Handle None
    if value is None:
        return ""
    
    # Handle string representations of null
    if isinstance(value, str) and value.lower() in ['nan', 'null', 'none', '']:
        return ""
    
    # Convert pandas/numpy types to native Python types
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    
    # Handle numpy bool
    if isinstance(value, (np.bool_, np.bool8)):
        return bool(value)
    
    # Handle numpy integers
    if isinstance(value, (np.integer, np.int_, np.int8, np.int16, np.int32, np.int64)):
        return int(value)
    
    # Handle numpy floats
    if isinstance(value, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        if math.isnan(value):
            return ""
        return float(value)
    
    # Return the value as-is if it's already a basic Python type
    return value

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
        import numpy as np
        import math
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
                    
                    # Convert to documents using universal schema with backward compatibility
                    documents = []
                    for _, row in df.iterrows():
                        # Universal schema mapping with backward compatibility
                        # Priority: new universal schema -> old CSV schema -> fallbacks
                        title = (row.get('title') or 
                                row.get('term') or 
                                'Untitled')
                        
                        description = (row.get('description') or 
                                     row.get('definition') or 
                                     '')
                        
                        # Handle content_type from universal schema or infer from context
                        content_type = (row.get('content_type') or 
                                      'definition')  # Default for definitions
                        
                        # Create rich content that preserves structure
                        content_parts = []
                        content_parts.append(f"TITLE: {title}")
                        if description:
                            content_parts.append(f"DESCRIPTION: {description}")
                        
                        # Add purpose if available (universal schema)
                        purpose = row.get('purpose', '')
                        if purpose:
                            content_parts.append(f"PURPOSE: {purpose}")
                            
                        # Add location if available (universal schema)
                        location = row.get('location', '')
                        if location:
                            content_parts.append(f"LOCATION: {location}")
                        
                        content = '\n'.join(content_parts)
                        
                        # Create comprehensive metadata using universal schema
                        metadata = {
                            # Core fields (always present)
                            "source_type": content_type,
                            "type": content_type,
                            "title": title,  # Always store title in metadata
                            "description": description,  # Always store description in metadata
                            "processed_at": datetime.now().isoformat(),
                            
                            # Universal schema fields (primary) - sanitize all values
                            "id": sanitize_value(row.get('id', '')),
                            "content_type": content_type,
                            "origin_url": sanitize_value(row.get('origin_url', '')),
                            "tags": sanitize_value(row.get('tags', [])),
                            "purpose": sanitize_value(purpose),
                            "location": sanitize_value(location),
                            "related_url": sanitize_value(row.get('related_url', '')),
                            "status": sanitize_value(row.get('status', '')),
                            "creator_id": sanitize_value(row.get('creator_id', '')),
                            "collaborators": sanitize_value(row.get('collaborators', '')),
                            "group_id": sanitize_value(row.get('group_id', '')),
                            "created_at": sanitize_value(row.get('created_at', '')),
                            "last_updated_at": sanitize_value(row.get('last_updated_at', '')),
                            "analysis_completed": sanitize_value(row.get('analysis_completed', '')),
                            "visibility": sanitize_value(row.get('visibility', '')),
                        }
                        
                        # Add backward compatibility fields (old CSV schema)
                        legacy_fields = {
                            'term': row.get('term', ''),
                            'definition': row.get('definition', ''),
                            'resource_url': row.get('resource_url', ''),
                            'source_text': row.get('source_text', ''),
                            'category': row.get('category', ''),
                            'source': row.get('source', ''),
                        }
                        
                        # Only add legacy fields if they have values
                        for key, value in legacy_fields.items():
                            sanitized_value = sanitize_value(value)
                            if sanitized_value:  # Only add if not empty after sanitization
                                metadata[key] = sanitized_value
                        
                        # Add any additional columns not covered above
                        excluded_cols = {
                            'title', 'description', 'term', 'definition', 'id', 'content_type',
                            'origin_url', 'tags', 'purpose', 'location', 'related_url', 'status',
                            'creator_id', 'collaborators', 'group_id', 'created_at', 
                            'last_updated_at', 'analysis_completed', 'visibility', 'resource_url',
                            'source_text', 'category', 'source'
                        }
                        
                        for col in df.columns:
                            if col not in excluded_cols:
                                value = row.get(col)
                                sanitized_value = sanitize_value(value)
                                if sanitized_value:  # Only add if not empty after sanitization
                                    metadata[col] = sanitized_value
                        
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
                    
                    # Convert to documents using universal schema with backward compatibility
                    documents = []
                    for _, row in df.iterrows():
                        # Universal schema mapping with backward compatibility
                        title = (row.get('title') or 
                                row.get('project_name') or 
                                'Untitled Project')
                        
                        description = (row.get('description') or 
                                     row.get('project_description') or 
                                     '')
                        
                        goals = row.get('goals', '')
                        
                        # Handle content_type from universal schema or infer from context
                        content_type = (row.get('content_type') or 
                                      'project')  # Default for projects
                        
                        # Create rich content that preserves structure
                        content_parts = []
                        content_parts.append(f"TITLE: {title}")
                        if description:
                            content_parts.append(f"DESCRIPTION: {description}")
                        if goals:
                            content_parts.append(f"GOALS: {goals}")
                        
                        # Add purpose if available (universal schema)
                        purpose = row.get('purpose', '')
                        if purpose:
                            content_parts.append(f"PURPOSE: {purpose}")
                            
                        # Add location if available (universal schema)
                        location = row.get('location', '')
                        if location:
                            content_parts.append(f"LOCATION: {location}")
                        
                        content = '\n'.join(content_parts)
                        
                        # Create comprehensive metadata using universal schema
                        metadata = {
                            # Core fields (always present)
                            "source_type": content_type,
                            "type": content_type,
                            "title": title,  # Always store title in metadata
                            "description": description,  # Always store description in metadata
                            "processed_at": datetime.now().isoformat(),
                            
                            # Universal schema fields (primary) - sanitize all values
                            "id": sanitize_value(row.get('id', '')),
                            "content_type": content_type,
                            "origin_url": sanitize_value(row.get('origin_url', '')),
                            "tags": sanitize_value(row.get('tags', [])),
                            "purpose": sanitize_value(purpose),
                            "location": sanitize_value(location),
                            "related_url": sanitize_value(row.get('related_url', '')),
                            "status": sanitize_value(row.get('status', '')),
                            "creator_id": sanitize_value(row.get('creator_id', '')),
                            "collaborators": sanitize_value(row.get('collaborators', '')),
                            "group_id": sanitize_value(row.get('group_id', '')),
                            "created_at": sanitize_value(row.get('created_at', '')),
                            "last_updated_at": sanitize_value(row.get('last_updated_at', '')),
                            "analysis_completed": sanitize_value(row.get('analysis_completed', '')),
                            "visibility": sanitize_value(row.get('visibility', '')),
                        }
                        
                        # Add backward compatibility fields (old CSV schema)
                        legacy_fields = {
                            'goals': goals,
                            'project_name': row.get('project_name', ''),
                            'project_description': row.get('project_description', ''),
                            'resource_url': row.get('resource_url', ''),
                            'source_text': row.get('source_text', ''),
                            'category': row.get('category', ''),
                            'source': row.get('source', ''),
                        }
                        
                        # Only add legacy fields if they have values
                        for key, value in legacy_fields.items():
                            if value and pd.notna(value):
                                # Convert numpy/pandas types to native Python types and handle NaN
                                if isinstance(value, (pd.Series, pd.DataFrame)):
                                    continue
                                if pd.isna(value):
                                    continue
                                # Convert numpy types to Python native types
                                if hasattr(value, 'item'):
                                    value = value.item()
                                metadata[key] = value
                        
                        # Add any additional columns not covered above
                        excluded_cols = {
                            'title', 'description', 'goals', 'project_name', 'project_description',
                            'id', 'content_type', 'origin_url', 'tags', 'purpose', 'location', 
                            'related_url', 'status', 'creator_id', 'collaborators', 'group_id', 
                            'created_at', 'last_updated_at', 'analysis_completed', 'visibility', 
                            'resource_url', 'source_text', 'category', 'source'
                        }
                        
                        for col in df.columns:
                            if col not in excluded_cols:
                                value = row.get(col)
                                if pd.notna(value) and isinstance(value, (str, int, float, bool)):
                                    # Convert numpy/pandas types to native Python types and handle NaN
                                    if isinstance(value, (pd.Series, pd.DataFrame)):
                                        continue
                                    if pd.isna(value):
                                        continue
                                    # Convert numpy types to Python native types
                                    if hasattr(value, 'item'):
                                        value = value.item()
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
                    
                    # Convert to documents using universal schema with backward compatibility
                    documents = []
                    for _, row in df.iterrows():
                        # Universal schema mapping with backward compatibility
                        title = (row.get('title') or 
                                row.get('name') or 
                                row.get('method_name') or 
                                'Untitled Method')
                        
                        description = (row.get('description') or 
                                     row.get('method_description') or 
                                     '')
                        
                        steps = row.get('steps', '')
                        
                        # Handle content_type from universal schema or infer from context
                        content_type = (row.get('content_type') or 
                                      'method')  # Default for methods
                        
                        # Create rich content that preserves structure
                        content_parts = []
                        content_parts.append(f"TITLE: {title}")
                        if description:
                            content_parts.append(f"DESCRIPTION: {description}")
                        if steps:
                            content_parts.append(f"STEPS: {steps}")
                        
                        # Add purpose if available (universal schema)
                        purpose = row.get('purpose', '')
                        if purpose:
                            content_parts.append(f"PURPOSE: {purpose}")
                            
                        # Add location if available (universal schema)
                        location = row.get('location', '')
                        if location:
                            content_parts.append(f"LOCATION: {location}")
                        
                        content = '\n'.join(content_parts)
                        
                        # Create comprehensive metadata using universal schema
                        metadata = {
                            # Core fields (always present)
                            "source_type": content_type,
                            "type": content_type,
                            "title": title,  # Always store title in metadata
                            "description": description,  # Always store description in metadata
                            "processed_at": datetime.now().isoformat(),
                            
                            # Universal schema fields (primary) - sanitize all values
                            "id": sanitize_value(row.get('id', '')),
                            "content_type": content_type,
                            "origin_url": sanitize_value(row.get('origin_url', '')),
                            "tags": sanitize_value(row.get('tags', [])),
                            "purpose": sanitize_value(purpose),
                            "location": sanitize_value(location),
                            "related_url": sanitize_value(row.get('related_url', '')),
                            "status": sanitize_value(row.get('status', '')),
                            "creator_id": sanitize_value(row.get('creator_id', '')),
                            "collaborators": sanitize_value(row.get('collaborators', '')),
                            "group_id": sanitize_value(row.get('group_id', '')),
                            "created_at": sanitize_value(row.get('created_at', '')),
                            "last_updated_at": sanitize_value(row.get('last_updated_at', '')),
                            "analysis_completed": sanitize_value(row.get('analysis_completed', '')),
                            "visibility": sanitize_value(row.get('visibility', '')),
                        }
                        
                        # Add backward compatibility fields (old CSV schema)
                        legacy_fields = {
                            'steps': steps,
                            'name': row.get('name', ''),
                            'method_name': row.get('method_name', ''),
                            'method_description': row.get('method_description', ''),
                            'resource_url': row.get('resource_url', ''),
                            'source_text': row.get('source_text', ''),
                            'category': row.get('category', ''),
                            'source': row.get('source', ''),
                        }
                        
                        # Only add legacy fields if they have values
                        for key, value in legacy_fields.items():
                            if value and pd.notna(value):
                                # Convert numpy/pandas types to native Python types and handle NaN
                                if isinstance(value, (pd.Series, pd.DataFrame)):
                                    continue
                                if pd.isna(value):
                                    continue
                                # Convert numpy types to Python native types
                                if hasattr(value, 'item'):
                                    value = value.item()
                                metadata[key] = value
                        
                        # Add any additional columns not covered above
                        excluded_cols = {
                            'title', 'description', 'steps', 'name', 'method_name', 'method_description',
                            'id', 'content_type', 'origin_url', 'tags', 'purpose', 'location', 
                            'related_url', 'status', 'creator_id', 'collaborators', 'group_id', 
                            'created_at', 'last_updated_at', 'analysis_completed', 'visibility', 
                            'resource_url', 'source_text', 'category', 'source'
                        }
                        
                        for col in df.columns:
                            if col not in excluded_cols:
                                value = row.get(col)
                                if pd.notna(value) and isinstance(value, (str, int, float, bool)):
                                    # Convert numpy/pandas types to native Python types and handle NaN
                                    if isinstance(value, (pd.Series, pd.DataFrame)):
                                        continue
                                    if pd.isna(value):
                                        continue
                                    # Convert numpy types to Python native types
                                    if hasattr(value, 'item'):
                                        value = value.item()
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
