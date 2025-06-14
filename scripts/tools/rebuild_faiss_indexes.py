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
import traceback
import pandas as pd
from datetime import datetime
from pathlib import Path
from langchain.schema import Document

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

def create_universal_field_mappings():
    """Create comprehensive field mappings for all content types."""
    return {
        # Universal schema fields (primary)
        'id': 'ID',
        'title': 'TITLE',
        'description': 'DESCRIPTION',
        'content_type': 'CONTENT_TYPE',
        'origin_url': 'SOURCE_URL',
        'related_url': 'RELATED_URL',
        'tags': 'TAGS',
        'purpose': 'PURPOSE',
        'location': 'LOCATION',
        'status': 'STATUS',
        'creator_id': 'CREATOR',
        'collaborators': 'COLLABORATORS',
        'group_id': 'GROUP',
        'visibility': 'VISIBILITY',
        'created_at': 'CREATED',
        'last_updated_at': 'UPDATED',
        'analysis_completed': 'ANALYZED',
        
        # Content-specific fields
        'steps': 'STEPS',
        'goals': 'GOALS',
        'achievement': 'ACHIEVEMENT',
        'documentation_url': 'DOCUMENTATION',
        'file_path': 'FILE_PATH',
        'authors': 'AUTHORS',
        'publication_date': 'PUBLICATION_DATE',
        'journal': 'JOURNAL',
        'doi': 'DOI',
        'abstract': 'ABSTRACT',
        'keywords': 'KEYWORDS',
        'methodology': 'METHODOLOGY',
        'findings': 'FINDINGS',
        'conclusion': 'CONCLUSION',
        'recommendations': 'RECOMMENDATIONS',
        'limitations': 'LIMITATIONS',
        'future_work': 'FUTURE_WORK',
        'references': 'REFERENCES',
        
        # Legacy backward compatibility fields
        'term': 'TERM',
        'definition': 'DEFINITION',
        'resource_url': 'RESOURCE_URL',
        'source_text': 'SOURCE_TEXT',
        'category': 'CATEGORY',
        'source': 'SOURCE',
        'name': 'NAME',
        'url': 'URL',
        'path': 'PATH',
        'content': 'CONTENT',
        'summary': 'SUMMARY',
        'notes': 'NOTES',
        'examples': 'EXAMPLES',
        'use_cases': 'USE_CASES',
        'best_practices': 'BEST_PRACTICES',
        'prerequisites': 'PREREQUISITES',
        'difficulty': 'DIFFICULTY',
        'time_required': 'TIME_REQUIRED',
        'tools_required': 'TOOLS_REQUIRED',
        'version': 'VERSION',
        'license': 'LICENSE',
        'maintainer': 'MAINTAINER',
        'contact': 'CONTACT',
        'feedback': 'FEEDBACK',
        'rating': 'RATING',
        'priority': 'PRIORITY',
        'risk_level': 'RISK_LEVEL',
        'dependencies': 'DEPENDENCIES',
        'outputs': 'OUTPUTS',
        'inputs': 'INPUTS',
        'parameters': 'PARAMETERS',
        'configuration': 'CONFIGURATION',
        'environment': 'ENVIRONMENT',
        'platform': 'PLATFORM',
        'language': 'LANGUAGE',
        'framework': 'FRAMEWORK',
        'library': 'LIBRARY',
        'api': 'API',
        'database': 'DATABASE',
        'schema': 'SCHEMA',
        'model': 'MODEL',
        'algorithm': 'ALGORITHM',
        'data_source': 'DATA_SOURCE',
        'data_format': 'DATA_FORMAT',
        'data_size': 'DATA_SIZE',
        'data_quality': 'DATA_QUALITY',
        'validation': 'VALIDATION',
        'testing': 'TESTING',
        'deployment': 'DEPLOYMENT',
        'monitoring': 'MONITORING',
        'troubleshooting': 'TROUBLESHOOTING',
        'faq': 'FAQ',
        'changelog': 'CHANGELOG',
        'roadmap': 'ROADMAP'
    }

def process_csv_to_documents(csv_path, content_type, field_mappings):
    """Process any CSV file to create rich documents with universal schema support.
    
    Args:
        csv_path (Path): Path to CSV file
        content_type (str): Type of content (from analysis-tasks-config.json)
        field_mappings (dict): Field mappings for labels
        
    Returns:
        list: List of Document objects with rich content and metadata
    """
    try:
        import pandas as pd
        import json
        from datetime import datetime
        
        df = pd.read_csv(csv_path)
        logger.info(f"Processing {len(df)} {content_type} entries from {csv_path}")
        logger.info(f"CSV columns: {list(df.columns)}")
        
        documents = []
        
        # Define intelligent title/description fallbacks based on content type
        title_fallbacks = {
            'definition': ['title', 'term', 'name', 'concept', 'keyword'],
            'method': ['title', 'name', 'method', 'procedure', 'technique'],
            'project': ['title', 'name', 'project', 'initiative', 'case_study'],
            'reference': ['title', 'name', 'paper', 'article', 'source'],
            'link': ['title', 'name', 'resource', 'tool', 'platform']
        }
        
        description_fallbacks = {
            'definition': ['description', 'definition', 'explanation', 'summary', 'abstract'],
            'method': ['description', 'summary', 'overview', 'abstract', 'procedure'],
            'project': ['description', 'summary', 'overview', 'abstract', 'objective'],
            'reference': ['description', 'abstract', 'summary', 'overview', 'content'],
            'link': ['description', 'summary', 'overview', 'abstract', 'purpose']
        }
        
        for _, row in df.iterrows():
            # Smart title detection with fallbacks
            title = None
            for field in title_fallbacks.get(content_type, ['title', 'name']):
                if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
                    title = str(row[field]).strip()
                    break
            
            if not title:
                title = f"Untitled {content_type.title()}"
            
            # Smart description detection with fallbacks
            description = None
            for field in description_fallbacks.get(content_type, ['description', 'summary']):
                if field in row.index and pd.notna(row[field]) and str(row[field]).strip():
                    description = str(row[field]).strip()
                    break
            
            if not description:
                description = f"No description available for this {content_type}"
            
            # Create rich content that includes ALL available data
            content_parts = []
            content_parts.append(f"TITLE: {title}")
            content_parts.append(f"DESCRIPTION: {description}")
            content_parts.append(f"TYPE: {content_type.upper()}")
            
            # Process ALL columns to preserve maximum data richness
            processed_fields = set()
            # Track which fields we've already used for title/description
            for field in title_fallbacks.get(content_type, []) + description_fallbacks.get(content_type, []):
                if field in row.index and pd.notna(row[field]):
                    processed_fields.add(field)
            
            # Process all remaining columns
            for column in df.columns:
                if column in processed_fields:
                    continue
                    
                value = row[column]
                if pd.isna(value):
                    continue
                    
                clean_value = sanitize_value(value)
                if not clean_value:
                    continue
                    
                # Get display label
                label = field_mappings.get(column, column.upper().replace('_', ' '))
                
                # Handle special field types
                if column in ['tags', 'keywords', 'collaborators', 'authors', 'references']:
                    # Handle JSON arrays or comma-separated values
                    if isinstance(clean_value, str):
                        try:
                            if clean_value.startswith('['):
                                items_list = json.loads(clean_value)
                                content_parts.append(f"{label}: {', '.join(items_list)}")
                            else:
                                # Comma-separated values
                                items_list = [item.strip() for item in clean_value.split(',') if item.strip()]
                                content_parts.append(f"{label}: {', '.join(items_list)}")
                        except:
                            content_parts.append(f"{label}: {clean_value}")
                    else:
                        content_parts.append(f"{label}: {clean_value}")
                        
                elif column in ['steps', 'goals', 'examples', 'use_cases', 'best_practices']:
                    # Handle structured lists
                    if isinstance(clean_value, str):
                        try:
                            if clean_value.startswith('['):
                                items_list = json.loads(clean_value)
                                numbered_items = [f"{i+1}. {item}" for i, item in enumerate(items_list)]
                                content_parts.append(f"{label}:\n" + '\n'.join(numbered_items))
                            else:
                                content_parts.append(f"{label}: {clean_value}")
                        except:
                            content_parts.append(f"{label}: {clean_value}")
                    else:
                        content_parts.append(f"{label}: {clean_value}")
                        
                else:
                    # Standard field handling
                    content_parts.append(f"{label}: {clean_value}")
            
            # Create the final rich content
            content = '\n'.join(content_parts)
            
            # Create comprehensive metadata
            metadata = {
                # Core identification
                "source_type": content_type,
                "type": content_type,
                "title": title,
                "description": description,
                "processed_at": datetime.now().isoformat(),
                "csv_source": str(csv_path),
                
                # Content type from universal schema
                "content_type": content_type,
            }
            
            # Add ALL CSV data to metadata (sanitized)
            for column in df.columns:
                value = row[column]
                if pd.notna(value):
                    sanitized_value = sanitize_value(value)
                    if sanitized_value:
                        metadata[column] = sanitized_value
            
            # Create document
            document = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(document)
            
        logger.info(f"Created {len(documents)} documents from {content_type} CSV")
        return documents
        
    except Exception as e:
        logger.error(f"Error processing {content_type} CSV at {csv_path}: {e}")
        return []

async def add_document_types(data_path, force_all=False, check_existing=False):
    """Add document types to reference_store from CSV files based on analysis-tasks-config.json.
    
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
        
        logger.info("Processing document types for reference_store using UNIVERSAL CSV processing...")
        
        # Load analysis tasks configuration to determine which CSV files to process
        tasks_config_path = Path(__file__).parent.parent / "settings" / "analysis-tasks-config.json"
        available_content_types = []
        
        if tasks_config_path.exists():
            try:
                with open(tasks_config_path, 'r', encoding='utf-8') as f:
                    tasks_config = json.load(f)
                available_content_types = list(tasks_config.keys())
                logger.info(f"Found {len(available_content_types)} content types in analysis config: {available_content_types}")
            except Exception as e:
                logger.warning(f"Could not load analysis tasks config: {e}")
                # Fallback to common types
                available_content_types = ['definition', 'method', 'project', 'reference', 'link']
        else:
            logger.warning("Analysis tasks config not found, using default types")
            available_content_types = ['definition', 'method', 'project', 'reference', 'link']
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

        # UNIVERSAL CSV PROCESSING - Process all content types dynamically
        added_docs = 0
        field_mappings = create_universal_field_mappings()
        
        logger.info(f"🚀 Starting UNIVERSAL CSV processing for content types: {available_content_types}")
        
        # Process each content type dynamically based on analysis-tasks-config.json
        for content_type in available_content_types:
            # Convert content type to plural for CSV filename (e.g., definition -> definitions)
            csv_name = content_type + 's' if not content_type.endswith('s') else content_type
            
            # Skip if already exists and we're checking
            if check_existing and csv_name in existing_types and not force_all:
                logger.info(f"✅ {csv_name} already in reference store, skipping")
                continue
            
            logger.info(f"🔄 Processing {content_type} -> {csv_name}.csv")
            
            # Try multiple possible paths for the CSV file
            csv_paths = [
                data_path / f"{csv_name}.csv",                    # Root directory
                data_path / "data" / f"{csv_name}.csv",           # Data subdirectory  
                data_path / "vector_stores" / f"{csv_name}.csv",  # Vector stores directory
            ]
            
            csv_path = None
            for path in csv_paths:
                if path.exists():
                    logger.info(f"📁 Found {csv_name} at {path}")
                    csv_path = path
                    break
            
            if not csv_path:
                logger.info(f"⚠️  No {csv_name}.csv file found, skipping")
                continue
            
            try:
                # Use universal CSV processor
                documents = process_csv_to_documents(csv_path, content_type, field_mappings)
                
                if not documents:
                    logger.warning(f"⚠️  No documents created from {csv_name}.csv")
                    continue
                
                logger.info(f"📊 Adding {len(documents)} {content_type} documents to reference_store")
                
                # Create metadata for tracking
                metadata = {
                    "data_type": csv_name,
                    "content_type": content_type,
                    "item_count": len(documents),
                    "source": "rebuild_faiss_indexes_universal",
                    "csv_path": str(csv_path),
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
                    logger.info(f"✅ Successfully added {len(documents)} {content_type} documents")
                    added_docs += len(documents)
                else:
                    logger.error(f"❌ Failed to add {content_type} documents")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {content_type}: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"🎉 UNIVERSAL CSV processing complete! Total documents added: {added_docs}")
        return added_docs
        
    except Exception as e:
        logger.error(f"💥 Error in universal document processing: {e}")
        logger.error(traceback.format_exc())
        return 0

def get_data_path_from_config():
    """Get data path from config file."""
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Prioritize config.json in the project root directory
    config_paths = [
        project_root / "config.json"  # Project root config
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

def create_universal_field_mappings():
    """Create comprehensive field mappings for all content types."""
    return {
        # Universal schema core fields
        'id': 'ID',
        'title': 'TITLE', 
        'description': 'DESCRIPTION',
        'content_type': 'TYPE',
        'purpose': 'PURPOSE',
        'location': 'LOCATION', 
        'tags': 'TAGS',
        'origin_url': 'SOURCE_URL',
        'related_url': 'RELATED_URL',
        'status': 'STATUS',
        'creator_id': 'CREATOR',
        'group_id': 'GROUP',
        'visibility': 'VISIBILITY',
        'created_at': 'CREATED',
        'last_updated_at': 'UPDATED',
        'analysis_completed': 'ANALYZED',
        'collaborators': 'COLLABORATORS',
        
        # Content type specific fields
        'steps': 'STEPS',                    # Methods
        'goals': 'GOALS',                    # Projects
        'achievement': 'ACHIEVEMENT',        # Projects
        'documentation_url': 'DOCUMENTATION', # Projects
        'file_path': 'FILE_PATH',           # Materials
        
        # Legacy/backward compatibility fields
        'term': 'TERM',                     # Old definitions
        'definition': 'DEFINITION',         # Old definitions  
        'project_name': 'PROJECT_NAME',     # Old projects
        'project_description': 'PROJECT_DESCRIPTION',  # Old projects
        'method_name': 'METHOD_NAME',       # Old methods
        'method_description': 'METHOD_DESCRIPTION',    # Old methods
        'usecase': 'USECASE',              # Old methods
        'resource_url': 'RESOURCE_URL',
        'source_text': 'SOURCE_TEXT',
        'category': 'CATEGORY',
        'source': 'SOURCE',
        'fields': 'FIELDS',                # Old tags
        'privacy': 'PRIVACY',              # Old visibility
    }

def process_csv_to_documents(csv_path: Path, content_type: str, field_mappings: dict) -> list:
    """Universal CSV to Document converter that preserves all data."""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Processing {csv_path.name}: Found {len(df)} {content_type} records")
        logger.debug(f"CSV columns: {list(df.columns)}")
        
        documents = []
        for _, row in df.iterrows():
            # Determine title with fallbacks based on content type
            title_fallbacks = {
                'definition': ['title', 'term'],
                'method': ['title', 'method_name', 'name'],
                'project': ['title', 'project_name', 'name'],
                'reference': ['title', 'name'],
                'material': ['title', 'name'],
            }
            
            title = None
            for field in title_fallbacks.get(content_type, ['title', 'name']):
                if row.get(field):
                    title = str(row[field]).strip()
                    break
            
            if not title:
                title = f'Untitled {content_type.title()}'
            
            # Determine description with fallbacks based on content type  
            description_fallbacks = {
                'definition': ['description', 'definition'],
                'method': ['description', 'method_description', 'usecase'],
                'project': ['description', 'project_description'],
                'reference': ['description'],
                'material': ['description', 'content'],
            }
            
            description = ''
            for field in description_fallbacks.get(content_type, ['description']):
                if row.get(field):
                    description = str(row[field]).strip()
                    break
            
            # Build rich content that includes ALL available CSV data
            content_parts = []
            content_parts.append(f"TITLE: {title}")
            
            if description:
                content_parts.append(f"DESCRIPTION: {description}")
            
            # Process ALL other columns to preserve maximum data richness
            processed_fields = {'title', 'description'}
            
            # Add title fallback fields to processed to avoid duplication
            processed_fields.update(title_fallbacks.get(content_type, []))
            processed_fields.update(description_fallbacks.get(content_type, []))
            
            for column, value in row.items():
                if column in processed_fields:
                    continue
                    
                # Clean and validate the value
                clean_value = sanitize_value(value)
                if not clean_value:  # Skip empty values
                    continue
                
                # Get the display label for this field
                label = field_mappings.get(column, column.upper().replace('_', ' '))
                
                # Special handling for JSON fields
                if column in ['tags', 'collaborators', 'steps', 'goals'] and clean_value:
                    if isinstance(clean_value, str):
                        try:
                            if clean_value.startswith('[') or clean_value.startswith('{'):
                                parsed_value = json.loads(clean_value)
                                if isinstance(parsed_value, list):
                                    if column == 'steps':
                                        # Format steps as numbered list
                                        numbered_steps = [f"{i+1}. {step}" for i, step in enumerate(parsed_value)]
                                        content_parts.append(f"{label}:\n" + '\n'.join(numbered_steps))
                                    else:
                                        content_parts.append(f"{label}: {', '.join(map(str, parsed_value))}")
                                else:
                                    content_parts.append(f"{label}: {parsed_value}")
                            else:
                                # Comma-separated string
                                if column == 'tags':
                                    tags_list = [tag.strip() for tag in clean_value.split(',')]
                                    content_parts.append(f"{label}: {', '.join(tags_list)}")
                                else:
                                    content_parts.append(f"{label}: {clean_value}")
                        except json.JSONDecodeError:
                            content_parts.append(f"{label}: {clean_value}")
                    elif isinstance(clean_value, list):
                        if column == 'steps':
                            numbered_steps = [f"{i+1}. {step}" for i, step in enumerate(clean_value)]
                            content_parts.append(f"{label}:\n" + '\n'.join(numbered_steps))
                        else:
                            content_parts.append(f"{label}: {', '.join(map(str, clean_value))}")
                    else:
                        content_parts.append(f"{label}: {clean_value}")
                else:
                    # Standard field handling
                    content_parts.append(f"{label}: {clean_value}")
            
            content = '\n'.join(content_parts)
            
            # Create comprehensive metadata preserving all CSV data
            metadata = {
                # Core fields (always present)
                "source_type": content_type,
                "type": content_type, 
                "title": title,
                "description": description,
                "processed_at": datetime.now().isoformat(),
                "content_type": content_type,
            }
            
            # Add all CSV columns as metadata (sanitized)
            for col in df.columns:
                value = row.get(col)
                sanitized_value = sanitize_value(value)
                if sanitized_value:  # Only add non-empty values
                    metadata[col] = sanitized_value
            
            # Create document
            documents.append(Document(page_content=content, metadata=metadata))
        
        logger.info(f"✅ Created {len(documents)} documents from {csv_path.name}")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error processing {csv_path}: {e}")
        return []

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
