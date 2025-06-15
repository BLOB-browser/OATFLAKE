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
from langchain_community.document_loaders import PyPDFLoader

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
        
        # STEP 1: Validate and cleanup existing vector stores
        logger.info("🔧 STEP 1: Validating and cleaning up existing vector stores...")
        cleanup_results = validate_and_cleanup_vector_stores(data_path)
        
        # STEP 2: If we're doing a complete reference store rebuild, handle that first
        if rebuild_reference:
            logger.info("🔄 STEP 2: Starting complete reference store rebuild with all document types...")
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
        
        # STEP 3: Rebuild all existing stores (or if --rebuild-reference isn't specified)
        logger.info("🔄 STEP 3: Rebuilding vector stores using modular architecture...")
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
                
            # STEP 4: If we're not forcing a complete rebuild of the reference store,
            # check for missing document types and add them
            if not rebuild_reference:
                logger.info("🔍 STEP 4: Checking for missing document types...")
                added_missing = await add_document_types(data_path, check_existing=True)
                if added_missing > 0:
                    logger.info(f"Successfully added {added_missing} missing documents to reference_store")
                else:
                    logger.info("No missing document types needed to be added")
            
            # STEP 5: Check if we need to generate topic stores
            # Count how many topic stores were rebuilt
            topic_stores = [store for store in result.get('stores_rebuilt', []) if store.startswith("topic_")]
            
            if len(topic_stores) < 3:
                logger.info(f"🏷️ STEP 5: Few topic stores created ({len(topic_stores)}), attempting to generate topic stores")
                
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
                        
                        # Try to create topic stores from these docs using tag-based approach
                        topic_results = await vector_store_manager.create_topic_stores(
                            rep_docs, 
                            use_clustering=False,  # Use individual tag stores instead of clustering
                            min_docs_per_topic=1   # Allow single-document topics per tag
                        )
                        
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
            
            # STEP 6: Create comprehensive processing summary
            duration = time.time() - start_time
            
            # Combine all results
            comprehensive_results = {
                **result,
                "cleanup_results": cleanup_results,
                "rebuild_reference": rebuild_reference,
                "rebuild_all": rebuild_all,
                "processing_duration_seconds": duration,
                "missing_documents_added": added_missing if not rebuild_reference else 0,
                "topic_stores_generated": len(topic_stores)
            }
            
            # Create processing summary
            create_processing_summary(data_path, comprehensive_results)
            
            # Log completion time
            logger.info(f"⏱️ Complete rebuild process finished in {duration:.2f} seconds")
            logger.info("🎉 REBUILD COMPLETE - Check processing_summary.txt for details")
            return True
        else:
            error_msg = result.get("message", "Unknown error") if result else "No result returned"
            logger.error(f"❌ Failed to rebuild indexes: {error_msg}")
            
            # Still create a summary for failed processing
            duration = time.time() - start_time
            error_results = {
                "status": "error",
                "error": error_msg,
                "cleanup_results": cleanup_results,
                "processing_duration_seconds": duration
            }
            create_processing_summary(data_path, error_results)
            
            return False
    except Exception as e:
        logger.error(f"❌ Error rebuilding indexes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Create error summary
        try:
            duration = time.time() - start_time
            error_results = {
                "status": "exception",
                "error": str(e),
                "processing_duration_seconds": duration
            }
            create_processing_summary(data_path, error_results)
        except:
            pass  # Don't fail on summary creation
            
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
        
        # Add materials/PDF processing to available content types
        if 'material' not in available_content_types:
            available_content_types.append('material')
            logger.info("Added 'material' content type for PDF processing")

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
                # Special handling for materials (PDFs)
                if content_type == 'material':
                    logger.info(f"🔄 Using specialized PDF processing for materials")
                    documents = await process_materials_to_documents(csv_path, data_path)
                else:
                    # Use universal CSV processor for other content types
                    documents = process_csv_to_documents(csv_path, content_type, field_mappings)
                
                if not documents:
                    logger.warning(f"⚠️  No documents created from {csv_name}.csv")
                    continue
                
                logger.info(f"📊 Adding {len(documents)} {content_type} documents to vector store")
                
                # Determine target store based on content type
                target_store = "content_store" if content_type == 'material' else "reference_store"
                logger.info(f"📍 Target vector store: {target_store}")
                
                # Create metadata for tracking
                metadata = {
                    "data_type": csv_name,
                    "content_type": content_type,
                    "item_count": len(documents),
                    "source": "rebuild_faiss_indexes_universal",
                    "csv_path": str(csv_path),
                    "added_at": datetime.now().isoformat()
                }
                
                # Add to appropriate vector store
                result = await vector_store_manager.add_documents_to_store(
                    target_store,
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

async def process_materials_to_documents(csv_path, data_path):
    """Process materials.csv to create PDF documents using PyPDFLoader with consistent chunking.
    Also handles PDFs in materials folder that don't appear in materials.csv (auto-downloaded ones).
    
    Args:
        csv_path (Path): Path to materials.csv file
        data_path (Path): Base data path for finding PDF files
        
    Returns:
        list: List of Document objects from PDF processing
    """
    try:
        import pandas as pd
        
        documents = []
        materials_path = data_path / "materials"
        processed_pdfs = set()  # Track which PDFs we've processed
        
        # First, process PDFs listed in materials.csv
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Processing {len(df)} PDF materials from {csv_path}")
            logger.info(f"Materials CSV columns: {list(df.columns)}")
            
            for _, row in df.iterrows():
                try:
                    # Get PDF file path from the CSV
                    file_path_str = row.get('file_path', '')
                    if not file_path_str:
                        logger.warning(f"No file_path for material: {row.get('title', 'Unknown')}")
                        continue
                    
                    # Handle both absolute and relative paths
                    file_path = Path(file_path_str)
                    if not file_path.is_absolute():
                        # Try relative to materials folder first
                        file_path = materials_path / file_path_str
                        if not file_path.exists():
                            # Try relative to data path
                            file_path = data_path / file_path_str
                    
                    if not file_path.exists():
                        logger.warning(f"PDF file not found: {file_path}")
                        continue
                    
                    if not file_path.suffix.lower() == '.pdf':
                        logger.warning(f"File is not a PDF: {file_path}")
                        continue
                    
                    # Mark this PDF as processed from CSV
                    processed_pdfs.add(str(file_path.resolve()))
                    
                    logger.info(f"Processing PDF from CSV: {file_path}")
                    
                    # Use PyPDFLoader to extract content
                    loader = PyPDFLoader(str(file_path))
                    pdf_docs = loader.load()
                    
                    # Apply consistent chunking to match reference store settings
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,    # Consistent with reference store chunking
                        chunk_overlap=200,  # Consistent with reference store chunking
                        separators=["\n\n", "\n", ". ", ", ", " ", ""]
                    )
                    
                    # Re-chunk PDF documents for consistency with reference store
                    chunked_docs = text_splitter.split_documents(pdf_docs)
                    pdf_docs = chunked_docs
                    
                    # Generate unique document ID for relationship tracking
                    material_id = row.get('id', '')
                    document_id = f"material_{material_id}_pdf_{file_path.name}"
                    document_title = row.get('title', file_path.stem)
                    
                    # Add comprehensive metadata to each document chunk for relationship tracking
                    for i, doc in enumerate(pdf_docs):
                        # Create enhanced metadata with document relationship labeling
                        doc.metadata.update({
                            # Core identification
                            'source_type': 'material',
                            'content_type': 'material',
                            'type': 'material',
                            'title': document_title,
                            'description': row.get('description', ''),
                            'fields': row.get('fields', '').split(',') if row.get('fields') else [],
                            'file_path': str(file_path),
                            'created_at': row.get('created_at', ''),
                            'processed_at': datetime.now().isoformat(),
                            'csv_source': str(csv_path),
                            
                            # Document relationship labeling for chunk grouping (CRITICAL FOR SUMMARIZATION)
                            'document_id': document_id,
                            'document_name': file_path.name,
                            'document_title': document_title,
                            'material_id': material_id,
                            'material_title': document_title,
                            
                            # Chunk relationship metadata for document summaries
                            'total_chunks': len(pdf_docs),
                            'chunk_index': i,
                            'chunk_id': f"{document_id}_chunk_{i}",
                            
                            # PDF-specific metadata
                            'pdf_page': doc.metadata.get('page', 0),
                            'pdf_source': str(file_path),
                            'pdf_chunks_total': len(pdf_docs),
                            
                            # Additional material-specific metadata
                            'original_source': 'materials_csv'
                        })
                        
                        # Sanitize all metadata to ensure JSON serialization
                        for key, value in doc.metadata.items():
                            doc.metadata[key] = sanitize_value(value)
                    
                    documents.extend(pdf_docs)
                    logger.info(f"✅ Extracted {len(pdf_docs)} chunks from PDF: {file_path.name} (chunks sized ~1500 chars for consistency)")
                    logger.info(f"📊 Document ID for summarization: {document_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing PDF material {row.get('title', 'Unknown')}: {e}")
                    continue
        else:
            logger.info(f"materials.csv not found at {csv_path}, will only process PDFs in materials folder")
        
        # Second, process any PDFs in materials folder that weren't in materials.csv
        # (These are typically auto-downloaded PDFs from URL processing)
        if materials_path.exists():
            logger.info(f"🔍 Scanning materials folder for additional PDFs: {materials_path}")
            
            # Find all PDF files in materials directory
            pdf_files = list(materials_path.glob("*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files in materials folder")
            
            # Filter out PDFs we already processed from CSV
            unprocessed_pdfs = []
            for pdf_file in pdf_files:
                pdf_path_str = str(pdf_file.resolve())
                if pdf_path_str not in processed_pdfs:
                    unprocessed_pdfs.append(pdf_file)
            
            logger.info(f"Found {len(unprocessed_pdfs)} PDFs not in materials.csv that need processing")
            
            # Process unprocessed PDFs
            for pdf_file in unprocessed_pdfs:
                try:
                    logger.info(f"Processing auto-downloaded PDF: {pdf_file.name}")
                    
                    # Use PyPDFLoader to extract content
                    loader = PyPDFLoader(str(pdf_file))
                    pdf_docs = loader.load()
                    
                    # Apply consistent chunking
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,    # Consistent with reference store chunking
                        chunk_overlap=200,  # Consistent with reference store chunking
                        separators=["\n\n", "\n", ". ", ", ", " ", ""]
                    )
                    
                    # Re-chunk PDF documents for consistency
                    chunked_docs = text_splitter.split_documents(pdf_docs)
                    pdf_docs = chunked_docs
                    
                    # Generate document ID for auto-downloaded PDFs
                    document_id = f"auto_pdf_{pdf_file.stem}_{int(datetime.now().timestamp())}"
                    document_title = pdf_file.stem
                    
                    # Add metadata for auto-downloaded PDFs
                    for i, doc in enumerate(pdf_docs):
                        doc.metadata.update({
                            # Core identification
                            'source_type': 'auto_downloaded_pdf',
                            'content_type': 'material',
                            'type': 'material',
                            'title': document_title,
                            'description': f"Auto-downloaded PDF: {pdf_file.name}",
                            'fields': [],
                            'file_path': str(pdf_file),
                            'created_at': '',
                            'processed_at': datetime.now().isoformat(),
                            'csv_source': 'not_in_csv',
                            
                            # Document relationship labeling
                            'document_id': document_id,
                            'document_name': pdf_file.name,
                            'document_title': document_title,
                            'material_id': '',
                            'material_title': document_title,
                            
                            # Chunk relationship metadata
                            'total_chunks': len(pdf_docs),
                            'chunk_index': i,
                            'chunk_id': f"{document_id}_chunk_{i}",
                            
                            # PDF-specific metadata
                            'pdf_page': doc.metadata.get('page', 0),
                            'pdf_source': str(pdf_file),
                            'pdf_chunks_total': len(pdf_docs),
                            
                            # Mark as auto-downloaded
                            'original_source': 'auto_downloaded',
                            'auto_downloaded': True
                        })
                        
                        # Sanitize all metadata
                        for key, value in doc.metadata.items():
                            doc.metadata[key] = sanitize_value(value)
                    
                    documents.extend(pdf_docs)
                    logger.info(f"✅ Processed auto-downloaded PDF: {pdf_file.name} ({len(pdf_docs)} chunks)")
                    
                    # Optionally, add to materials.csv for future reference
                    # This can be enabled as needed
                    await add_pdf_to_materials_csv(pdf_file, csv_path, data_path)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing auto-downloaded PDF {pdf_file.name}: {e}")
                    continue
        
        logger.info(f"🎉 Created {len(documents)} total document chunks from PDF materials")
        logger.info(f"📋 All chunks labeled with document relationships for future summarization")
        return documents
        
    except Exception as e:
        logger.error(f"💥 Error processing materials CSV at {csv_path}: {e}")
        return []

async def add_pdf_to_materials_csv(pdf_file, csv_path, data_path):
    """Add an auto-downloaded PDF to materials.csv for future reference.
    
    Args:
        pdf_file (Path): Path to the PDF file
        csv_path (Path): Path to materials.csv
        data_path (Path): Base data path
    """
    try:
        import pandas as pd
        from datetime import datetime
        
        # Create materials.csv if it doesn't exist
        if not csv_path.exists():
            # Create directory if needed
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create empty CSV with standard columns
            df = pd.DataFrame(columns=[
                'id', 'title', 'description', 'fields', 'file_path', 'created_at'
            ])
        else:
            df = pd.read_csv(csv_path)
        
        # Check if this PDF is already in the CSV
        relative_path = pdf_file.relative_to(data_path / "materials")
        existing_entry = df[df['file_path'].str.contains(pdf_file.name, na=False)]
        
        if len(existing_entry) > 0:
            logger.info(f"PDF {pdf_file.name} already exists in materials.csv")
            return
        
        # Generate new ID
        max_id = df['id'].max() if len(df) > 0 and pd.notna(df['id'].max()) else 0
        new_id = int(max_id) + 1 if pd.notna(max_id) else 1
        
        # Create new row for the auto-downloaded PDF
        new_row = {
            'id': new_id,
            'title': f"Auto-downloaded: {pdf_file.stem}",
            'description': f"Automatically downloaded PDF from URL processing: {pdf_file.name}",
            'fields': 'auto-downloaded',
            'file_path': str(relative_path),
            'created_at': datetime.now().isoformat()
        }
        
        # Add the new row
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Added auto-downloaded PDF to materials.csv: {pdf_file.name}")
        
    except Exception as e:
        logger.warning(f"⚠️ Could not add PDF to materials.csv: {e}")

def should_rebuild_vector_stores(data_path):
    """
    Check if vector stores need rebuilding based on content changes.
    This function now checks all CSV files mentioned in analysis-tasks-config.json
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Dict with rebuild decision and reasons
    """
    from pathlib import Path
    
    logger.info("🔍 Checking if vector stores need rebuilding based on analysis tasks config...")
    
    reasons = []
    force_rebuild = False
    
    # Load analysis tasks configuration to check for relevant CSVs
    tasks_config_path = Path(__file__).parent.parent / "settings" / "analysis-tasks-config.json"
    content_types = []
    
    if tasks_config_path.exists():
        try:
            with open(tasks_config_path, 'r', encoding='utf-8') as f:
                tasks_config = json.load(f)
            content_types = list(tasks_config.keys())
            logger.info(f"Checking {len(content_types)} content types from analysis config: {content_types}")
        except Exception as e:
            logger.warning(f"Could not load analysis tasks config: {e}")
            # Fallback to common types
            content_types = ['definition', 'method', 'project', 'reference', 'link']
    else:
        logger.warning("Analysis tasks config not found, using default types")
        content_types = ['definition', 'method', 'project', 'reference', 'link']
    
    # Add materials for PDF processing
    content_types.append('material')
    
    # Check for new or modified CSV files based on content types
    last_rebuild = get_last_rebuild_time(data_path)
    for content_type in content_types:
        csv_name = content_type + 's' if not content_type.endswith('s') else content_type
        csv_paths = [
            Path(data_path) / f"{csv_name}.csv",
            Path(data_path) / "data" / f"{csv_name}.csv",
            Path(data_path) / "vector_stores" / f"{csv_name}.csv",
        ]
        
        for csv_path in csv_paths:
            if csv_path.exists():
                if csv_path.stat().st_mtime > last_rebuild:
                    reasons.append(f"Modified CSV: {csv_path.name}")
                    force_rebuild = True
                break
    
    # Check for new or modified PDF files in materials directory
    materials_path = Path(data_path) / "materials"
    if materials_path.exists():
        pdf_files = list(materials_path.glob("*.pdf"))
        for pdf_file in pdf_files:
            if pdf_file.stat().st_mtime > last_rebuild:
                reasons.append(f"New/Modified PDF: {pdf_file.name}")
                force_rebuild = True
    
    # Check if vector stores exist
    vector_stores_path = Path(data_path) / "vector_stores" / "default"
    if not vector_stores_path.exists():
        reasons.append("Vector stores directory missing")
        force_rebuild = True
    else:
        # Check for missing essential stores
        essential_stores = ["content_store", "reference_store"]
        for store in essential_stores:
            store_path = vector_stores_path / store
            if not (store_path.exists() and (store_path / "index.faiss").exists()):
                reasons.append(f"Missing vector store: {store}")
                force_rebuild = True
    
    return {
        "should_rebuild": force_rebuild,
        "reasons": reasons,
        "last_rebuild_time": last_rebuild,
        "checked_files": len(content_types),
        "content_types_checked": content_types
    }

def get_last_rebuild_time(data_path):
    """Get the timestamp of the last rebuild."""
    rebuild_info_file = Path(data_path) / "vector_stores" / "last_rebuild.json"
    if rebuild_info_file.exists():
        try:
            with open(rebuild_info_file, 'r') as f:
                info = json.load(f)
                return datetime.fromisoformat(info["timestamp"]).timestamp()
        except Exception as e:
            logger.warning(f"Could not read last rebuild info: {e}")
    return 0  # Force rebuild if no info available

def update_last_rebuild_time(data_path):
    """Update the timestamp of the last rebuild."""
    rebuild_info_file = Path(data_path) / "vector_stores" / "last_rebuild.json"
    rebuild_info_file.parent.mkdir(parents=True, exist_ok=True)
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat()
    }
    
    with open(rebuild_info_file, 'w') as f:
        json.dump(info, f, indent=2)

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

def validate_and_cleanup_vector_stores(data_path):
    """
    Validate existing vector stores and clean up any corrupted or inconsistent data.
    This ensures a clean slate before rebuilding.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Dict with cleanup results
    """
    from pathlib import Path
    import shutil
    
    logger.info("🔧 Validating and cleaning up existing vector stores...")
    
    cleanup_results = {
        "stores_checked": 0,
        "stores_cleaned": 0,
        "corrupted_stores_removed": 0,
        "empty_stores_removed": 0,
        "issues_found": []
    }
    
    vector_stores_path = Path(data_path) / "vector_stores" / "default"
    
    if not vector_stores_path.exists():
        logger.info("No existing vector stores found - starting fresh")
        return cleanup_results
    
    try:
        # Check each store directory
        for store_dir in vector_stores_path.iterdir():
            if not store_dir.is_dir():
                continue
                
            cleanup_results["stores_checked"] += 1
            store_name = store_dir.name
            
            # Check for required files
            documents_json = store_dir / "documents.json"
            index_faiss = store_dir / "index.faiss"
            index_pkl = store_dir / "index.pkl"
            
            store_corrupted = False
            store_empty = False
            
            # Validate documents.json
            if documents_json.exists():
                try:
                    with open(documents_json, 'r', encoding='utf-8') as f:
                        docs = json.load(f)
                        if not isinstance(docs, list):
                            cleanup_results["issues_found"].append(f"{store_name}: documents.json is not a list")
                            store_corrupted = True
                        elif len(docs) == 0:
                            store_empty = True
                            logger.info(f"Store {store_name} is empty")
                        else:
                            logger.info(f"Store {store_name} has {len(docs)} documents")
                except Exception as e:
                    cleanup_results["issues_found"].append(f"{store_name}: corrupted documents.json - {e}")
                    store_corrupted = True
            else:
                cleanup_results["issues_found"].append(f"{store_name}: missing documents.json")
                store_corrupted = True
            
            # Check FAISS index files consistency
            if index_faiss.exists() and not index_pkl.exists():
                cleanup_results["issues_found"].append(f"{store_name}: has index.faiss but missing index.pkl")
                store_corrupted = True
            elif index_pkl.exists() and not index_faiss.exists():
                cleanup_results["issues_found"].append(f"{store_name}: has index.pkl but missing index.faiss")
                store_corrupted = True
            
            # Clean up corrupted or empty stores
            if store_corrupted:
                logger.warning(f"🗑️ Removing corrupted store: {store_name}")
                shutil.rmtree(store_dir)
                cleanup_results["corrupted_stores_removed"] += 1
                cleanup_results["stores_cleaned"] += 1
            elif store_empty:
                logger.info(f"🧹 Removing empty store: {store_name}")
                shutil.rmtree(store_dir)
                cleanup_results["empty_stores_removed"] += 1
                cleanup_results["stores_cleaned"] += 1
        
        # Log cleanup summary
        if cleanup_results["stores_cleaned"] > 0:
            logger.info(f"✅ Cleanup complete: removed {cleanup_results['corrupted_stores_removed']} corrupted and {cleanup_results['empty_stores_removed']} empty stores")
        else:
            logger.info("✅ All existing stores are valid - no cleanup needed")
            
        if cleanup_results["issues_found"]:
            logger.info(f"Issues addressed: {len(cleanup_results['issues_found'])}")
            for issue in cleanup_results["issues_found"]:
                logger.debug(f"  - {issue}")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Error during vector store cleanup: {e}")
        cleanup_results["issues_found"].append(f"Cleanup error: {e}")
        return cleanup_results

def create_processing_summary(data_path, processing_results):
    """
    Create a comprehensive summary of the processing results and save it to a file.
    
    Args:
        data_path: Path to data directory
        processing_results: Dict with all processing results
    """
    try:
        summary = {
            "processing_timestamp": datetime.now().isoformat(),
            "processing_results": processing_results,
            "data_path": str(data_path),
            "system_info": {
                "python_version": sys.version,
                "platform": os.name
            }
        }
        
        # Save summary to file
        summary_path = Path(data_path) / "vector_stores" / "last_processing_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Processing summary saved to: {summary_path}")
        
        # Also create a human-readable summary
        readable_summary_path = Path(data_path) / "vector_stores" / "processing_summary.txt"
        with open(readable_summary_path, 'w', encoding='utf-8') as f:
            f.write("OATFLAKE Knowledge Processing Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Path: {data_path}\n\n")
            
            # Write processing results
            if "status" in processing_results:
                f.write(f"Overall Status: {processing_results['status']}\n\n")
            
            if "stores_rebuilt" in processing_results:
                f.write(f"Vector Stores Rebuilt: {len(processing_results['stores_rebuilt'])}\n")
                for store in processing_results["stores_rebuilt"]:
                    f.write(f"  - {store}\n")
                f.write("\n")
            
            if "document_counts" in processing_results:
                f.write("Document Counts by Store:\n")
                for store, count in processing_results["document_counts"].items():
                    f.write(f"  - {store}: {count} documents\n")
                f.write("\n")
            
            if "total_documents" in processing_results:
                f.write(f"Total Documents Processed: {processing_results['total_documents']}\n\n")
            
            # Add recommendations
            f.write("Next Steps:\n")
            f.write("- Use document_summarizer.py to generate PDF summaries\n")
            f.write("- Check for duplicate URLs in resources.csv if needed\n")
            f.write("- Monitor processing logs for any warnings\n")
        
        logger.info(f"📝 Human-readable summary saved to: {readable_summary_path}")
        
    except Exception as e:
        logger.error(f"Error creating processing summary: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rebuild FAISS indexes for vector stores")
    parser.add_argument("--data-path", type=str, help="Path to data directory (defaults to config.json setting)")
    parser.add_argument("--rebuild-all", action="store_true", help="Force complete rebuild of all vector stores")
    parser.add_argument("--rebuild-reference", action="store_true", help="Force complete rebuild of reference store with all document types")
    parser.add_argument('--force', action='store_true', help='Force rebuild even if no changes detected')
    parser.add_argument('--check-only', action='store_true', help='Only check if rebuild is needed, don\'t rebuild')
    
    args = parser.parse_args()
    
    # Get data path from arguments or config
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_data_path_from_config()
    
    logger.info(f"Using data path: {data_path}")
    
    # Check if rebuild is needed (unless forcing a complete rebuild)
    if not args.rebuild_all and not args.rebuild_reference and not args.force:
        rebuild_check = should_rebuild_vector_stores(data_path)
        
        logger.info(f"📊 Rebuild Check Results:")
        logger.info(f"   Should rebuild: {rebuild_check['should_rebuild']}")
        logger.info(f"   Reasons: {', '.join(rebuild_check['reasons']) if rebuild_check['reasons'] else 'None'}")
        logger.info(f"   Files checked: {rebuild_check['checked_files']}")
        
        if args.check_only:
            return 0 if rebuild_check['should_rebuild'] else 1
        
        if not rebuild_check['should_rebuild']:
            logger.info("✅ Vector stores are up to date, no rebuild needed")
            return 0
    
    if args.force:
        logger.info("🔄 Force rebuild requested")
    
    # Run the rebuild process
    try:
        # Validate and clean up existing vector stores first
        cleanup_results = validate_and_cleanup_vector_stores(data_path)
        
        logger.info(f"🔧 Cleanup Results: {cleanup_results}")
        
        result = asyncio.run(rebuild_indexes(data_path, rebuild_all=args.rebuild_all, rebuild_reference=args.rebuild_reference))
        
        if result:
            # Update rebuild timestamp on success
            update_last_rebuild_time(data_path)
            
            if args.rebuild_reference:
                logger.info("🎉 Complete rebuild successful")
            else:
                logger.info("🎉 FAISS index rebuild successful")
            
            # Create processing summary
            processing_results = {
                "status": "success",
                "stores_rebuilt": cleanup_results.get("stores_checked", 0),
                "total_documents": sum(cleanup_results.get("document_counts", {}).values()),
                "document_counts": cleanup_results.get("document_counts", {})
            }
            
            create_processing_summary(data_path, processing_results)
            
            return 0
        else:
            logger.error("💥 FAISS index rebuild failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error during rebuild: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
