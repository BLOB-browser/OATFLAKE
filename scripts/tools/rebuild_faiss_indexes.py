#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rebuild FAISS Indexes

This script rebuilds all FAISS indexes from stored document data without 
needing to reprocess or re-embed documents. It's designed to be run after 
knowledge processing is complete or interrupted, to ensure vector store
consistency.

Usage:
    python rebuild_faiss_indexes.py [--data-path PATH] [--rebuild-all] [--rebuild-reference] [--content-only]

Options:
    --data-path PATH       Path to data directory (defaults to config.json setting)
    --rebuild-all          Force complete rebuild of all vector stores
    --rebuild-reference    Force complete rebuild of reference store with all document types
    --content-only         Force rebuild of content store only (materials/PDFs)
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
import numpy as np
import math
import shutil
import inspect
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

async def rebuild_indexes(data_path, rebuild_all=False, rebuild_reference=False, content_only=False):
    """
    Completely rebuild FAISS indexes from scratch in a single flow.
    This function always does a complete rebuild - never incremental.
    
    Args:
        data_path (Path): Path to data directory
        rebuild_all (bool): Force rebuild of all vector stores (default behavior)
        rebuild_reference (bool): Force rebuild of reference store with all document types
        content_only (bool): Force rebuild of content store only (materials/PDFs)
    """
    try:
        start_time = time.time()
        
        # STEP 1: Complete cleanup - remove ALL existing stores for fresh start
        logger.info("🔧 STEP 1: Complete cleanup - removing ALL existing stores for fresh rebuild...")
        cleanup_results = complete_vector_store_cleanup(data_path, content_only=content_only)
        
        # Initialize vector store manager
        from scripts.storage.vector_store_manager import VectorStoreManager
        vector_store_manager = VectorStoreManager(base_path=data_path)
        
        result = {
            "status": "success",
            "stores_rebuilt": [],
            "total_documents": 0,
            "document_counts": {}
        }
        
        # STEP 2: Rebuild stores based on requested scope
        if content_only:
            logger.info("🔄 STEP 2: Rebuilding ONLY content store (materials/PDFs)...")
            
            # Process PDF materials for content store
            materials_csv_path = data_path / "materials.csv"
            test_documents = await process_materials_to_documents(materials_csv_path, data_path)
            
            if test_documents:
                logger.info(f"Creating fresh content_store with {len(test_documents)} material documents")
                
                # Create metadata for tracking
                metadata = {
                    "data_type": "materials",
                    "content_type": "material", 
                    "item_count": len(test_documents),
                    "source": "rebuild_faiss_indexes_complete",
                    "added_at": datetime.now().isoformat()
                }
                
                # Create content store from scratch (not add to existing)
                content_result = await vector_store_manager.create_or_update_store(
                    "content_store",
                    test_documents,
                    metadata=metadata,
                    update_stats=True
                )
                
                if content_result:
                    logger.info(f"✅ Successfully created content_store with {len(test_documents)} documents")
                    result["stores_rebuilt"].append("content_store")
                    result["document_counts"]["content_store"] = len(test_documents)
                    result["total_documents"] += len(test_documents)
                else:
                    raise Exception("Failed to create content_store")
        
        elif rebuild_reference:
            logger.info("🔄 STEP 2: Rebuilding ONLY reference store with all document types...")
            
            # Add all document types to reference store
            total_docs = await add_document_types(data_path, force_all=True)
            
            if total_docs > 0:
                logger.info(f"✅ Successfully created reference_store with {total_docs} documents")
                result["stores_rebuilt"].append("reference_store")
                result["document_counts"]["reference_store"] = total_docs
                result["total_documents"] += total_docs
            else:
                raise Exception("Failed to create reference_store")
        
        else:
            # Default: Rebuild ALL stores (content_store + reference_store + topic_stores)
            logger.info("🔄 STEP 2: Rebuilding ALL stores (content + reference + topics)...")
            
            # 2A: Create content_store with materials
            materials_csv_path = data_path / "materials.csv"
            test_documents = await process_materials_to_documents(materials_csv_path, data_path)
            if test_documents:
                logger.info(f"Creating fresh content_store with {len(test_documents)} material documents")
                
                metadata = {
                    "data_type": "materials",
                    "content_type": "material",
                    "item_count": len(test_documents),
                    "source": "rebuild_faiss_indexes_complete",
                    "added_at": datetime.now().isoformat()
                }
                
                content_result = await vector_store_manager.create_or_update_store(
                    "content_store",
                    test_documents,
                    metadata=metadata,
                    update_stats=True
                )
                
                if content_result:
                    logger.info(f"✅ Successfully created content_store with {len(test_documents)} documents")
                    result["stores_rebuilt"].append("content_store")
                    result["document_counts"]["content_store"] = len(test_documents)
                    result["total_documents"] += len(test_documents)
            
            # 2B: Create reference_store with all document types
            logger.info("Creating fresh reference_store with all document types...")
            ref_docs = await add_document_types(data_path, force_all=True)
            
            if ref_docs > 0:
                logger.info(f"✅ Successfully created reference_store with {ref_docs} documents")
                result["stores_rebuilt"].append("reference_store")
                result["document_counts"]["reference_store"] = ref_docs
                result["total_documents"] += ref_docs
        
        # STEP 3: Generate topic stores from newly created stores
        # Create topic stores when doing a full rebuild (not content_only and not rebuild_reference only)
        if not content_only and not rebuild_reference:
            logger.info("🏷️ STEP 3: Generating topic stores from newly created content...")
            topic_stores_created = 0
            
            try:
                # Get updated store list after rebuilding
                stores = vector_store_manager.list_stores()
                store_names = [store.get("name") for store in stores]
                
                # Use reference_store for topic generation (contains documents with proper tags)
                # content_store contains raw materials without tags - not suitable for topic generation
                source_store = None
                if "reference_store" in store_names:
                    source_store = "reference_store"
                    logger.info("Using reference_store for topic generation (contains documents with tags)")
                elif "content_store" in store_names:
                    source_store = "content_store"
                    logger.warning("Falling back to content_store for topic generation (may not have proper tags)")
                
                if source_store:
                    logger.info(f"Getting representative docs from {source_store} for topics")
                    
                    # Get representative chunks to create topic stores
                    rep_docs = await vector_store_manager.get_representative_chunks(
                        store_name=source_store,
                        num_chunks=100
                    )
                    
                    if rep_docs:
                        logger.info(f"Got {len(rep_docs)} representative documents for topic generation")
                        
                        # Create topic stores using tag-based approach
                        topic_results = await vector_store_manager.create_topic_stores(
                            rep_docs,
                            use_clustering=False,  # Use individual tag stores
                            min_docs_per_topic=1   # Allow single-document topics per tag
                        )
                        
                        if topic_results:
                            topic_stores_created = len(topic_results)
                            logger.info(f"Created {topic_stores_created} topic stores")
                            
                            # Add topic stores to results
                            for topic, success in topic_results.items():
                                if success:
                                    topic_store_name = f"topic_{topic}"
                                    result["stores_rebuilt"].append(topic_store_name)
                                    logger.info(f"  - Created topic store: {topic_store_name}")
                        else:
                            logger.info("No topic stores were created from representative documents")
                    else:
                        logger.warning(f"No representative documents found in {source_store} for topic generation")
                else:
                    logger.warning("No source store found for topic generation")
            except Exception as e:
                logger.warning(f"Topic store generation failed: {e}")
                logger.debug(f"Topic store generation error details: {traceback.format_exc()}")
        
        # STEP 4: Create comprehensive processing summary
        duration = time.time() - start_time
        
        # Combine all results
        comprehensive_results = {
            **result,
            "cleanup_results": cleanup_results,
            "rebuild_reference": rebuild_reference,
            "rebuild_all": rebuild_all,
            "content_only": content_only,
            "processing_duration_seconds": duration,
            "topic_stores_created": len([s for s in result["stores_rebuilt"] if s.startswith("topic_")]),
            "total_stores": len(result["stores_rebuilt"])
        }
        
        # Create processing summary
        create_processing_summary(data_path, comprehensive_results)
        
        # Log completion time
        logger.info(f"⏱️ Complete rebuild process finished in {duration:.2f} seconds")
        logger.info(f"📊 Rebuilt {len(result['stores_rebuilt'])} stores with {result['total_documents']} total documents")
        logger.info("🎉 COMPLETE REBUILD FINISHED - Check processing_summary.txt for details")
        
        return comprehensive_results
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding indexes: {e}")
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
            
        return {
            "status": "exception",
            "error": str(e),
            "stores_rebuilt": [],
            "total_documents": 0,
            "document_counts": {}
        }
        start_time = time.time()
        
        # STEP 1: Validate and cleanup existing vector stores
        logger.info("🔧 STEP 1: Validating and cleaning up existing vector stores...")
        cleanup_results = validate_and_cleanup_vector_stores(data_path, content_only=content_only)
        
        # STEP 2: If we're doing content-only rebuild, handle that first
        if content_only:
            logger.info("🔄 STEP 2: Starting content store only rebuild (materials/PDFs)...")
            from scripts.storage.vector_store_manager import VectorStoreManager
            
            # Initialize vector store manager
            vector_store_manager = VectorStoreManager(base_path=data_path)
            
            # Get content store path
            vector_stores_path = data_path / "vector_stores" / "default"
            content_store_path = vector_stores_path / "content_store"
            
            # Ensure the directory exists
            content_store_path.mkdir(parents=True, exist_ok=True)
            
            # Create a clean content store by removing existing documents.json
            if (content_store_path / "documents.json").exists():
                logger.info("Removing existing content store documents")
                (content_store_path / "documents.json").unlink()
                
            # Create empty documents.json
            with open(content_store_path / "documents.json", "w") as f:
                json.dump([], f)
                
            # Now process only materials (PDFs) for content store
            logger.info("Creating content store with materials/PDFs from scratch")
            
            # Check for materials folder
            materials_path = data_path / "materials"
            if not materials_path.exists() or not list(materials_path.glob("*.pdf")):
                logger.warning("No PDF files found in materials folder for content store creation")
                # Create a test PDF document for testing
                logger.info("Creating test PDF document for content store testing...")
                test_documents = [Document(
                    page_content="This is a test PDF document for content store testing. It contains sample content to verify that the content store creation process works correctly.",
                    metadata={
                        'source_type': 'material',
                        'content_type': 'material',
                        'type': 'material',
                        'title': 'Test PDF Document',
                        'description': 'Test PDF for content store creation',
                        'file_path': 'test_document.pdf',
                        'processed_at': datetime.now().isoformat(),
                        'document_id': f"test_pdf_{int(datetime.now().timestamp())}",
                        'chunk_index': 0,
                        'total_chunks': 1
                    }
                )]
            else:
                # Process actual PDF materials
                materials_csv_path = data_path / "materials.csv"
                test_documents = await process_materials_to_documents(materials_csv_path, data_path)
            
            if test_documents:
                logger.info(f"Adding {len(test_documents)} material documents to content store")
                
                # Create metadata for tracking
                metadata = {
                    "data_type": "materials",
                    "content_type": "material",
                    "item_count": len(test_documents),
                    "source": "rebuild_faiss_indexes_content_only",
                    "added_at": datetime.now().isoformat()
                }
                
                # Add to content store
                result = await vector_store_manager.add_documents_to_store(
                    "content_store",
                    test_documents,
                    metadata=metadata,
                    update_stats=True
                )
                
                if result:
                    logger.info(f"✅ Successfully created content store with {len(test_documents)} documents")
                    
                    # Create comprehensive results for content-only rebuild
                    duration = time.time() - start_time
                    comprehensive_results = {
                        "status": "success",
                        "stores_rebuilt": ["content_store"],
                        "total_documents": len(test_documents),
                        "document_counts": {"content_store": len(test_documents)},
                        "cleanup_results": cleanup_results,
                        "content_only": True,
                        "processing_duration_seconds": duration
                    }
                    
                    # Create processing summary
                    create_processing_summary(data_path, comprehensive_results)
                    
                    logger.info(f"⏱️ Content store rebuild finished in {duration:.2f} seconds")
                    logger.info("🎉 CONTENT STORE REBUILD COMPLETE")
                    return comprehensive_results
                else:
                    logger.error("❌ Failed to create content store")
                    error_results = {
                        "status": "error",
                        "error": "Failed to create content store",
                        "stores_rebuilt": [],
                        "total_documents": 0,
                        "document_counts": {},
                        "cleanup_results": cleanup_results,
                        "content_only": True,
                        "processing_duration_seconds": time.time() - start_time
                    }
                    create_processing_summary(data_path, error_results)
                    return error_results
            else:
                logger.error("❌ No documents to add to content store")
                error_results = {
                    "status": "error",
                    "error": "No documents to add to content store",
                    "stores_rebuilt": [],
                    "total_documents": 0,
                    "document_counts": {},
                    "cleanup_results": cleanup_results,
                    "content_only": True,
                    "processing_duration_seconds": time.time() - start_time
                }
                create_processing_summary(data_path, error_results)
                return error_results
        
        # STEP 2: If we're doing a complete reference store rebuild, handle that first
        elif rebuild_reference:
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
        
        # STEP 3: Create missing stores (content_store if not exists) without redundant rebuilding
        logger.info("🔄 STEP 3: Creating missing stores and adding missing document types...")
        
        # Import vector store manager
        from scripts.storage.vector_store_manager import VectorStoreManager
        vector_store_manager = VectorStoreManager(base_path=data_path)
        
        # Check which stores exist
        stores = vector_store_manager.list_stores()
        store_names = [store.get("name") for store in stores]
        
        result = {
            "status": "success",
            "stores_rebuilt": [],
            "total_documents": 0,
            "document_counts": {}
        }
        
        # Create content_store if it doesn't exist or is empty
        if "content_store" not in store_names or rebuild_all:
            logger.info("Creating content_store with PDF materials...")
            
            # Process PDF materials for content store
            materials_csv_path = data_path / "materials.csv"
            test_documents = await process_materials_to_documents(materials_csv_path, data_path)
            
            if test_documents:
                # Create metadata for tracking
                metadata = {
                    "data_type": "materials",
                    "content_type": "material",
                    "item_count": len(test_documents),
                    "source": "rebuild_faiss_indexes_materials",
                    "added_at": datetime.now().isoformat()
                }
                
                # Add to content store (will create if not exists)
                content_result = await vector_store_manager.add_documents_to_store(
                    "content_store",
                    test_documents,
                    metadata=metadata,
                    update_stats=True
                )
                
                if content_result:
                    logger.info(f"✅ Successfully created content_store with {len(test_documents)} documents")
                    result["stores_rebuilt"].append("content_store")
                    result["document_counts"]["content_store"] = len(test_documents)
                    result["total_documents"] += len(test_documents)
        
        # Use existing modular architecture only for reference_store if needed
        if "reference_store" not in store_names or rebuild_all:
            logger.info("Creating/rebuilding reference_store using modular architecture...")
            if use_new_modules:
                logger.info("Using new modular architecture for reference store")
                faiss_builder = FAISSBuilder(data_path)
                # Check if FAISSBuilder has a rebuild_all_indexes method with a rebuild_all parameter
                import inspect
                faiss_rebuild_params = inspect.signature(faiss_builder.rebuild_all_indexes).parameters
                if 'rebuild_all' in faiss_rebuild_params:
                    ref_result = await faiss_builder.rebuild_all_indexes(rebuild_all=rebuild_all)
                else:
                    # Fallback if the parameter isn't supported
                    ref_result = await faiss_builder.rebuild_all_indexes()
            else:
                logger.info("Using legacy DataProcessor for reference store")
                data_processor = DataProcessor(data_path)
                # Check if DataProcessor has a rebuild_all_vector_stores method with a rebuild_all parameter
                import inspect
                data_processor_params = inspect.signature(data_processor.rebuild_all_vector_stores).parameters
                if 'rebuild_all' in data_processor_params:
                    ref_result = await data_processor.rebuild_all_vector_stores(rebuild_all=rebuild_all)
                else:
                    # Fallback if the parameter isn't supported
                    ref_result = await data_processor.rebuild_all_vector_stores()
            
            # Merge results
            if ref_result and ref_result.get("status") == "success":
                for store in ref_result.get("stores_rebuilt", []):
                    if store not in result["stores_rebuilt"]:
                        result["stores_rebuilt"].append(store)
                
                for store, count in ref_result.get("document_counts", {}).items():
                    result["document_counts"][store] = count
                    result["total_documents"] += count
        
        # STEP 4: Add missing document types to reference_store (not rebuild_reference case)
        if not rebuild_reference:
            logger.info("🔍 STEP 4: Checking for missing document types...")
            added_missing = await add_document_types(data_path, check_existing=True)
            if added_missing > 0:
                logger.info(f"Successfully added {added_missing} missing documents to reference_store")
                # Update results
                if "reference_store" in result["document_counts"]:
                    result["document_counts"]["reference_store"] += added_missing
                else:
                    result["document_counts"]["reference_store"] = added_missing
                result["total_documents"] += added_missing
            else:
                logger.info("No missing document types needed to be added")
        
        # STEP 5: Generate topic stores from existing content (without rebuilding content stores)
        logger.info(f"🏷️ STEP 5: Generating topic stores from existing content")
        topic_stores_created = 0
        
        try:
            # Get updated store list after our changes
            stores = vector_store_manager.list_stores()
            store_names = [store.get("name") for store in stores]
            
            # Try reference_store first (has documents with tags), then content_store as fallback
            source_store = None
            if "reference_store" in store_names:
                source_store = "reference_store"
                logger.info("Using reference_store for topic generation (contains documents with tags)")
            elif "content_store" in store_names:
                source_store = "content_store"
                logger.warning("Falling back to content_store for topic generation (may not have proper tags)")
            
            if source_store:
                logger.info(f"Getting representative docs from {source_store} for topics")
                
                # Get representative chunks to create topic stores
                rep_docs = await vector_store_manager.get_representative_chunks(
                    store_name=source_store, 
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
                        topic_stores_created = len(topic_results)
                        logger.info(f"Created {topic_stores_created} topic stores")
                        # Log the created topic stores
                        for topic, success in topic_results.items():
                            if success:
                                logger.info(f"  - Created topic store for: {topic}")
                    else:
                        logger.info("No topic stores were created from representative documents")
                else:
                    logger.warning(f"No representative documents found in {source_store} for topic generation")
            else:
                logger.warning("Neither content_store nor reference_store found, cannot generate topic stores")
        except Exception as e:
            logger.warning(f"Topic store generation failed: {e}")
            logger.debug(f"Topic store generation error details: {traceback.format_exc()}")
        
        # STEP 6: Create comprehensive processing summary
        duration = time.time() - start_time
        
        # Count existing topic stores for reporting
        existing_topic_stores = [store for store in result.get('stores_rebuilt', []) if store.startswith("topic_")]
        total_topic_stores = len(existing_topic_stores) + topic_stores_created
        
        # Combine all results
        comprehensive_results = {
            **result,
            "cleanup_results": cleanup_results,
            "rebuild_reference": rebuild_reference,
            "rebuild_all": rebuild_all,
            "processing_duration_seconds": duration,
            "missing_documents_added": added_missing if not rebuild_reference else 0,
            "existing_topic_stores_rebuilt": len(existing_topic_stores),
            "new_topic_stores_created": topic_stores_created,
            "total_topic_stores": total_topic_stores
        }
        
        # Create processing summary
        create_processing_summary(data_path, comprehensive_results)
        
        # Log completion time with topic store information
        logger.info(f"⏱️ Complete rebuild process finished in {duration:.2f} seconds")
        logger.info(f"🏷️ Topic stores: {len(existing_topic_stores)} existing + {topic_stores_created} newly created = {total_topic_stores} total")
        logger.info("🎉 REBUILD COMPLETE - Check processing_summary.txt for details")
        return comprehensive_results
        
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
            
        return {
            "status": "exception",
            "error": str(e),
            "stores_rebuilt": [],
            "total_documents": 0,
            "document_counts": {}
        }

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
        'origin_url': 'ORIGIN_URL',  # Keep as ORIGIN_URL, not SOURCE_URL
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
                
                # Create fresh vector store (not add to existing)
                result = await vector_store_manager.create_or_update_store(
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
    """Process PDF files that are referenced in materials.csv.
    Only processes PDFs that have entries in the materials.csv file for proper cataloguing.
    
    Args:
        csv_path (Path): Path to materials.csv file (now required!)
        data_path (Path): Base data path for finding PDF files
        
    Returns:
        list: List of Document objects from PDF processing
    """
    try:
        documents = []
        materials_path = data_path / "materials"
        
        # Check if materials folder exists
        if not materials_path.exists():
            logger.info(f"Materials folder not found at {materials_path}, creating it")
            materials_path.mkdir(parents=True, exist_ok=True)
            return documents
        
        # Check if materials.csv exists
        if not csv_path or not csv_path.exists():
            logger.warning(f"materials.csv not found at {csv_path}")
            logger.info("📋 No materials.csv found - skipping PDF processing")
            logger.info("💡 Create materials.csv to catalog PDFs for processing")
            return documents
        
        # Read materials.csv to get list of PDFs to process
        import pandas as pd
        try:
            materials_df = pd.read_csv(csv_path)
            logger.info(f"📋 Found materials.csv with {len(materials_df)} entries")
        except Exception as e:
            logger.error(f"❌ Error reading materials.csv: {e}")
            return documents
        
        if materials_df.empty:
            logger.info("📋 materials.csv is empty - no PDFs to process")
            return documents
        
        # Get list of PDF files from CSV
        pdf_files_to_process = []
        for _, row in materials_df.iterrows():
            file_path = row.get('file_path', '')
            if file_path and file_path.lower().endswith('.pdf'):
                full_pdf_path = materials_path / file_path
                if full_pdf_path.exists():
                    pdf_files_to_process.append((full_pdf_path, row))
                    logger.info(f"✅ Found PDF to process: {file_path}")
                else:
                    logger.warning(f"⚠️ PDF listed in CSV but not found: {file_path}")
        
        if not pdf_files_to_process:
            logger.info("📂 No valid PDF files found from materials.csv entries")
            return documents
        
        logger.info(f"📂 Processing {len(pdf_files_to_process)} PDFs from materials.csv")
        
        # Process PDFs that are catalogued in materials.csv
        for pdf_file, csv_row in pdf_files_to_process:
            try:
                logger.info(f"📄 Processing catalogued PDF: {pdf_file.name}")
                
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
                
                # Generate document ID
                document_id = f"pdf_{pdf_file.stem}_{int(datetime.now().timestamp())}"
                
                # Use metadata from CSV if available, otherwise use filename
                document_title = csv_row.get('title', pdf_file.stem.replace('_', ' ').title())
                material_description = csv_row.get('description', f"PDF Document: {pdf_file.name}")
                material_fields = csv_row.get('fields', '')
                created_at = csv_row.get('created_at', '')
                material_id = csv_row.get('id', '')
                
                # Add metadata combining CSV data with PDF processing
                for i, doc in enumerate(pdf_docs):
                    doc.metadata.update({
                        # Core identification
                        'source_type': 'material',
                        'content_type': 'material',
                        'type': 'material',
                        'title': document_title,
                        'description': material_description,
                        'fields': material_fields,  # Keep as string for consistency
                        'file_path': str(pdf_file),
                        'created_at': created_at,
                        'processed_at': datetime.now().isoformat(),
                        'csv_source': 'materials_csv',
                        
                        # URL for accessing the PDF via the API
                        'origin_url': f"/api/data/materials/pdf/{pdf_file.name}",
                        
                        # Document relationship labeling
                        'document_id': document_id,
                        'document_name': pdf_file.name,
                        'document_title': document_title,
                        'material_id': str(material_id),
                        'material_title': document_title,
                        
                        # Chunk relationship metadata
                        'total_chunks': len(pdf_docs),
                        'chunk_index': i,
                        'chunk_id': f"{document_id}_chunk_{i}",
                        
                        # PDF-specific metadata
                        'pdf_page': doc.metadata.get('page', 0),
                        'pdf_source': str(pdf_file),
                        'pdf_chunks_total': len(pdf_docs),
                        
                        # Mark as processed from CSV
                        'original_source': 'materials_csv',
                        'auto_downloaded': False
                    })
                    
                    # Sanitize all metadata
                    for key, value in doc.metadata.items():
                        doc.metadata[key] = sanitize_value(value)
                
                documents.extend(pdf_docs)
                logger.info(f"✅ Processed catalogued PDF: {pdf_file.name} ({len(pdf_docs)} chunks)")
                
            except Exception as e:
                logger.error(f"❌ Error processing PDF {pdf_file.name}: {e}")
                continue
        
        # Check for uncatalogued PDFs and report them
        all_pdfs_in_folder = list(materials_path.glob("*.pdf"))
        catalogued_filenames = {pdf_file.name for pdf_file, _ in pdf_files_to_process}
        uncatalogued_pdfs = [pdf for pdf in all_pdfs_in_folder if pdf.name not in catalogued_filenames]
        
        if uncatalogued_pdfs:
            logger.info(f"📋 Found {len(uncatalogued_pdfs)} uncatalogued PDFs in materials folder:")
            for pdf in uncatalogued_pdfs:
                logger.info(f"   📄 {pdf.name} (not in materials.csv)")
            logger.info("💡 Add these PDFs to materials.csv to include them in search results")
        
        logger.info(f"🎉 Created {len(documents)} total document chunks from {len(pdf_files_to_process)} catalogued PDF materials")
        logger.info(f"📋 All chunks labeled with document relationships and CSV metadata")
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
        
        # Check for topic stores - if content_store or reference_store exists but no topic stores, we should rebuild
        if vector_stores_path.exists():
            existing_stores = [d.name for d in vector_stores_path.iterdir() if d.is_dir()]
            topic_stores = [store for store in existing_stores if store.startswith("topic_")]
            
            # If we have content/reference stores but no topic stores, force rebuild for topic generation
            has_main_stores = any(store in existing_stores for store in ["content_store", "reference_store"])
            if has_main_stores and not topic_stores:
                reasons.append("Topic stores missing - need to generate topic stores from existing content")
                force_rebuild = True
                logger.info(f"Found main stores {[s for s in essential_stores if s in existing_stores]} but no topic stores")
            elif topic_stores:
                logger.info(f"Found {len(topic_stores)} topic stores: {topic_stores[:3]}{'...' if len(topic_stores) > 3 else ''}")
    
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
        'origin_url': 'ORIGIN_URL',  # Keep as ORIGIN_URL, not SOURCE_URL
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

def complete_vector_store_cleanup(data_path, content_only=False):
    """
    Completely remove all existing vector stores for a fresh start.
    This ensures we always do a complete rebuild, never incremental.
    
    Args:
        data_path: Path to data directory
        content_only (bool): If True, only remove content_store, preserve others
        
    Returns:
        Dict with cleanup results
    """
    from pathlib import Path
    import shutil
    
    logger.info("🧹 Performing complete vector store cleanup for fresh rebuild...")
    
    cleanup_results = {
        "stores_removed": 0,
        "stores_preserved": 0,
        "action": "complete_cleanup",
        "issues_found": []
    }
    
    vector_stores_path = Path(data_path) / "vector_stores" / "default"
    
    if not vector_stores_path.exists():
        logger.info("No existing vector stores found - starting completely fresh")
        return cleanup_results
    
    try:
        # Remove all store directories for fresh start
        for store_dir in vector_stores_path.iterdir():
            if not store_dir.is_dir():
                continue
                
            store_name = store_dir.name
            
            # If content_only mode, only remove content_store, preserve all other stores
            if content_only and store_name != "content_store":
                logger.info(f"🔒 Content-only mode: Preserving {store_name}")
                cleanup_results["stores_preserved"] += 1
                continue
            
            # Remove this store completely for fresh rebuild
            logger.info(f"🗑️ Removing {store_name} for fresh rebuild")
            shutil.rmtree(store_dir)
            cleanup_results["stores_removed"] += 1
        
        logger.info(f"✅ Cleanup complete: removed {cleanup_results['stores_removed']} stores, preserved {cleanup_results['stores_preserved']} stores")
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Error during complete vector store cleanup: {e}")
        cleanup_results["issues_found"].append(f"Cleanup error: {e}")
        return cleanup_results

def validate_and_cleanup_vector_stores(data_path, content_only=False):
    """
    Validate existing vector stores and clean up ONLY corrupted or inconsistent data.
    Preserves working stores to avoid unnecessary rebuilds.
    
    Args:
        data_path: Path to data directory
        content_only (bool): If True, only validate content_store, preserve all others
        
    Returns:
        Dict with cleanup results
    """
    from pathlib import Path
    import shutil
    
    logger.info("🔧 Validating existing vector stores (removing only corrupted stores)...")
    
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
            
            # If content_only mode, only clean up content_store, preserve all other stores
            if content_only and store_name != "content_store":
                logger.info(f"🔒 Content-only mode: Preserving {store_name} (not cleaning)")
                continue
            
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
                document_counts = processing_results["document_counts"]
                
                # Handle different types of document_counts data
                if isinstance(document_counts, dict):
                    for store, count in document_counts.items():
                        f.write(f"  - {store}: {count} documents\n")
                elif isinstance(document_counts, (int, float)):
                    f.write(f"  - Total: {document_counts} documents\n")
                else:
                    f.write(f"  - Document counts: {document_counts}\n")
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
    parser.add_argument("--content-only", action="store_true", help="Force rebuild of content store only (materials/PDFs)")
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
        cleanup_results = validate_and_cleanup_vector_stores(data_path, content_only=args.content_only)
        
        logger.info(f"🔧 Cleanup Results: {cleanup_results}")
        
        result = asyncio.run(rebuild_indexes(data_path, rebuild_all=args.rebuild_all, rebuild_reference=args.rebuild_reference, content_only=args.content_only))
        
        if result:
            # Update rebuild timestamp on success
            update_last_rebuild_time(data_path)
            
            if args.rebuild_reference:
                logger.info("🎉 Complete rebuild successful")
            elif args.content_only:
                logger.info("🎉 Content store rebuild successful")
            else:
                logger.info("🎉 FAISS index rebuild successful")
            
            # Create processing summary using the actual rebuild results, not cleanup results
            processing_results = {
                "status": "success",
                "stores_rebuilt": result.get("stores_rebuilt", []),
                "total_documents": result.get("total_documents", 0),
                "document_counts": result.get("document_counts", {}),
                "cleanup_results": cleanup_results  # Include cleanup info separately
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
