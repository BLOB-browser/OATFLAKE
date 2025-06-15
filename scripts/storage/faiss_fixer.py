#!/usr/bin/env python3
"""
FAISS Vector Store Fixer

This module provides functionality to fix broken FAISS vector stores
by regenerating missing index.pkl files from existing index.faiss and documents.json files.
"""

import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import faiss

logger = logging.getLogger(__name__)

def check_faiss_corruption(store_path: Path) -> bool:
    """
    Check if a FAISS vector store has index/docstore synchronization issues.
    
    Args:
        store_path: Path to the vector store directory
        
    Returns:
        True if corruption is detected, False otherwise
    """
    try:
        index_faiss_file = store_path / "index.faiss"
        index_pkl_file = store_path / "index.pkl"
        
        if not index_faiss_file.exists() or not index_pkl_file.exists():
            return True  # Missing files indicate corruption
        
        # Load FAISS index
        index = faiss.read_index(str(index_faiss_file))
        
        # Load docstore and index mapping
        with open(index_pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        docstore = data.get("docstore")
        index_to_docstore_id = data.get("index_to_docstore_id", {})
        
        if not docstore or not index_to_docstore_id:
            logger.warning(f"Missing docstore or index mapping in {store_path}")
            return True
        
        # Check if index references non-existent documents
        missing_docs = []
        for i in range(index.ntotal):
            doc_id = index_to_docstore_id.get(i, str(i))
            if doc_id not in docstore._dict:
                missing_docs.append(doc_id)
        
        if missing_docs:
            logger.warning(f"Found {len(missing_docs)} missing documents in docstore: {missing_docs[:10]}...")
            return True
        
        # Check for count mismatches
        if len(docstore._dict) != index.ntotal:
            logger.warning(f"Count mismatch: docstore has {len(docstore._dict)} docs, index has {index.ntotal} vectors")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking corruption in {store_path}: {e}")
        return True  # Assume corruption if we can't check

def fix_single_faiss_store(store_path: Path) -> Dict[str, Any]:
    """
    Fix a single FAISS vector store by regenerating missing index.pkl file.
    
    Args:
        store_path: Path to the vector store directory
        
    Returns:
        Dictionary with fix results
    """
    try:
        store_name = store_path.name
        logger.info(f"Checking vector store: {store_name}")
        
        # Check required files
        index_faiss_file = store_path / "index.faiss"
        index_pkl_file = store_path / "index.pkl"
        documents_file = store_path / "documents.json"
        
        # Check for corruption even if index.pkl exists
        corruption_detected = False
        if index_pkl_file.exists():
            logger.info(f"index.pkl exists, checking for corruption in {store_path}")
            corruption_detected = check_faiss_corruption(store_path)
            
            if not corruption_detected:
                logger.info(f"No corruption detected in {store_path}, skipping")
                return {
                    "success": True,
                    "action": "skipped",
                    "reason": "No corruption detected"
                }
            else:
                logger.warning(f"Corruption detected in {store_path}, will rebuild")
        
        # Check if required files exist
        if not index_faiss_file.exists():
            logger.error(f"Missing index.faiss file in {store_path}")
            return {
                "success": False,
                "action": "failed",
                "reason": "Missing index.faiss file"
            }
            
        if not documents_file.exists():
            logger.error(f"Missing documents.json file in {store_path}")
            return {
                "success": False,
                "action": "failed", 
                "reason": "Missing documents.json file"
            }
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_faiss_file}")
        index = faiss.read_index(str(index_faiss_file))
        logger.info(f"FAISS index has {index.ntotal} vectors")
        
        # Load documents
        logger.info(f"Loading documents from {documents_file}")
        with open(documents_file, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)
        logger.info(f"Found {len(documents_data)} documents")
        
        # Create document objects - handle both old and new document formats
        documents = []
        for doc_data in documents_data:
            # Handle both formats: new format uses 'content' and old format uses 'page_content'
            content = doc_data.get('content', doc_data.get('page_content', ''))
            metadata = doc_data.get('metadata', {})
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        # Verify we have the right number of documents vs vectors
        if len(documents) != index.ntotal:
            logger.warning(f"Document count ({len(documents)}) doesn't match vector count ({index.ntotal})")
            # Adjust to the smaller count to prevent KeyErrors
            min_count = min(len(documents), index.ntotal)
            documents = documents[:min_count]
            logger.info(f"Adjusted document count to {min_count} to match vector index")
        
        # Create docstore
        logger.info(f"Created docstore with {len(documents)} documents")
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        
        # Create index to docstore mapping - ensure it matches the actual index size
        index_to_docstore_id = {i: str(i) for i in range(len(documents))}
        logger.info(f"Created index mapping with {len(index_to_docstore_id)} entries")
        
        # Save docstore and index mapping as pickle
        logger.info(f"Saving docstore and index mapping to {index_pkl_file}")
        data_to_save = {
            "docstore": docstore,
            "index_to_docstore_id": index_to_docstore_id
        }
        
        with open(index_pkl_file, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        logger.info(f"✅ Successfully fixed vector store: {store_name}")
        action_type = "corruption_fixed" if corruption_detected else "regenerated"
        return {
            "success": True,
            "action": action_type,
            "documents_count": len(documents),
            "vectors_count": index.ntotal
        }
        
    except Exception as e:
        logger.error(f"Error fixing vector store {store_path}: {e}")
        return {
            "success": False,
            "action": "failed",
            "reason": str(e)
        }

def fix_all_faiss_stores(data_folder: str) -> Dict[str, Any]:
    """
    Fix all FAISS vector stores in the data folder.
    
    Args:
        data_folder: Path to the data directory
        
    Returns:
        Dictionary with overall fix results
    """
    try:
        logger.info("🔧 Starting FAISS vector store fix process")
        
        data_path = Path(data_folder)
        vector_stores_path = data_path / "vector_stores" / "default"
        
        if not vector_stores_path.exists():
            logger.warning(f"Vector stores directory doesn't exist: {vector_stores_path}")
            return {
                "success": True,
                "message": "No vector stores directory found",
                "fixed_stores": 0,
                "failed_stores": 0,
                "results": []
            }
        
        # Find all vector store directories
        store_dirs = [d for d in vector_stores_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(store_dirs)} vector store directories")
        
        results = []
        fixed_count = 0
        failed_count = 0
        
        for store_dir in store_dirs:
            result = fix_single_faiss_store(store_dir)
            result["store_name"] = store_dir.name
            results.append(result)
            
            if result["success"] and result["action"] in ["regenerated", "corruption_fixed"]:
                fixed_count += 1
            elif not result["success"]:
                failed_count += 1
        
        logger.info("=" * 50)
        logger.info("FAISS Vector Store Fix Summary:")
        logger.info(f"  Total stores processed: {len(store_dirs)}")
        logger.info(f"  Successfully fixed: {fixed_count}")
        logger.info(f"  Failed to fix: {failed_count}")
        logger.info("=" * 50)
        
        if failed_count > 0:
            logger.warning(f"⚠️  {failed_count} vector stores failed to fix")
        
        return {
            "success": True,
            "message": f"Fixed {fixed_count} vector stores, {failed_count} failed",
            "fixed_stores": fixed_count,
            "failed_stores": failed_count,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error during FAISS fix process: {e}")
        return {
            "success": False,
            "message": str(e),
            "fixed_stores": 0,
            "failed_stores": 0,
            "results": []
        }
