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
        
        # Skip if index.pkl already exists
        if index_pkl_file.exists():
            logger.info(f"index.pkl already exists in {store_path}, skipping")
            return {
                "success": True,
                "action": "skipped",
                "reason": "index.pkl already exists"
            }
        
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
        
        # Create document objects
        documents = []
        for doc_data in documents_data:
            doc = Document(
                page_content=doc_data.get('page_content', ''),
                metadata=doc_data.get('metadata', {})
            )
            documents.append(doc)
        
        # Create docstore
        logger.info(f"Created docstore with {len(documents)} documents")
        docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
        
        # Create index to docstore mapping
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
        return {
            "success": True,
            "action": "fixed",
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
            
            if result["success"] and result["action"] == "fixed":
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
