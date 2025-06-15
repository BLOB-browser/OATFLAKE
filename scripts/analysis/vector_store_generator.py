#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class VectorStoreGenerator:
    """
    Generates vector stores from processed content.
    This component handles STEP 5 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the vector store generator.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
    async def generate_vector_stores(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Generate vector stores from processed content.
        
        Args:
            force_update: If True, forces generation even with no new content
            
        Returns:
            Dictionary with processing results
        """
        logger.info("STEP 5: GENERATING VECTOR STORES FROM PROCESSED CONTENT")
        logger.info("======================================================")
        
        try:
            from scripts.storage.content_storage_service import ContentStorageService
            content_storage = ContentStorageService(self.data_folder)
            
            # Look for JSONL files in multiple locations
            content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
            temp_dir = Path(self.data_folder) / "temp"
            if temp_dir.exists():
                content_paths.extend(list(temp_dir.glob("*.jsonl")))
                # Also check vector_data folder if it exists
                vector_data_path = temp_dir / "vector_data"
                if vector_data_path.exists():
                    content_paths.extend(list(vector_data_path.glob("*.jsonl")))
            
            # Check if we should proceed even if no content files found
            if not content_paths and force_update:
                logger.info("No new content files found. Looking for existing content to vectorize.")
                # Look for existing content files
                existing_content_dir = Path(self.data_folder) / "content"
                if existing_content_dir.exists():
                    content_paths.extend(list(existing_content_dir.glob("*.jsonl")))
                    logger.info(f"Found {len(content_paths)} existing content files to vectorize from content directory")
            
            if content_paths:
                logger.info(f"Found {len(content_paths)} content files for vector generation")
                
                # Run the rebuild_faiss_indexes.py script for vector generation
                logger.info("Running rebuild_faiss_indexes.py script to generate vector stores")
                
                # Get the path to the rebuild script
                script_path = Path(__file__).parents[2] / "scripts" / "tools" / "rebuild_faiss_indexes.py"
                
                # Run the script as a separate process
                logger.info(f"Executing rebuild script: {script_path}")
                rebuild_start_time = time.time()
                
                try:
                    # Run with the same Python interpreter
                    python_path = sys.executable
                    cmd = [python_path, str(script_path), "--data-path", self.data_folder]
                    
                    # Execute with real-time output
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                        bufsize=1
                    )
                    
                    # Stream the output in real-time
                    logger.info("===== BEGIN REBUILD SCRIPT OUTPUT =====")
                    rebuild_output = []
                    for line in process.stdout:
                        line = line.strip()
                        rebuild_output.append(line)
                        if line:
                            logger.info(f"REBUILD: {line}")
                    
                    # Wait for process to complete
                    return_code = process.wait()
                    logger.info("===== END REBUILD SCRIPT OUTPUT =====")
                    rebuild_duration = time.time() - rebuild_start_time
                    
                    # Parse output to extract stats
                    rebuilt_stores = []
                    total_documents = 0
                    
                    for line in rebuild_output:
                        if "Successfully rebuilt" in line and "store" in line:
                            store_name = line.split("Successfully rebuilt")[-1].strip()
                            rebuilt_stores.append(store_name)
                        elif "documents" in line and ":" in line:
                            try:
                                store_info = line.strip().split(":")
                                if len(store_info) >= 2 and "documents" in store_info[1]:
                                    doc_count = int(store_info[1].split("documents")[0].strip())
                                    total_documents += doc_count
                            except:
                                pass
                    
                    if return_code == 0:
                        return {
                            "status": "success",
                            "documents_processed": total_documents,
                            "embeddings_generated": total_documents,
                            "stores_created": rebuilt_stores,
                            "topic_stores_created": [store for store in rebuilt_stores if store.startswith("topic_")],
                            "duration_seconds": rebuild_duration
                        }
                    else:
                        return {
                            "status": "error",
                            "message": f"Rebuild script failed with return code {return_code}",
                            "duration_seconds": rebuild_duration
                        }
                
                except Exception as e:
                    logger.error(f"Error executing rebuild script: {e}")
                    return {
                        "status": "error",
                        "message": str(e),
                        "duration_seconds": time.time() - rebuild_start_time
                    }
            else:
                # Even if no content files found, try to rebuild existing stores
                logger.info("No content files found, attempting to rebuild existing vector stores")
                
                # Use VectorStoreManager's capabilities directly
                try:
                    from scripts.storage.vector_store_manager import VectorStoreManager
                    
                    # Initialize VectorStoreManager
                    vector_store_manager = VectorStoreManager(base_path=self.data_folder)
                    
                    # List existing stores
                    stores = vector_store_manager.list_stores()
                    
                    # Record which stores were successfully rebuilt
                    rebuilt_stores = []
                    topic_stores = []
                    
                    # Force rebuild of existing stores
                    for store in stores:
                        store_name = store.get("name")
                        
                        # Skip if not a real store (metadata only)
                        if not store_name:
                            continue
                            
                        # For each store, try to rebuild topic stores
                        if store_name == "content_store":
                            logger.info("Attempting to rebuild topic stores from content_store")
                            
                            # Get representative chunks
                            rep_docs = await vector_store_manager.get_representative_chunks(
                                store_name=store_name, 
                                num_chunks=100
                            )
                            
                            if rep_docs:
                                # Create topic stores using tag-based approach
                                topic_results = await vector_store_manager.create_topic_stores(
                                    rep_docs,
                                    use_clustering=False,  # Use individual tag stores
                                    min_docs_per_topic=1
                                )
                                
                                # Track rebuilt topic stores
                                for topic, success in topic_results.items():
                                    if success:
                                        topic_stores.append(f"topic_{topic}")
                        
                        # Track this store as rebuilt
                        rebuilt_stores.append(store_name)
                    
                    # Record rebuild results
                    return {
                        "status": "success",
                        "message": "Rebuilt existing vector stores",
                        "stores_created": rebuilt_stores,
                        "topic_stores_created": topic_stores,
                        "duration_seconds": 0
                    }
                    
                except Exception as e:
                    logger.error(f"Error rebuilding vector stores: {e}")
                    return {
                        "status": "error",
                        "message": "Error rebuilding vector stores",
                        "error": str(e)
                    }
                    
        except Exception as e:
            logger.error(f"Error during vector generation: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_temporary_files(self):
        """
        Clean up temporary files after vector generation.
        """
        try:
            from scripts.analysis.cleanup_manager import CleanupManager
            cleanup_manager = CleanupManager(self.data_folder)
            cleanup_manager.cleanup_temporary_files()
            logger.info("Cleaned up temporary files after vector generation")
        except Exception as e:
            logger.error(f"Error during cleanup after vector generation: {e}")

# Standalone function for easy import
async def generate_vector_stores(data_folder: str, force_update: bool = False) -> Dict[str, Any]:
    """
    Generate vector stores from processed content.
    
    Args:
        data_folder: Path to data folder
        force_update: If True, forces generation even with no new content
        
    Returns:
        Dictionary with processing results
    """
    generator = VectorStoreGenerator(data_folder)
    return await generator.generate_vector_stores(force_update)