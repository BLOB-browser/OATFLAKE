#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from langchain.schema import Document

logger = logging.getLogger(__name__)

class VectorGenerator:
    """
    Simplified vector generator that delegates to VectorStoreManager.
    Acts as a wrapper to maintain backward compatibility.
    """
    
    def __init__(self, data_folder: str, chunk_size: int = 200, chunk_overlap: int = 20):
        """Initialize the vector generator."""
        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    async def generate_vector_stores(self, content_paths: List[Path]) -> Dict[str, Any]:
        """
        Generate vector stores from content files by delegating to VectorStoreManager.
        
        Args:
            content_paths: Paths to JSONL files containing content to vectorize
            
        Returns:
            Dictionary with statistics about the vector generation process
        """
        # Import VectorStoreManager
        from scripts.storage.vector_store_manager import VectorStoreManager
        
        logger.info("Starting vector store generation using VectorStoreManager...")
        start_time = time.time()
        
        try:
            # Initialize the vector store manager
            vector_store_manager = VectorStoreManager(base_path=Path(self.data_folder))
            
            # Stats to track
            stats = {
                "documents_processed": 0,
                "embeddings_generated": 0,
                "stores_created": set(),
                "topic_stores_created": set(),
                "duration": 0
            }
            
            # Extract documents from all content files
            all_documents = await self._extract_documents_from_files(content_paths)
            
            if not all_documents:
                logger.warning("No documents found in content files")
                return {
                    "status": "warning",
                    "message": "No documents found in content files",
                    "duration_seconds": time.time() - start_time
                }
            
            logger.info(f"Extracted {len(all_documents)} documents from content files")
            
            # Group documents by store based on their metadata
            store_documents = self._group_documents_by_store(all_documents)
            
            # Process each store using VectorStoreManager
            for store_name, documents in store_documents.items():
                logger.info(f"Creating/updating store '{store_name}' with {len(documents)} documents")
                
                # Use VectorStoreManager to create the store
                success = await vector_store_manager.create_or_update_store(
                    store_name=store_name,
                    documents=documents,
                    metadata={"source": "vector_generator"}
                )
                
                if success:
                    stats["stores_created"].add(store_name)
                    stats["documents_processed"] += len(documents)
                    logger.info(f"Successfully created/updated store: {store_name}")
                
            # Create topic stores using documents with tags
            logger.info("Creating topic-specific stores...")
            
            # Filter documents that have tags
            tagged_docs = [doc for doc in all_documents 
                          if hasattr(doc, 'metadata') and 'tags' in doc.metadata 
                          and isinstance(doc.metadata['tags'], list)]
            
            if tagged_docs:
                logger.info(f"Found {len(tagged_docs)} documents with tags for topic stores")
                
                # Use VectorStoreManager's built-in method to create topic stores
                topic_results = await vector_store_manager.create_topic_stores(tagged_docs)
                
                # Add topic stores to stats
                for topic, success in topic_results.items():
                    if success:
                        topic_store_name = f"topic_{topic}"
                        stats["topic_stores_created"].add(topic_store_name)
                        logger.info(f"Created topic store: {topic_store_name}")
            else:
                logger.warning("No documents with tags found for topic stores")
            
            # Calculate final stats
            stats["duration"] = time.time() - start_time
            stats["stores_created"] = list(stats["stores_created"])  # Convert set to list for JSON
            stats["topic_stores_created"] = list(stats["topic_stores_created"])  # Convert set to list
            
            # Delete processed files
            for path in content_paths:
                try:
                    path.unlink()
                    logger.info(f"Deleted processed file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to delete processed file {path}: {e}")
            
            logger.info(f"Vector store generation completed in {stats['duration']:.2f} seconds")
            logger.info(f"Created {len(stats['stores_created'])} regular stores and {len(stats['topic_stores_created'])} topic stores")
            
            return {
                "status": "success",
                "documents_processed": stats["documents_processed"],
                "embeddings_generated": stats["documents_processed"],  # Estimate since we don't track chunks
                "stores_created": stats["stores_created"],
                "topic_stores_created": stats["topic_stores_created"],
                "duration_seconds": stats["duration"]
            }
            
        except Exception as e:
            logger.error(f"Error generating vector stores: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "duration_seconds": time.time() - start_time
            }
    
    async def _extract_documents_from_files(self, content_paths: List[Path]) -> List[Document]:
        """Extract documents from JSONL files."""
        documents = []
        
        for content_path in content_paths:
            if not content_path.exists():
                logger.warning(f"Content file not found: {content_path}")
                continue
                
            try:
                with open(content_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            doc_dict = json.loads(line.strip())
                            document = Document(
                                page_content=doc_dict["page_content"],
                                metadata=doc_dict["metadata"]
                            )
                            documents.append(document)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in {content_path}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading {content_path}: {e}")
                
        return documents
    
    def _group_documents_by_store(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Group documents by their target store based on metadata."""
        store_documents = {
            "content_store": [],  # Default store for content
            "reference_store": []  # Default store for references
        }
        
        for doc in documents:
            # Determine which store this document belongs to
            if "source_type" in doc.metadata:
                source_type = doc.metadata["source_type"].lower()
                
                # References go to reference_store, content to content_store
                if source_type in ["definition", "method", "project"]:
                    store_documents["reference_store"].append(doc)
                else:
                    store_documents["content_store"].append(doc)
            else:
                # If no source_type, use content_type
                content_type = doc.metadata.get("content_type", "").lower()
                
                if content_type in ["definition", "method", "project"]:
                    store_documents["reference_store"].append(doc)
                else:
                    # Default to content_store
                    store_documents["content_store"].append(doc)
        
        return store_documents
