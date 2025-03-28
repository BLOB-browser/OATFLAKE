from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import json
from datetime import datetime
import os
import faiss
import numpy as np

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class FAISSBuilder:
    """Builds and manages FAISS indexes."""
    
    def __init__(self, base_path: Path):
        """
        Initialize FAISS builder.
        
        Args:
            base_path: Base path where vector stores should be stored
        """
        self.base_path = Path(base_path)
        self.vectors_path = self.base_path / "vector_stores"
        self.vectors_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService()
        self.embeddings = self.embedding_service.embeddings
    
    def _get_store_path(self, store_name: str, group_id: str = "default") -> Path:
        """
        Get path for a vector store.
        
        Args:
            store_name: Name of the store
            group_id: Group ID (defaults to "default")
            
        Returns:
            Path to the store directory
        """
        group_to_use = group_id if group_id else "default"
        logger.info(f"Using vector store path: {self.vectors_path}/{group_to_use}/{store_name}")
        return self.vectors_path / group_to_use / store_name
    
    async def create_index(self, 
        texts: List[str], 
        embeddings_list: List[List[float]], 
        metadatas: List[Dict[str, Any]], 
        store_name: str, 
        group_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Create a new FAISS index from embeddings.
        
        Args:
            texts: List of text content
            embeddings_list: List of embeddings
            metadatas: List of metadata dictionaries
            store_name: Name of the store to create
            group_id: Group ID (defaults to "default")
            
        Returns:
            Dictionary with status information
        """
        try:
            store_path = self._get_store_path(store_name, group_id)
            store_path.mkdir(parents=True, exist_ok=True)
            
            # Check if we have valid embeddings
            if not embeddings_list or len(embeddings_list) == 0:
                logger.error(f"No embeddings provided for store {store_name}")
                return {"status": "error", "message": "No embeddings provided"}
            
            # Create FAISS vector store
            logger.info(f"Creating new FAISS index for {store_name} with {len(texts)} documents")
            vector_store = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings_list)),
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # Save the FAISS index
            vector_store.save_local(str(store_path))
            
            # Save documents separately for easier rebuilding
            docs_data = []
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                docs_data.append({
                    "content": text,
                    "metadata": metadata,
                    "embedding_id": i
                })
            
            with open(store_path / "documents.json", "w") as f:
                json.dump(docs_data, f, indent=2)
            
            # Create embedding stats
            embedding_stats = {
                "document_count": len(texts),
                "chunk_count": len(texts),
                "embedding_count": len(embeddings_list),
                "dimension": len(embeddings_list[0]) if embeddings_list else 0,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "model": self.embeddings.model_name
            }
            
            # Save stats
            with open(store_path / "embedding_stats.json", "w") as f:
                json.dump(embedding_stats, f, indent=2)
                
            logger.info(f"Successfully created vector store: {store_name}")
            return {
                "status": "success",
                "message": f"Created vector store {store_name}",
                "document_count": len(texts),
                "embedding_count": len(embeddings_list)
            }
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def rebuild_all_indexes(self) -> Dict[str, Any]:
        """
        Rebuild all FAISS indexes from stored document data.
        
        Returns:
            Dictionary with status information
        """
        try:
            # Get default vector store path
            vector_path = self.vectors_path / "default"
            if not vector_path.exists():
                logger.warning(f"Vector store path not found: {vector_path}")
                return {"status": "error", "message": "Vector store path not found"}
            
            # Find all stores
            store_paths = [p for p in vector_path.iterdir() if p.is_dir()]
            logger.info(f"Found {len(store_paths)} vector stores to rebuild")
            
            # Track rebuild statistics
            stores_rebuilt = []
            document_counts = {}
            total_documents = 0
            
            # Process each store
            for store_path in store_paths:
                store_name = store_path.name
                documents_file = store_path / "documents.json"
                
                # Skip if documents.json doesn't exist
                if not documents_file.exists():
                    logger.warning(f"Store {store_name} missing documents.json, skipping")
                    continue
                
                try:
                    # Load documents from JSON
                    with open(documents_file, 'r') as f:
                        doc_data = json.load(f)
                    
                    document_count = len(doc_data)
                    logger.info(f"Rebuilding {store_name} with {document_count} documents")
                    
                    if document_count == 0:
                        logger.warning(f"Store {store_name} has no documents, skipping")
                        continue
                    
                    # Extract texts and metadata from document data
                    texts = []
                    metadatas = []
                    for doc in doc_data:
                        texts.append(doc["content"])
                        metadatas.append(doc["metadata"])
                    
                    # Generate embeddings
                    logger.info(f"Generating embeddings for {len(texts)} documents")
                    embeddings_list = await self.embedding_service.generate_embeddings(texts)
                    
                    # Create new FAISS index
                    vector_store = FAISS.from_embeddings(
                        text_embeddings=list(zip(texts, embeddings_list)),
                        embedding=self.embeddings,
                        metadatas=metadatas
                    )
                    
                    # Save vector store
                    vector_store.save_local(str(store_path))
                    
                    # Update metadata
                    if (store_path / "embedding_stats.json").exists():
                        try:
                            with open(store_path / "embedding_stats.json", 'r') as f:
                                stats_data = json.load(f)
                            
                            # Update stats
                            stats_data["rebuilt_at"] = datetime.now().isoformat()
                            stats_data["updated_at"] = datetime.now().isoformat()
                            stats_data["embedding_count"] = len(embeddings_list)
                            
                            with open(store_path / "embedding_stats.json", 'w') as f:
                                json.dump(stats_data, f, indent=2)
                        except Exception as stats_err:
                            logger.error(f"Error updating stats for {store_name}: {stats_err}")
                    
                    # Update tracking
                    stores_rebuilt.append(store_name)
                    document_counts[store_name] = document_count
                    total_documents += document_count
                    
                    logger.info(f"Successfully rebuilt {store_name}")
                    
                except Exception as e:
                    logger.error(f"Error rebuilding store {store_name}: {e}")
            
            return {
                "status": "success",
                "message": f"Rebuilt {len(stores_rebuilt)} vector stores with {total_documents} total documents",
                "stores_rebuilt": stores_rebuilt,
                "document_counts": document_counts,
                "total_documents": total_documents
            }
            
        except Exception as e:
            logger.error(f"Error in rebuild_all_indexes: {e}")
            return {
                "status": "error",
                "message": str(e)
            }