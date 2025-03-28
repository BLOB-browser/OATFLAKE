from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os

from langchain.schema import Document

from .document_loader import DocumentLoader
from .document_processor import DocumentProcessor
from .embedding_service import EmbeddingService
from .faiss_builder import FAISSBuilder

logger = logging.getLogger(__name__)

class ProcessingManager:
    """Orchestrates the knowledge processing pipeline."""
    
    def __init__(self, data_path: Path, group_id: str = "default"):
        """
        Initialize processing manager.
        
        Args:
            data_path: Base path for data files
            group_id: Group ID for vector stores
        """
        self.base_path = Path(data_path)
        self.group_id = group_id
        
        # Initialize components
        self.document_loader = DocumentLoader(self.base_path)
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.faiss_builder = FAISSBuilder(self.base_path)
        
        # Stats tracking
        self.stats = {
            "documents_processed": 0,
            "chunks_generated": 0,
            "embeddings_generated": 0,
            "stores_updated": 0
        }
    
    async def process_critical_content(self, incremental=True) -> Dict[str, Any]:
        """
        Process high-priority content (PDFs and methods).
        
        Args:
            incremental: Whether to use incremental processing
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            logger.info("Processing critical content (PDFs and methods)...")
            reference_docs = []
            content_docs = []
            
            # Load PDF materials
            # This would be implemented to load documents with document_loader
            # For example:
            # pdf_docs = await self.document_loader.load_pdf_materials()
            # content_docs.extend(pdf_docs)
            
            # Process reference documents (simplified example)
            # In a real implementation, you would call the appropriate document_loader methods
            # reference_docs.extend(self.document_loader.load_methods())
            
            # Process documents in reference store
            if reference_docs:
                await self._process_store(reference_docs, "reference_store")
            
            # Process documents in content store
            if content_docs:
                await self._process_store(content_docs, "content_store")
            
            return {
                "status": "success",
                "reference_docs_processed": len(reference_docs),
                "content_docs_processed": len(content_docs)
            }
        except Exception as e:
            logger.error(f"Error processing critical content: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _process_store(self, documents: List[Document], store_name: str) -> Dict[str, Any]:
        """
        Process documents and create/update a vector store.
        
        Args:
            documents: List of documents to process
            store_name: Name of the store to update
            
        Returns:
            Dictionary with processing statistics
        """
        try:
            # Chunk documents
            chunks = self.document_processor.chunk_documents(documents)
            self.stats["chunks_generated"] += len(chunks)
            
            # Get texts and metadata
            texts, metadatas = self.document_processor.get_text_and_metadata(chunks)
            
            # Generate embeddings
            embeddings = await self.embedding_service.generate_embeddings(texts)
            self.stats["embeddings_generated"] += len(embeddings)
            
            # Create FAISS index
            result = await self.faiss_builder.create_index(
                texts=texts,
                embeddings_list=embeddings,
                metadatas=metadatas,
                store_name=store_name,
                group_id=self.group_id
            )
            
            if result["status"] == "success":
                self.stats["stores_updated"] += 1
                
            return result
        except Exception as e:
            logger.error(f"Error processing store {store_name}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def rebuild_all_indexes(self) -> Dict[str, Any]:
        """
        Rebuild all FAISS indexes from document data.
        
        Returns:
            Dictionary with rebuild statistics
        """
        try:
            logger.info("Rebuilding all FAISS indexes...")
            result = await self.faiss_builder.rebuild_all_indexes()
            return result
        except Exception as e:
            logger.error(f"Error rebuilding FAISS indexes: {e}")
            return {
                "status": "error",
                "message": str(e)
            }