from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
import platform
import psutil

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles text processing and document chunking."""
    def __init__(self):
        """Initialize document processor with optimized chunking configurations."""
        # Use optimized chunking settings for better performance and analysis quality
        logger.info("Using optimized chunking settings for better performance")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,      # Larger chunk size for better context and fewer chunks
            chunk_overlap=200,    # Moderate overlap for context preservation
            separators=[
                "\n\n",          # First try to split on double newlines
                "\n",            # Then single newlines
                ". ",            # Then sentences
                ", ",            # Then clauses
                " ",             # Then words
                ""               # Finally characters
            ],
            length_function=len,
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        # Skip if documents are already chunked
        already_chunked = any("chunk_index" in doc.metadata for doc in documents)
        
        if already_chunked:
            # Documents are already chunked, use them directly
            logger.info("Documents appear to be pre-chunked, using existing chunks")
            chunks = documents
        else:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
        
        # Log chunking stats
        if documents:
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            
            # Calculate chunk size statistics
            chunk_sizes = [len(chunk.page_content) for chunk in chunks]
            avg_size = sum(chunk_sizes) / len(chunks) if chunks else 0
            min_size = min(chunk_sizes) if chunks else 0
            max_size = max(chunk_sizes) if chunks else 0
            
            logger.info(f"Chunk size statistics: min={min_size}, max={max_size}, avg={avg_size:.1f}")
            
        return chunks
    
    def get_text_and_metadata(self, documents: List[Document]) -> tuple:
        """
        Extract texts and metadata from documents.
        
        Args:
            documents: List of documents to extract from
            
        Returns:
            Tuple of (texts, metadatas)
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return texts, metadatas