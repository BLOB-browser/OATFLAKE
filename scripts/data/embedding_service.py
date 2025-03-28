from pathlib import Path
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
from datetime import datetime
import platform
import psutil

from langchain.schema import Document
from ..llm.ollama_embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Centralized service for generating embeddings."""
    
    def __init__(self):
        """Initialize embedding service with consistent configurations."""
        # Consistent embeddings configuration for all devices
        logger.info("Using consistent embedding settings for all devices")
        self.embeddings = OllamaEmbeddings(batch_size=20, timeout=60.0)
        
        logger.info(f"Embedding service initialized using {self.embeddings.model_name}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Process with detailed timing
            start_time = datetime.now()
            logger.info(f"Embedding process started at {start_time.isoformat()} for {len(texts)} texts")
            
            # Generate embeddings
            embeddings = await self.embeddings.aembeddings(texts)
            
            # Log completion time and performance
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            text_lengths = [len(t) for t in texts]
            total_chars = sum(text_lengths)
            
            logger.info(f"Embedding completed in {duration:.1f} seconds")
            logger.info(f"Performance: {duration/len(texts):.2f} seconds per text, {total_chars/duration:.1f} chars/second")
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model being used."""
        return self.embeddings.model_name