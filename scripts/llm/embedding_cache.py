"""
Embedding cache module to optimize embedding generation.

This module provides a simple in-memory cache for embeddings to avoid
regenerating embeddings for the same query.
"""

import time
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """Cache for storing and retrieving embeddings to avoid regeneration."""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            ttl: Time-to-live in seconds for cache entries (default: 1 hour)
        """
        self.cache: Dict[str, Tuple[List[float], float]] = {}  # {query: (embedding, timestamp)}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        logger.info(f"Initialized embedding cache with max_size={max_size}, ttl={ttl}s")
    
    def get(self, query: str) -> Optional[List[float]]:
        """
        Get embedding from cache if it exists and is not expired.
        
        Args:
            query: The query string to look up
            
        Returns:
            The embedding vector if found and valid, None otherwise
        """
        if query in self.cache:
            embedding, timestamp = self.cache[query]
            # Check if the entry is still valid
            if time.time() - timestamp <= self.ttl:
                self.hits += 1
                logger.debug(f"Cache hit for query: {query[:30]}... [{self.hits} hits, {self.misses} misses]")
                return embedding
            else:
                # Remove expired entry
                logger.debug(f"Removing expired cache entry for: {query[:30]}...")
                del self.cache[query]
                
        self.misses += 1
        logger.debug(f"Cache miss for query: {query[:30]}... [{self.hits} hits, {self.misses} misses]")
        return None
    
    def set(self, query: str, embedding: List[float]) -> None:
        """
        Store embedding in the cache.
        
        Args:
            query: The query string to use as key
            embedding: The embedding vector to store
        """
        # If we're at capacity, remove the oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            logger.debug(f"Cache full, removing oldest entry: {oldest_key[:30]}...")
            del self.cache[oldest_key]
        
        # Store the new entry with current timestamp
        self.cache[query] = (embedding, time.time())
        logger.debug(f"Added to cache: {query[:30]}... [cache size: {len(self.cache)}]")
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }
