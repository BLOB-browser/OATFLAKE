# Simple in-memory cache for search results between /api/references and /api/web endpoints
import time
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SimpleSearchCache:
    """
    A simple in-memory cache to store search results between API endpoints.
    This prevents duplicate searches when users click Search then Generate.
    """
    
    def __init__(self, ttl_minutes: int = 30, max_entries: int = 100):
        """
        Initialize the cache.
        
        Args:
            ttl_minutes: Time to live for cache entries in minutes
            max_entries: Maximum number of entries to keep in cache
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl_minutes = ttl_minutes
        self._max_entries = max_entries
        logger.info(f"SimpleSearchCache initialized with TTL={ttl_minutes}min, max_entries={max_entries}")
    
    def _generate_key(self, query: str, k_value: int = 10) -> str:
        """Generate a cache key from query and parameters."""
        # Create a unique key based on query and k_value
        key_data = f"{query.strip().lower()}|k={k_value}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        now = datetime.now()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if now - entry['timestamp'] > timedelta(minutes=self._ttl_minutes):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _enforce_size_limit(self):
        """Remove oldest entries if cache exceeds max size."""
        if len(self._cache) <= self._max_entries:
            return
            
        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(
            self._cache.items(), 
            key=lambda x: x[1]['timestamp']
        )
        
        entries_to_remove = len(self._cache) - self._max_entries
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self._cache[key]
            
        logger.debug(f"Removed {entries_to_remove} oldest cache entries to enforce size limit")
    
    def put(self, query: str, search_results: Dict[str, Any], k_value: int = 10):
        """
        Store search results in cache.
        
        Args:
            query: The search query
            search_results: The results from /api/references endpoint
            k_value: The k parameter used in the search
        """
        try:
            # Clean up before adding new entry
            self._cleanup_expired()
            self._enforce_size_limit()
            
            key = self._generate_key(query, k_value)
            
            # Store the complete search results with metadata
            cache_entry = {
                'query': query,
                'k_value': k_value,
                'results': search_results,
                'timestamp': datetime.now(),
                'hits': 0  # Track how many times this entry is used
            }
            
            self._cache[key] = cache_entry
            logger.info(f"Cached search results for query: '{query[:50]}...' (key: {key})")
            
        except Exception as e:
            logger.error(f"Error caching search results: {e}")
    
    def get(self, query: str, k_value: int = 10) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached search results.
        
        Args:
            query: The search query
            k_value: The k parameter used in the search
            
        Returns:
            Cached search results or None if not found/expired
        """
        try:
            key = self._generate_key(query, k_value)
            
            if key not in self._cache:
                logger.debug(f"Cache miss for query: '{query[:50]}...' (key: {key})")
                return None
            
            entry = self._cache[key]
            
            # Check if entry is expired
            if datetime.now() - entry['timestamp'] > timedelta(minutes=self._ttl_minutes):
                del self._cache[key]
                logger.debug(f"Cache entry expired for query: '{query[:50]}...' (key: {key})")
                return None
            
            # Increment hit counter and return results
            entry['hits'] += 1
            logger.info(f"Cache hit for query: '{query[:50]}...' (key: {key}, hits: {entry['hits']})")
            
            return entry['results']
            
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None
    
    def clear(self):
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} entries from search cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        active_entries = 0
        total_hits = 0
        
        for entry in self._cache.values():
            if now - entry['timestamp'] <= timedelta(minutes=self._ttl_minutes):
                active_entries += 1
                total_hits += entry['hits']
        
        return {
            'total_entries': len(self._cache),
            'active_entries': active_entries,
            'total_hits': total_hits,
            'ttl_minutes': self._ttl_minutes,
            'max_entries': self._max_entries
        }

# Global cache instance
_search_cache = None

def get_search_cache() -> SimpleSearchCache:
    """Get the global search cache instance."""
    global _search_cache
    if _search_cache is None:
        _search_cache = SimpleSearchCache()
    return _search_cache
