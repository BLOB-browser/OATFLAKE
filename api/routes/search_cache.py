from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import logging

logger = logging.getLogger(__name__)

class SearchResultCache:
    """Simple in-memory cache for search results between API endpoints"""
    
    def __init__(self, ttl_minutes: int = 30, max_entries: int = 100):
        """
        Initialize the cache
        
        Args:
            ttl_minutes: Time to live for cache entries in minutes
            max_entries: Maximum number of entries to keep in cache
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.ttl_minutes = ttl_minutes
        self.max_entries = max_entries
        logger.info(f"SearchResultCache initialized with TTL={ttl_minutes}min, max_entries={max_entries}")
    
    def _generate_key(self, query: str, k_value: int = 10) -> str:
        """Generate a cache key from query and parameters"""
        # Create consistent cache key from query and k_value
        cache_input = f"{query.lower().strip()}|{k_value}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if current_time - entry['timestamp'] > timedelta(minutes=self.ttl_minutes):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _enforce_size_limit(self):
        """Remove oldest entries if cache exceeds max size"""
        if len(self._cache) <= self.max_entries:
            return
            
        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(
            self._cache.items(), 
            key=lambda x: x[1]['timestamp']
        )
        
        # Keep only the most recent max_entries
        entries_to_remove = len(sorted_entries) - self.max_entries
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self._cache[key]
            
        logger.debug(f"Cache size limit enforced, removed {entries_to_remove} oldest entries")
    
    def store(self, query: str, references: list, content: list, context: str, k_value: int = 10) -> str:
        """
        Store search results in cache
        
        Args:
            query: The search query
            references: List of reference results
            content: List of content results  
            context: Combined context string
            k_value: The k parameter used for search
            
        Returns:
            str: The cache key used for storage
        """
        # Clean up expired entries first
        self._cleanup_expired()
        
        cache_key = self._generate_key(query, k_value)
        
        self._cache[cache_key] = {
            'query': query,
            'references': references,
            'content': content,
            'context': context,
            'k_value': k_value,
            'timestamp': datetime.now(),
            'hit_count': 0
        }
        
        # Enforce size limit
        self._enforce_size_limit()
        
        logger.info(f"Stored search results in cache: query='{query[:50]}...', key={cache_key}, cache_size={len(self._cache)}")
        return cache_key
    
    def get(self, query: str, k_value: int = 10) -> Optional[Dict[str, Any]]:
        """
        Retrieve search results from cache
        
        Args:
            query: The search query
            k_value: The k parameter used for search
            
        Returns:
            Dict with search results if found, None otherwise
        """
        # Clean up expired entries first
        self._cleanup_expired()
        
        cache_key = self._generate_key(query, k_value)
        
        if cache_key not in self._cache:
            logger.debug(f"Cache miss for query: '{query[:50]}...', key={cache_key}")
            return None
            
        entry = self._cache[cache_key]
        
        # Check if entry is still valid (double-check after cleanup)
        if datetime.now() - entry['timestamp'] > timedelta(minutes=self.ttl_minutes):
            del self._cache[cache_key]
            logger.debug(f"Cache entry expired during retrieval: {cache_key}")
            return None
        
        # Increment hit count and update access time
        entry['hit_count'] += 1
        entry['last_accessed'] = datetime.now()
        
        logger.info(f"Cache hit for query: '{query[:50]}...', key={cache_key}, hit_count={entry['hit_count']}")
        
        return {
            'references': entry['references'],
            'content': entry['content'], 
            'context': entry['context'],
            'k_value': entry['k_value'],
            'cached_at': entry['timestamp'],
            'hit_count': entry['hit_count']
        }
    
    def clear(self):
        """Clear all cache entries"""
        cache_size = len(self._cache)
        self._cache.clear()
        logger.info(f"Cache cleared, removed {cache_size} entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        self._cleanup_expired()
        
        if not self._cache:
            return {
                'size': 0,
                'max_size': self.max_entries,
                'ttl_minutes': self.ttl_minutes,
                'total_hits': 0,
                'entries': []
            }
        
        total_hits = sum(entry.get('hit_count', 0) for entry in self._cache.values())
        
        entries_info = []
        for key, entry in list(self._cache.items())[:5]:  # Show first 5 entries
            entries_info.append({
                'key': key,
                'query_preview': entry['query'][:50] + '...' if len(entry['query']) > 50 else entry['query'],
                'cached_at': entry['timestamp'].isoformat(),
                'hit_count': entry.get('hit_count', 0),
                'references_count': len(entry.get('references', [])),
                'content_count': len(entry.get('content', []))
            })
        
        return {
            'size': len(self._cache),
            'max_size': self.max_entries,
            'ttl_minutes': self.ttl_minutes,
            'total_hits': total_hits,
            'entries': entries_info
        }

# Global cache instance
_search_cache = None

def get_search_cache() -> SearchResultCache:
    """Get the global search cache instance"""
    global _search_cache
    if _search_cache is None:
        _search_cache = SearchResultCache()
    return _search_cache
