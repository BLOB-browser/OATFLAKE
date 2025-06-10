"""
Embedding Cache Inspector for OATFLAKE.

This utility allows you to inspect the internal state of the embedding cache,
including cache statistics, contents, and integration with the search API endpoints.

Usage:
    python -m tests.inspect_embedding_cache

The script can be used both programmatically and as a standalone tool to:
1. Inspect cache statistics
2. Check cache contents and memory usage
3. Test cache eviction policies
4. Verify cache is properly shared across API endpoints
"""

import asyncio
import time
import logging
import sys
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import statistics

# Add the project root to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

# Create a logger for this tool
logger = logging.getLogger("embedding_cache_inspector")

class EmbeddingCacheInspector:
    """Utility class to inspect the embedding cache in OATFLAKE."""
    
    def __init__(self):
        """Initialize the inspector with references to the embedding cache instances."""
        # Import required modules
        from scripts.llm.open_router_client import OpenRouterClient, _embedding_cache
        from scripts.llm.embedding_cache import EmbeddingCache
        from scripts.llm.ollama_embeddings import OllamaEmbeddings
        
        self.shared_cache = _embedding_cache  # Access the global shared cache
        logger.info(f"Found shared embedding cache: {self.shared_cache}")
        
        # Create a client to access the search functionality
        self.client = OpenRouterClient()
        logger.info(f"Initialized OpenRouter client")
    
    def inspect_cache_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the embedding cache."""
        stats = self.shared_cache.stats()
        
        logger.info("Cache Statistics:")
        logger.info(f"  Size: {stats['size']} / {stats['max_size']} entries ({stats['size'] / stats['max_size'] * 100:.1f}% full)")
        logger.info(f"  Hits: {stats['hits']}")
        logger.info(f"  Misses: {stats['misses']}")
        logger.info(f"  Hit ratio: {stats['hit_ratio']:.2%}")
        
        return stats
    
    def inspect_cache_contents(self, max_items: int = 10) -> None:
        """Inspect the contents of the cache."""
        logger.info("Cache Contents:")
        
        # Get a list of items sorted by timestamp (newest first)
        cache_items = sorted(
            [(query, embedding, timestamp) for query, (embedding, timestamp) in self.shared_cache.cache.items()],
            key=lambda x: x[2],
            reverse=True
        )
        
        # Calculate cache memory usage
        total_bytes = sum(sys.getsizeof(emb) for _, emb, _ in cache_items)
        logger.info(f"  Total approximate memory usage: {total_bytes / 1024:.1f} KB")
        
        # Display most recent entries
        logger.info(f"  Most recent {min(max_items, len(cache_items))} entries:")
        for i, (query, embedding, timestamp) in enumerate(cache_items[:max_items]):
            age = time.time() - timestamp
            logger.info(f"  {i+1}. '{query[:30]}...' - {len(embedding)} dimensions, {age:.1f}s old")
        
        # Display oldest entries
        if len(cache_items) > max_items * 2:
            logger.info(f"  Oldest {min(max_items, len(cache_items))} entries:")
            for i, (query, embedding, timestamp) in enumerate(cache_items[-max_items:]):
                age = time.time() - timestamp
                logger.info(f"  {len(cache_items) - max_items + i + 1}. '{query[:30]}...' - {len(embedding)} dimensions, {age:.1f}s old")
    
    async def test_cache_eviction(self, num_entries: int = 10) -> None:
        """Test the cache eviction policy by filling it beyond capacity."""
        logger.info("Testing cache eviction policy:")
        
        # Get current cache size
        initial_stats = self.shared_cache.stats()
        logger.info(f"  Initial cache size: {initial_stats['size']} / {initial_stats['max_size']}")
        
        # Calculate how many entries to add to trigger eviction
        entries_to_add = max(1, initial_stats['max_size'] - initial_stats['size'] + num_entries)
        logger.info(f"  Adding {entries_to_add} entries to trigger eviction")
        
        # Add entries to fill the cache
        for i in range(entries_to_add):
            unique_query = f"Test query for eviction testing {datetime.now().isoformat()} {i}"
            embedding = await self.client.embeddings.aembeddings([unique_query])
            logger.info(f"  Added entry {i+1}/{entries_to_add}")
            await asyncio.sleep(0.1)  # Small delay to prevent overloading
        
        # Check final cache state
        final_stats = self.shared_cache.stats()
        logger.info(f"  Final cache size: {final_stats['size']} / {final_stats['max_size']}")
        logger.info(f"  Entries evicted: {initial_stats['size'] + entries_to_add - final_stats['size']}")
        
        # Verify the cache size hasn't exceeded max_size
        if final_stats['size'] > final_stats['max_size']:
            logger.error(f"  PROBLEM: Cache size ({final_stats['size']}) exceeds max_size ({final_stats['max_size']})")
        else:
            logger.info("  Cache eviction policy working correctly")
    
    async def test_cache_expiration(self, ttl_factor: float = 0.5) -> None:
        """Test the cache expiration policy by waiting for entries to expire."""
        logger.info("Testing cache expiration policy:")
        
        # Get cache TTL
        ttl = self.shared_cache.ttl
        logger.info(f"  Cache TTL: {ttl} seconds")
        
        # Create a test entry
        test_query = f"Test query for expiration testing {datetime.now().isoformat()}"
        logger.info(f"  Adding test entry: '{test_query[:30]}...'")
        
        # Add to cache directly to control timestamp
        embedding = await self.client.embeddings.aembeddings([test_query])
        
        # Manipulate the timestamp to make it almost expired
        wait_time = ttl * ttl_factor
        current_time = time.time()
        fake_timestamp = current_time - (ttl - wait_time)
        self.shared_cache.cache[test_query] = (embedding[0], fake_timestamp)
        
        logger.info(f"  Entry added with artificial age of {ttl - wait_time:.1f}s (TTL: {ttl}s)")
        logger.info(f"  Waiting {wait_time:.1f}s for entry to expire...")
        
        # Wait for entry to expire
        await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
        
        # Check if entry expired
        result = self.shared_cache.get(test_query)
        if result is None:
            logger.info("  ✓ Entry expired as expected")
        else:
            logger.error("  ✕ Entry did not expire when it should have!")
    
    async def test_cache_sharing_across_endpoints(self) -> None:
        """Test if the cache is properly shared across different API endpoints."""
        logger.info("Testing cache sharing across API endpoints:")
        
        # Create a test query
        test_query = f"Test query for API endpoint sharing {datetime.now().isoformat()}"
        
        # First, use unified_search to populate the cache
        logger.info(f"  Step 1: Unified search with query '{test_query[:30]}...'")
        t1_start = time.time()
        unified_results = await self.client.unified_search(query=test_query, k_reference=3, k_content=3)
        t1_duration = time.time() - t1_start
        
        unified_from_cache = unified_results.get("metadata", {}).get("embedding_from_cache", False)
        logger.info(f"  Unified search took {t1_duration:.3f}s (from cache: {unified_from_cache})")
        
        # Get cache stats after first query
        stats_after_unified = self.shared_cache.stats()
        
        # Now try a deeper search with the same query - should hit cache
        logger.info(f"  Step 2: Deeper search with same query")
        t2_start = time.time()
        deeper_results = await self.client.search_deeper(
            query=test_query,
            pinned_items=[],
            k=3
        )
        t2_duration = time.time() - t2_start
        
        deeper_from_cache = deeper_results.get("metadata", {}).get("embedding_from_cache", False)
        logger.info(f"  Deeper search took {t2_duration:.3f}s (from cache: {deeper_from_cache})")
        
        # Get cache stats after second query
        stats_after_deeper = self.shared_cache.stats()
        
        # Check if the second query hit the cache
        if deeper_from_cache:
            logger.info("  ✓ Cache successfully shared between endpoints")
            logger.info(f"  Performance improvement: {t1_duration / t2_duration:.1f}x faster")
        else:
            logger.error("  ✕ Cache not shared between endpoints!")
            logger.info(f"  Cache hits before: {stats_after_unified['hits']}, after: {stats_after_deeper['hits']}")
    
    async def run_all_tests(self, skip_eviction: bool = False) -> None:
        """Run all embedding cache tests."""
        logger.info("=" * 80)
        logger.info("OATFLAKE EMBEDDING CACHE INSPECTION")
        logger.info("=" * 80)
        
        # Check cache statistics
        self.inspect_cache_stats()
        
        # Check cache contents
        self.inspect_cache_contents()
        
        # Test cache sharing across endpoints
        await self.test_cache_sharing_across_endpoints()
        
        # Test cache expiration
        await self.test_cache_expiration()
        
        # Test cache eviction (skip if requested)
        if not skip_eviction:
            await self.test_cache_eviction()
        
        # Final cache statistics
        logger.info("=" * 80)
        logger.info("FINAL CACHE STATISTICS")
        logger.info("=" * 80)
        self.inspect_cache_stats()
        
        logger.info("Cache inspection complete")


async def main():
    parser = argparse.ArgumentParser(description='Inspect OATFLAKE embedding cache')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--skip-eviction', action='store_true', help='Skip cache eviction test (which adds many entries)')
    parser.add_argument('--stats-only', action='store_true', help='Only show cache statistics')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    inspector = EmbeddingCacheInspector()
    
    if args.stats_only:
        inspector.inspect_cache_stats()
        inspector.inspect_cache_contents()
    else:
        await inspector.run_all_tests(skip_eviction=args.skip_eviction)


if __name__ == "__main__":
    asyncio.run(main())
