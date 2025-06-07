"""
Search API Interface Benchmark Tool for OATFLAKE.

This script tests the interface between the search functionality and the API endpoints,
focusing on how the embedding cache is integrated with the API layer.

Usage:
    python -m tests.benchmark_search_api_interface

The script simulates frontend client requests to the search API endpoints and measures:
1. Response times for different endpoints (unified search, deeper search)
2. Cache effectiveness across repeated queries
3. API response structure and correctness
4. The end-to-end flow from API to search implementation
"""

import asyncio
import time
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import httpx
import argparse
import statistics
import uuid

# Add the project root to the path so we can import modules properly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

# Create a logger for this test
logger = logging.getLogger("search_api_interface_benchmark")

# Default API server configuration
DEFAULT_API_URL = "http://localhost:8000"

class APIInterfaceBenchmark:
    def __init__(self, base_url: str = DEFAULT_API_URL):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)  # Longer timeout for search requests
        self.test_queries = [
            "What is machine learning?",
            "How do neural networks work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain the concept of embedding vectors",
            "What are the applications of natural language processing?",
            # Add duplicate queries to test cache effectiveness
            "What is machine learning?",
            "How do neural networks work?"
        ]
        self.results = {
            "unified_search": [],
            "deeper_search": [],
            "summary": {}
        }
        self.pinned_items = []
    
    async def close(self):
        await self.client.aclose()
    
    async def benchmark_unified_search(self):
        """Test the unified search endpoint"""
        logger.info("=" * 80)
        logger.info("TESTING UNIFIED SEARCH API ENDPOINT")
        logger.info("=" * 80)
        
        for i, query in enumerate(self.test_queries):
            logger.info(f"Query {i+1}/{len(self.test_queries)}: '{query}'")
            
            # Call the unified search API
            start_time = time.time()
            
            response = await self.client.post(
                f"{self.base_url}/api/search/unified",
                json={
                    "query": query,
                    "k_reference": 3,
                    "k_content": 3
                }
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract metadata
                from_cache = result.get("metadata", {}).get("embedding_from_cache", False)
                reference_count = result.get("metadata", {}).get("reference_count", 0)
                content_count = result.get("metadata", {}).get("content_count", 0)
                
                cache_status = "CACHE HIT" if from_cache else "CACHE MISS"
                logger.info(f"  Response: {response.status_code} - {duration:.3f}s ({cache_status})")
                logger.info(f"  Results: {reference_count} references, {content_count} content items")
                
                # Store the test result
                self.results["unified_search"].append({
                    "query": query,
                    "duration": duration,
                    "from_cache": from_cache,
                    "reference_count": reference_count,
                    "content_count": content_count,
                    "status_code": response.status_code,
                    "success": True
                })
                
                # Pin the first result for deeper search testing if available
                if (reference_count > 0 or content_count > 0) and len(self.pinned_items) < 3:
                    if "references" in result and result["references"]:
                        item_to_pin = result["references"][0]
                        await self.pin_item(item_to_pin)
                    elif "content" in result and result["content"]:
                        item_to_pin = result["content"][0]
                        await self.pin_item(item_to_pin)
            else:
                logger.error(f"  Error: {response.status_code} - {response.text}")
                self.results["unified_search"].append({
                    "query": query,
                    "duration": duration,
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                })
            
            # Short delay between requests
            await asyncio.sleep(0.5)
        
        # Calculate summary statistics for unified search
        durations = [r["duration"] for r in self.results["unified_search"] if r["success"]]
        cache_hits = [r for r in self.results["unified_search"] if r["success"] and r.get("from_cache", False)]
        cache_misses = [r for r in self.results["unified_search"] if r["success"] and not r.get("from_cache", False)]
        
        self.results["summary"]["unified_search"] = {
            "total_requests": len(self.results["unified_search"]),
            "successful_requests": len([r for r in self.results["unified_search"] if r["success"]]),
            "avg_duration": statistics.mean(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "cache_hits": len(cache_hits),
            "cache_misses": len(cache_misses),
            "avg_cache_hit_duration": statistics.mean([r["duration"] for r in cache_hits]) if cache_hits else 0,
            "avg_cache_miss_duration": statistics.mean([r["duration"] for r in cache_misses]) if cache_misses else 0
        }
        
        logger.info("\nUnified Search Summary:")
        logger.info(f"Total requests: {self.results['summary']['unified_search']['total_requests']}")
        logger.info(f"Successful requests: {self.results['summary']['unified_search']['successful_requests']}")
        logger.info(f"Average duration: {self.results['summary']['unified_search']['avg_duration']:.3f}s")
        logger.info(f"Cache hits: {self.results['summary']['unified_search']['cache_hits']}")
        logger.info(f"Cache misses: {self.results['summary']['unified_search']['cache_misses']}")
        if cache_hits and cache_misses:
            speedup = self.results['summary']['unified_search']['avg_cache_miss_duration'] / self.results['summary']['unified_search']['avg_cache_hit_duration']
            logger.info(f"Cache speedup: {speedup:.1f}x faster with cache")
    
    async def pin_item(self, item: Dict[str, Any]) -> str:
        """Pin a search result item and return its ID"""
        try:
            # Ensure the item has an ID
            if "id" not in item:
                item["id"] = str(uuid.uuid4())
            
            response = await self.client.post(
                f"{self.base_url}/api/search/pin",
                json={
                    "item": item,
                    "item_type": "search_result"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                item_id = result.get("item_id")
                logger.info(f"  Pinned item with ID: {item_id}")
                self.pinned_items.append({"id": item_id, "title": item.get("title", "Untitled")})
                return item_id
            else:
                logger.error(f"  Error pinning item: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            logger.error(f"  Error pinning item: {e}")
            return ""
    
    async def benchmark_deeper_search(self):
        """Test the deeper search endpoint using pinned items"""
        if not self.pinned_items:
            logger.warning("No pinned items available for deeper search test")
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("TESTING DEEPER SEARCH API ENDPOINT")
        logger.info("=" * 80)
        
        # Use a subset of queries for deeper search
        deeper_queries = self.test_queries[:4] + [self.test_queries[0]]  # Add a duplicate to test caching
        
        for i, query in enumerate(deeper_queries):
            logger.info(f"Query {i+1}/{len(deeper_queries)}: '{query}'")
            logger.info(f"  Using {len(self.pinned_items)} pinned items")
            
            # Call the deeper search API
            start_time = time.time()
            
            response = await self.client.post(
                f"{self.base_url}/api/search/deeper",
                json={
                    "query": query,
                    "pinned_item_ids": [item["id"] for item in self.pinned_items],
                    "k": 5
                }
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract metadata
                from_cache = result.get("metadata", {}).get("embedding_from_cache", False)
                total_results = result.get("metadata", {}).get("total_results", 0)
                pinned_items_used = result.get("metadata", {}).get("pinned_items_used", 0)
                
                cache_status = "CACHE HIT" if from_cache else "CACHE MISS"
                logger.info(f"  Response: {response.status_code} - {duration:.3f}s ({cache_status})")
                logger.info(f"  Results: {total_results} items, {pinned_items_used} pinned items used")
                
                # Store the test result
                self.results["deeper_search"].append({
                    "query": query,
                    "duration": duration,
                    "from_cache": from_cache,
                    "total_results": total_results,
                    "pinned_items_used": pinned_items_used,
                    "status_code": response.status_code,
                    "success": True
                })
            else:
                logger.error(f"  Error: {response.status_code} - {response.text}")
                self.results["deeper_search"].append({
                    "query": query,
                    "duration": duration,
                    "success": False,
                    "error": response.text,
                    "status_code": response.status_code
                })
            
            # Short delay between requests
            await asyncio.sleep(0.5)
        
        # Calculate summary statistics for deeper search
        durations = [r["duration"] for r in self.results["deeper_search"] if r["success"]]
        cache_hits = [r for r in self.results["deeper_search"] if r["success"] and r.get("from_cache", False)]
        cache_misses = [r for r in self.results["deeper_search"] if r["success"] and not r.get("from_cache", False)]
        
        self.results["summary"]["deeper_search"] = {
            "total_requests": len(self.results["deeper_search"]),
            "successful_requests": len([r for r in self.results["deeper_search"] if r["success"]]),
            "avg_duration": statistics.mean(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "cache_hits": len(cache_hits),
            "cache_misses": len(cache_misses),
            "avg_cache_hit_duration": statistics.mean([r["duration"] for r in cache_hits]) if cache_hits else 0,
            "avg_cache_miss_duration": statistics.mean([r["duration"] for r in cache_misses]) if cache_misses else 0
        }
        
        logger.info("\nDeeper Search Summary:")
        logger.info(f"Total requests: {self.results['summary']['deeper_search']['total_requests']}")
        logger.info(f"Successful requests: {self.results['summary']['deeper_search']['successful_requests']}")
        logger.info(f"Average duration: {self.results['summary']['deeper_search']['avg_duration']:.3f}s")
        logger.info(f"Cache hits: {self.results['summary']['deeper_search']['cache_hits']}")
        logger.info(f"Cache misses: {self.results['summary']['deeper_search']['cache_misses']}")
        if cache_hits and cache_misses:
            speedup = self.results['summary']['deeper_search']['avg_cache_miss_duration'] / self.results['summary']['deeper_search']['avg_cache_hit_duration']
            logger.info(f"Cache speedup: {speedup:.1f}x faster with cache")
    
    async def verify_cache_propagation(self):
        """Test if cache entries are properly propagated between endpoints"""
        logger.info("\n" + "=" * 80)
        logger.info("TESTING CACHE PROPAGATION BETWEEN ENDPOINTS")
        logger.info("=" * 80)
        
        # Use a fresh query for this test
        test_query = "How does embedding caching improve search performance?"
        
        # First, send to unified search to populate the cache
        logger.info(f"Step 1: Initial unified search query - '{test_query}'")
        unified_start = time.time()
        unified_response = await self.client.post(
            f"{self.base_url}/api/search/unified",
            json={"query": test_query, "k_reference": 3, "k_content": 3}
        )
        unified_duration = time.time() - unified_start
        
        if unified_response.status_code != 200:
            logger.error(f"  Error on initial query: {unified_response.status_code}")
            return
        
        unified_result = unified_response.json()
        unified_cache_hit = unified_result.get("metadata", {}).get("embedding_from_cache", False)
        logger.info(f"  Initial query took {unified_duration:.3f}s (cache hit: {unified_cache_hit})")
        
        # Wait a moment
        await asyncio.sleep(1)
        
        # Now try a deeper search with the same query - should hit cache
        logger.info(f"Step 2: Deeper search with same query - '{test_query}'")
        deeper_start = time.time()
        deeper_response = await self.client.post(
            f"{self.base_url}/api/search/deeper",
            json={
                "query": test_query, 
                "pinned_item_ids": [item["id"] for item in self.pinned_items[:1]] if self.pinned_items else [],
                "k": 3
            }
        )
        deeper_duration = time.time() - deeper_start
        
        if deeper_response.status_code != 200:
            logger.error(f"  Error on deeper search: {deeper_response.status_code}")
            return
            
        deeper_result = deeper_response.json()
        deeper_cache_hit = deeper_result.get("metadata", {}).get("embedding_from_cache", False)
        logger.info(f"  Deeper search took {deeper_duration:.3f}s (cache hit: {deeper_cache_hit})")
        
        # Verify that the second query was faster and used the cache
        logger.info(f"Cache propagation results:")
        logger.info(f"  Original query time: {unified_duration:.3f}s (cache hit: {unified_cache_hit})")
        logger.info(f"  Second query time:   {deeper_duration:.3f}s (cache hit: {deeper_cache_hit})")
        logger.info(f"  Cache shared properly: {deeper_cache_hit}")
        
        if deeper_cache_hit:
            speedup = unified_duration / deeper_duration
            logger.info(f"  Cache speedup: {speedup:.1f}x faster")
        else:
            logger.warning("  Cache not properly shared between endpoints!")
    
    async def run_all_tests(self):
        """Run all benchmark tests"""
        logger.info(f"Starting Search API Interface benchmark tests against {self.base_url}")
        logger.info("=" * 80)
        
        try:
            # Check API availability first
            try:
                response = await self.client.get(f"{self.base_url}/api/health")
                if response.status_code != 200:
                    logger.error(f"API server not available at {self.base_url} (status: {response.status_code})")
                    return
                logger.info(f"API server is available at {self.base_url}")
            except Exception as e:
                logger.error(f"API server not available at {self.base_url}: {e}")
                return
            
            # Run unified search tests
            await self.benchmark_unified_search()
            
            # Run deeper search tests
            await self.benchmark_deeper_search()
            
            # Test cache propagation between endpoints
            await self.verify_cache_propagation()
            
            # Print overall summary
            logger.info("\n" + "=" * 80)
            logger.info("INTERFACE BENCHMARK SUMMARY")
            logger.info("=" * 80)
            
            # Compare unified vs deeper search
            unified_avg = self.results["summary"].get("unified_search", {}).get("avg_duration", 0)
            deeper_avg = self.results["summary"].get("deeper_search", {}).get("avg_duration", 0)
            
            logger.info(f"Unified search average time: {unified_avg:.3f}s")
            logger.info(f"Deeper search average time: {deeper_avg:.3f}s")
            if unified_avg > 0 and deeper_avg > 0:
                logger.info(f"Deeper search penalty: {deeper_avg/unified_avg:.1f}x slower than unified search")
            
            # Analyze cache effectiveness
            unified_cache_speedup = (
                self.results["summary"].get("unified_search", {}).get("avg_cache_miss_duration", 0) / 
                self.results["summary"].get("unified_search", {}).get("avg_cache_hit_duration", 1)
            ) if self.results["summary"].get("unified_search", {}).get("avg_cache_hit_duration", 0) > 0 else 0
            
            deeper_cache_speedup = (
                self.results["summary"].get("deeper_search", {}).get("avg_cache_miss_duration", 0) / 
                self.results["summary"].get("deeper_search", {}).get("avg_cache_hit_duration", 1)
            ) if self.results["summary"].get("deeper_search", {}).get("avg_cache_hit_duration", 0) > 0 else 0
            
            if unified_cache_speedup > 0:
                logger.info(f"Unified search cache speedup: {unified_cache_speedup:.1f}x")
            if deeper_cache_speedup > 0:
                logger.info(f"Deeper search cache speedup: {deeper_cache_speedup:.1f}x")
            
            # Recommendations
            logger.info("\nRECOMMENDATIONS:")
            if unified_cache_speedup > 2:
                logger.info("✓ Cache is working effectively for unified search")
            else:
                logger.info("⚠ Cache may not be working optimally for unified search")
                
            if deeper_cache_speedup > 2:
                logger.info("✓ Cache is working effectively for deeper search")
            else:
                logger.info("⚠ Cache may not be working optimally for deeper search")
            
            # Save results to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"search_api_interface_benchmark_{timestamp}.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Benchmark results saved to {results_file}")
            
        finally:
            # Ensure we close the client
            await self.close()


async def main():
    parser = argparse.ArgumentParser(description='Benchmark search API interface')
    parser.add_argument('--url', default=DEFAULT_API_URL, help='Base URL of the API server')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    benchmark = APIInterfaceBenchmark(base_url=args.url)
    await benchmark.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
